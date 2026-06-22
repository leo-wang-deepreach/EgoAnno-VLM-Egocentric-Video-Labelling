#!/usr/bin/env python3
"""ground_simple.py — LEAN object grounder (sam3py). ONE segmentation source, ONE decision rule.

  STEP 1  SAM3 concept text-seg for each verified object -> all instance masks, named + scored.
          (This is SAM3's native "segment everything"; unlike a point-grid/AMG it also finds TRANSPARENT
           objects, and the masks come pre-named.)
  STEP 2  RULE 1 near-hand filter: keep a mask only if it is AT the pinch, INTERSECTS the hand box, and is
          NOT the wrist/arm (and a sane size).
  STEP 3  LLM names: show Claude the numbered near-hand candidates + the reference sheet; it picks which one
          the hand manipulates and names it (disambiguating visually-similar look-alike objects), or N/A.
          SAFETY NET: if Claude says N/A but a STRONG (>= TRUST_THR) RULE-1 candidate exists, ground it
          anyway (Claude was vetoing correct transparent detections that SAM3 had clearly found).
  STEP 4  The chosen text-seg mask IS the clean mask.

Dropped vs the old pipeline: custom point-grid AMG, depth/FG gate, container-preference, retry loop,
edge-recovery, advisory verify. Easier to debug, more robust.

Run (sam3py): ground_simple.py <outdir> <inventory.json> <prompts.json>
  env: REFS_DIR (refsheet + _refs verified vocab), TRUST_THR (0.6), PINCH (0.045), DBG.
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

HERE = os.path.dirname(os.path.abspath(__file__))
PY310 = "/home/ubuntu/local/.venv/bin/python"
TRUST_THR = float(os.environ.get("TRUST_THR", "0.6"))
PINCH = float(os.environ.get("PINCH", "0.045"))
REFS_DIR = os.environ.get("REFS_DIR", "")
DBG = os.environ.get("DBG", "")
ZOOM = os.environ.get("ZOOM", "1") != "0"               # STEP 3.5: re-seg the chosen name on a zoomed crop
COLORS = [(80, 255, 80), (80, 160, 255), (255, 255, 60), (255, 80, 255), (80, 255, 255), (255, 160, 80)]
_DBGF = None


def _dbg(m):
    if _DBGF:
        _DBGF.write(m + "\n"); _DBGF.flush()


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def _on_hand(m, hand_mask, thr=0.70):
    """True if >thr of the candidate mask sits on the segmented hand -> it's a hand/finger mask, not the object."""
    if hand_mask is None:
        return False
    a = float(m.sum())
    return a > 0 and float(np.logical_and(m, hand_mask).sum()) / a > thr


def _clean(m, ax, ay):
    """Single clean blob: keep only the connected component at the grasp anchor (the held object), then
    close small gaps. SAM point-prompts on thin objects in clutter return scattered multi-part masks; this
    drops the stray pieces and (bonus) isolates the held instance from a text-seg mask that merged a pile."""
    mm = m.astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mm, 8)
    if n > 2:
        ai = max(0, min(m.shape[0] - 1, int(ay))); aj = max(0, min(m.shape[1] - 1, int(ax)))
        sl = int(lab[ai, aj])
        if sl == 0:                                   # anchor not inside any component -> nearest centroid
            sl = min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
        mm = (lab == sl).astype(np.uint8)
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mm.astype(bool)


def _frac_in(a, b):
    """Fraction of mask a that lies inside mask/region b."""
    sa = float(a.sum())
    return float(np.logical_and(a, b).sum()) / sa if sa else 0.0


def _pinch_region(pts, W, H, dilate=10):
    """Filled bool mask of the convex hull of the thumb+index FINGER-LINE keypoints (dilated): the pinch
    region BETWEEN the two fingers where a small precision-held object sits. None if too few valid pts."""
    pts = [p for p in pts if p and len(p) == 2]
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(np.array(pts, np.int32))
    reg = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(reg, hull, 1)
    if dilate > 0:
        reg = cv2.dilate(reg, np.ones((dilate, dilate), np.uint8))
    return reg.astype(bool)


def _ti_band_score(mk, tp, ip):
    """Fraction of mask pixels inside the THUMB<->INDEX band — the gap where a pinched object sits.
    Prioritizes (and lightly filters) candidates: the held object is between the two fingers, while
    finger/forearm/background blobs are not."""
    if not (tp and ip):
        return 0.0
    ys, xs = np.nonzero(mk)
    if len(xs) == 0:
        return 0.0
    ax, ay = float(tp[0]), float(tp[1]); bx, by = float(ip[0]), float(ip[1])
    vx, vy = bx - ax, by - ay; L2 = vx * vx + vy * vy
    if L2 < 1.0:
        d = ((xs - ax) ** 2 + (ys - ay) ** 2) ** 0.5
    else:
        tt = np.clip(((xs - ax) * vx + (ys - ay) * vy) / L2, 0.0, 1.0)
        d = ((xs - (ax + tt * vx)) ** 2 + (ys - (ay + tt * vy)) ** 2) ** 0.5
    band = max(10.0, 0.6 * (L2 ** 0.5))                # band radius ~ the pinch aperture
    return float((d <= band).mean())


def _split_at_anchor(mask, ax, ay):
    """Separate two TOUCHING instances: erode to break the thin junction between adjacent objects, keep the
    connected component containing the anchor, dilate back, intersect with the original for exact boundaries.
    No-op for a single clean blob (erosion still yields one component) — fixes a held mask that merged the
    adjacent touching part (two parts touching end-to-end)."""
    H, W = mask.shape
    k = max(3, int(round(0.006 * W)))
    er = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8))
    n, lab, _, cents = cv2.connectedComponentsWithStats(er, 8)
    if n <= 2:
        return mask                                          # 0/1 component after erosion -> already single
    ai = max(0, min(H - 1, int(ay))); aj = max(0, min(W - 1, int(ax)))
    sl = int(lab[ai, aj])
    if sl == 0:
        sl = min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
    comp = cv2.dilate((lab == sl).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    out = np.logical_and(comp, mask)
    return out if out.any() else mask


def _skin_mask(rgb, box):
    """Hand region by SKIN COLOR — robust where the keypoint/box detection is fooled by clutter (a metallic
    pile of small elongated parts looks finger-like, so the pose model lands keypoints on it; skin color separates the real
    hand from the silver pile). Adaptive: sample the box bottom-centre (reliably wrist/arm skin) as reference,
    keep pixels close in (Cr,Cb), take the largest connected blob. Returns a full-frame bool mask or None."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    ex0, ey0 = max(0, x0 - 12), max(0, y0 - 12); ex1, ey1 = min(W, x1 + 12), min(H, y1 + 12)
    if ex1 - ex0 < 10 or ey1 - ey0 < 10:
        return None
    ycc = cv2.cvtColor(rgb[ey0:ey1, ex0:ex1], cv2.COLOR_RGB2YCrCb)
    bh, bw = ycc.shape[:2]
    patch = ycc[int(bh * 0.7):bh, int(bw * 0.3):int(bw * 0.7)].reshape(-1, 3)   # arm/wrist = reliable skin
    if len(patch) < 30:
        return None
    ref = np.median(patch, axis=0)
    d = np.abs(ycc[:, :, 1].astype(int) - ref[1]) + np.abs(ycc[:, :, 2].astype(int) - ref[2])
    skin = (d < 18).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin, 8)
    if n <= 1:
        return None
    big = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA])
    if st[big, cv2.CC_STAT_AREA] < 0.05 * skin.size:        # too little skin found -> unreliable
        return None
    out = np.zeros((H, W), bool); out[ey0:ey1, ex0:ex1] = (lab == big)
    return cv2.dilate(out.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def call_llm(imgs, hand, n, names, feedback=""):
    """STEP 3: Claude picks the manipulated candidate # + names it from the vocab (or -1/N/A)."""
    try:
        out = subprocess.run([PY310, os.path.join(HERE, "llm_pick.py"), ",".join(imgs), hand, str(n),
                              ",".join(names), feedback], capture_output=True, text=True, timeout=150)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        return int(r.get("choice", -1)), (r.get("name") or "N/A").strip()
    except Exception as e:
        _dbg(f"  call_llm error: {e}")
        return -1, "N/A"


def call_manip(imgs, hand, k=3):
    """MANIPULATION GATE: broad hold-vs-empty decision (transparent-aware, k-vote). -> (manip, q1, q2, transp)
    where transp = the transparent question (q1) reached majority -> used to STEER naming toward a clear object."""
    try:
        out = subprocess.run([PY310, os.path.join(HERE, "llm_manip.py"), ",".join(imgs), hand, str(k)],
                             capture_output=True, text=True, timeout=240)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        q1 = int(r.get("q1", 0)); kk = int(r.get("k", k)) or k
        return bool(r.get("manip", False)), q1, int(r.get("q2", 0)), (q1 * 2 > kk)
    except Exception as e:
        _dbg(f"  call_manip error: {e}")
        return False, 0, 0, False


def main():
    global _DBGF
    outdir, inv_file, prompts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    if DBG:
        _DBGF = open(os.path.join(outdir, "_dbgtrace.log"), "w")
    inv = json.load(open(inv_file)); tag = inv["tag"]
    names = [o["name"] if isinstance(o, dict) else o for o in inv["objects"]
             if not (isinstance(o, dict) and o.get("role") == "fixture")]
    all_names = list(names)            # FULL inventory — used for NAMING (so objects SAM couldn't verify,
    #                                    e.g. thin pen parts, can still be named on a point-prompt mask)
    refsheet = ""
    if REFS_DIR:
        rs = os.path.abspath(os.path.join(REFS_DIR, f"_refsheet_{tag}.png"))
        refsheet = rs if os.path.exists(rs) else ""
        rm = os.path.join(REFS_DIR, f"_refs_{tag}.json")
        if os.path.exists(rm):
            verified = set(json.load(open(rm)).keys())
            kept = [n for n in names if n in verified]
            if kept:
                print(f"{tag}: vocab restricted to SAM-verified: {kept}"); names = kept
    pj = json.load(open(prompts_file)); video = pj["video"]; W, H = pj["W"], pj["H"]
    model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    cap = cv2.VideoCapture(video); tmpd = tempfile.mkdtemp()
    by_t = {}
    for p in pj["prompts"]:
        by_t.setdefault(round(p["t"], 3), []).append(p)

    def _anchor_of(p):
        """Reproduce the main loop's grasp anchor / box / pinch-window for a prompt (for re-seg + re-render)."""
        x0, y0, x1, y1 = p["box_px"]
        gx, gy = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
        tpx, ipx = p.get("thumb_px"), p.get("index_px")
        mid = ((int((tpx[0] + ipx[0]) / 2), int((tpx[1] + ipx[1]) / 2)) if (tpx and ipx) else (gx, gy))
        return float(mid[0]), float(mid[1]), (x0, y0, x1, y1), max(20, int(PINCH * W)), mid

    def _reseg(rgb2, nm, ax, ay, box, pw):
        """TEMPORAL: text-seg a carried NAME at the primary frame; return the at-grasp blob (mask, sam, dist)
        or None when the object can't be re-segmented here (occluded — identity still carried by the caller)."""
        x0, y0, x1, y1 = box
        yl, yh = max(0, int(ay) - pw), min(H, int(ay) + pw + 1)
        xl, xh = max(0, int(ax) - pw), min(W, int(ax) + pw + 1)
        with torch.inference_mode(), ac:
            st = proc.set_image(Image.fromarray(rgb2))
            out = proc.set_text_prompt(state=st, prompt=nm.split(" with ")[0].strip())
        tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
        if tm.ndim == 4:
            tm = tm[:, 0]
        best = None
        for k in range(len(tm) if tm.ndim == 3 else 0):
            mk = tm[k] > 0.5
            if mk.ndim > 2:
                mk = mk.squeeze()
            if mk.shape != (H, W) or not mk.any():
                continue
            mk = _clean(mk, ax, ay)
            if (not mk.any() or not (0.0005 < float(mk.mean()) < 0.40)
                    or not mk[yl:yh, xl:xh].any() or mk[y0:y1, x0:x1].sum() == 0):
                continue
            ys, xs = np.nonzero(mk)
            d = (((xs.mean() - ax) ** 2 + (ys.mean() - ay) ** 2) ** 0.5) / max(pw, 1)
            if d > 1.2:                                  # must be AT the grasp (same gate as selection)
                continue
            if best is None or d < best[2]:
                best = (mk, float(ts[k]), d)
        return best

    def _render(rgb2, mask, p, cap_txt, path):
        """Re-render a primary frame with a carried mask + caption (matches the main-loop overlay style)."""
        vis = rgb2.copy()
        if mask is not None:
            f = vis.astype(np.float32); f[mask] = 0.5 * f[mask] + 0.5 * np.array(COLORS[0], np.float32)
            vis = f.clip(0, 255).astype(np.uint8)
            cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cont, -1, COLORS[0], 2)
        x0, y0, x1, y1 = p["box_px"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1)
        _, _, _, _, mid = _anchor_of(p)
        cv2.circle(vis, (int(mid[0]), int(mid[1])), 4, (255, 140, 0), -1)
        cv2.circle(vis, (int(mid[0]), int(mid[1])), 10, (60, 255, 60), 2)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _zoom_reseg(rgb2, nm, gx, gy, ax, ay, hand_mask):
        """STEP 3.5 — ZOOMED TEXT-SEG. Re-segment the CHOSEN name on a TIGHT upscaled crop around the grasp
        (0.12·W, anchor-centred so the held object is in frame). A small/occluded object is a few pixels at
        full-frame scale (ragged mask) but fills far more of SAM's input when the crop is upscaled, so text-seg
        of its name returns a CLEAN mask. Returns (full-frame mask, sam) or None (keep the original mask)."""
        rad = int(0.12 * W)
        cx0, cy0 = max(0, gx - rad), max(0, gy - rad); cx1, cy1 = min(W, gx + rad), min(H, gy + rad)
        crop = rgb2[cy0:cy1, cx0:cx1]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        f = max(1, int(512 / max(crop.shape[:2])))            # upscale the crop to ~512px
        up = cv2.resize(crop, (crop.shape[1] * f, crop.shape[0] * f), interpolation=cv2.INTER_CUBIC)
        Hu, Wu = up.shape[:2]; acx, acy = (ax - cx0) * f, (ay - cy0) * f   # anchor in upscaled-crop coords
        with torch.inference_mode(), ac:
            st = proc.set_image(Image.fromarray(up))
            out = proc.set_text_prompt(state=st, prompt=nm.split(" with ")[0].strip())
        tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
        if tm.ndim == 4:
            tm = tm[:, 0]
        best = None
        for k in range(len(tm) if tm.ndim == 3 else 0):
            mk = tm[k] > 0.5
            if mk.ndim > 2:
                mk = mk.squeeze()
            if mk.shape != (Hu, Wu) or not mk.any() or not (0.0003 < float(mk.mean()) < 0.6):
                continue
            ys, xs = np.nonzero(mk); d = ((xs.mean() - acx) ** 2 + (ys.mean() - acy) ** 2) ** 0.5   # nearest the grasp
            if best is None or d < best[2]:
                best = (mk, float(ts[k]), d)
        if best is None:
            return None
        small = cv2.resize(best[0].astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((H, W), bool); full[cy0:cy1, cx0:cx1] = small.astype(bool)
        full = _clean(full, ax, ay)
        # NO off-hand guard here: this is TEXT-SEG of a NAMED object (returns the object, not the hand), and a
        # small object held in a precision pinch legitimately overlaps the hand silhouette. The name + the
        # nearest-anchor selection + _clean already localize it; an off-hand reject would kill the real held part.
        if not full.any() or not (0.0002 < float(full.mean()) < 0.40):
            return None
        return full, best[1]

    index = []
    prim_prompt = {}                                     # seg_id -> the primary prompt (for temporal re-seg)
    for t, plist in sorted(by_t.items()):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), ac:
            state = proc.set_image(Image.fromarray(rgb))
            segs = []                                            # STEP 1: (name, mask, score) per verified object
            for nm in names:
                out = proc.set_text_prompt(state=state, prompt=nm.split(" with ")[0].strip())
                tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
                if tm.ndim == 4:
                    tm = tm[:, 0]
                for k in range(len(tm) if tm.ndim == 3 else 0):
                    mk = tm[k] > 0.5
                    if mk.ndim > 2:
                        mk = mk.squeeze()
                    if mk.any() and mk.shape == (H, W):
                        segs.append((nm, mk, float(ts[k])))

        for p in plist:
            gx, gy = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
            x0, y0, x1, y1 = p["box_px"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            wpx, wpy = _clamp(2 * cx - gx, x0, x1), _clamp(2 * cy - gy, y0, y1)
            hand = "LEFT" if p["hand"] == "L" else "RIGHT"
            pw = max(20, int(PINCH * W)); yl, yh = max(0, gy - pw), min(H, gy + pw + 1)
            xl, xh = max(0, gx - pw), min(W, gx + pw + 1)

            # --- HAND MASK + SEED RELOCATION ---------------------------------------------------
            # The red pinch point is the midpoint of the fingertips, so on a top-down PRECISION grasp it
            # lands on the finger surface OCCLUDING the small held object -> a point-prompt there segments
            # the HAND. So: (1) SAM-segment this hand from its keypoints; (2) RELOCATE the seed onto the
            # nearest visible non-hand pixel at the grasp (the object peeking between the fingers);
            # (3) REJECT any candidate mask that still mostly sits on the hand.
            hull = np.array(p.get("hull_px", []), float).reshape(-1, 2) if p.get("hull_px") else None
            tpx, ipx = p.get("thumb_px"), p.get("index_px")      # thumb-tip kp4, index-tip kp8 (object in the gap)
            # HAND MASK = SAM keypoint-segmentation (skin-mask-as-guard was tried but over-rejects small objects
            # held BETWEEN skin fingers; _skin_mask is kept for keypoint VALIDATION, not the guard).
            hand_mask = None
            if hull is not None and len(hull) >= 3:
                with torch.inference_mode(), ac:
                    hm, hs, _ = model.predict_inst(state, point_coords=hull.astype(np.float32),
                                                   point_labels=np.ones(len(hull), int), multimask_output=True)
                hm = _np(hm); hs = _np(hs).ravel()
                cxi, cyi = _clamp(cx, 0, W - 1), _clamp(cy, 0, H - 1)
                for i in range(len(hs) if hm.ndim == 3 else 0):
                    m = hm[i] > 0.5
                    if m.ndim > 2:
                        m = m.squeeze()
                    if m.shape == (H, W) and m[cyi, cxi] and 0.003 < float(m.mean()) < 0.30:
                        if hand_mask is None or m.sum() > hand_mask.sum():
                            hand_mask = m

            # ============ UNIFIED POOL + balanced MANIPULATION-CONFIDENCE selection ============
            # Segment the held object THREE ways — TEXT-SEG (by name), the RED pinch point, the ORANGE
            # thumb-index midpoint — then score EVERY mask with ONE balanced "manipulation" confidence so the
            # sources are comparable. Geometry (at the grasp, off the hand) dominates so text-seg's higher raw
            # SAM score doesn't auto-win; we keep the highest-manip mask and gate out far (pile/background) ones.
            # grasp anchor = thumb-index midpoint (single noisy keypoint dropped: the all-finger pinch).
            mid = ((int((tpx[0] + ipx[0]) / 2), int((tpx[1] + ipx[1]) / 2)) if (tpx and ipx) else (gx, gy))
            shared = False
            ax, ay = float(mid[0]), float(mid[1])              # robust grasp anchor
            # ANCHOR-FIX (bad keypoints): if the thumb-index anchor is splayed FAR from the hand's actual
            # SKIN (an elongated-clutter pile fooled the pose model), relocate it onto the nearest hand-skin
            # pixel so candidates come from the REAL grasp, not random pile parts. DISTANCE-gated, so a normal
            # grasp (anchor in the small finger gap, ~object-width from skin) is NOT moved.
            _skin = _skin_mask(rgb, (x0, y0, x1, y1))
            if _skin is not None:
                _dts = cv2.distanceTransform((~_skin).astype(np.uint8), cv2.DIST_L2, 3)
                if _dts[_clamp(ay, 0, H - 1), _clamp(ax, 0, W - 1)] > 0.05 * W:        # far from any skin -> displaced
                    sy_, sx_ = np.nonzero(_skin)
                    i_ = int(np.argmin((sx_ - ax) ** 2 + (sy_ - ay) ** 2))
                    ax, ay = float(sx_[i_]), float(sy_[i_])
                    gx, gy = _clamp(ax, 0, W - 1), _clamp(ay, 0, H - 1)               # re-centre crop/seeds/window
                    wpx, wpy = _clamp(2 * cx - gx, x0, x1), _clamp(2 * cy - gy, y0, y1)
                    yl, yh = max(0, gy - pw), min(H, gy + pw + 1); xl, xh = max(0, gx - pw), min(W, gx + pw + 1)
                    _dbg(f"[ANCHOR-FIX] {tag} t={t:.1f} {hand}: anchor splayed into clutter -> relocated to skin ({gx},{gy})")
            palmc = hull.mean(0) if (hull is not None and len(hull)) else (cx, cy)
            neg_all = [[wpx, wpy], [_clamp(palmc[0], 0, W - 1), _clamp(palmc[1], 0, H - 1)]]
            for q in plist:                                    # the OTHER hand as a NEGATIVE (don't grab it)
                if q is not p and q.get("hull_px"):
                    oarr = np.array(q["hull_px"], float)
                    neg_all.append([int(oarr[:, 0].mean()), int(oarr[:, 1].mean())])
            GATE = float(os.environ.get("ANCHOR_GATE", "0.7"))

            def _manip(mk, sam):                              # balanced manipulation confidence (0..1)
                ys4, xs4 = np.nonzero(mk)
                dist = (((xs4.mean() - ax) ** 2 + (ys4.mean() - ay) ** 2) ** 0.5) / max(pw, 1)
                oh = (float(np.logical_and(mk, hand_mask).sum()) / float(mk.sum())) if hand_mask is not None else 0.0
                area = float(mk.mean())
                sized = 1.0 if area < 0.12 else max(0.0, 1.0 - (area - 0.12) / 0.18)
                prox = max(0.0, 1.0 - dist / 1.2)
                return 0.42 * prox + 0.30 * (1.0 - oh) + 0.12 * sized + 0.16 * float(sam), dist

            def _seg_at(sx, sy):                               # SAM point-prompt + single-blob cleanup + validity
                with torch.inference_mode(), ac:
                    pm, ps, _ = model.predict_inst(state, point_coords=np.array([[int(sx), int(sy)]] + neg_all, float),
                                                   point_labels=np.array([1] + [0] * len(neg_all), int),
                                                   multimask_output=True)
                pm = _np(pm); ps = _np(ps).ravel(); res = []
                for i in range(len(ps) if pm.ndim == 3 else 0):
                    m = pm[i] > 0.5
                    if m.ndim > 2:
                        m = m.squeeze()
                    if m.shape != (H, W) or not m.any():
                        continue
                    m = _clean(m, ax, ay)
                    if (not m.any() or m[wpy, wpx] or not m[yl:yh, xl:xh].any() or m[y0:y1, x0:x1].sum() == 0
                            or not (0.0002 < float(m.mean()) < 0.40) or _on_hand(m, hand_mask)):
                        continue
                    res.append((m, float(ps[i])))
                return res

            # PINCH REGION = the gap BETWEEN the two finger lines (thumb kp1-4, index kp5-8), where a
            # small precision-held object sits. Used to KEEP small-object candidates (below).
            tchain = list(p.get("thumb_chain_px") or []); ichain = list(p.get("index_chain_px") or [])
            # Acceptance region around the orange grasp anchor / finger lines. A LITTLE BIGGER than before so a
            # near point mask that the tight region rejected still counts — safe now that the
            # manipulation gate (not this filter) decides N/A, so a bigger region can't cause false-grounds.
            pinch_region = _pinch_region(tchain + ichain, W, H, dilate=max(16, int(0.30 * pw)))
            N_SEED = int(os.environ.get("N_SEED", "10"))
            PIN_THR = float(os.environ.get("PIN_THR", "0.25"))      # point mask must OVERLAP the pinch region (>=25%)

            pool = []                                          # (name, mask, sam, src, manip, dist)
            for nm, mk, sc in segs:                            # (1) TEXT-SEG: clean named masks (by object name)
                mk = _clean(mk, ax, ay)
                if (not mk.any() or mk[wpy, wpx] or not mk[yl:yh, xl:xh].any() or mk[y0:y1, x0:x1].sum() == 0
                        or not (0.0005 < float(mk.mean()) < 0.40) or _on_hand(mk, hand_mask)):
                    continue
                mn, dd = _manip(mk, sc); pool.append((nm, mk, sc, "TS", mn, dd))
            # (2) POINT-SEG: segment AT the orange dot + RANDOM points around it, re-segment. Keep a point mask
            # if it OVERLAPS the thumb<->index pinch region and is OFF the hand. We DON'T drop ones that match a
            # text mask — the VLM is SHOWN both the text-seg and point-seg versions and CHOOSES the better one.
            seed_pts = [(int(ax), int(ay))]
            if mid is not None:
                ap = (((tpx[0] - ipx[0]) ** 2 + (tpx[1] - ipx[1]) ** 2) ** 0.5) if (tpx and ipx) else float(pw)
                sigp = max(6.0, min(0.6 * ap, 0.5 * pw))       # random spread ~ the pinch aperture
                rng = np.random.default_rng(int(ax) * 100003 + int(ay))   # deterministic per frame
                for _ in range(N_SEED):
                    dx, dy = rng.normal(0.0, sigp, 2)
                    seed_pts.append((_clamp(ax + dx, 0, W - 1), _clamp(ay + dy, 0, H - 1)))
            seen_xy, n_kept, rej = set(), 0, {"pin": 0, "hand": 0}
            for sx, sy in seed_pts:
                if (sx, sy) in seen_xy:
                    continue
                seen_xy.add((sx, sy))
                for m, s in _seg_at(sx, sy):
                    if pinch_region is not None:
                        if _frac_in(m, pinch_region) < PIN_THR:              # OVERLAPS the finger-line gap
                            rej["pin"] += 1; continue
                    elif tpx and ipx and _ti_band_score(m, tpx, ipx) < PIN_THR:  # fallback when chains missing
                        rej["pin"] += 1; continue
                    if hand_mask is not None and _on_hand(m, hand_mask, thr=0.40):   # off the hand
                        rej["hand"] += 1; continue
                    mn, dd = _manip(m, s); pool.append(("", m, s, "POINT", mn, dd)); n_kept += 1
            _dbg(f"[POINTSEG] {tag} t={t:.1f} {hand}: seeds={len(seen_xy)} region={'Y' if pinch_region is not None else 'N'} kept={n_kept} rej={rej}")
            pool = [c for c in pool if c[5] <= GATE]           # GATE: drop far-from-grasp (pile/background)
            if not pool and hand_mask is not None:             # (3) ON-HAND RECOVERY: re-seed nearest NON-hand blob
                sub = (~hand_mask)[y0:y1, x0:x1].astype(np.uint8)
                nC, _, statsC, centsC = cv2.connectedComponentsWithStats(sub, 8)
                blob = None
                for i in range(1, nC):
                    a = int(statsC[i, cv2.CC_STAT_AREA]); cxg = centsC[i][0] + x0; cyg = centsC[i][1] + y0
                    d = (((cxg - ax) ** 2 + (cyg - ay) ** 2) ** 0.5) / max(pw, 1)
                    if a < 40 or d > GATE:                     # skip tiny specks / far-from-grasp blobs
                        continue
                    s_ = a / (d + 0.3)                         # prefer a near, sizable non-hand region (object peeking)
                    if blob is None or s_ > blob[0]:
                        blob = (s_, int(cxg), int(cyg))
                if blob:
                    for m, s in _seg_at(blob[1], blob[2]):
                        mn, dd = _manip(m, s)
                        if dd <= GATE:
                            pool.append(("", m, s, "RECOVER", mn, dd))
                    _dbg(f"[RECOVER] {tag} t={t:.1f} {hand}: reseed=({blob[1]},{blob[2]}) -> cands={len(pool)}")
            pool.sort(key=lambda c: -c[4])                     # highest manipulation confidence first
            cands = []                                         # dedup near-identical masks, keep higher manip
            for c in pool:
                if all(_iou(c[1], k[1]) < 0.8 for k in cands):
                    cands.append(c)
            cands = cands[:5]                                  # cap for a clean chooser panel
            _dbg(f"[POOL] {tag} t={t:.1f} {hand} anchor=({int(ax)},{int(ay)}) shared={shared}: "
                 + " | ".join(f"{c[3]}{(':' + c[0][:10]) if c[0] else ''} manip={c[4]:.2f} d={c[5]:.2f} sam={c[2]:.2f}"
                              for c in pool))
            seed_viz = [(sx, sy, (255, 140, 0)) for (sx, sy) in seed_pts]   # orange = all multi-seed probe points
            win_seed = (int(ax), int(ay))                      # green ring = grasp anchor
            cand_src = ["T" if c[3] == "TS" else "P" for c in cands]   # T=text-seg, P=point-seg (shown to the VLM)
            cands = [(n, m, s) for n, m, s, src, mn, dd in cands]
            rad = int(0.12 * W)                                # PINCH-CENTERED crop bounds (chooser + DBG panel)
            ex0, ey0 = max(0, gx - rad), max(0, gy - rad); ex1, ey1 = min(W, gx + rad), min(H, gy + rad)

            def _build_imgs(cc, csrc, suffix=""):
                """Chooser images: a CLEAN crop (objects visible, no marks) + the SAME crop with numbered
                candidate outlines (letter: T=text-seg, P=point-seg, Z=zoom). So the VLM sees the real objects
                AND picks the right outline when several candidates overlap the same region."""
                clean = rgb[ey0:ey1, ex0:ex1].copy()
                cv2.circle(clean, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
                ov = rgb[ey0:ey1, ex0:ex1].copy()
                for j, (nm, mk, sc) in enumerate(cc):
                    col = COLORS[j % len(COLORS)]
                    cont, _ = cv2.findContours(mk[ey0:ey1, ex0:ex1].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(ov, cont, -1, col, 2)
                    yx = np.argwhere(mk[ey0:ey1, ex0:ex1])
                    if len(yx):
                        yc, xc = map(int, yx.mean(0))
                        cv2.putText(ov, f"{j}{csrc[j]}", (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)
                cv2.circle(ov, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
                outp = []
                for tn, im in (("c" + suffix, clean), ("o" + suffix, ov)):
                    im = cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2))
                    pth = os.path.join(tmpd, f"{tn}_{tag}_{t:.1f}_{p['hand']}.jpg")
                    cv2.imwrite(pth, cv2.cvtColor(im, cv2.COLOR_RGB2BGR)); outp.append(pth)
                return outp + ([refsheet] if refsheet else [])

            # STEP 3: MANIPULATION GATE (broad hold-vs-empty, transparent-aware, k-vote) -> then NAME.
            # Decide manip FIRST; if yes we MUST ground (pick/force/zoom-recover), if no -> N/A. Replaces the
            # flaky single-pick + geometry auto-accept that over-fired AND under-fired on transparent holds.
            name, conf, chosen, src = "N/A", 0.0, None, ""
            gimgs = _build_imgs(cands, cand_src)               # clean + overlay crops (overlay empty if no cands)
            manip, q1, q2, transp = call_manip(gimgs, hand)
            _dbg(f"[GATE] {tag} t={t:.1f} {hand}: manip={manip} (q1_transparent={q1} q2_object={q2} transp={transp})")
            # LEVER 1: when the gate votes TRANSPARENT, steer naming to a clear object (fixes labelling a clear
            # a clear vessel as an opaque object — the transparent signal otherwise never reaches naming).
            _tn = (" The object is TRANSPARENT / clear see-through — name it as a CLEAR/transparent object from "
                   "the list, NOT an opaque one.") if transp else ""
            choice = -1
            if manip and cands:
                choice, nm = call_llm(gimgs, hand, len(cands), all_names,
                                      "The hand IS manipulating an object here. Pick which numbered candidate it is "
                                      "manipulating and NAME it from the list. Do NOT answer N/A." + _tn)
                _dbg(f"[PICK] {tag} t={t:.1f} {hand}: choice={choice} name={nm!r}")
                if 0 <= choice < len(cands):
                    cnm = nm if (nm and nm.upper() not in ("N/A", "NA")) else cands[choice][0]
                    if cnm:
                        name, chosen, conf, src = cnm, cands[choice][1], cands[choice][2], "llm"
                # FORCE-PICK: the gate already decided the hand IS manipulating, so do NOT fall back to N/A. If the
                # VLM still vetoed every candidate, take the top one (highest manip score) and name it (forced).
                if chosen is None and cands:
                    tn, tm, ts_ = cands[0]; cnm = tn
                    if not cnm:                                  # unnamed point candidate -> force a name from the list
                        _, fn2 = call_llm(gimgs, hand, len(cands), all_names,
                                          f"The {hand} hand IS manipulating the highlighted object — NAME it from the "
                                          "list; N/A is NOT allowed." + _tn)
                        cnm = fn2 if (fn2 and fn2.upper() not in ("N/A", "NA")) else None
                    if cnm:
                        name, chosen, conf, src, choice = cnm, tm, ts_, "gate-pick", 0
                        _dbg(f"[FORCE] {tag} t={t:.1f} {hand}: gate=manip, VLM vetoed -> top cand {cnm!r} (sam={ts_:.2f})")
                if DBG:                                              # CANDIDATE PANEL: every candidate + source
                    panel = rgb[ey0:ey1, ex0:ex1].copy(); rep = []
                    for j, (nm2, mk2, sc2) in enumerate(cands):
                        col = COLORS[j % len(COLORS)]
                        sub = mk2[ey0:ey1, ex0:ex1].astype(np.uint8)
                        cont, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(panel, cont, -1, col, 2)
                        ys3, xs3 = np.nonzero(mk2)
                        dcc = (((xs3.mean() - ax) ** 2 + (ys3.mean() - ay) ** 2) ** 0.5) / max(pw, 1)
                        oh = (float(np.logical_and(mk2, hand_mask).sum()) / float(mk2.sum())
                              if hand_mask is not None else 0.0)
                        srcj = "TEXT" if nm2 else "POINT"; star = "*" if j == choice else ""
                        rep.append(f"{j}{star}:{srcj} s={sc2:.2f} d={dcc:.2f} oh={oh:.2f}")
                        yx = np.argwhere(sub)
                        if len(yx):
                            yc, xc = map(int, yx.mean(0))
                            cv2.putText(panel, f"{j}{cand_src[j]}{star}", (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
                    cv2.circle(panel, (_clamp(int(ax) - ex0, 0, panel.shape[1] - 1),
                                       _clamp(int(ay) - ey0, 0, panel.shape[0] - 1)), 6, (255, 140, 0), -1)
                    panel = cv2.resize(panel, (panel.shape[1] * 2, panel.shape[0] * 2))
                    cv2.imwrite(os.path.join(outdir, f"{tag}_t{p['t']:05.1f}_{p['hand']}_cand.png"),
                                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    _dbg(f"[CAND] {tag} t={t:.1f} {hand} choice={choice}: " + " | ".join(rep))

            # RECOVERY: the gate says the hand IS manipulating but nothing grounded (empty pool / unnamed). The
            # object may be small/occluded/transparent and only findable ZOOMED. Text-seg each inventory name on
            # the upscaled crop -> clean candidates at the grasp -> VLM picks (gate already said manip, so ground).
            if manip and chosen is None:
                rc = []
                for nm0 in all_names:
                    zr0 = _zoom_reseg(rgb, nm0, gx, gy, ax, ay, hand_mask)
                    if zr0 is not None:
                        rc.append((nm0, zr0[0], float(zr0[1])))
                rc2 = []
                for c in sorted(rc, key=lambda c: -c[2]):
                    if all(_iou(c[1], kk[1]) < 0.8 for kk in rc2):
                        rc2.append(c)
                rc2 = rc2[:5]
                if rc2:
                    rcs = [(n, m, s) for n, m, s in rc2]; rsrc = ["Z"] * len(rcs)
                    ch, nm2 = call_llm(_build_imgs(rcs, rsrc, "r"), hand, len(rcs), all_names,
                                       "The hand IS manipulating an object. These are ZOOMED segmentations of "
                                       "candidates at the grasp — pick the one it manipulates and NAME it (no N/A)." + _tn)
                    pick = ch if (0 <= ch < len(rcs)) else 0   # gate=manip -> fall back to the best zoom candidate
                    cnm = nm2 if (nm2 and nm2.upper() not in ("N/A", "NA")) else rcs[pick][0]
                    _dbg(f"[RECOVER-ZOOM] {tag} t={t:.1f} {hand}: {len(rcs)} cands -> choice={ch} name={nm2!r} -> {cnm!r}")
                    if cnm:
                        name, chosen, conf, src = cnm, rcs[pick][1], round(rcs[pick][2], 3), "zoom-recover"

            # STEP 3.5: ZOOMED TEXT-SEG RE-SEGMENT — refine the chosen mask on a tight upscaled crop (helps
            # small/occluded parts). The result is the POINT-side mask (P) for the mask chooser below.
            if ZOOM and chosen is not None and "zoom" not in src and name and name.upper() not in ("N/A", "NA"):
                zr = _zoom_reseg(rgb, name, gx, gy, ax, ay, hand_mask)
                if zr is not None:
                    chosen, conf = zr[0], round(float(zr[1]), 3); src = (src + "+zoom") if src else "zoom"
                    _dbg(f"[ZOOM] {tag} t={t:.1f} {hand}: re-seg '{name}' -> clean zoomed mask (sam={zr[1]:.2f})")
                else:
                    _dbg(f"[ZOOM] {tag} t={t:.1f} {hand}: '{name}' not re-found in zoom crop -> kept original mask")

            # MASK CHOOSER (user: keep BOTH point-seg and text-seg, then choose the more accurate). Build the
            # full-frame TEXT-SEG mask of the named object at the grasp (T); if it differs from the current
            # point/zoom mask (P), let the VLM pick whichever outline most accurately + completely fits the object.
            if chosen is not None and name and name.upper() not in ("N/A", "NA"):
                base = name.split(" with ")[0].strip().lower()
                ts_alt = None
                for nm_s, mk_s, sc_s in segs:
                    if base not in nm_s.lower() and nm_s.lower() not in base:
                        continue
                    mks = _clean(mk_s, ax, ay)
                    if (not mks.any() or mks[wpy, wpx] or not mks[yl:yh, xl:xh].any()
                            or mks[y0:y1, x0:x1].sum() == 0 or not (0.0005 < float(mks.mean()) < 0.40)):
                        continue
                    ys_s, xs_s = np.nonzero(mks)
                    d_s = (((xs_s.mean() - ax) ** 2 + (ys_s.mean() - ay) ** 2) ** 0.5) / max(pw, 1)
                    if d_s > 1.2:
                        continue
                    if ts_alt is None or sc_s > ts_alt[1]:
                        ts_alt = (mks, float(sc_s))
                if ts_alt is not None and _iou(ts_alt[0], chosen) < 0.8:
                    two = [(name, chosen, conf), (name, ts_alt[0], ts_alt[1])]
                    ch2, _ = call_llm(_build_imgs(two, ["P", "T"], "m"), hand, 2, all_names,
                                      f"Both outlines mark the SAME object ('{name}'): 0P is a point-prompt mask, "
                                      "1T is a text-segmentation mask. Pick the NUMBER whose outline most ACCURATELY "
                                      "and COMPLETELY fits that object — cleanest and full extent, NOT collapsed onto "
                                      "its contents and NOT spilled onto background/hand.")
                    if ch2 == 1:
                        chosen, conf, src = ts_alt[0], round(ts_alt[1], 3), (src + "+T") if src else "ts"
                        _dbg(f"[MASKPICK] {tag} t={t:.1f} {hand}: chose TEXT-seg mask for '{name}' (sam={ts_alt[1]:.2f})")
                    else:
                        _dbg(f"[MASKPICK] {tag} t={t:.1f} {hand}: kept point/zoom mask for '{name}'")

            if chosen is not None:                             # drop an adjacent touching instance the mask merged
                chosen = _split_at_anchor(chosen, ax, ay)
            vis = rgb.copy(); locked = chosen is not None
            if locked:
                f = vis.astype(np.float32); f[chosen] = 0.5 * f[chosen] + 0.5 * np.array(COLORS[0], np.float32)
                vis = f.clip(0, 255).astype(np.uint8)
                cont, _ = cv2.findContours(chosen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, COLORS[0], 2)
                cap_txt = f"{tag} {hand}: {name}  (conf {conf:.2f}, {src})"
            else:
                cap_txt = f"{tag} {hand}: N/A"
            cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1)
            if pinch_region is not None:                               # cyan = thumb<->index pinch region (small-obj keep)
                pc, _ = cv2.findContours(pinch_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, pc, -1, (0, 220, 220), 1)
            for sxv, syv, col in seed_viz:                             # orange = random probe seeds
                cv2.circle(vis, (int(sxv), int(syv)), 3, col, -1)
            if win_seed is not None:                                   # green ring = grasp anchor (thumb-index)
                cv2.circle(vis, (int(win_seed[0]), int(win_seed[1])), 10, (60, 255, 60), 2)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            sid = p.get("seg_id", f"s00{p['hand']}"); is_prim = bool(p.get("primary", True))
            fn = f"{tag}_{sid}_t{p['t']:05.1f}_{p['hand']}_{'P' if is_prim else 'N'}.png"
            cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            row = {"file": fn, "tag": tag, "hand": hand, "t": p["t"], "seg_id": sid, "primary": is_prim,
                   "obj": bool(locked), "name": name if locked else "N/A", "conf": round(conf, 3),
                   "low_conf": bool(locked and conf < TRUST_THR), "src": src, "_mask": chosen}
            index.append(row)
            if is_prim:
                prim_prompt[sid] = p
            print(cap_txt, flush=True)

    # ===================== LAYER-2 TEMPORAL CARRY =====================
    # A PRIMARY frame that came back N/A is often a single-frame failure (Mode-2 empty pool, or the bad-keypoint
    # pile splaying the pose) while a NEIGHBOR frame in the SAME segment grounded the held object cleanly. Carry
    # that neighbor's IDENTITY onto the primary, and re-segment the name AT the primary frame for a clean mask
    # (fall back to the neighbor's mask if it's occluded here). Conservative: only FILLS false-N/A primaries —
    # never overrides a confident primary — so recall rises without trading away precision.
    by_seg = {}
    for r in index:
        by_seg.setdefault(r["seg_id"], []).append(r)
    carried = 0
    for sid, rows in by_seg.items():
        prim = next((r for r in rows if r["primary"]), None)
        if prim is None or prim["obj"] or sid not in prim_prompt:
            continue
        nbrs = [r for r in rows if not r["primary"] and r["obj"] and r["conf"] >= TRUST_THR]
        if not nbrs:
            continue
        nbrs.sort(key=lambda r: (r["src"] != "llm", -r["conf"], abs(r["t"] - prim["t"])))
        nb = nbrs[0]; cname = nb["name"]; p = prim_prompt[sid]
        ax, ay, box, pw, _ = _anchor_of(p)
        cap.set(cv2.CAP_PROP_POS_MSEC, prim["t"] * 1000.0); ok, bgr = cap.read()
        cmask, cconf = None, nb["conf"]
        if ok:
            rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            best = _reseg(rgb2, cname, ax, ay, box, pw)
            if best is not None:
                cmask, cconf = best[0], best[1]
        if cmask is None:
            cmask = nb.get("_mask")                       # occluded at primary -> approximate w/ neighbor mask
        prim.update(obj=True, name=cname, conf=round(float(cconf), 3), src="temporal",
                    low_conf=bool(cconf < TRUST_THR), _mask=cmask)
        if ok:
            hnd = "LEFT" if p["hand"] == "L" else "RIGHT"
            _render(rgb2, cmask, p, f"{tag} {hnd}: {cname}  (conf {cconf:.2f}, temporal)",
                    os.path.join(outdir, prim["file"]))
        carried += 1
        _dbg(f"[TEMPORAL] {tag} {sid} {p['hand']} t={prim['t']:.1f}: false-N/A <- neighbor t={nb['t']:.1f} "
             f"'{cname}' (conf {nb['conf']:.2f} src {nb['src']})")
    if carried:
        print(f"{tag}: temporal carry filled {carried} false-N/A primary frame(s)")
    cap.release()

    # Final index = PRIMARY (output) frames only; strip private fields (e.g. cached masks).
    prim_index = [{k: v for k, v in r.items() if not k.startswith("_")} for r in index if r["primary"]]
    json.dump(prim_index, open(os.path.join(outdir, f"_index_{tag}.json"), "w"), indent=2)
    n = sum(1 for r in prim_index if r["obj"])
    print(f"\n{tag}: {len(prim_index)} primary frames, {n} grounded, {len(prim_index) - n} N/A")


if __name__ == "__main__":
    main()
