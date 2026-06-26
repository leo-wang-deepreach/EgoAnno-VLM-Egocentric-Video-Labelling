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


_RING8 = [(1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
          (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071)]


def _seed_points(mid, tp, ip, pw, W, H):
    """MULTI-SEED. The orange thumb-index midpoint is imprecise, so probe a deterministic Gaussian
    neighborhood AROUND it (centre + a ring at the pinch-aperture scale) PLUS points ALONG the
    thumb->index segment (the held object sits in that gap). Each point -> a SAM mask candidate, so
    one of them lands on the actual object even when the midpoint itself misses it."""
    cx0, cy0 = float(mid[0]), float(mid[1])
    ap = (((tp[0] - ip[0]) ** 2 + (tp[1] - ip[1]) ** 2) ** 0.5) if (tp and ip) else float(pw)
    sig = max(6.0, min(0.6 * ap, 0.5 * pw))            # spread ~ aperture, capped by the pinch window
    pts = [(cx0, cy0)] + [(cx0 + sig * dx, cy0 + sig * dy) for dx, dy in _RING8]
    if tp and ip:
        for f in (0.3, 0.5, 0.7):
            pts.append((tp[0] + f * (ip[0] - tp[0]), tp[1] + f * (ip[1] - tp[1])))
    out, seen = [], set()
    for x, y in pts:
        xy = (_clamp(x, 0, W - 1), _clamp(y, 0, H - 1))
        if xy not in seen:
            seen.add(xy); out.append(xy)
    return out


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


def _skin_mask(rgb, box):
    """Hand region by SKIN COLOR — robust where the keypoint/box detection is fooled by clutter (a metallic
    pile of bolts looks finger-like, so the pose model lands keypoints on it; skin color separates the real
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
                ti = _ti_band_score(mk, tpx, ipx)            # between-fingers prior (the held-object gap)
                return 0.34 * prox + 0.26 * (1.0 - oh) + 0.10 * sized + 0.14 * float(sam) + 0.16 * ti, dist

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

            pool = []                                          # (name, mask, sam, src, manip, dist)
            for nm, mk, sc in segs:                            # (1) TEXT-SEG, cleaned to a single blob at the grasp
                mk = _clean(mk, ax, ay)
                if (not mk.any() or mk[wpy, wpx] or not mk[yl:yh, xl:xh].any() or mk[y0:y1, x0:x1].sum() == 0
                        or not (0.0005 < float(mk.mean()) < 0.40) or _on_hand(mk, hand_mask)):
                    continue
                mn, dd = _manip(mk, sc); pool.append((nm, mk, sc, "TS", mn, dd))
            # (2) MULTI-SEED: probe a Gaussian neighborhood AROUND the orange midpoint + points ALONG
            # the thumb->index segment (the midpoint alone is imprecise — more seeds find the real object).
            seed_pts = _seed_points(mid, tpx, ipx, pw, W, H) if mid is not None else []
            for sx, sy in seed_pts:
                for m, s in _seg_at(sx, sy):
                    mn, dd = _manip(m, s); pool.append(("", m, s, "ORANGE", mn, dd))
            pool = [c for c in pool if c[5] <= GATE]           # GATE: drop far-from-grasp (pile/background)
            # THUMB-INDEX PRIOR: keep masks sitting BETWEEN the fingers; drop ones with no overlap with
            # that band (sheds finger/forearm/background blobs). Lenient (>0.05) so the VLM still chooses.
            if tpx and ipx:
                tis = [_ti_band_score(c[1], tpx, ipx) for c in pool]
                if any(v > 0.05 for v in tis):
                    pool = [c for c, v in zip(pool, tis) if v > 0.05]
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
            _dbg(f"[POOL] {tag} t={t:.1f} {hand} anchor=({int(ax)},{int(ay)}) shared={shared}: "
                 + " | ".join(f"{c[3]}{(':' + c[0][:10]) if c[0] else ''} manip={c[4]:.2f} d={c[5]:.2f} sam={c[2]:.2f}"
                              for c in pool))
            seed_viz = [(sx, sy, (255, 140, 0)) for (sx, sy) in seed_pts]   # orange = all multi-seed probe points
            win_seed = (int(ax), int(ay))                      # green ring = grasp anchor
            cands = [(n, m, s) for n, m, s, src, mn, dd in cands]

            # STEP 3: LLM names; SAFETY NET overrides a false N/A when a strong RULE-1 candidate exists
            name, conf, chosen, src = "N/A", 0.0, None, ""
            if cands:
                # PINCH-CENTERED crop: zoom into the grasp so a small held object (pen part, bolt) is
                # isolated for Claude, not lost in a cluttered hand-box view of the whole pile.
                rad = int(0.12 * W)
                ex0, ey0 = max(0, gx - rad), max(0, gy - rad)
                ex1, ey1 = min(W, gx + rad), min(H, gy + rad)
                zoom = rgb[ey0:ey1, ex0:ex1].copy()
                for j, (nm, mk, sc) in enumerate(cands):
                    col = COLORS[j % len(COLORS)]
                    cont, _ = cv2.findContours(mk[ey0:ey1, ex0:ex1].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(zoom, cont, -1, col, 2)
                    yx = np.argwhere(mk[ey0:ey1, ex0:ex1])
                    if len(yx):
                        yc, xc = map(int, yx.mean(0))
                        cv2.putText(zoom, str(j), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)
                cv2.circle(zoom, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
                zoom = cv2.resize(zoom, (zoom.shape[1] * 2, zoom.shape[0] * 2))
                zpath = os.path.join(tmpd, f"z_{tag}_{t:.1f}_{p['hand']}.jpg")
                cv2.imwrite(zpath, cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
                imgs = [zpath] + ([refsheet] if refsheet else [])
                choice, nm = call_llm(imgs, hand, len(cands), all_names)
                _dbg(f"[DBG] {tag} t={t:.1f} {hand}: LLM choice={choice} name={nm!r}")
                # RE-ASK once if Claude rejected a candidate that is RIGHT at the grasp (a point-prompt mask
                # contains the pinch). Claude over-rejects small in-place manipulation as "not lifted out";
                # this focused prompt counts it as holding, while still allowing N/A for a truly open/empty hand.
                if choice < 0 and any(c[0] == "" for c in cands):
                    # TIGHT crop on the candidate(s) — isolate + scale up the held part so Claude can name a
                    # tiny object (a pen refill) that's lost in the wider pile view.
                    union = np.logical_or.reduce([c[1] for c in cands])
                    ys2, xs2 = np.nonzero(union)
                    imgs2 = imgs
                    if len(xs2):
                        mg = int(0.05 * W)
                        tx0, ty0 = max(0, int(xs2.min()) - mg), max(0, int(ys2.min()) - mg)
                        tx1, ty1 = min(W, int(xs2.max()) + mg), min(H, int(ys2.max()) + mg)
                        tz = rgb[ty0:ty1, tx0:tx1].copy()
                        for j, (nm2, mk2, sc2) in enumerate(cands):
                            col = COLORS[j % len(COLORS)]
                            cont, _ = cv2.findContours(mk2[ty0:ty1, tx0:tx1].astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(tz, cont, -1, col, 2)
                            yx = np.argwhere(mk2[ty0:ty1, tx0:tx1])
                            if len(yx):
                                yc, xc = map(int, yx.mean(0))
                                cv2.putText(tz, str(j), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                        if 0 <= gy - ty0 < tz.shape[0] and 0 <= gx - tx0 < tz.shape[1]:
                            cv2.circle(tz, (gx - tx0, gy - ty0), 5, (255, 40, 40), -1)
                        scl = max(2, int(560 / max(1, tz.shape[1])))
                        tz = cv2.resize(tz, (tz.shape[1] * scl, tz.shape[0] * scl), interpolation=cv2.INTER_LINEAR)
                        tzp = os.path.join(tmpd, f"zt_{tag}_{t:.1f}_{p['hand']}.jpg")
                        cv2.imwrite(tzp, cv2.cvtColor(tz, cv2.COLOR_RGB2BGR))
                        imgs2 = [tzp] + ([refsheet] if refsheet else [])
                    fb = ("The highlighted object is RIGHT AT the grasp point (red dot), shown zoomed in. If the "
                          "hand is touching / holding / manipulating it — INCLUDING small in-place work like "
                          "assembling, threading, or picking a part from a pile — pick it and NAME it from the "
                          "list. Answer N/A ONLY if the hand is clearly OPEN or EMPTY with nothing in it.")
                    choice, nm = call_llm(imgs2, hand, len(cands), all_names, fb)
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: RE-ASK(tight) choice={choice} name={nm!r}")
                # The LLM is the sole decider: it picks the manipulated candidate (+ name from the FULL
                # inventory) or N/A. A point-prompt candidate has no built-in name, so it relies on the LLM
                # naming it; if the LLM gives no valid name, it stays N/A.
                if 0 <= choice < len(cands):
                    cnm = nm if nm and nm.upper() not in ("N/A", "NA") else cands[choice][0]
                    if cnm:
                        name, chosen, conf, src = cnm, cands[choice][1], cands[choice][2], "llm"
                # AUTO-ACCEPT a clearly-held object the LLM over-vetoed. Geometry + measured grip override the
                # flaky veto: the top candidate must be strong (at the contact, off the hand), the grip a CLOSED
                # grasp, AND it must TOUCH the SKIN hand — which validates it's a real held object adjacent to the
                # actual hand, not a pile instance the (mis-detected) keypoints anchored on. True open/empty hands
                # have a high grip + weaker candidate and don't pass.
                if chosen is None and cands:
                    grip_v = p.get("grip"); grip_v = grip_v if grip_v is not None else 1.0
                    tn, tm, ts_ = cands[0]
                    ys5, xs5 = np.nonzero(tm)
                    td = (((xs5.mean() - ax) ** 2 + (ys5.mean() - ay) ** 2) ** 0.5) / max(pw, 1)
                    toh = (float(np.logical_and(tm, hand_mask).sum()) / float(tm.sum())) if hand_mask is not None else 0.0
                    tman = (0.42 * max(0.0, 1 - td / 1.2) + 0.30 * (1 - toh)
                            + 0.12 * (1.0 if float(tm.mean()) < 0.12 else 0.0) + 0.16 * ts_)
                    skin = _skin_mask(rgb, (x0, y0, x1, y1))
                    touch_skin = anchor_ok = False
                    if skin is not None:
                        sd = cv2.dilate(skin.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                        touch_skin = bool(np.logical_and(tm, sd).any())          # candidate adjacent to the real hand
                        dt = cv2.distanceTransform((~skin).astype(np.uint8), cv2.DIST_L2, 3)
                        anchor_ok = dt[_clamp(ay, 0, H - 1), _clamp(ax, 0, W - 1)] <= 0.05 * W   # keypoints near real hand
                    if tman >= 0.75 and grip_v < 0.40 and toh < 0.35 and td < 0.4 and touch_skin and anchor_ok:
                        cnm = tn or (all_names[0] if len(all_names) == 1 else None)
                        if cnm is None:                          # unnamed + multi-vocab -> force a name (no N/A)
                            _, fn2 = call_llm(imgs, hand, len(cands), all_names,
                                              f"The {hand} hand IS firmly holding the highlighted object — NAME it "
                                              "from the list; N/A is NOT allowed.")
                            cnm = fn2 if (fn2 and fn2.upper() not in ("N/A", "NA")) else None
                        if cnm:
                            name, chosen, conf, src = cnm, tm, ts_, "auto"
                            _dbg(f"[AUTO] {tag} t={t:.1f} {hand}: LLM-veto override -> {cnm!r} "
                                 f"(manip={tman:.2f} grip={grip_v:.2f} d={td:.2f} oh={toh:.2f})")
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
                        srcj = "TS" if nm2 else "PT"; star = "*" if j == choice else ""
                        rep.append(f"{j}{star}:{srcj} s={sc2:.2f} d={dcc:.2f} oh={oh:.2f}")
                        yx = np.argwhere(sub)
                        if len(yx):
                            yc, xc = map(int, yx.mean(0))
                            cv2.putText(panel, f"{j}{star}", (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
                    cv2.circle(panel, (_clamp(int(ax) - ex0, 0, panel.shape[1] - 1),
                                       _clamp(int(ay) - ey0, 0, panel.shape[0] - 1)), 6, (255, 140, 0), -1)
                    panel = cv2.resize(panel, (panel.shape[1] * 2, panel.shape[0] * 2))
                    cv2.imwrite(os.path.join(outdir, f"{tag}_t{p['t']:05.1f}_{p['hand']}_cand.png"),
                                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    _dbg(f"[CAND] {tag} t={t:.1f} {hand} choice={choice}: " + " | ".join(rep))

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
            for sxv, syv, col in seed_viz:                             # all candidate seeds, color-coded
                cv2.circle(vis, (int(sxv), int(syv)), 4, col, -1)
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
