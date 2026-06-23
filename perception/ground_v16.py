#!/usr/bin/env python3
"""ground_v16.py — NEW ARCHITECTURE (sam3py 3.12). Per grasp-time frame, PER HAND:
  [1] SAM3 segment-everything (localized point-grid AMG) near the hand box -> class-agnostic masks
  [2] YOLO hand box/pinch (from the _grasp prompts)
  [3] drop the hand's own mask + depth-filter (DA3) to the hand's depth -> tightened candidate list
  [4] Claude (subprocess llm_pick.py) picks which candidate is manipulated -> name, or N/A
  [5] SAM3 text-seg that name around the hand -> mask + confidence
  [6] confidence >= 0.60 -> accept; else re-ask Claude (low-conf feedback) and rerun [5] (<=2 retries)

env: DEPTH_DIR (DA3 .npy), CONF_MIN (0.60), DEPTH_BAND (0.20), GRID (7), MARGIN (0.6), MAX_RETRY (2)
Run (sam3py): ground_v16.py <outdir> <_inventory_tag.json> <_grasp_tag.json>
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PY310 = "/home/ubuntu/local/.venv/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 255, 60), (255, 80, 255),
          (80, 255, 255), (255, 160, 40), (160, 80, 255), (40, 200, 120), (200, 120, 200)]
DEPTH_DIR = os.environ.get("DEPTH_DIR", "")
CONF_MIN = float(os.environ.get("CONF_MIN", "0.60"))
BAND = float(os.environ.get("DEPTH_BAND", "0.20"))
GRID = int(os.environ.get("GRID", "7"))
MARGIN = float(os.environ.get("MARGIN", "0.6"))
MAX_RETRY = int(os.environ.get("MAX_RETRY", "1"))
PINCH_TOL = int(os.environ.get("PINCH_TOL", "0"))   # 0 => auto (4.5% of W): object must sit this near the pinch
FG_GATE = os.environ.get("FG_GATE", "1") == "1"     # reject candidates FARTHER than the hand (background rack)
FG_BAND = float(os.environ.get("FG_BAND", "0.30"))  # "farther than hand by > FG_BAND*range" = background (loose: held objects tilt back)
DEPTH_FG = os.environ.get("DEPTH_FG", "0") == "1"   # OFF: depth sees THROUGH transparent objects -> blob lands on the background
TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"   # OFF: motion across frames is too unstable; judge the moment
DELTA = float(os.environ.get("DELTA", "0.2"))       # seconds for the optical-flow motion window
DBG = os.environ.get("DBG", "")                     # trace per-hand stage outcomes (which branch -> N/A)
TRANSP = os.environ.get("TRANSP", "1") == "1"       # transparent-object GAP check on N/A hands
TRUST_THR = float(os.environ.get("TRUST_THR", "0.6"))  # trust a SAM3 text-seg RULE-1 hit at/above this score (skip LLM veto)
REFS_DIR = os.environ.get("REFS_DIR", "")           # dir with _refsheet_<tag>.png (clear-view object references)
TRACKS_DIR = os.environ.get("TRACKS_DIR", "")       # dir with per-grasp-time <tag>_t<t>.npz video-tracked masks (Phase 2)
_DBGF = None                                        # debug trace file (set in main) — stdout is eaten by the launcher


def _dbg(m, **kw):
    print(m, flush=True)
    if _DBGF:
        _DBGF.write(str(m) + "\n"); _DBGF.flush()


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _mostly_in(m, masks, frac=0.5):
    """True if mask m is mostly (>= frac) inside ANY of `masks` — e.g. the recovered blob IS the hand/arm."""
    a = float(m.sum())
    return a > 0 and any(float(np.logical_and(m, k).sum()) >= frac * a for k in masks)


def amg_near_hand(model, state, box, W, H, min_a=0.0006, max_a=0.25):
    """STEP 1: grid points around the hand box -> per-point best masks -> NMS = class-agnostic set."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ex0, ey0 = max(0, int(x0 - MARGIN * bw)), max(0, int(y0 - MARGIN * bh))
    ex1, ey1 = min(W, int(x1 + MARGIN * bw)), min(H, int(y1 + MARGIN * bh))
    xs = np.linspace(ex0 + 4, ex1 - 4, GRID).astype(int)
    ys = np.linspace(ey0 + 4, ey1 - 4, GRID).astype(int)
    raw = []
    for yy in ys:
        for xx in xs:
            masks, scores, _ = model.predict_inst(
                state, point_coords=np.array([[xx, yy]]), point_labels=np.array([1]),
                multimask_output=True)
            masks = _np(masks); scores = _np(scores).ravel()
            for i in range(len(scores)):
                m = masks[i] > 0.5
                if m.ndim > 2:
                    m = m.squeeze()
                a = float(m.mean())
                if min_a <= a <= max_a:
                    raw.append((float(scores[i]), a, m))
    raw.sort(key=lambda r: -r[0])
    keep = []
    for sc, a, m in raw:
        if all(_iou(m, k[2]) < 0.7 for k in keep):
            keep.append((sc, a, m))
    return keep[:12]


def call_llm(img_paths, hand, n, names, feedback=""):
    """STEP 4: subprocess into the 3.10 venv (Claude). Returns (choice, name)."""
    try:
        out = subprocess.run(
            [PY310, os.path.join(HERE, "llm_pick.py"), ",".join(img_paths), hand, str(n),
             ",".join(names), feedback],
            capture_output=True, text=True, timeout=120, cwd=HERE)
        for line in reversed(out.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                return int(d["choice"]), d["name"]
    except Exception as e:
        print(f"  [llm_pick error] {e}")
    return -1, "N/A"


def edge_box(rgb, gx, gy, hbox, W, H):
    """EDGE ANALYSIS for transparent objects: their interior is see-through (depth/region-seg miss
    them) but their RIM/walls show as edges. Canny in the grasp region, then the container-contour
    nearest the pinch -> a bounding box to hand SAM as a box prompt. Returns [x0,y0,x1,y1] or None."""
    x0, y0, x1, y1 = hbox; bw, bh = x1 - x0, y1 - y0
    ex0, ey0 = max(0, int(x0 - 0.6 * bw)), max(0, int(y0 - 0.6 * bh))
    ex1, ey1 = min(W, int(x1 + 0.6 * bw)), min(H, int(y1 + 0.6 * bh))
    sub = rgb[ey0:ey1, ex0:ex1]
    if sub.size == 0:
        return None
    gray = cv2.bilateralFilter(cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY), 7, 50, 50)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # close rim gaps
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pxl, pyl = gx - ex0, gy - ey0; area = sub.shape[0] * sub.shape[1]
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c); a = w * h
        if a < 0.01 * area or a > 0.45 * area:
            continue                                            # object-scale, not a sliver / the whole region
        if not (x - 6 <= pxl <= x + w + 6 and y - 6 <= pyl <= y + h + 6):
            continue                                            # the box must CONTAIN the pinch (the HELD object)
        if best is None or a < best[0]:                         # SMALLEST such box = the tight held object
            best = (a, [ex0 + x, ey0 + y, ex0 + x + w, ey0 + y + h])   # (not the larger container/surface/hand around it)
    return best[1] if best else None


def refine_point(model, state, cmask, negs, H, W):
    """STEP 1: re-segment the held object cleanly — point-prompt seeded INSIDE the rough mask, with
    hand-body NEGATIVES, and pick the smallest object-scale mask that contains the seed and excludes
    the negatives. Returns (clean_mask, score) or None. Fixes rough depth-fg / AMG masks."""
    ys, xs = np.where(cmask)
    if len(xs) == 0:
        return None
    cxo, cyo = int(np.median(xs)), int(np.median(ys))
    if not cmask[_clamp(cyo, 0, H - 1), _clamp(cxo, 0, W - 1)]:
        i = len(xs) // 2; cxo, cyo = int(xs[i]), int(ys[i])      # ensure the seed is inside the mask
    pts = np.array([[cxo, cyo]] + negs); labs = np.array([1] + [0] * len(negs))
    masks, scores, _ = model.predict_inst(state, point_coords=pts, point_labels=labs, multimask_output=True)
    masks = _np(masks); scores = _np(scores).ravel()
    best = None
    for i in range(len(scores)):
        m = masks[i] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if not m[_clamp(cyo, 0, H - 1), _clamp(cxo, 0, W - 1)]:
            continue                                            # must contain the seed
        if any(m[_clamp(ny, 0, H - 1), _clamp(nx, 0, W - 1)] for nx, ny in negs):
            continue                                            # must exclude the hand/arm
        a = float(m.mean())
        if not (0.0006 < a < 0.30):
            continue
        if best is None or a < best[2]:
            best = (m, float(scores[i]), a)                     # smallest object-scale
    return (best[0], best[1]) if best else None


def temporal_held(cap, t, dt, mask, box, W, H):
    """C3: HELD if the object moves WITH the hand (or both are still). REACHING/release if the hand
    moves but the object does NOT follow -> returns False (=> N/A). Optical flow between t and t+dt."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, a = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, (t + dt) * 1000.0); ok2, b = cap.read()
    if not (ok and ok2):
        return True
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    x0, y0, x1, y1 = box
    handm = np.zeros((H, W), bool); handm[y0:y1, x0:x1] = True
    handm = handm & ~mask                                    # hand region, excluding the object
    if handm.sum() < 50 or mask.sum() < 50:
        return True
    hmag = float(np.linalg.norm(flow[handm].reshape(-1, 2).mean(0)))   # hand motion magnitude
    omag = float(np.linalg.norm(flow[mask].reshape(-1, 2).mean(0)))    # object motion magnitude
    # LENIENT: only call it "reaching" when the hand CLEARLY moves but the object is essentially
    # STATIC (a fixed rack item). Any object motion (held + scooping/pouring/adjusting) = held.
    if hmag < 4.0:
        return True                                          # hand not clearly moving -> held (steadying)
    return omag > 1.5                                        # object moving -> held; static -> reaching (N/A)


def call_verify(img_path, hand, name):
    """STEP 6b: subprocess into the 3.10 venv (Claude) -> does the highlighted mask really cover <name>?"""
    try:
        out = subprocess.run(
            [PY310, os.path.join(HERE, "llm_verify.py"), img_path, hand, name],
            capture_output=True, text=True, timeout=90, cwd=HERE)
        for line in reversed(out.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                return bool(json.loads(line.strip()).get("ok", False))
    except Exception as e:
        print(f"  [llm_verify error] {e}")
    return True                                              # fail-open (don't drop on verifier error)


def call_transparent(img_paths, hand, names):
    """TRANSPARENT GAP CHECK: subprocess into the 3.10 venv (Claude) -> name of a held clear/transparent
    object the segmentation missed, or 'N/A'."""
    try:
        out = subprocess.run(
            [PY310, os.path.join(HERE, "llm_transparent.py"), ",".join(img_paths), hand, ",".join(names)],
            capture_output=True, text=True, timeout=90, cwd=HERE)
        for line in reversed(out.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                return json.loads(line.strip()).get("name", "N/A")
    except Exception as e:
        print(f"  [llm_transparent error] {e}")
    return "N/A"


def main():
    outdir, inv_file, prompts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    tmpd = tempfile.mkdtemp(prefix="v16_")
    global _DBGF
    if DBG:
        _DBGF = open(os.path.join(outdir, "_dbgtrace.log"), "w")   # reliable trace (launcher eats stdout)
    inv = json.load(open(inv_file)); tag = inv["tag"]
    objects = [{"name": o, "role": "manipulable"} if isinstance(o, str) else o for o in inv["objects"]]
    names = [o["name"] for o in objects if o.get("role") != "fixture"]   # canonical vocab (no fixtures)
    refsheet = os.path.join(REFS_DIR, f"_refsheet_{tag}.png") if REFS_DIR else ""
    refsheet = os.path.abspath(refsheet) if refsheet and os.path.exists(refsheet) else ""   # ABS: LLM subprocess runs in perception/
    refman = os.path.join(REFS_DIR, f"_refs_{tag}.json") if REFS_DIR else ""
    if refman and os.path.exists(refman):                # SAM-verified vocab: drop objects with no clear
        verified = set(json.load(open(refman)).keys())   # view anywhere (hallucinations like a phantom bowl)
        kept = [n for n in names if n in verified]
        if kept:
            print(f"{tag}: vocab restricted to SAM-verified objects: {kept} (dropped {sorted(set(names) - set(kept))})")
            names = kept
    pj = json.load(open(prompts_file)); video = pj["video"]; W, H = pj["W"], pj["H"]
    model = build_sam3_image_model(enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    cap = cv2.VideoCapture(video)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    by_t = defaultdict(list)
    for p in pj["prompts"]:
        by_t[round(p["t"], 3)].append(p)

    index = []
    seed_masks = {}        # SEED HARVEST (Stage 1): raw chosen_mask of every confident detection, for tracking seeds
    for t, plist in sorted(by_t.items()):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # depth (precomputed DA3) for STEP 3
        dep = None
        if DEPTH_DIR:
            dpath = os.path.join(DEPTH_DIR, f"{tag}_t{t:05.1f}.npy")
            if os.path.exists(dpath):
                dep = np.load(dpath).astype(np.float32).squeeze()
                if dep.shape != (H, W):
                    dep = cv2.resize(dep, (W, H))
        far_tol = None; rng = None
        if dep is not None:
            lo, hiv = np.percentile(dep, [5, 95]); rng = float(hiv - lo) + 1e-6
            far_tol = FG_BAND * rng
        tracks = {}                                              # Phase-2 video tracks for this grasp time
        if TRACKS_DIR:
            tp = os.path.join(TRACKS_DIR, f"{tag}_t{t:05.1f}.npz")
            if os.path.exists(tp):
                tz = np.load(tp)
                for nm in tz.files:
                    mk = tz[nm].astype(bool)
                    if mk.shape != (H, W):
                        mk = cv2.resize(mk.astype(np.uint8), (W, H)).astype(bool)
                    tracks[nm] = mk

        with torch.inference_mode(), ac:
            state = proc.set_image(Image.fromarray(rgb))   # AMG (segment-everything) is computed LAZILY per hand below
            # TEXT-SEG SWEEP: SAM3 text-seg labels each object well (incl. TRANSPARENT cups/jars the
            # point-grid AMG can't click). Run it once per verified object; per hand we add the masks
            # that pass RULE 1 to the candidate pool, selecting by LOCATION (not top class-score).
            textseg_all = []                                # (name, mask, score) for every verified object this frame
            for nm in names:
                out = proc.set_text_prompt(state=state, prompt=nm.split(" with ")[0].strip())
                tmm = _np(out["masks"]); tss = _np(out["scores"]).ravel()
                if tmm.ndim == 4:
                    tmm = tmm[:, 0]
                for k in range(len(tmm) if tmm.ndim == 3 else 0):
                    mk = tmm[k] > 0.5
                    if mk.ndim > 2:
                        mk = mk.squeeze()
                    if mk.any() and mk.shape == (H, W):
                        textseg_all.append((nm, mk, float(tss[k])))

        for p in plist:
            gx, gy = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
            x0, y0, x1, y1 = p["box_px"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            wpx, wpy = int(_clamp(2 * cx - gx, x0, x1)), int(_clamp(2 * cy - gy, y0, y1))  # wrist/arm ref
            hand = "LEFT" if p["hand"] == "L" else "RIGHT"

            # VIDEO-TRACK PRIMARY (Phase 2): if a tracked object's mask is AT THE PINCH (and not the
            # hand/arm), ground it directly — consistent identity + clean mask carried from a clearer
            # frame. The pinch-association handles object SWITCHES (a different track is under the
            # fingers once the held object changes). Falls through to the per-frame path if no track here.
            if tracks:
                tw = max(20, int(0.045 * W))
                yl2, yh2 = max(0, gy - tw), min(H, gy + tw + 1); xl2, xh2 = max(0, gx - tw), min(W, gx + tw + 1)
                th = None
                for nm, mk in tracks.items():
                    if nm not in names:
                        continue                                   # only verified MANIPULABLE objects (no fixture tracks)
                    # RULE 1: the tracked object must be AT the pinch, INTERSECT the hand box, and not be
                    # the hand/arm itself — i.e. the hand is actually on it, not merely near a passing track.
                    at_pinch = mk[yl2:yh2, xl2:xh2].any()
                    in_handbox = mk[y0:y1, x0:x1].any()
                    is_wrist = mk[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)]
                    if at_pinch and in_handbox and not is_wrist and 0.0005 < float(mk.mean()) < 0.40:
                        if th is None or int(mk.sum()) < th[2]:    # the pinch is INSIDE the smallest such track
                            th = (nm, mk, int(mk.sum()))
                if th is not None:
                    nm, mk, _ = th; col = COLORS[0]
                    f = rgb.copy().astype(np.float32); f[mk] = 0.5 * f[mk] + 0.5 * np.array(col, np.float32)
                    vis = f.clip(0, 255).astype(np.uint8)
                    cont, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cont, -1, col, 2)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1); cv2.circle(vis, (gx, gy), 7, (255, 40, 40), -1)
                    cap_txt = f"{tag} {hand}: {nm}  (track)"
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
                    cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                    fn = f"{tag}_t{p['t']:05.1f}_{p['hand']}.png"
                    cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    index.append({"file": fn, "tag": tag, "hand": hand, "t": p["t"], "obj": True,
                                  "name": nm, "conf": 0.9, "n_cand": len(tracks), "low_conf": False, "src": "track"})
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: TRACK -> {nm}")
                    continue
            # LAZY AMG (speedup): run segment-everything near THIS hand only now that no track grounded it.
            with torch.inference_mode(), ac:
                amg_hand = amg_near_hand(model, state, p["box_px"], W, H)
            hd = None
            if dep is not None:
                hd = float(np.median(dep[max(0, gy - 3):gy + 4, max(0, gx - 3):gx + 4]))

            # STEP 3: keep only masks that are (a) at the grasp (near the pinch), (b) intersect the
            # hand box [RULE 1: object must be IN the hand box], (c) not the hand/arm, (d) at hand depth
            pt = PINCH_TOL or max(20, int(0.045 * W))
            ylo, yhi = max(0, gy - pt), min(H, gy + pt + 1)
            xlo, xhi = max(0, gx - pt), min(W, gx + pt + 1)
            hand_masks = []
            cands = []
            for sc, a, m in amg_hand:
                if m[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)]:
                    hand_masks.append(m)                        # this mask is the hand/arm itself
                    continue
                if not m[ylo:yhi, xlo:xhi].any():
                    continue                                    # no object AT the grasp point (RULE 1)
                if m[y0:y1, x0:x1].sum() == 0:
                    continue                                    # mask does not intersect the hand box (RULE 1)
                if FG_GATE and dep is not None:
                    if float(np.median(dep[m])) - hd > far_tol:   # object is BEHIND the hand = background (rack)
                        continue                                  # one-sided: keep foreground, drop background
                cands.append((sc, a, m))

            # TEXT-SEG candidates: add SAM3 text-seg masks (transparent objects the AMG missed) that pass
            # the same RULE 1 — at the pinch, in the hand box, not the wrist/arm — deduped against the AMG set.
            # textseg_hits[name] = (best score, mask) at THIS hand -> used to TRUST a strong SAM3 detection.
            textseg_hits = {}
            for nm, mk, sc in textseg_all:
                if mk[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)]:
                    continue
                if not mk[ylo:yhi, xlo:xhi].any() or mk[y0:y1, x0:x1].sum() == 0:
                    continue
                if not (0.0005 < float(mk.mean()) < 0.40):
                    continue
                if nm not in textseg_hits or sc > textseg_hits[nm][0]:
                    textseg_hits[nm] = (float(sc), mk)
                if any(_iou(mk, c[2]) > 0.7 for c in cands):
                    continue
                cands.append((float(sc), float(mk.mean()), mk))

            # CONTAINER-PREFERENCE (STEP 3): drop a candidate whose mask is mostly INSIDE a larger
            # candidate (loose contents sit inside the held container -> keep the container).
            if len(cands) > 1:
                cands.sort(key=lambda c: -c[1])              # largest area first
                kept = []
                for sc, a, m in cands:
                    inside = any(np.logical_and(m, mk).sum() > 0.8 * m.sum() for _s, ak, mk in kept)
                    if not inside:
                        kept.append((sc, a, m))
                cands = kept

            # FIX A: transparent/occluded held object -> no AMG candidate. Recover the FOREGROUND blob
            # at the hand depth around the pinch (DA3 sees it through transparency), minus the hand/arm.
            if not cands and DEPTH_FG and dep is not None and rng:
                bx0 = max(0, int(x0 - 0.25 * (x1 - x0))); by0 = max(0, int(y0 - 0.25 * (y1 - y0)))
                bx1 = min(W, int(x1 + 0.25 * (x1 - x0))); by1 = min(H, int(y1 + 0.25 * (y1 - y0)))
                boxm = np.zeros((H, W), bool); boxm[by0:by1, bx0:bx1] = True   # CLIP to the hand box (no arm/tray)
                fg = (np.abs(dep - hd) <= 0.10 * rng).astype(np.uint8)         # tight band: object at hand depth
                _, lbl = cv2.connectedComponents(fg)
                pid = lbl[gy, gx]
                if pid:
                    blob = (lbl == pid) & boxm
                    for hm in hand_masks:
                        blob = blob & ~hm                        # drop the hand/arm
                    if blob[ylo:yhi, xlo:xhi].any() and 0.001 < blob.mean() < 0.10:
                        cands.append((0.5, float(blob.mean()), blob))    # synthetic foreground candidate

            # render numbered candidates (zoomed hand region) + full frame, for the LLM
            ex0, ey0 = max(0, int(x0 - 1.2 * (x1 - x0))), max(0, int(y0 - 1.2 * (y1 - y0)))
            ex1, ey1 = min(W, int(x1 + 1.2 * (x1 - x0))), min(H, int(y1 + 1.2 * (y1 - y0)))
            zoom = rgb[ey0:ey1, ex0:ex1].copy()
            for j, (sc, a, m) in enumerate(cands):
                col = COLORS[j % len(COLORS)]
                cont, _ = cv2.findContours(m[ey0:ey1, ex0:ex1].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(zoom, cont, -1, col, 2)
                yx = np.argwhere(m[ey0:ey1, ex0:ex1])
                if len(yx):
                    yc, xc = map(int, yx.mean(0))
                    cv2.putText(zoom, str(j), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)
            cv2.circle(zoom, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
            zoom = cv2.resize(zoom, (zoom.shape[1] * 2, zoom.shape[0] * 2))
            zpath = os.path.join(tmpd, f"z_{tag}_{t:.1f}_{p['hand']}.jpg")
            cv2.imwrite(zpath, cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))

            name, conf, chosen_mask, choice = "N/A", 0.0, None, -1
            na_reason = "no_candidates_after_filter"
            if DBG:
                _dbg(f"[DBG] {tag} t={t:.1f} {hand}: {len(cands)} candidates after filter "
                      f"(areas={[round(c[1]*100,1) for c in cands]}); textseg_hits="
                      f"{ {n: round(s, 2) for n, (s, _) in textseg_hits.items()} }", flush=True)
            # TRUST SAM3 TEXT-SEG: if exactly ONE verified object has a STRONG (>= TRUST_THR) RULE-1 text-seg
            # hit at this hand, ground it directly — SAM3 labels each frame well, and the LLM-pick was
            # vetoing correct transparent detections (e.g. the t33.4-L cup @0.83). If 2+ objects match
            # strongly (cup vs jar look-alikes), fall through to the LLM to disambiguate.
            trusted = {n: (s, m) for n, (s, m) in textseg_hits.items() if s >= TRUST_THR}
            if len(trusted) == 1:
                nm, (sc, mk) = next(iter(trusted.items()))
                name, conf, chosen_mask, choice, na_reason = nm, float(sc), mk, 0, ""
                if DBG:
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: TRUST text-seg -> {nm} ({sc:.2f})", flush=True)
            elif cands:
                na_reason = ""
                feedback = ""
                for attempt in range(MAX_RETRY + 1):
                    imgs = [zpath] + ([refsheet] if refsheet else [])
                    choice, name = call_llm(imgs, hand, len(cands), names, feedback)
                    if DBG:
                        _dbg(f"[DBG] {tag} t={t:.1f} {hand}: LLM choice={choice} name={name!r}", flush=True)
                    if choice < 0:
                        name, conf, chosen_mask = "N/A", 0.0, None
                        na_reason = "llm_declined_N/A"
                        break
                    cmask = cands[choice][2]
                    cont_name = name.split(" with ")[0].strip()   # the CONTAINER part, for text-seg
                    # STEP 5 + STEP 1 refinement: prefer a CLEAN text-seg mask of the container that
                    # overlaps the pick; else point-prompt re-segment seeded inside the rough mask.
                    with torch.inference_mode(), ac:
                        out = proc.set_text_prompt(state=state, prompt=cont_name)
                    tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
                    if tm.ndim == 4:
                        tm = tm[:, 0]
                    # Accept a text-seg mask as the clean mask ONLY if it is a GRASP-LOCALIZED match to
                    # the picked candidate — at the pinch, in the hand box, not the hand/arm, and a real
                    # overlap with the pick (not a 10%-clip). Otherwise SAM's high CLASS score for a
                    # different instance / mislocated blob yields high conf on a WRONG mask.
                    best_sc, best_m = 0.0, None
                    for k in range(len(tm) if tm.ndim == 3 else 0):
                        mk = tm[k] > 0.5
                        if not mk.sum():
                            continue
                        if not mk[ylo:yhi, xlo:xhi].any():                 # must sit at the pinch
                            continue
                        if mk[y0:y1, x0:x1].sum() == 0:                    # must intersect the hand box (RULE 1)
                            continue
                        if mk[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)]:   # must not be the hand/arm
                            continue
                        if _iou(mk, cmask) < 0.3 or float(mk.mean()) > 0.35:  # match the pick; not a giant blob
                            continue
                        if float(ts[k]) > best_sc:
                            best_sc, best_m = float(ts[k]), mk
                    if best_m is not None and best_sc >= 0.30:
                        chosen_mask, conf = best_m, best_sc        # clean, grasp-localized named mask
                    else:
                        with torch.inference_mode(), ac:
                            rf = refine_point(model, state, cmask, [(int(wpx), int(wpy))], H, W)
                        rf_ok = (rf is not None and rf[0][ylo:yhi, xlo:xhi].any()
                                 and not rf[0][_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)])
                        if rf_ok:
                            chosen_mask, conf = rf[0], max(rf[1], 0.6)   # refined clean mask -> real conf
                        else:
                            chosen_mask, conf = cmask, float(cands[choice][0])   # fallback: candidate (already at grasp)
                    if conf >= CONF_MIN:
                        break
                    feedback = (f"The object you chose ('{name}') could not be cleanly segmented "
                                f"(conf {conf:.2f}). Reconsider what the {hand} hand is manipulating; "
                                f"pick the best numbered candidate or -1 for N/A.")
                if conf < CONF_MIN and choice >= 0 and chosen_mask is None:
                    chosen_mask = cands[choice][2]              # keep the held-object mask, flag low conf

            # STEP 6b: verify the chosen MASK actually covers the named object (else N/A)
            if choice >= 0 and chosen_mask is not None and name != "N/A":
                vz = rgb[ey0:ey1, ex0:ex1].copy()
                f = vz.astype(np.float32); sub = chosen_mask[ey0:ey1, ex0:ex1]
                f[sub] = 0.45 * f[sub] + 0.55 * np.array((80, 255, 80), np.float32)
                vz = f.clip(0, 255).astype(np.uint8)
                cont, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vz, cont, -1, (40, 255, 40), 2)
                cv2.circle(vz, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
                vz = cv2.resize(vz, (vz.shape[1] * 2, vz.shape[0] * 2))
                vpath = os.path.join(tmpd, f"v_{tag}_{t:.1f}_{p['hand']}.jpg")
                cv2.imwrite(vpath, cv2.cvtColor(vz, cv2.COLOR_RGB2BGR))
                verify_ok = call_verify(vpath, hand, name)
                if DBG:
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: verify({name!r})={verify_ok}", flush=True)
                # ADVISORY ONLY (no longer forces N/A — it was killing correct transparent/rough masks).
                # A failed verify just lowers confidence so it shows as low-conf for review.
                if not verify_ok:
                    conf = min(conf, 0.5)

            # STEP 6c (C3): temporal held-vs-reaching — drop objects the hand only reaches toward
            if TEMPORAL and choice >= 0 and chosen_mask is not None:
                held = temporal_held(cap, t, DELTA, chosen_mask, p["box_px"], W, H)
                if DBG:
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: temporal_held={held}", flush=True)
                if not held:
                    choice, name, conf, chosen_mask = -1, "N/A", 0.0, None   # reaching, not holding -> N/A
                    na_reason = "temporal_reaching_N/A"
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)                   # restore cap position
            # EDGE-DRIVEN TRANSPARENT RECOVERY — only when SAM3 text-seg found NOTHING at this hand. If
            # text-seg DID return a RULE-1 hit (even one the LLM/trust didn't pick), edge-recovery must NOT
            # fabricate a different label over it (it was mislabeling e.g. t33.4-L cup as "jar"). Gated on
            # `not textseg_hits` so the reliable text-seg evidence wins over the noisier edge guess.
            if TRANSP and (choice < 0 or name == "N/A") and names and not textseg_hits:
                rec = None
                eb = edge_box(rgb, gx, gy, p["box_px"], W, H)            # rim box near the pinch
                if DBG:
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: edge_box={eb}", flush=True)
                if eb is not None:                                       # SAM BOX prompt on the edge box
                    with torch.inference_mode(), ac:
                        mks, scr, _ = model.predict_inst(state, box=np.array(eb), multimask_output=True)
                    mks = _np(mks); scr = _np(scr).ravel(); bb = None
                    for i in range(len(scr)):
                        mm = mks[i] > 0.5
                        if mm.ndim > 2:
                            mm = mm.squeeze()
                        if mm[_clamp(gy, 0, H - 1), _clamp(gx, 0, W - 1)] and 0.0008 < mm.mean() < 0.25 \
                           and not mm[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)] \
                           and not _mostly_in(mm, hand_masks):       # the recovered blob must not BE the hand/arm
                            if bb is None or float(scr[i]) > bb[1]:
                                bb = (mm, float(scr[i]))
                    if bb is not None:
                        rec = bb[0]
                if rec is None:                                          # fallback: point-prompt the rim/contents
                    with torch.inference_mode(), ac:
                        mks, scr, _ = model.predict_inst(
                            state, point_coords=np.array([[gx, gy], [int(wpx), int(wpy)]]),
                            point_labels=np.array([1, 0]), multimask_output=True)
                    mks = _np(mks); scr = _np(scr).ravel(); best_a = 1.0
                    for i in range(len(scr)):
                        mm = mks[i] > 0.5
                        if mm.ndim > 2:
                            mm = mm.squeeze()
                        if mm[_clamp(gy, 0, H - 1), _clamp(gx, 0, W - 1)] and \
                           not mm[_clamp(wpy, 0, H - 1), _clamp(wpx, 0, W - 1)] and 0.0006 < mm.mean() < 0.20 \
                           and not _mostly_in(mm, hand_masks):       # not the hand/arm
                            if mm.mean() < best_a:
                                rec, best_a = mm, float(mm.mean())
                if rec is not None:                                      # NAME the recovered region (LLM) -> vocab or N/A
                    rz = rgb[ey0:ey1, ex0:ex1].copy(); sub = rec[ey0:ey1, ex0:ex1]
                    f = rz.astype(np.float32); f[sub] = 0.45 * f[sub] + 0.55 * np.array((80, 255, 80), np.float32)
                    rz = f.clip(0, 255).astype(np.uint8)
                    cont, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rz, cont, -1, (40, 255, 40), 2)
                    cv2.circle(rz, (gx - ex0, gy - ey0), 6, (255, 40, 40), -1)
                    rz = cv2.resize(rz, (rz.shape[1] * 2, rz.shape[0] * 2))
                    tpath = os.path.join(tmpd, f"tr_{tag}_{t:.1f}_{p['hand']}.jpg")
                    cv2.imwrite(tpath, cv2.cvtColor(rz, cv2.COLOR_RGB2BGR))
                    tname = call_transparent([tpath] + ([refsheet] if refsheet else []), hand, names)
                    if DBG:
                        _dbg(f"[DBG] {tag} t={t:.1f} {hand}: edge-recovered -> name {tname!r}", flush=True)
                    if tname != "N/A":
                        chosen_mask, name, conf, choice, na_reason = rec, tname, 0.7, 0, ""

            if DBG and na_reason:
                _dbg(f"[DBG] {tag} t={t:.1f} {hand}: FINAL N/A -> reason = {na_reason}", flush=True)

            # render final
            vis = rgb.copy()
            locked = choice >= 0 and chosen_mask is not None
            if locked:
                col = COLORS[choice % len(COLORS)]
                f = vis.astype(np.float32); f[chosen_mask] = 0.5 * f[chosen_mask] + 0.5 * np.array(col, np.float32)
                vis = f.clip(0, 255).astype(np.uint8)
                cont, _ = cv2.findContours(chosen_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, col, 2)
                lowflag = " LOWCONF" if conf < CONF_MIN else ""
                cap_txt = f"{tag} {hand}: {name}  (conf {conf:.2f}){lowflag}"
            else:
                cap_txt = f"{tag} {hand}: N/A"
            cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1)
            cv2.circle(vis, (gx, gy), 7, (255, 40, 40), -1)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            fn = f"{tag}_t{p['t']:05.1f}_{p['hand']}.png"
            cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            index.append({"file": fn, "tag": tag, "hand": hand, "t": p["t"], "obj": bool(locked),
                          "name": name if locked else "N/A", "conf": round(conf, 3),
                          "n_cand": len(cands), "low_conf": bool(locked and conf < CONF_MIN)})
            if locked and chosen_mask is not None:   # harvest seed mask (only main-path reaches here; track path continued)
                seed_masks[f"{p['t']:05.1f}_{p['hand']}"] = chosen_mask.astype(bool)
            print(cap_txt + f"  [{len(cands)} cands]")
    cap.release()
    shard = "_".join(sorted({r["tag"] for r in index})) or "shard"
    json.dump(index, open(os.path.join(outdir, f"_index_{shard}.json"), "w"), indent=2)
    if seed_masks:           # Stage-1 seeds: confident-detection masks (key "<t>_<hand>") for the tracker
        np.savez_compressed(os.path.join(outdir, f"_seeds_{shard}.npz"), **seed_masks)
    n_obj = sum(1 for r in index if r["obj"])
    print(f"\n{tag}: {len(index)} hand-frames, {n_obj} grounded, {len(index) - n_obj} N/A "
          f"({len(seed_masks)} seed masks saved)")


if __name__ == "__main__":
    main()
