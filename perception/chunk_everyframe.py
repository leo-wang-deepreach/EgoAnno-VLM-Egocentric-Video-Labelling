#!/usr/bin/env python3
"""chunk_everyframe.py — run the candidate pipeline on EVERY frame (no grounding/segment gating). Per frame,
per hand: encode once, build candidates = TEXT-seg of each verified object + POINT-seg at the grasp, score
them by a fast manipulation-confidence heuristic (at-grasp + off-far + sane-size + SAM score) and draw the
best one with its name. Shows whatever each hand is holding throughout the clip. No LLM (fast), so the
per-frame name can flip — this is the "show everything" view.

Run (sam3py): chunk_everyframe.py <chunk.mp4> <chunk_start_sec> <refs.json> <inventory.json> <out.mp4>
  env: HANDPOSE_PKL ; MANIP_MIN (0.34 floor to draw)
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

chunk, cstart, refs_f, inv_f, out_mp4 = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
GREEN, BLUE = (80, 230, 80), (90, 160, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MANIP_MIN = float(os.environ.get("MANIP_MIN", "0.34"))

cap = cv2.VideoCapture(chunk); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30; cap.release()
sig = pickle.load(open(os.environ["HANDPOSE_PKL"], "rb")); T = sig["t"]
verified = set(json.load(open(refs_f)).keys())
inv = json.load(open(inv_f))
names = [o["name"] for o in inv["objects"] if o.get("role") != "fixture" and o["name"] in verified]
print(f"every-frame text+point pick; objects={names}; {len(T)} frames", flush=True)
PW = max(20, int(0.05 * W))


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _clean(m, ax, ay):
    mm = m.astype(np.uint8); n, lab, _, cents = cv2.connectedComponentsWithStats(mm, 8)
    if n > 2:
        ai = _clamp(ay, 0, m.shape[0] - 1); aj = _clamp(ax, 0, m.shape[1] - 1); sl = int(lab[ai, aj])
        if sl == 0:
            sl = min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
        mm = (lab == sl).astype(np.uint8)
    return cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


ON_HAND_REJECT = float(os.environ.get("ON_HAND_REJECT", "0.55"))    # drop a candidate that's mostly the hand
PIN_THR = float(os.environ.get("PIN_THR", "0.15"))                  # candidate must fill the thumb<->index gap


def _pinch_region(side, i, dilate=12):
    """The region BETWEEN the thumb finger-line (kp1-4) and index finger-line (kp5-8) — where a held object
    sits. Convex hull of those 8 keypoints, dilated. A candidate that doesn't overlap it isn't actually held."""
    ch = sig["chain"][side][i] if (isinstance(sig.get("chain"), dict) and i < len(sig["chain"][side])) else None
    if ch is None:
        return None
    pts = [[int(p[0] * W), int(p[1] * H)] for line in ch for p in line if np.isfinite(p).all()]
    if len(pts) < 3:
        return None
    reg = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(reg, cv2.convexHull(np.array(pts, np.int32)), 1)
    return cv2.dilate(reg, np.ones((dilate, dilate), np.uint8)).astype(bool)


def _onhand(m, hm):
    a = float(m.sum())
    return 0.0 if (hm is None or a == 0) else float(np.logical_and(m, hm).sum()) / a


def _manip(m, ax, ay, sam, oh):
    ys, xs = np.nonzero(m); dist = (((xs.mean() - ax) ** 2 + (ys.mean() - ay) ** 2) ** 0.5) / PW
    area = float(m.mean()); sized = 1.0 if area < 0.12 else max(0.0, 1.0 - (area - 0.12) / 0.18)
    prox = max(0.0, 1.0 - dist / 2.0)
    return 0.40 * prox + 0.27 * (1.0 - oh) + 0.18 * float(sam) + 0.15 * sized


def anchor_negs(i, side):
    tip = sig["tip"][side][i] if i < len(sig["tip"][side]) else None
    if tip is not None and np.isfinite(tip).all():
        ax, ay = float((tip[0][0] + tip[1][0]) / 2 * W), float((tip[0][1] + tip[1][1]) / 2 * H)
    else:
        gz = sig["grasp"][side][i]
        if not np.isfinite(gz).all():
            return None
        ax, ay = float(gz[0] * W), float(gz[1] * H)
    negs = []
    b = sig["box"][side][i] if i < len(sig["box"][side]) else None
    if b is not None and np.isfinite(b).all():
        cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
        negs.append([_clamp(2 * cx - ax, 0, W - 1), _clamp(2 * cy - ay, 0, H - 1)])
    oth = "R" if side == "L" else "L"
    go = sig["grasp"][oth][i] if i < len(sig["grasp"][oth]) else None
    if go is not None and np.isfinite(go).all():
        negs.append([_clamp(go[0] * W, 0, W - 1), _clamp(go[1] * H, 0, H - 1)])
    return ax, ay, negs


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def hand_mask(state, i, side):
    """Segment THIS hand from its handpose hull keypoints, so candidates that are mostly the hand are rejected."""
    hl = sig["hull"][side][i] if (isinstance(sig.get("hull"), dict) and i < len(sig["hull"][side])) else None
    if not hl or len(hl) < 3:
        return None
    pts = (np.array(hl, float) * [W, H]).astype(np.float32)
    with torch.inference_mode(), ac:
        hm, hs, _ = model.predict_inst(state, point_coords=pts, point_labels=np.ones(len(pts), int),
                                       multimask_output=True)
    hm = _np(hm); hs = _np(hs).ravel()
    b = sig["box"][side][i] if i < len(sig["box"][side]) else None
    if b is not None and np.isfinite(b).all():
        cxi, cyi = _clamp((b[0] + b[2]) / 2 * W, 0, W - 1), _clamp((b[1] + b[3]) / 2 * H, 0, H - 1)
    else:
        cxi, cyi = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    best = None
    for k in range(len(hs) if hm.ndim == 3 else 0):
        m = hm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.003 < float(m.mean()) < 0.35 and m[cyi, cxi]:
            if best is None or m.sum() > best.sum():
                best = m
    return best


def best_held(state, i, side):
    an = anchor_negs(i, side)
    if an is None:
        return None
    ax, ay, negs = an; cands = []                                  # (manip, mask, name)
    hm = hand_mask(state, i, side)                                 # this hand's silhouette (to reject hand grabs)
    pr = _pinch_region(side, i)                                    # the thumb<->index gap (held object sits here)

    def in_pinch(m):
        return pr is None or np.logical_and(m, pr).sum() >= PIN_THR * float(pr.sum())
    for nm in names:                                               # TEXT-seg each verified object
        with torch.inference_mode(), ac:
            out = proc.set_text_prompt(state=state, prompt=nm.split(" with ")[0].strip())
        tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
        if tm.ndim == 4:
            tm = tm[:, 0]
        for k in range(len(tm) if tm.ndim == 3 else 0):
            m = tm[k] > 0.5
            if m.ndim > 2:
                m = m.squeeze()
            if m.shape != (H, W) or not m.any():
                continue
            m = _clean(m, ax, ay); a = float(m.mean())
            if not (0.0004 < a < 0.35):
                continue
            oh = _onhand(m, hm)
            if oh > ON_HAND_REJECT or not in_pinch(m):             # mostly the hand, or not in the thumb-index gap
                continue
            cands.append((_manip(m, ax, ay, float(ts[k]), oh), m, nm))
    pts = np.array([[int(ax), int(ay)]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:                               # POINT-seg at the grasp
        pm, ps, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    pm = _np(pm); ps = _np(ps).ravel()
    for k in range(len(ps) if pm.ndim == 3 else 0):
        m = pm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W) or not m.any():
            continue
        m = _clean(m, ax, ay); a = float(m.mean())
        if not (0.0004 < a < 0.35):
            continue
        oh = _onhand(m, hm)
        if oh > ON_HAND_REJECT or not in_pinch(m):                 # the hand, or not in the thumb-index gap -> drop
            continue
        nm = "object"                                              # name a point mask by best text overlap
        for _, tmk, tn in cands:
            if np.logical_and(m, tmk).sum() > 0.4 * m.sum():
                nm = tn; break
        cands.append((_manip(m, ax, ay, float(ps[k]), oh) - 0.03, m, nm))   # tiny tie-break toward text
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    return cands[0] if cands[0][0] >= MANIP_MIN else None


def short(n):
    return n.split(" with ")[0] if n else n


def draw(img, mask, col, lbl):
    cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return
    f = img.astype(np.float32); f[mask] = 0.55 * f[mask] + 0.45 * np.array(col, np.float32)
    img[:] = f.clip(0, 255).astype(np.uint8)
    cv2.drawContours(img, cont, -1, col, 2)
    ys, xs = np.nonzero(mask); cx = int(xs.mean())
    x, y = cx, max(16, int(ys.min()) - 6); (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.5, 1)
    x = min(max(2, x - tw // 2), img.shape[1] - tw - 2)
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(img, lbl, (x, y), FONT, 0.5, col, 1, cv2.LINE_AA)


cap = cv2.VideoCapture(chunk); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
outd = tempfile.mkdtemp(); drawn = 0
n = min(len(frames), len(T))
for i in range(n):
    rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB); img = rgb.copy()
    with torch.inference_mode(), ac:
        state = proc.set_image(Image.fromarray(rgb))
    for side, col in (("L", GREEN), ("R", BLUE)):
        b = best_held(state, i, side)
        if b is not None:
            draw(img, b[1], col, f"{side}: {short(b[2])}"); drawn += 1
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"t={cstart + float(T[i]):5.1f}s", (8, 17), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if i % 60 == 0:
        print(f"frame {i}/{n} drawn={drawn}", flush=True)
print(f"rendered {n} frames, {drawn} masks", flush=True)
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%04d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
