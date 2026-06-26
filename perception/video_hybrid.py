#!/usr/bin/env python3
"""video_hybrid.py — WHOLE-video segmented output (no chunking; image model runs per frame). Combines the
LLM IDENTITY from ground_simple (run at the action segments) with DENSE per-frame segmentation:
  * per frame, per hand: identity = the grounded object of the segment at this time (carried across gaps),
  * dense seg = TEXT-seg that object (preferred) + POINT-seg at the grasp, filtered to the thumb<->index
    PINCH gap and OFF the hand, best by manip score,
  * draw it labeled with the grounded name; if nothing valid is in the pinch this frame -> no mask (hold gate).
So names stay consistent (from the LLM grounding) and masks are dense + only what's actually held.

Run (sam3py): video_hybrid.py <video> <grasp.json> <grounding_index.json> <out.mp4> [fps]
  env: HANDPOSE_PKL ; PIN_THR(0.15) ; ON_HAND_REJECT(0.55) ; MANIP_MIN(0.30) ; OUT_W(960)
"""
import glob
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

video, grasp_f, idx_f, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
GREEN, BLUE = (80, 230, 80), (90, 160, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PIN_THR = float(os.environ.get("PIN_THR", "0.15"))
ON_HAND_REJECT = float(os.environ.get("ON_HAND_REJECT", "0.55"))
MANIP_MIN = float(os.environ.get("MANIP_MIN", "0.30"))
OUT_W = int(os.environ.get("OUT_W", "960"))

sig = pickle.load(open(os.environ["HANDPOSE_PKL"], "rb")); T = sig["t"]
fps = float(sig["fps"])   # render every native handpose frame (frame i <-> T[i]); no resampling drift

# grounded identity: per segment (global span) -> L/R object (carried across gaps)
g = json.load(open(grasp_f)); segs = {}
for p in g["prompts"]:
    if p.get("primary", True):
        segs.setdefault(p["seg_id"][:-1], {"span": p["seg"], "L": None, "R": None})
for r in json.load(open(idx_f)):
    num, h = r["seg_id"][:-1], r["seg_id"][-1]
    if num in segs:
        segs[num][h] = r["name"] if r["obj"] else None
seglist = sorted(segs.values(), key=lambda s: s["span"][0])


def object_at(side, t):
    inside = [s for s in seglist if s["span"][0] <= t <= s["span"][1]]
    s = inside[0] if inside else (min(seglist, key=lambda s: abs((s["span"][0] + s["span"][1]) / 2 - t))
                                  if seglist else None)
    return s[side] if s else None


# frames (downscaled) for the whole video
tmpd = tempfile.mkdtemp()
cap = cv2.VideoCapture(video); VW = int(cap.get(3)); VH = int(cap.get(4)); cap.release()
OH = int(round(VH * OUT_W / VW / 2) * 2)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", f"scale={OUT_W}:{OH}",
                "-an", os.path.join(tmpd, "f%05d.png")], check=True)   # every native frame -> aligns with sig["t"]
frames = sorted(glob.glob(os.path.join(tmpd, "f*.png")))
W, H = OUT_W, OH
PW = max(20, int(0.05 * W))
print(f"{len(frames)} frames vs {len(T)} handpose @ {fps:.2f}fps {W}x{H}; {len(seglist)} grounded segments", flush=True)


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


def _onhand(m, hm):
    a = float(m.sum())
    return 0.0 if (hm is None or a == 0) else float(np.logical_and(m, hm).sum()) / a


def _manip(m, ax, ay, sam, oh):
    ys, xs = np.nonzero(m); dist = (((xs.mean() - ax) ** 2 + (ys.mean() - ay) ** 2) ** 0.5) / PW
    area = float(m.mean()); sized = 1.0 if area < 0.12 else max(0.0, 1.0 - (area - 0.12) / 0.18)
    return 0.40 * max(0.0, 1.0 - dist / 2.0) + 0.27 * (1.0 - oh) + 0.18 * float(sam) + 0.15 * sized


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


def _pinch_region(side, i, dilate=12):
    ch = sig["chain"][side][i] if (isinstance(sig.get("chain"), dict) and i < len(sig["chain"][side])) else None
    if ch is None:
        return None
    pts = [[int(p[0] * W), int(p[1] * H)] for line in ch for p in line if np.isfinite(p).all()]
    if len(pts) < 3:
        return None
    reg = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(reg, cv2.convexHull(np.array(pts, np.int32)), 1)
    return cv2.dilate(reg, np.ones((dilate, dilate), np.uint8)).astype(bool)


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def hand_mask(state, i, side):
    hl = sig["hull"][side][i] if (isinstance(sig.get("hull"), dict) and i < len(sig["hull"][side])) else None
    if not hl or len(hl) < 3:
        return None
    pts = (np.array(hl, float) * [W, H]).astype(np.float32)
    with torch.inference_mode(), ac:
        hm, hs, _ = model.predict_inst(state, point_coords=pts, point_labels=np.ones(len(pts), int),
                                       multimask_output=True)
    hm = _np(hm); hs = _np(hs).ravel()
    b = sig["box"][side][i] if i < len(sig["box"][side]) else None
    cxi, cyi = ((_clamp((b[0] + b[2]) / 2 * W, 0, W - 1), _clamp((b[1] + b[3]) / 2 * H, 0, H - 1))
                if (b is not None and np.isfinite(b).all()) else (int(pts[:, 0].mean()), int(pts[:, 1].mean())))
    best = None
    for k in range(len(hs) if hm.ndim == 3 else 0):
        m = hm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.003 < float(m.mean()) < 0.35 and m[cyi, cxi]:
            if best is None or m.sum() > best.sum():
                best = m
    return best


def held(state, i, side, obj):
    """text-seg the grounded object + point-seg at the grasp; keep only in-pinch, off-hand; best by manip."""
    an = anchor_negs(i, side)
    if an is None:
        return None
    ax, ay, negs = an; hm = hand_mask(state, i, side); pr = _pinch_region(side, i); cands = []

    def good(m):
        return (0.0004 < float(m.mean()) < 0.35 and _onhand(m, hm) <= ON_HAND_REJECT
                and (pr is None or np.logical_and(m, pr).sum() >= PIN_THR * float(pr.sum())))
    with torch.inference_mode(), ac:                               # TEXT-seg the grounded object
        out = proc.set_text_prompt(state=state, prompt=obj.split(" with ")[0].strip())
    tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
    if tm.ndim == 4:
        tm = tm[:, 0]
    for k in range(len(tm) if tm.ndim == 3 else 0):
        m = tm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W) or not m.any():
            continue
        m = _clean(m, ax, ay)
        if good(m):
            cands.append((_manip(m, ax, ay, float(ts[k]), _onhand(m, hm)), m))
    pts = np.array([[int(ax), int(ay)]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:                               # POINT-seg fallback at the grasp
        pm, ps, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    pm = _np(pm); ps = _np(ps).ravel()
    for k in range(len(ps) if pm.ndim == 3 else 0):
        m = pm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W) or not m.any():
            continue
        m = _clean(m, ax, ay)
        if good(m):
            cands.append((_manip(m, ax, ay, float(ps[k]), _onhand(m, hm)) - 0.03, m))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    return cands[0][1] if cands[0][0] >= MANIP_MIN else None


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


outd = tempfile.mkdtemp(); drawn = 0
START_N = int(os.environ.get("START_N", "0"))
n = min(len(frames), len(T))
n = min(n, START_N + int(os.environ["MAX_N"])) if os.environ.get("MAX_N") else n
for j, i in enumerate(range(START_N, n)):
    rgb = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB); img = rgb.copy(); t = float(T[i])
    objs = {"L": object_at("L", t), "R": object_at("R", t)}
    if objs["L"] or objs["R"]:
        with torch.inference_mode(), ac:
            state = proc.set_image(Image.fromarray(rgb))
        for side, col in (("L", GREEN), ("R", BLUE)):
            if objs[side]:
                m = held(state, i, side, objs[side])
                if m is not None:
                    draw(img, m, col, f"{side}: {short(objs[side])}"); drawn += 1
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:5.1f}s", (8, 17), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{j:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if j % 100 == 0:
        print(f"frame {i} ({j}/{n - START_N}) drawn={drawn}", flush=True)
n = n - START_N
print(f"rendered {n} frames, {drawn} masks", flush=True)
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%05d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
