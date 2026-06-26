#!/usr/bin/env python3
"""chunk_pointseg.py — DENSE held-object masks via per-frame POINT-SEG at the hand (the point-seg lane of
ground_simple, run every frame), labeled by the grounding's per-segment object. Appearance-agnostic: it
segments whatever is held between the fingertips (bead-filled cup included), so the mask = what the hand
manipulates. No video-track (which lost the held object), no per-frame LLM (identity reused from grounding).

Run (sam3py): chunk_pointseg.py <chunk.mp4> <chunk_start_sec> <grasp.json> <grounding_index.json> <out.mp4>
  env: HANDPOSE_PKL (per-frame hand keypoints)
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

chunk, cstart, grasp_f, idx_f, out_mp4 = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
GREEN, BLUE, YELLOW = (80, 230, 80), (90, 160, 255), (70, 220, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(chunk); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30
cn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
sig = pickle.load(open(os.environ["HANDPOSE_PKL"], "rb")); T = sig["t"]

g = json.load(open(grasp_f)); segs = {}
for p in g["prompts"]:
    if p.get("primary", True):
        segs.setdefault(p["seg_id"][:-1], {"span": p["seg"], "L": None, "R": None})
for r in json.load(open(idx_f)):
    num, h = r["seg_id"][:-1], r["seg_id"][-1]
    if num in segs:
        segs[num][h] = r["name"] if r["obj"] else None
cend = cstart + cn / fps
seglist = sorted([s for s in segs.values() if s["span"][1] >= cstart and s["span"][0] <= cend],
                 key=lambda s: s["span"][0])


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


def anchor_negs(i, side):
    """grasp anchor (thumb-index midpoint) + negatives (wrist, palm, other hand) in pixels, for frame i."""
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
        cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H               # wrist ~ reflect anchor thru box center
        negs.append([_clamp(2 * cx - ax, 0, W - 1), _clamp(2 * cy - ay, 0, H - 1)])
    hl = sig["hull"][side][i] if (isinstance(sig.get("hull"), dict) and i < len(sig["hull"][side])) else None
    if hl:
        ar = np.array(hl, float) * [W, H]
        negs.append([_clamp(ar[:, 0].mean(), 0, W - 1), _clamp(ar[:, 1].mean(), 0, H - 1)])      # palm
    oth = "R" if side == "L" else "L"
    go = sig["grasp"][oth][i] if i < len(sig["grasp"][oth]) else None
    if go is not None and np.isfinite(go).all():
        negs.append([_clamp(go[0] * W, 0, W - 1), _clamp(go[1] * H, 0, H - 1)])                  # other hand
    return ax, ay, negs


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)

cap = cv2.VideoCapture(chunk); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()


def _ok(m):
    a = float(m.mean()); return 0.0003 < a < 0.35                       # sane size (not speck, not background)


def seg_held(state, i, side, text):
    """TEXT-seg first: use an instance of this object that sits AT the hand (cleaner mask). Only if text
    finds nothing at the hand, fall back to POINT-seg at the grasp (appearance-agnostic). state = encoded once."""
    an = anchor_negs(i, side)
    if an is None:
        return None
    ax, ay, negs = an
    aj, ai = _clamp(ax, 0, W - 1), _clamp(ay, 0, H - 1)
    # --- TEXT-SEG: instance AT the hand? ---
    with torch.inference_mode(), ac:
        out = proc.set_text_prompt(state=state, prompt=text.split(" with ")[0].strip())
    tm = _np(out["masks"]); tbest = None
    if tm.ndim == 4:
        tm = tm[:, 0]
    for k in range(len(tm) if tm.ndim == 3 else 0):
        m = tm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W) or not m.any():
            continue
        m = _clean(m, ax, ay)
        if not _ok(m):
            continue
        ys, xs = np.nonzero(m); d = ((xs.mean() - ax) ** 2 + (ys.mean() - ay) ** 2) ** 0.5
        if not (m[ai, aj] or d < 0.10 * W):                            # must be AT the hand (else it's a rack/far one)
            continue
        if tbest is None or d < tbest[0]:
            tbest = (d, m)
    if tbest is not None:
        return tbest[1], "text"
    # --- POINT-SEG fallback: segment whatever is held at the grasp ---
    pts = np.array([[int(ax), int(ay)]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:
        pm, ps, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    pm = _np(pm); ps = _np(ps).ravel(); best = None
    for k in range(len(ps) if pm.ndim == 3 else 0):
        m = pm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W) or not m.any():
            continue
        m = _clean(m, ax, ay)
        if not _ok(m):
            continue
        ys, xs = np.nonzero(m); d = ((xs.mean() - ax) ** 2 + (ys.mean() - ay) ** 2) ** 0.5
        score = float(ps[k]) - 0.002 * d
        if best is None or score > best[0]:
            best = (score, m)
    return (best[1], "point") if best else None


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
for i, f in enumerate(frames):
    if i >= len(T):
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); tg = cstart + float(T[i])
    s = next((s for s in seglist if s["span"][0] <= tg <= s["span"][1]), None)
    img = rgb.copy()
    if s and (s["L"] or s["R"]):
        with torch.inference_mode(), ac:                               # encode the frame ONCE (shared)
            state = proc.set_image(Image.fromarray(rgb))
        rl = seg_held(state, i, "L", s["L"]) if s["L"] else None
        rr = seg_held(state, i, "R", s["R"]) if s["R"] else None
        ml = rl[0] if rl else None; mr = rr[0] if rr else None
        same = (s["L"] and s["L"] == s["R"] and ml is not None and mr is not None
                and np.logical_and(ml, mr).sum() > 0.2 * min(ml.sum(), mr.sum()))
        if same:
            draw(img, np.logical_or(ml, mr), YELLOW, short(s["L"]) + f" [{rl[1]}]"); drawn += 1
        else:
            if ml is not None:
                draw(img, ml, GREEN, "L: " + short(s["L"]) + f" [{rl[1]}]"); drawn += 1
            if mr is not None:
                draw(img, mr, BLUE, "R: " + short(s["R"]) + f" [{rr[1]}]"); drawn += 1
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"t={tg:5.1f}s", (8, 17), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if i % 60 == 0:
        print(f"frame {i}/{len(frames)} drawn={drawn}", flush=True)
print(f"rendered {min(len(frames), len(T))} frames, {drawn} draws", flush=True)
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%04d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
