#!/usr/bin/env python3
"""chunk_full2.py — HAND-ANCHORED full pipeline on one chunk (correct + faster).

Per (segment, hand) with a grounded object:
  1. SEED the object's text prompt AT THE GRASP FRAME (so the held instance is in the tracked set),
  2. propagate ONLY over the segment span (bounded forward+backward from the grasp frame),
  3. take the UNION of instances; in the render, isolate the held one via NEAREST-BLOB-TO-THE-HAND
     (per-frame hand anchor) — robust, no reliance on unstable obj_ids.
So the tracked + labeled mask is the object the hand actually manipulates, and we only track the
manipulated objects over their segments (not all inventory over the whole chunk).

Run (sam3py): chunk_full2.py <chunk.mp4> <chunk_start_sec> <grasp.json> <grounding_index.json> <out.mp4>
  env: HANDPOSE_PKL (per-frame hand anchors, precomputed in the 3.10 venv)
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

from sam3.model_builder import build_sam3_video_predictor

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
print(f"chunk {os.path.basename(chunk)} [{cstart:.0f}-{cend:.0f}s], {len(seglist)} segments overlap", flush=True)


def union(out):
    mks = out.get("out_binary_masks")
    if mks is None:
        return None
    u = None
    for m in mks:
        m = np.asarray(m)
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m.astype(bool)
        if m.any():
            u = m if u is None else (u | m)
    return u


def nearest_blob(mask, ax, ay):
    m = mask.astype(np.uint8); n, lab, _, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return mask
    ai = min(max(int(ay), 0), m.shape[0] - 1); aj = min(max(int(ax), 0), m.shape[1] - 1); sl = int(lab[ai, aj])
    if sl == 0:
        sl = min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
    return lab == sl


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
pred = build_sam3_video_predictor()
sid = pred.handle_request(dict(type="start_session", resource_path=chunk,
                               offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]

seg_tracks = {}                                            # (seg_idx, hand) -> {frame: union_mask}
for si, s in enumerate(seglist):
    gf = max(0, min(cn - 1, int(round(((s["span"][0] + s["span"][1]) / 2 - cstart) * fps))))
    fa = max(0, int(round((s["span"][0] - cstart) * fps)))
    fb = min(cn - 1, int(round((s["span"][1] - cstart) * fps)))
    for hand in ("L", "R"):
        obj = s[hand]
        if not obj:
            continue
        text = obj.split(" with ")[0].strip()
        dd = {}
        # forward + backward from the grasp frame, BOUNDED to the segment span (re-seed for each direction)
        for direction, span in (("forward", fb - gf), ("backward", gf - fa)):
            pred.handle_request(dict(type="reset_session", session_id=sid))
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=gf, text=text))
            for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                    start_frame_index=gf, max_frame_num_to_track=max(1, span), propagation_direction=direction)):
                fi = resp["frame_index"]
                if fi < fa or fi > fb:
                    continue
                u = union(resp["outputs"])
                if u is not None and u.any():
                    dd[fi] = u
        seg_tracks[(si, hand)] = dd
        print(f"seg{si} {hand} {obj}: span [{fa},{fb}] gf={gf} -> {len(dd)} frames (union)", flush=True)
pred.handle_request(dict(type="close_session", session_id=sid)); torch.cuda.empty_cache()

# render — isolate the held instance per frame via nearest-blob to that hand's grasp anchor
cap = cv2.VideoCapture(chunk); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()


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


def held_at(si, hand, i):
    dd = seg_tracks.get((si, hand))
    if not dd:
        return None
    k = min(dd, key=lambda kk: abs(kk - i))
    if abs(k - i) > 3:
        return None
    ga = sig["grasp"][hand][i] if i < len(sig["grasp"][hand]) else [np.nan, np.nan]
    if not np.isfinite(ga).all():
        return None
    return nearest_blob(dd[k], ga[0] * W, ga[1] * H)


outd = tempfile.mkdtemp(); drawn = 0
for i, f in enumerate(frames):
    if i >= len(T):
        break
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); tg = cstart + float(T[i])
    si = next((j for j, s in enumerate(seglist) if s["span"][0] <= tg <= s["span"][1]), None)
    if si is not None:
        s = seglist[si]; ml = held_at(si, "L", i); mr = held_at(si, "R", i)
        same = (s["L"] and s["L"] == s["R"] and ml is not None and mr is not None
                and np.logical_and(ml, mr).sum() > 0.2 * min(ml.sum(), mr.sum()))
        if same:
            draw(img, np.logical_or(ml, mr), YELLOW, short(s["L"])); drawn += 1
        else:
            if ml is not None:
                draw(img, ml, GREEN, "L: " + short(s["L"])); drawn += 1
            if mr is not None:
                draw(img, mr, BLUE, "R: " + short(s["R"])); drawn += 1
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"t={tg:5.1f}s", (8, 17), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"rendered {min(len(frames), len(T))} frames, {drawn} draws", flush=True)
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%04d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
