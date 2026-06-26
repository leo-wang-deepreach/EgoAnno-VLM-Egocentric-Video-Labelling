#!/usr/bin/env python3
"""viz_chunk.py — track ONE object on ONE chunk sub-video and burn the SAM3-video mask onto every frame,
so we can eyeball the dense per-frame mask quality. Run (sam3py):
  viz_chunk.py <chunk.mp4> "<object text>" <out.mp4>
"""
import subprocess
import sys
import tempfile
import os

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

chunk, name, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(chunk); W = int(cap.get(3)); H = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f"chunk: {len(frames)} frames {W}x{H} @ {fps:.0f}fps; tracking {name!r}", flush=True)

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
pred = build_sam3_video_predictor()
sid = pred.handle_request(dict(type="start_session", resource_path=chunk,
                               offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]


def union(out):
    mks = out.get("out_binary_masks")
    if mks is None or len(mks) == 0:
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


seed = None
for sf in range(0, len(frames), max(1, len(frames) // 8)):
    pred.handle_request(dict(type="reset_session", session_id=sid))
    r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=sf, text=name))
    if union(r["outputs"]) is not None:
        seed = sf; break
print(f"seed frame: {seed}", flush=True)
masks = {}
if seed is not None:
    for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
            start_frame_index=seed, propagation_direction="both")):
        u = union(resp["outputs"])
        if u is not None and u.any():
            masks[resp["frame_index"]] = u
print(f"tracked {len(masks)}/{len(frames)} frames", flush=True)

outd = tempfile.mkdtemp()
for i, f in enumerate(frames):
    vis = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    m = masks.get(i)
    if m is not None:
        ff = vis.astype(np.float32); ff[m] = 0.5 * ff[m] + 0.5 * np.array((80, 230, 80), np.float32)
        vis = ff.clip(0, 255).astype(np.uint8)
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (80, 230, 80), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, f"{name}  frame {i}/{len(frames)}" + ("" if m is not None else "  (no mask)"),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{i:04d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%04d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
