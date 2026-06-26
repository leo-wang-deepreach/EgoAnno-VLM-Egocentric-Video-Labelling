#!/usr/bin/env python3
"""emit_dense_prompts.py — like emit_grasp_prompts but DENSE: one prompt per frame at <fps> per hand over a
time window [t0,t1], every frame PRIMARY with a unique seg_id (so ground_simple runs the FULL per-frame
pipeline with no cross-frame temporal carry). Feeds ground_simple.py for the "every frame through the good
grounder" experiment.

Run (3.10 venv): emit_dense_prompts.py <video> <out.json> <t0> <t1> [fps=10]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perception as PCP                                       # noqa: E402
from emit_grasp_prompts import _robust_idx                     # noqa: E402

video, out = sys.argv[1], sys.argv[2]
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
fps = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0

sig = PCP.extract(video, fps_target=fps)
W, H = sig["W"], sig["H"]; t = sig["t"]


def _px(pt):
    return None if np.any(np.isnan(pt)) else [int(pt[0] * W), int(pt[1] * H)]


def _chain_px(pts):
    return [[int(px[0] * W), int(px[1] * H)] for px in pts if not np.any(np.isnan(px))]


def _make(idx0, hand):
    idx = _robust_idx(sig, hand, idx0)                         # same keypoint sanity-gate as the sparse path
    g = sig["pinch"][hand][idx]; b = sig["box"][hand][idx]
    if np.any(np.isnan(g)) or np.any(np.isnan(b)):
        return None
    tp = sig["tip"][hand][idx]; ch = sig["chain"][hand][idx]
    return {
        "t": round(float(t[idx0]), 3), "hand": hand,
        "seg_id": f"d{idx0:04d}{hand}", "primary": True,       # unique tracklet/frame -> no temporal carry
        "x_px": int(g[0] * W), "y_px": int(g[1] * H),
        "box_px": [int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)],
        "hull_px": [[int(px[0] * W), int(px[1] * H)] for px in sig["hull"][hand][idx]],
        "thumb_px": _px(tp[0]), "index_px": _px(tp[1]),
        "thumb_chain_px": _chain_px(ch[0]), "index_chain_px": _chain_px(ch[1]),
        "grip": (round(float(sig["grip"][hand][idx]), 3) if not np.isnan(sig["grip"][hand][idx]) else None),
        "seg": [t0, t1],
    }


prompts = []
for idx0 in range(len(t)):
    if not (t0 <= t[idx0] <= t1):
        continue
    for hand in ("L", "R"):
        p = _make(idx0, hand)
        if p is not None:
            prompts.append(p)
Path(out).write_text(json.dumps({"video": video, "W": W, "H": H, "fps": sig["fps"], "prompts": prompts}))
nf = len({p["t"] for p in prompts})
print(f"emitted {len(prompts)} dense prompts over [{t0},{t1}]s @ {fps}fps ({nf} frames, both hands) -> {out}")
