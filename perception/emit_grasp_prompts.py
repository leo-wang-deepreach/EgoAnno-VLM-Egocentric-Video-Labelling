#!/usr/bin/env python3
"""emit_grasp_prompts.py — S5 bridge step A (runs in the factsfirst 3.10 venv).

The measured spine already knows, per frame per hand, the fingertip "grasp" centroid (where the
held object sits) and the hand box. This emits those as SAM3 point prompts at the midpoint of the
longest ACTION segments — the moments a hand is most likely holding something. SAM3 itself runs in
a DIFFERENT venv, so this writes a JSON the SAM3-venv step (ground_probe.py) consumes; no SAM here.

Run: ../.venv/bin/python perception/emit_grasp_prompts.py <video> [out.json] [n_segs] [fps]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perception as PCP                                     # noqa: E402


def _robust_idx(sig, hand, idx0, K=3):
    """Sanity-gate the handpose: if frame idx0's keypoints for `hand` look mis-detected (box-center jumps
    from the local median, keypoints spread = grip outlier, or tips missing), substitute the nearest
    temporally-consistent frame in a +/-K window. Returns the index to read this hand's keypoints from."""
    cx, cy = sig["cx"][hand], sig["cy"][hand]; pres = sig["pres"][hand]
    grip, tip = sig["grip"][hand], sig["tip"][hand]
    n = len(cx); lo, hi = max(0, idx0 - K), min(n, idx0 + K + 1)
    win = [i for i in range(lo, hi) if pres[i] and not np.isnan(cx[i])]
    if not win:
        return idx0
    mcx, mcy = np.median([cx[i] for i in win]), np.median([cy[i] for i in win])
    gw = [grip[i] for i in win if not np.isnan(grip[i])]
    mg = np.median(gw) if gw else np.nan
    tw = [i for i in win if not np.any(np.isnan(tip[i]))]    # frames with valid tips
    mth = np.median([tip[i][0] for i in tw], axis=0) if tw else None   # local median thumb-tip [x,y]
    mix = np.median([tip[i][1] for i in tw], axis=0) if tw else None   # local median index-tip [x,y]

    def good(i):
        if np.any(np.isnan(tip[i])):                         # tips must be detected
            return False
        if ((cx[i] - mcx) ** 2 + (cy[i] - mcy) ** 2) ** 0.5 > 0.06:   # box-center jumped from local median
            return False
        if not np.isnan(mg) and not np.isnan(grip[i]) and grip[i] > mg + 0.08:   # keypoints spread (mis-detect)
            return False
        if mth is not None and (((tip[i][0][0] - mth[0]) ** 2 + (tip[i][0][1] - mth[1]) ** 2) ** 0.5 > 0.05
                                or ((tip[i][1][0] - mix[0]) ** 2 + (tip[i][1][1] - mix[1]) ** 2) ** 0.5 > 0.05):
            return False                                     # thumb/index tip jumped from its local median
        return True
    if good(idx0):
        return idx0
    cand = [i for i in win if good(i)]
    return min(cand, key=lambda i: abs(i - idx0)) if cand else idx0


def main():
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "perception/_grasp_prompts.json"
    n_segs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0   # probe: 10fps is plenty for sampling

    sig = PCP.extract(video, fps_target=fps)
    energy, _ = PCP.motion_energy(sig)
    res = PCP.detect_segments(sig, energy)
    W, H = sig["W"], sig["H"]
    t = sig["t"]
    actions = sorted([s for s in res["segments"] if s["type"] == "action"],
                     key=lambda s: s["end"] - s["start"], reverse=True)[:n_segs]
    if not actions:                                          # fallback: even sampling
        ts = np.linspace(t[0], t[-1], n_segs)
        actions = [{"start": float(x), "end": float(x), "type": "action"} for x in ts]

    def _px(pt):
        return None if np.any(np.isnan(pt)) else [int(pt[0] * W), int(pt[1] * H)]

    def _make_prompt(idx0, hand, seg_id, primary, s):
        """One grasp prompt at nominal frame idx0 for `hand` (keypoints robust-gated). None if keypoints nan."""
        idx = _robust_idx(sig, hand, idx0)   # sanity-gate keypoints (fix mis-detections), keep the frame time
        g = sig["pinch"][hand][idx]          # PINCH point (thumb↔fingers) — kept as a reference dot
        b = sig["box"][hand][idx]
        if np.any(np.isnan(g)) or np.any(np.isnan(b)):
            return None
        hull_px = [[int(px[0] * W), int(px[1] * H)] for px in sig["hull"][hand][idx]]  # all-fingers polygon
        gr = sig["grip"][hand][idx]                          # fingertip-spread / box-diag (low = closed grip)
        tp = sig["tip"][hand][idx]                           # [[thumb x,y],[index x,y]] normalized (nan if low-conf)
        ch = sig["chain"][hand][idx]                         # [thumb kp1-4, index kp5-8] normalized finger lines

        def _chain_px(pts):
            return [[int(px[0] * W), int(px[1] * H)] for px in pts if not np.any(np.isnan(px))]
        return {
            "t": round(float(t[idx0]), 3), "hand": hand,    # segment frame time (both hands paired)
            "seg_id": seg_id, "primary": primary,           # TEMPORAL: tracklet id + which frame is the OUTPUT frame
            "x_px": int(g[0] * W), "y_px": int(g[1] * H),
            "box_px": [int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)],
            "hull_px": hull_px,                  # ALL-FINGERS grasp region (wrist+MCPs+fingertips)
            "thumb_px": _px(tp[0]), "index_px": _px(tp[1]),  # precision-pinch pair (object in the gap)
            "thumb_chain_px": _chain_px(ch[0]), "index_chain_px": _chain_px(ch[1]),  # the two FINGER LINES
            "grip": (round(float(gr), 3) if not np.isnan(gr) else None),  # hold-vs-hover signal
            "seg": [round(s["start"], 2), round(s["end"], 2)],
        }

    # TEMPORAL window: per action segment per hand emit the MIDPOINT (primary = the output frame) plus neighbor
    # frames at 1/4 and 3/4 of the span. A neighbor with cleaner keypoints / less occlusion lets the grounder
    # recover an object the midpoint frame misses (Mode-2 empty pool, bad-keypoint pile), then CARRY that
    # identity back to the primary (Layer-2). Primary listed FIRST so a short-segment dup can't shadow it.
    NEIGHBORS = [(0.5, True), (0.25, False), (0.75, False)]
    prompts = []
    for si, s in enumerate(actions):
        span = max(0.0, s["end"] - s["start"])
        for hand in ("L", "R"):
            seg_id = f"s{si:02d}{hand}"
            seen = set()
            for frac, primary in NEIGHBORS:
                idx0 = int(np.argmin(np.abs(t - (s["start"] + frac * span))))
                if idx0 in seen:                 # short segment: neighbor collapsed onto an emitted frame
                    continue
                pr = _make_prompt(idx0, hand, seg_id, primary, s)
                if pr is None:
                    continue
                seen.add(idx0)
                prompts.append(pr)
    Path(out).write_text(json.dumps({"video": video, "W": W, "H": H, "fps": sig["fps"],
                                     "prompts": prompts}, indent=2))
    print(f"emitted {len(prompts)} grasp prompts from {len(actions)} action segs -> {out}")


if __name__ == "__main__":
    main()
