#!/usr/bin/env python3
"""eval.py — measure the measured-spine (perception) WITHOUT any VLM calls.

Two things you cannot eyeball:
  1. DETERMINISM — run perception twice per clip; the boundaries must be identical
     (this is the whole point: same input -> same segmentation, unlike the 3-7 VLM spread).
  2. BOUNDARY ACCURACY — recall/precision vs held-out boundaries where we have them
     (76a67a82 only, for now). GT is EVALUATION-ONLY and never enters any model.

Run: ../.venv/bin/python perception/eval.py [clip.mp4 ...]   (default: all in videos_4)
"""
from __future__ import annotations
import sys
from pathlib import Path

import perception as PCP

VIDS = Path("/home/ubuntu/local/not_for_testing_videos_4")
# held-out boundary structure — lives HERE in the eval module only, never in the pipeline.
GT_BOUNDARIES = {"76a67a82": [11.5, 13.0, 15.0, 27.0, 27.8, 29.0]}


def _interior(b):
    return [x for x in b if 0.3 < x < (b[-1] - 0.3 if b else 0)]


def _boundaries(video, fps=30.0):
    sig = PCP.extract(video, fps_target=fps)
    energy, _ = PCP.motion_energy(sig)
    return PCP.detect_segments(sig, energy)["boundaries"]


def _match(a, b, tol=0.06):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    clips = [Path(p) for p in sys.argv[1:]] or sorted(VIDS.glob("*.mp4"))
    print(f"{'clip':<14} {'#segs':>5} {'determ':>7} {'recall':>7} {'prec':>6}")
    print("-" * 46)
    det_ok = 0
    for v in clips:
        key = v.stem.split("-")[0]
        try:
            b1 = _boundaries(str(v))
            b2 = _boundaries(str(v))
        except Exception as e:                          # noqa: BLE001
            print(f"{key:<14}  ERROR {e}")
            continue
        determ = _match(b1, b2)
        det_ok += determ
        gtk = key if key in GT_BOUNDARIES else None
        if gtk:
            rec, prec, _ = PCP._score_boundaries(b1, GT_BOUNDARIES[gtk])
            rs, ps = f"{rec:.2f}", f"{prec:.2f}"
        else:
            rs = ps = "  -"
        print(f"{key:<14} {len(b1)-1:>5} {('YES' if determ else 'NO'):>7} {rs:>7} {ps:>6}")
    print("-" * 46)
    print(f"determinism: {det_ok}/{len(clips)} clips identical across 2 runs")


if __name__ == "__main__":
    main()
