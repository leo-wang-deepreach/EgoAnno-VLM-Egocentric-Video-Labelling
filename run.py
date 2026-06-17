#!/usr/bin/env python3
"""run.py — CLI entry for the facts-first egoanno pipeline.

  python run.py VIDEO.mp4 --out out/ep.json [--workdir logs/x] [--passes 2]

Keys are read from this dir or the egoanno root (.gemini_key/.openai_key/.anthropic_key).
"""
from __future__ import annotations

import argparse
import sys

from pipeline import annotate


def main():
    ap = argparse.ArgumentParser(description="facts-first egocentric annotation")
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--passes", type=int, default=2, help="max rough->refine passes")
    ap.add_argument("--attempts", type=int, default=3,
                    help="max full Phase 4-5 quality-rerun attempts (facts cached)")
    ap.add_argument("--apply-overrides", action="store_true",
                    help="apply out/overrides/<id>.yaml at export (OFF for honest eval)")
    ap.add_argument("--no-hand-overlay", action="store_true",
                    help="disable the YOLO green=L/blue=R hand-circle overlay pre-step")
    ap.add_argument("--facts-first", action="store_true",
                    help="use the legacy facts-first flow (1A/1C-seeded); DEFAULT is the lean circle-grounded flow")
    args = ap.parse_args()

    ep = annotate(args.video, args.out, workdir=args.workdir,
                  max_passes=args.passes, max_attempts=args.attempts,
                  apply_overrides=args.apply_overrides,
                  hand_overlay=not args.no_hand_overlay, lean=not args.facts_first)
    print(f"\nclip={ep['clip']} direction={ep['direction']} "
          f"segments={len(ep['segments'])} qa={len(ep['_qa']['violations'])}")
    print(f"goal: {ep['goal']}")
    for i, seg in enumerate(ep["segments"]):
        print(f"  #{i+1} [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}] "
              f"L: {seg['left']} | R: {seg['right']}  ({seg['boundary_provenance']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
