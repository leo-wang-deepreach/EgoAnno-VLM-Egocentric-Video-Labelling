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
    args = ap.parse_args()

    ep = annotate(args.video, args.out, workdir=args.workdir)
    print(f"\nclip={ep['clip']} direction={ep['direction']} "
          f"segments={len(ep['segments'])} qa={len(ep['_qa']['violations'])}")
    print(f"goal: {ep['goal']}")
    for i, seg in enumerate(ep["segments"]):
        print(f"  #{i+1} [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}] "
              f"L: {seg['left']} | R: {seg['right']}  ({seg['boundary_provenance']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
