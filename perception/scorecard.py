#!/usr/bin/env python3
"""scorecard.py — per-STAGE accuracy scorecard vs the human GT (leo_edited).

Stage-isolated testing: one number per pipeline stage so each fix must move ITS number.
This file implements the stages that score WITHOUT a VLM (zero API cost):

  STAGE 1  segmentation : boundary recall / precision / F1 vs GT segment times,
                          + run-to-run determinism.

The VLM stages (direction / objects / labels) are graded separately by a fresh codex
agent comparing the v2 timeline to the GT timeline (built next). GT is read as TEXT only;
no video is opened here.

Run: ../.venv/bin/python perception/scorecard.py [<clip-tag> ...]   (default: short GT clips)
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import perception as PCP

GT_DIR = Path("/home/ubuntu/local/factsfirst/out/leo_edited")
VID_DIR = Path("/home/ubuntu/local/Videos-Leo")
TOL = 1.0          # boundary match tolerance (s) — GT segments here are seconds long


def _gt_for(tag):
    g = list(GT_DIR.glob(f"*{tag}*.json"))
    return g[0] if g else None


def _vid_for(tag):
    for ext in ("mp4", "MP4"):
        v = list(VID_DIR.glob(f"*{tag}*camera_1*.{ext}"))
        if v:
            return v[0]
    return None


def _gt_boundaries(gt_json):
    d = json.loads(Path(gt_json).read_text())
    segs = d.get("segments", [])
    bs = sorted({round(s["start_sec"], 2) for s in segs} | {round(s["end_sec"], 2) for s in segs})
    cover = (min(s["start_sec"] for s in segs), max(s["end_sec"] for s in segs)) if segs else (0, 0)
    return bs, cover, len(segs), d.get("direction", "")


def _score(detected, gt, lo, hi, tol=TOL):
    """recall/precision/F1 over INTERIOR boundaries inside the GT-covered [lo,hi]."""
    det = [b for b in detected if lo - 0.5 < b < hi + 0.5]
    g = [b for b in gt if lo + 0.3 < b < hi - 0.3]
    det_i = [b for b in det if lo + 0.3 < b < hi - 0.3]
    used = set(); m = 0
    for x in g:
        for j, y in enumerate(det_i):
            if j not in used and abs(x - y) <= tol:
                used.add(j); m += 1; break
    rec = m / len(g) if g else 0.0
    prec = m / len(det_i) if det_i else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
    return rec, prec, f1, len(g), len(det_i)


def run(tags, fps=15.0, determinism=False):
    print(f"{'clip':<8} {'dir(GT)':<22} {'GTseg':>5} {'v2seg':>5} "
          f"{'recall':>7} {'prec':>6} {'F1':>6} {'det':>5}")
    print("-" * 72)
    rows = []
    for tag in tags:
        gt = _gt_for(tag); vid = _vid_for(tag)
        if not gt or not vid:
            print(f"{tag:<8} MISSING gt={bool(gt)} vid={bool(vid)}"); continue
        gtb, (lo, hi), gtn, gdir = _gt_boundaries(gt)
        sig = PCP.extract(str(vid), fps_target=fps)
        energy, _ = PCP.motion_energy(sig)
        res = PCP.detect_segments(sig, energy)
        b1 = res["boundaries"]
        rec, prec, f1, ng, nd = _score(b1, gtb, lo, hi)
        det = "-"
        if determinism:
            energy2, _ = PCP.motion_energy(PCP.extract(str(vid), fps_target=fps))
            b2 = PCP.detect_segments(sig, energy2)["boundaries"]
            det = "YES" if (len(b1) == len(b2) and all(abs(x - y) < 0.06 for x, y in zip(b1, b2))) else "NO"
        print(f"{tag:<8} {gdir[:22]:<22} {gtn:>5} {len(res['segments']):>5} "
              f"{rec:>7.2f} {prec:>6.2f} {f1:>6.2f} {det:>5}")
        rows.append((tag, rec, prec, f1))
    if rows:
        n = len(rows)
        print("-" * 72)
        print(f"{'MEAN':<8} {'':<22} {'':>5} {'':>5} "
              f"{sum(r[1] for r in rows)/n:>7.2f} {sum(r[2] for r in rows)/n:>6.2f} "
              f"{sum(r[3] for r in rows)/n:>6.2f}")


if __name__ == "__main__":
    tags = sys.argv[1:]                                       # clip tags to score (substring of the GT filename)
    if not tags:
        sys.exit("usage: scorecard.py <clip-tag> [<clip-tag> ...]")
    run(tags)
