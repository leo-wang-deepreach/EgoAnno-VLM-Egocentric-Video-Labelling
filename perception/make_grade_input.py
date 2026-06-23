#!/usr/bin/env python3
"""make_grade_input.py — build the v2-vs-GT comparison file a codex grader reads.

For each clip tag: pair out/v2/<tag>.json with its GT in out/leo_edited/*<tag>*.json,
emit out/v2/_grade_<tag>.json with both timelines as text + the object/direction headers.
GT is read as TEXT only (no video).
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/local/factsfirst")


def _lines(segs):
    return "\n".join(f"[{s['start_sec']:.1f}-{s['end_sec']:.1f}] "
                     f"L:{s.get('left', 'N/A')} | R:{s.get('right', 'N/A')}" for s in segs)


def _align(gt_segs, v2_segs):
    """DETERMINISTIC time-alignment: for each GT segment, the v2 label that covers its
    MIDPOINT (single, unambiguous). Removes the alignment variance that made codex grading
    noisy — codex then only judges per-row equivalence with a fixed denominator (#GT segs)."""
    rows = []
    for g in gt_segs:
        mid = (g["start_sec"] + g["end_sec"]) / 2.0
        cov = [v for v in v2_segs if v["start_sec"] - 0.01 <= mid <= v["end_sec"] + 0.01]
        v = cov[0] if cov else min(v2_segs, key=lambda x: abs((x["start_sec"] + x["end_sec"]) / 2 - mid))
        rows.append({"t": f"{g['start_sec']:.1f}-{g['end_sec']:.1f}",
                     "GT_L": g.get("left", "N/A"), "GT_R": g.get("right", "N/A"),
                     "V2_L": v.get("left", "N/A"), "V2_R": v.get("right", "N/A")})
    return rows


def build(tag):
    vp = ROOT / "out" / "v2" / f"{tag}.json"
    gps = glob.glob(str(ROOT / "out" / "leo_edited" / f"*{tag}*.json"))
    if not vp.exists() or not gps:
        print(f"{tag}: MISSING v2={vp.exists()} gt={bool(gps)}")
        return None
    v = json.loads(vp.read_text()); g = json.loads(Path(gps[0]).read_text())
    aligned = _align(g["segments"], v["segments"])
    out = {"clip": tag, "gt_direction": g.get("direction"), "v2_direction": v.get("direction"),
           "gt_objects": [o.get("name") for o in g.get("objects", []) if isinstance(o, dict)],
           "v2_objects": [o.get("name") for o in v.get("objects", [])],
           "n_gt": len(g["segments"]),
           "ALIGNED": "\n".join(f"{i+1}. [{r['t']}] GT: L={r['GT_L']} | R={r['GT_R']}  <=>  "
                                f"V2: L={r['V2_L']} | R={r['V2_R']}" for i, r in enumerate(aligned))}
    op = ROOT / "out" / "v2" / f"_grade_{tag}.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"{tag}: {len(g['segments'])} GT / {len(v['segments'])} v2 segs | "
          f"dir GT={g.get('direction')} v2={v.get('direction')} -> {op.name}")
    return str(op)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        build(t)
