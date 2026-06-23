#!/usr/bin/env python3
"""grounded_eval.py — NORTH-STAR number for a grounding run: of the frames where we grounded an
object, how often is it the RIGHT object per GT? (object-identity precision on the grounded subset).
GT-aware, eval-only. Judge = GPT-5.5 (reuses object_eval._judge). Also reports false-positives
(grounded an object where GT says the hand is doing nothing).

Run: ../.venv/bin/python perception/grounded_eval.py <review_dir>
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import object_eval as OE  # noqa: E402  (reuse _judge + GPT-5.5 judge)

GT_DIR = Path("/home/ubuntu/local/factsfirst/out/leo_edited")


def gt_label(gt_segs, t, hand):
    key = "left" if hand.upper().startswith("L") else "right"
    for s in gt_segs:
        if s["start_sec"] - 0.05 <= t <= s["end_sec"] + 0.05:
            return s.get(key) or "N/A"
    return None                                           # t outside GT coverage


def main():
    revdir = sys.argv[1]
    rows = []
    for f in glob.glob(f"{revdir}/_index_*.json"):
        rows += json.load(open(f))
    items, fp = [], 0
    for r in rows:
        if not r["obj"]:
            continue                                      # only judge frames where we grounded something
        gts = glob.glob(str(GT_DIR / f"*{r['tag']}*.json"))
        if not gts:
            continue
        glab = gt_label(json.loads(Path(gts[0]).read_text())["segments"], r["t"], r["hand"])
        if glab is None:
            continue                                      # no GT coverage at this time
        if glab.strip().upper() == "N/A":
            fp += 1                                        # GT: hand idle here, but we grounded an object
            continue
        items.append({"gt": glab, "pred": r["name"], "tag": r["tag"], "t": r["t"], "hand": r["hand"]})

    print(f"grounded frames judged vs GT-object: {len(items)} | false-positive (GT=N/A): {fp}")
    verdicts = OE._judge(items)
    n = sum(1 for v in verdicts if v is not None)
    good = sum(1 for v in verdicts if v is True)
    print(f"\nOBJECT-IDENTITY PRECISION (grounded subset): {good}/{n} = {good / n:.2f}" if n else "no judgeable items")
    # show the mismatches for inspection
    print("\nmismatches:")
    for it, v in zip(items, verdicts):
        if v is False:
            print(f"  {it['tag']} t={it['t']:.1f} {it['hand']:5} GT={it['gt']!r:38} PRED={it['pred']!r}")


if __name__ == "__main__":
    main()
