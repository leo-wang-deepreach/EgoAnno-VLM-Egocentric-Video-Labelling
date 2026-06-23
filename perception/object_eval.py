#!/usr/bin/env python3
"""object_eval.py — NORTH-STAR metric: OBJECT-IDENTITY ACCURACY of the measured-spine
labeler vs human GT (does the labeler name the RIGHT object the hand manipulates?).

EVAL-ONLY and GT-aware BY DESIGN: reads out/leo_edited GT, but is NEVER imported by the
pipeline, so it cannot leak GT into annotation (same contract as eval.py / scorecard.py).
The judge is GPT-5.5 — a DIFFERENT family from the Claude labeler, so the labeler does not
grade itself; consistent with the held-out-review doctrine.

Method (per clip):
  1. Align each GT segment to the predicted label covering its midpoint (reuse make_grade
     _align — deterministic, isolates object naming from segmentation error).
  2. For every hand-slot where GT names an object (non-N/A), ask GPT-5.5 whether the PREDICTED
     label names the SAME physical object — ignoring verb, phrasing, and L/R.
  3. Score = matches / (GT hand-slots that name an object). Pred=N/A on such a slot = miss.

Run: ../.venv/bin/python perception/object_eval.py [<clip-tag> ...]
     (default: every clip with BOTH a prediction in out/v2 and GT in out/leo_edited)
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # perception/ for make_grade_input + models
from make_grade_input import _align                          # noqa: E402  (deterministic alignment)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # factsfirst/ for models
from models import GPT_MODEL, USAGE, gpt_call                # noqa: E402

ROOT = Path("/home/ubuntu/local/factsfirst")
GT_DIR = ROOT / "out" / "leo_edited"
PRED_DIR = ROOT / "out" / "v2"
CHUNK = 40                                                   # judge rows per GPT call

SYSTEM = ("You grade OBJECT-IDENTITY for an egocentric robot-manipulation labeler. You are "
          "given, per row, a GROUND-TRUTH hand label and a PREDICTED hand label. Judge ONLY "
          "whether the OBJECT the hand manipulates is the SAME physical thing in both — IGNORE "
          "the verb/action, ignore wording/phrasing differences (e.g. '<color> <object>' == "
          "'the <object>' == '<object>'), ignore left/right. If a hand-held TOOL acts on a part, the manipulated "
          "object is the TOOL. match=false if the predicted object is a different thing, or if "
          "the prediction names no object (N/A) while GT names one. Output strict JSON only.")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "gt_object": {"type": "string"},
                    "pred_object": {"type": "string"},
                    "match": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["idx", "gt_object", "pred_object", "match", "reason"],
                "additionalProperties": False,
            },
        }
    },
}


def _is_obj(label):
    return bool(label) and label.strip().upper() != "N/A"


def _items_for(tag):
    """Return judge items [{idx, clip, t, hand, gt, pred}] for one clip (GT slots that name an object)."""
    vp = PRED_DIR / f"{tag}.json"
    gps = glob.glob(str(GT_DIR / f"*{tag}*.json"))
    if not vp.exists() or not gps:
        return None
    v = json.loads(vp.read_text()); g = json.loads(Path(gps[0]).read_text())
    rows = _align(g["segments"], v["segments"])
    items = []
    for r in rows:
        for hand, gt, pred in (("L", r["GT_L"], r["V2_L"]), ("R", r["GT_R"], r["V2_R"])):
            if _is_obj(gt):                                  # denominator = GT slots that name an object
                items.append({"clip": tag, "t": r["t"], "hand": hand,
                              "gt": gt, "pred": pred if _is_obj(pred) else "N/A"})
    return items


def _judge(items):
    """Run GPT-5.5 over judge items in chunks; return per-item match bool (aligned to items)."""
    verdicts = [None] * len(items)
    for s in range(0, len(items), CHUNK):
        chunk = items[s:s + CHUNK]
        rows = "\n".join(f"{i}) GT: {it['gt']}  ||  PRED: {it['pred']}" for i, it in enumerate(chunk))
        prompt = ("Judge object-identity for each row. Return one verdict per row (same idx).\n\n" + rows)
        try:
            r = gpt_call(prompt, [], SYSTEM, SCHEMA, model=GPT_MODEL, max_tokens=8000)
            for v in r.get("verdicts", []):
                j = v.get("idx")
                if isinstance(j, int) and 0 <= j < len(chunk):
                    verdicts[s + j] = bool(v.get("match"))
        except Exception as e:                               # noqa: BLE001
            print(f"  [judge] chunk {s}-{s+len(chunk)} failed: {e}")
    return verdicts


def main():
    global PRED_DIR
    argv = sys.argv[1:]
    if "--pred-dir" in argv:                                  # score a different prediction set (e.g. grounded)
        i = argv.index("--pred-dir")
        PRED_DIR = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    tags = argv
    if not tags:
        preds = {p.stem for p in PRED_DIR.glob("*.json") if not p.stem.startswith("_")}
        gts = {Path(g).name for g in glob.glob(str(GT_DIR / "*.json"))}
        tags = sorted(t for t in preds if any(t in name for name in gts))
    print(f"object-identity eval (judge={GPT_MODEL}) on: {', '.join(tags) or '(none)'}\n")
    print(f"{'clip':<8} {'slots':>5} {'match':>5} {'acc':>6} {'pred=N/A miss':>13}")
    print("-" * 42)
    all_items, all_match, report = [], [], {}
    for tag in tags:
        items = _items_for(tag)
        if items is None:
            print(f"{tag:<8}  MISSING pred or GT"); continue
        if not items:
            print(f"{tag:<8} {0:>5} {0:>5} {'-':>6}"); continue
        m = _judge(items)
        n = sum(1 for x in m if x is not None)
        good = sum(1 for x in m if x is True)
        miss_na = sum(1 for it, x in zip(items, m) if x is False and it["pred"] == "N/A")
        acc = good / n if n else 0.0
        print(f"{tag:<8} {n:>5} {good:>5} {acc:>6.2f} {miss_na:>13}")
        report[tag] = {"slots": n, "match": good, "acc": round(acc, 4), "pred_na_miss": miss_na,
                       "rows": [{**it, "match": x} for it, x in zip(items, m)]}
        all_items += items; all_match += m
    n = sum(1 for x in all_match if x is not None)
    good = sum(1 for x in all_match if x is True)
    overall = good / n if n else 0.0
    print("-" * 42)
    print(f"{'OVERALL':<8} {n:>5} {good:>5} {overall:>6.2f}")
    print(f"\n{USAGE.summary()}")
    out = PRED_DIR / "_object_eval.json"
    out.write_text(json.dumps({"judge": GPT_MODEL, "overall_acc": round(overall, 4),
                               "n_slots": n, "n_match": good, "by_clip": report}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
