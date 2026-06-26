#!/usr/bin/env python3
"""perception.py — measured event/segment spine from the handpose model.

NO VLM. Reads a video, runs the fine-tuned handpose detector, and derives a
DETERMINISTIC structure from physical signals:
  * per-frame per-hand: centroid, grip ratio (fingertip-spread / box-diag), wrist
  * motion energy   = max-over-hands of smoothed centroid speed (units/sec)
  * transition burst = sustained high-motion interval (a place / pick-up / handoff)
  * action plateau   = the quiet span between bursts (a continuous manipulation)
Boundaries = burst edges. This is the segmentation spine; the VLM only LABELS the
fixed spans later.

Run:  ../.venv/bin/python perception/perception.py <video> [--fps 30] [--eval]
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent / "yolo_hands" / "yolo_bundle"
sys.path.insert(0, str(BUNDLE))
MODEL = BUNDLE / "hand_yolo_detector@20260314.pt"

TIPS = (4, 8, 12, 16, 20)
WRIST = 0


def _grip(kpts, box):
    tips = [(float(kpts[t][0]), float(kpts[t][1])) for t in TIPS if kpts[t][2] > 0.3]
    if len(tips) < 3:
        return np.nan
    xs = [p[0] for p in tips]; ys = [p[1] for p in tips]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    x1, y1, x2, y2 = box
    return spread / max(1.0, math.hypot(x2 - x1, y2 - y1))


def _smooth(a, win):
    """centered moving average, NaN-aware."""
    a = np.asarray(a, float)
    out = np.full_like(a, np.nan)
    half = win // 2
    for i in range(len(a)):
        seg = a[max(0, i - half):i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def extract(video, fps_target=30.0, conf=0.3):
    """Run the detector and return per-frame normalized signals."""
    from hand_yolo_infer import detect_hands  # noqa
    from ultralytics import YOLO
    det = YOLO(str(MODEL))
    cap = cv2.VideoCapture(video)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(src / fps_target))
    fps = src / step
    T, fi = [], 0
    cx = {"L": [], "R": []}; cy = {"L": [], "R": []}
    grip = {"L": [], "R": []}; pres = {"L": [], "R": []}
    box = {"L": [], "R": []}                          # normalized [x0,y0,x1,y1] per frame
    grasp = {"L": [], "R": []}                         # normalized fingertip centroid (where the
    #                                                    object held by this hand sits) per frame
    pinch = {"L": [], "R": []}                          # normalized PINCH point: midpoint between the
    #                                                    thumb tip and the other-4-fingertip centroid —
    #                                                    where a grasped object actually sits
    hull = {"L": [], "R": []}                            # normalized wrist+MCP+fingertip points per frame —
    #                                                    the ALL-FINGERS grasp region (convex hull downstream)
    tip = {"L": [], "R": []}                              # normalized [thumb-tip kp4, index-tip kp8] per frame —
    #                                                    the precision-pinch pair (object sits in their gap)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi % step == 0:
            T.append(fi / src)
            hands = detect_hands(det, bgr, conf=conf)
            for side, cl in (("L", 0), ("R", 1)):
                h = max([x for x in hands if x["cls"] == cl],
                        key=lambda x: x["conf"], default=None)
                if h is None:
                    cx[side].append(np.nan); cy[side].append(np.nan)
                    grip[side].append(np.nan); pres[side].append(0)
                    box[side].append([np.nan] * 4); grasp[side].append([np.nan, np.nan])
                    pinch[side].append([np.nan, np.nan]); hull[side].append([])
                    tip[side].append([[np.nan, np.nan], [np.nan, np.nan]])
                else:
                    x1, y1, x2, y2 = [float(v) for v in h["box"]]
                    cx[side].append((x1 + x2) / 2 / W); cy[side].append((y1 + y2) / 2 / H)
                    grip[side].append(_grip(h["kpts"], h["box"]) if h["kpts"] is not None else np.nan)
                    pres[side].append(1)
                    box[side].append([x1 / W, y1 / H, x2 / W, y2 / H])
                    kp = h["kpts"]
                    tips = ([(float(kp[t][0]) / W, float(kp[t][1]) / H)
                             for t in TIPS if kp[t][2] > 0.3] if kp is not None else [])
                    grasp[side].append([float(np.mean([p[0] for p in tips])),
                                        float(np.mean([p[1] for p in tips]))]
                                       if len(tips) >= 2 else [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H])
                    # PINCH: midpoint between the THUMB tip (kp4) and the other-4-fingertip centroid
                    # (kp8/12/16/20) — where a held object sits, not the all-finger mean (which is on skin)
                    th = kp[4] if kp is not None else None
                    oth = ([(float(kp[t][0]) / W, float(kp[t][1]) / H)
                            for t in (8, 12, 16, 20) if kp[t][2] > 0.3] if kp is not None else [])
                    if th is not None and th[2] > 0.3 and oth:
                        oxm = float(np.mean([p[0] for p in oth])); oym = float(np.mean([p[1] for p in oth]))
                        pinch[side].append([0.5 * (float(th[0]) / W + oxm),
                                            0.5 * (float(th[1]) / H + oym)])
                    else:
                        pinch[side].append(grasp[side][-1])   # fallback to fingertip centroid
                    # ALL-FINGERS grasp hull: wrist + MCPs + fingertips (normalized) — the region the
                    # hand encloses (where the grasp focus is), not a single point
                    HULL_KP = (0, 5, 9, 13, 17, 4, 8, 12, 16, 20)
                    hull[side].append([[float(kp[i][0]) / W, float(kp[i][1]) / H]
                                       for i in HULL_KP if kp is not None and kp[i][2] > 0.3])
                    # thumb tip (kp4) + index tip (kp8) — the precision-pinch pair; nan if low-confidence
                    tip[side].append([[float(kp[i][0]) / W, float(kp[i][1]) / H]
                                      if kp is not None and kp[i][2] > 0.3 else [np.nan, np.nan]
                                      for i in (4, 8)])
        fi += 1
    cap.release()
    return {"t": np.array(T), "fps": fps, "W": W, "H": H,
            "cx": {k: np.array(v) for k, v in cx.items()},
            "cy": {k: np.array(v) for k, v in cy.items()},
            "grip": {k: np.array(v) for k, v in grip.items()},
            "pres": {k: np.array(v) for k, v in pres.items()},
            "box": {k: np.array(v, dtype=float) for k, v in box.items()},
            "grasp": {k: np.array(v, dtype=float) for k, v in grasp.items()},
            "pinch": {k: np.array(v, dtype=float) for k, v in pinch.items()},
            "tip": {k: np.array(v, dtype=float) for k, v in tip.items()},   # [thumb, index] tips
            "hull": hull}                                # variable-length point lists (not np.array)


def motion_energy(sig):
    """Per-frame transition energy = max-over-hands smoothed centroid speed (per sec)."""
    fps = sig["fps"]
    speeds = {}
    for side in ("L", "R"):
        x = _smooth(sig["cx"][side], 5); y = _smooth(sig["cy"][side], 5)
        dx = np.gradient(x); dy = np.gradient(y)
        sp = np.hypot(dx, dy) * fps                  # normalized units / second
        speeds[side] = _smooth(sp, max(3, int(round(0.25 * fps)) | 1))
    energy = np.nanmax(np.vstack([speeds["L"], speeds["R"]]), axis=0)
    return energy, speeds


def _split_long_burst(a, b, t, e, fps, min_sub=0.6):
    """A burst spanning a place+pick-up has two motion peaks with a still dip between.
    If a long burst has an interior local minimum that drops near the action floor,
    split there (object-free recovery of the two events). Returns a list of (a,b)."""
    if b - a < 1.2:
        return [(a, b)]
    i0 = int(np.searchsorted(t, a)); i1 = int(np.searchsorted(t, b))
    seg = e[i0:i1 + 1]
    if len(seg) < 5:
        return [(a, b)]
    k = int(np.argmin(seg))                               # deepest interior dip
    tk = t[i0 + k]
    # require a real valley: dip well below the two flanking peaks, and both subs long enough
    left_pk = seg[:k].max() if k > 0 else 0
    right_pk = seg[k + 1:].max() if k + 1 < len(seg) else 0
    if (tk - a) >= min_sub and (b - tk) >= min_sub \
            and seg[k] < 0.55 * min(left_pk, right_pk):
        return [(a, round(tk, 2)), (round(tk, 2), b)]
    return [(a, b)]


def detect_segments(sig, energy, min_action=0.6, min_burst=0.25, merge_gap=0.45):
    """Adaptive threshold on motion energy -> transition bursts; the spans between
    are action plateaus. Returns boundaries + typed segments (deterministic)."""
    t = sig["t"]; fps = sig["fps"]
    e = energy.copy(); e[np.isnan(e)] = 0.0
    med = np.median(e); mad = np.median(np.abs(e - med)) + 1e-6
    thr = max(med + 3.0 * mad, np.percentile(e, 75))      # adaptive, robust
    hot = e > thr
    # group consecutive hot frames into bursts
    bursts = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            bursts.append([t[i], t[j]])
            i = j + 1
        else:
            i += 1
    # merge bursts closer than merge_gap; drop bursts shorter than min_burst
    merged = []
    for b in bursts:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    merged = [b for b in merged if (b[1] - b[0]) >= min_burst]
    # split any long burst that bundles two events (place+pick-up) at its motion dip
    split = []
    for (a, b) in merged:
        split.extend(_split_long_burst(a, b, t, e, fps))
    merged = split
    # build typed segment list that TILES [0, dur] with NO gaps. When the action gap before a
    # transition is too small to be its own segment, the transition simply STARTS at `cur`
    # (absorbs the gap) — never leave an unlabelled hole in the timeline.
    dur = float(t[-1]) + 1.0 / fps
    segs, cur = [], 0.0
    for (a, b) in merged:
        if a - cur >= min_action:
            segs.append({"start": round(cur, 2), "end": round(a, 2), "type": "action"})
            ts = a
        else:
            ts = cur                                  # no gap: transition covers [cur, b]
        segs.append({"start": round(ts, 2), "end": round(b, 2), "type": "transition"})
        cur = b
    if dur - cur >= 0.05:
        segs.append({"start": round(cur, 2), "end": round(dur, 2), "type": "action"})
    elif segs:
        segs[-1]["end"] = round(dur, 2)               # cover the tail (no trailing gap)
    boundaries = sorted({round(s["start"], 2) for s in segs} |
                        {round(s["end"], 2) for s in segs})
    return {"threshold": float(thr), "segments": segs, "boundaries": boundaries,
            "transitions": [{"start": round(a, 2), "end": round(b, 2)} for a, b in merged]}


def run(video, fps=30.0):
    sig = extract(video, fps_target=fps)
    energy, speeds = motion_energy(sig)
    res = detect_segments(sig, energy)
    return sig, energy, res


# NOTE: ground truth lives ONLY in the eval modules (eval.py / scorecard.py), NEVER here in
# the annotation pipeline — so this module cannot leak GT into anything it produces.
def _score_boundaries(detected, gt, tol=0.8):
    matched, used = 0, set()
    for g in gt:
        for i, d in enumerate(detected):
            if i not in used and abs(d - g) <= tol:
                matched += 1; used.add(i); break
    rec = matched / len(gt) if gt else 0.0
    interior = [d for d in detected if 0.3 < d < detected[-1] - 0.3]
    prec = matched / len(interior) if interior else 0.0
    return rec, prec, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=str(HERE / "facts.json"))
    a = ap.parse_args()
    sig, energy, res = run(a.video, a.fps)
    print(f"thr={res['threshold']:.2f}  {len(res['segments'])} segments "
          f"({sum(1 for s in res['segments'] if s['type']=='transition')} transitions)")
    for s in res["segments"]:
        tag = "»TRANS«" if s["type"] == "transition" else "       "
        print(f"  [{s['start']:6.2f}-{s['end']:6.2f}s] {tag} {s['type']}")
    Path(a.out).write_text(json.dumps({"video": a.video, "fps": sig["fps"],
                                       **res}, indent=2))
    # GT scoring lives in eval.py / scorecard.py — keep this module GT-free.


if __name__ == "__main__":
    main()
