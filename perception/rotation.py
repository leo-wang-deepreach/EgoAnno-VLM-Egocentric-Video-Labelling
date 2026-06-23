#!/usr/bin/env python3
"""rotation.py — SPEC B: measure rotation per hand per segment from optical flow.

The VLM must never decide screw-vs-unscrew (it flips constantly). Instead, at 10fps we
measure the ROTATIONAL component (curl) of dense optical flow in each hand's crop:
  * the hand with the larger rotational energy is the TURNER (the other holds) — also fixes
    the holder/turner labeling wobble;
  * the SIGN of the turner's curl is the rotation direction; mapped to screw/unscrew by the
    clip's global direction (engagement-as-prior): on a disassembly clip the dominant turning
    sign = UNSCREW, the opposite sign = screw, and vice-versa for assembly. So one clip can
    contain BOTH, and a non-turning span is simply 'not fastening' (labeled by translation).

`segment_rotation(video, sig, segs)` returns one dict per segment:
  {L_curl, R_curl, L_mag, R_mag, turner: 'left'|'right'|None, fastening: bool}
`calibrate(seg_rots, global_direction)` adds {direction: 'screw'|'unscrew'|None} per segment.
"""
from __future__ import annotations
import math

import cv2
import numpy as np


def _crop_gray(frame, box, pad=0.30, size=176):
    """Crop a BGR frame to a normalized box (expanded), grayscale, fixed size. None if no box."""
    if box is None or np.any(np.isnan(box)):
        return None
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0), (y1 - y0)
    x0 -= pad * bw; x1 += pad * bw; y0 -= pad * bh; y1 += pad * bh
    x0 = max(0, int(x0 * W)); x1 = min(W, int(x1 * W))
    y0 = max(0, int(y0 * H)); y1 = min(H, int(y1 * H))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size))


def _curl_mag(g0, g1):
    """From dense flow, with the TRANSLATIONAL component removed (so hand drift doesn't fake
    rotation): return (signed_curl, rot_energy). signed_curl = mean tangential velocity about
    the centre (sign = CW/CCW); rot_energy = mean |tangential| (always >=0, so it does NOT
    cancel over a long span where direction reverses) — used to detect that turning happened."""
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = g0.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    u, v = flow[..., 0], flow[..., 1]
    if float(np.mean(np.hypot(u, v))) < 1e-6:
        return 0.0, 0.0
    u = u - float(np.mean(u)); v = v - float(np.mean(v))    # remove translation -> pure rotation
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + 1.0)
    tang = ((xs - cx) * v - (ys - cy) * u) / r
    return float(np.mean(tang)), float(np.mean(np.abs(tang)))


def segment_rotation(video, sig, segs, fps=10.0, anchors=6, min_len=0.6):
    """Per segment, per hand: mean signed curl + mean flow magnitude, from `anchors` close
    frame pairs (t, t+1/fps) sampled across the span, in that hand's crop."""
    cap = cv2.VideoCapture(video)
    dt = 1.0 / fps
    t_arr = sig["t"]
    out = []
    for s in segs:
        a, b = s["start"], s["end"]
        rec = {"L_curl": 0.0, "R_curl": 0.0, "L_mag": 0.0, "R_mag": 0.0,
               "turner": None, "fastening": False}
        if b - a < min_len:
            out.append(rec); continue
        ts = np.linspace(a + 0.05, b - dt - 0.05, max(2, anchors))
        acc = {"L": [], "R": []}; macc = {"L": [], "R": []}
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok0, f0 = cap.read()
            cap.set(cv2.CAP_PROP_POS_MSEC, (t + dt) * 1000.0)
            ok1, f1 = cap.read()
            if not (ok0 and ok1):
                continue
            idx = int(np.argmin(np.abs(t_arr - t)))
            for side in ("L", "R"):
                box = sig["box"][side][idx]
                g0 = _crop_gray(f0, box); g1 = _crop_gray(f1, box)
                if g0 is None or g1 is None:
                    continue
                c, m = _curl_mag(g0, g1)
                acc[side].append(c); macc[side].append(m)
        for side, key in (("L", "L_curl"), ("R", "R_curl")):
            if acc[side]:
                rec[key] = float(np.mean(acc[side]))
                rec[side + "_mag"] = float(np.mean(macc[side]))
        out.append(rec)
    cap.release()
    return out


def _energy_at(cap, sig, t, dt=0.1):
    """Max-over-hands rotational energy at time t (one close frame pair)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok0, f0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, (t + dt) * 1000.0); ok1, f1 = cap.read()
    if not (ok0 and ok1):
        return 0.0
    idx = int(np.argmin(np.abs(sig["t"] - t)))
    best = 0.0
    for side in ("L", "R"):
        g0 = _crop_gray(f0, sig["box"][side][idx]); g1 = _crop_gray(f1, sig["box"][side][idx])
        if g0 is not None and g1 is not None:
            _, e = _curl_mag(g0, g1)
            best = max(best, e)
    return best


def refine_long_segments(video, sig, segs, max_len=8.0, step=1.0, min_sub=2.5, max_chunk=5.0):
    """SEGMENTATION FIX: motion-energy is blind to in-place fastening (no translation), so
    long unscrew/screw spans collapse into one mega-segment. Here we densely sample ROTATIONAL
    energy inside every long segment and split it — at energy VALLEYS (pauses between fastener
    cycles), and, for stretches that stay high with no clear pause, at regular max_chunk steps.
    Only splits segments that are actually fastening (median rotation energy above a floor)."""
    cap = cv2.VideoCapture(video)
    out = []
    for s in segs:
        if s["end"] - s["start"] <= max_len:
            out.append(s); continue
        ts = np.arange(s["start"] + 0.3, s["end"] - step, step)
        es = np.array([_energy_at(cap, sig, float(t)) for t in ts])
        valid = es[es > 0]
        med = float(np.median(valid)) if len(valid) else 0.0
        if med < 0.02:                                   # not fastening -> leave intact
            out.append(s); continue
        # split ONLY at real rotational PAUSES (valleys) — never at arbitrary time intervals
        # (arbitrary interval cuts chop one continuous unscrew into pieces = over-segmentation).
        cuts, last = [], s["start"]
        for i in range(1, len(es) - 1):
            if es[i] < 0.55 * med and es[i] <= es[i - 1] and es[i] <= es[i + 1] \
                    and ts[i] - last >= min_sub:
                cuts.append(float(ts[i])); last = float(ts[i])
        allb = sorted(set(round(x, 2) for x in ([s["start"]] + cuts + [s["end"]])))
        for a, b in zip(allb[:-1], allb[1:]):
            d = dict(s); d["start"] = round(a, 2); d["end"] = round(b, 2)
            d["type"] = "action"; out.append(d)
    cap.release()
    return out


def calibrate(seg_rots, global_direction, energy_frac=0.5):
    """ENGAGEMENT-PRIMARY, rotation-SUPPORT (the user's spec): fastening = a hand shows clear
    ROTATIONAL ENERGY (|tangential|, which doesn't cancel over long spans); the turner = the
    hand with more of it. Direction DEFAULTS to the clip's global direction (disassembly ->
    unscrew, assembly -> screw) and only FLIPS to the reverse on STRONG, confident opposite
    rotation (top-quartile signed curl against the dominant sign) — so flow-sign noise can't
    sprinkle wrong screws across a teardown."""
    energy = [max(r["L_mag"], r["R_mag"]) for r in seg_rots]   # _mag now holds rot_energy
    pos = [e for e in energy if e > 1e-6]
    bar = max((np.median(pos) * energy_frac) if pos else 1e9, 0.025)
    for r, e in zip(seg_rots, energy):
        if e >= bar:
            r["fastening"] = True
            r["turner"] = "left" if r["L_mag"] >= r["R_mag"] else "right"
    fs = [r for r in seg_rots if r["fastening"]]
    signs = [math.copysign(1, r["L_curl"] if r["turner"] == "left" else r["R_curl"]) for r in fs]
    dom = 1.0 if sum(signs) >= 0 else -1.0
    disasm = (global_direction or "").startswith("dis")
    primary, reverse = ("unscrew", "screw") if disasm else ("screw", "unscrew")
    curls = [abs(r["L_curl"] if r["turner"] == "left" else r["R_curl"]) for r in fs]
    strong = float(np.percentile(curls, 75)) if curls else 1e9      # confident reversal only
    for r in seg_rots:
        if not r["fastening"]:
            r["direction"] = None
            continue
        c = r["L_curl"] if r["turner"] == "left" else r["R_curl"]
        r["direction"] = reverse if (math.copysign(1, c) != dom and abs(c) >= strong) else primary
    return seg_rots
