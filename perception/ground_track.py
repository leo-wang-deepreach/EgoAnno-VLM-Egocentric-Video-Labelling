#!/usr/bin/env python3
"""ground_track.py — VIDEO-version per-hand grounder (3.10 venv; numpy/cv2 only, no GPU).
HYBRID: the per-frame MANIPULATION GATE decides hold-vs-empty (N/A) — proven, kept from Lever A — while
the SAM3-video TRACKS supply CONSISTENT identity + a CLEAN mask carried from each object's clear frame.

For each hand at each grasp moment:
  - if the per-frame pass said N/A (gate = not manipulating) -> N/A.
  - else pick the TRACKED object whose mask is AT THE PINCH (in the hand box, near the anchor, not wrist,
    sane size); among those, the SMALLEST/tightest. Its name+mask are temporally consistent (the track
    keeps one identity across every frame). If no track sits at the pinch, fall back to the per-frame NAME.

Run: ../.venv/bin/python perception/ground_track.py <tag> <tracks_dir> <prompts.json> <perframe_index.json> <outdir>
  env: PINCH (0.045 of W)
"""
import json
import os
import sys

import cv2
import numpy as np

COLORS = [(80, 255, 80), (80, 160, 255), (255, 255, 60)]


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def main():
    tag, tracks_dir, prompts_file, perframe_file, outdir = sys.argv[1:6]
    os.makedirs(outdir, exist_ok=True)
    pj = json.load(open(prompts_file)); video = pj["video"]; W, H = pj["W"], pj["H"]
    pinch_w = max(20, int(float(os.environ.get("PINCH", "0.045")) * W))

    # per-frame gate decision + fallback name, keyed by (hand, round(t,1))
    pf = {}
    for r in json.load(open(perframe_file)):
        pf[(r["hand"], round(float(r["t"]), 1))] = (bool(r.get("obj")), r.get("name", "N/A"))

    by_t = {}
    for p in pj["prompts"]:
        by_t.setdefault(round(p["t"], 3), []).append(p)

    cap = cv2.VideoCapture(video)
    index = []
    for t, plist in sorted(by_t.items()):
        npz = os.path.join(tracks_dir, f"{tag}_t{t:05.1f}.npz")
        tracks = {}
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            tracks = {k: z[k].astype(bool) for k in z.files}
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for p in plist:
            gx, gy = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
            x0, y0, x1, y1 = p["box_px"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            wpx, wpy = _clamp(2 * cx - gx, x0, x1), _clamp(2 * cy - gy, y0, y1)
            hand = "LEFT" if p["hand"] == "L" else "RIGHT"
            yl, yh = max(0, gy - pinch_w), min(H, gy + pinch_w + 1)
            xl, xh = max(0, gx - pinch_w), min(W, gx + pinch_w + 1)

            grounded_pf, name_pf = pf.get((hand, round(t, 1)), (False, "N/A"))
            name, mask, src = "N/A", None, ""
            if grounded_pf:                                    # gate said the hand IS manipulating
                held = None                                    # (name, mask, area) — track at the pinch
                for nm, m in tracks.items():
                    if not m.any() or m.shape != (H, W):
                        continue
                    if not m[yl:yh, xl:xh].any() or m[y0:y1, x0:x1].sum() == 0 or m[wpy, wpx]:
                        continue                               # RULE 1: at pinch, in box, not wrist
                    if not (0.0005 < float(m.mean()) < 0.40):
                        continue
                    a = int(m.sum())
                    if held is None or a < held[2]:
                        held = (nm, m, a)
                if held is not None:
                    name, mask, src = held[0], held[1], "track"
                else:
                    name, src = name_pf, "perframe"            # gate=manip but no track at pinch -> keep per-frame name

            vis = rgb.copy()
            if mask is not None:
                f = vis.astype(np.float32); f[mask] = 0.5 * f[mask] + 0.5 * np.array(COLORS[0], np.float32)
                vis = f.clip(0, 255).astype(np.uint8)
                cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, COLORS[0], 2)
            cap_txt = f"{tag} {hand}: {name}" + (f"  ({src})" if name != "N/A" else "")
            cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1); cv2.circle(vis, (gx, gy), 7, (255, 40, 40), -1)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            fn = f"{tag}_t{p['t']:05.1f}_{p['hand']}.png"
            cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            index.append({"file": fn, "tag": tag, "hand": hand, "t": p["t"],
                          "obj": name != "N/A", "name": name, "src": src})
            print(cap_txt, flush=True)
    cap.release()
    json.dump(index, open(os.path.join(outdir, f"_index_{tag}.json"), "w"), indent=2)
    n = sum(1 for r in index if r["obj"])
    ntrk = sum(1 for r in index if r["src"] == "track")
    print(f"\n{tag}: {len(index)} hand-frames · {n} grounded ({ntrk} from track) · {len(index)-n} N/A")


if __name__ == "__main__":
    main()
