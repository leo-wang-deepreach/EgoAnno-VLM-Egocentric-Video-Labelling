#!/usr/bin/env python3
"""ground_review.py — EVIDENCE batch: render SAM grounding overlays across many frames/clips so a
human can judge whether the per-hand masked area is accurate, BEFORE any labeling is built on it.

Builds the SAM model once, then for each prompts file (one per clip) grounds every prompt with the
refined fingertip-positive + hand-body-negative logic and draws a full-frame overlay (green mask +
outline = what the hand is judged to manipulate; red dot = measured fingertip; blue x = negative
hand points; caption = clip / hand / area% / score). Writes a scrollable index.html gallery.

Run (sam3py): /home/ubuntu/local/sam3/sam3py perception/ground_review.py <outdir> <prompts1.json> [prompts2.json ...]
"""
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from ground_probe2 import _negs, _select


def overlay(rgb, m, gx, gy, negs, caption):
    ov = rgb.astype(np.float32)
    ov[m] = 0.55 * ov[m] + 0.45 * np.array([40, 255, 40], np.float32)
    ov = ov.clip(0, 255).astype(np.uint8)
    cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cont, -1, (0, 255, 0), 2)
    cv2.circle(ov, (gx, gy), 8, (255, 40, 40), -1)
    for nx, ny in negs:
        cv2.drawMarker(ov, (nx, ny), (60, 120, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.rectangle(ov, (0, 0), (ov.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(ov, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return ov


def main():
    outdir = sys.argv[1]
    pfiles = sys.argv[2:]
    os.makedirs(outdir, exist_ok=True)
    model = build_sam3_image_model(enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    index = []
    for pf in pfiles:
        tag = re.sub(r"^_?grasp_|\.json$", "", os.path.basename(pf))
        pj = json.load(open(pf))
        cap = cv2.VideoCapture(pj["video"])
        for p in pj["prompts"]:
            cap.set(cv2.CAP_PROP_POS_MSEC, p["t"] * 1000.0)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gx, gy = p["x_px"], p["y_px"]
            negs = _negs(p["box_px"], gx, gy)
            with torch.inference_mode(), ac:
                state = proc.set_image(Image.fromarray(rgb))
                masks, scores, _ = model.predict_inst(
                    state, point_coords=np.array([[gx, gy]] + negs),
                    point_labels=np.array([1, 0, 0]), multimask_output=True)
            _, m, area, sc, ok_sel = _select(masks, scores, gx, gy, negs)
            hand = "LEFT" if p["hand"] == "L" else "RIGHT"
            cap_txt = (f"{tag}  {hand} hand  t={p['t']:.1f}s  area={area*100:.1f}%  "
                       f"score={sc:.2f}{'' if ok_sel else '  [fallback]'}")
            fn = f"{tag}_t{p['t']:05.1f}_{p['hand']}.png"
            cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(overlay(rgb, m, gx, gy, negs, cap_txt),
                                                               cv2.COLOR_RGB2BGR))
            index.append({"file": fn, "tag": tag, "hand": hand, "t": p["t"],
                          "area_pct": round(area * 100, 2), "score": round(float(sc), 3), "obj": ok_sel})
            print(cap_txt)
        cap.release()

    # write a UNIQUE shard index (parallel workers each write their own; build_gallery.py merges)
    shard = "_".join(sorted({r["tag"] for r in index})) or "shard"
    json.dump(index, open(os.path.join(outdir, f"_index_{shard}.json"), "w"), indent=2)
    nobj = sum(1 for r in index if r["obj"])
    print(f"\n{len(index)} overlays ({nobj} object-locked, {len(index)-nobj} fallback) -> shard {shard}")


if __name__ == "__main__":
    main()
