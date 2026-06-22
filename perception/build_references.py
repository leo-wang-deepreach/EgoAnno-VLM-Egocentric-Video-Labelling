#!/usr/bin/env python3
"""build_references.py — CLEAR-REFERENCE step (sam3py). For each inventory object, scan the clip and
find its CLEAREST view (highest SAM text-seg score), crop+outline it, and assemble a single labeled
REFERENCE SHEET. The labeler is shown this sheet so it identifies/names the manipulated object
consistently from its clear view, even when that object is transparent/occluded in the grasp frame.

Run: build_references.py <tag> <inventory.json> <video> <outdir> [n_frames=18]
Writes: <outdir>/_refsheet_<tag>.png  and  <outdir>/_refs_<tag>.json
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    tag, inv_file, video, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    n_frames = int(sys.argv[5]) if len(sys.argv) > 5 else 18
    os.makedirs(outdir, exist_ok=True)
    inv = json.load(open(inv_file))
    names = [o["name"] if isinstance(o, dict) else o for o in inv["objects"]
             if not (isinstance(o, dict) and o.get("role") == "fixture")]
    model = build_sam3_image_model(); proc = Sam3Processor(model)
    cap = cv2.VideoCapture(video); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    frames = []
    for i in np.linspace(0, nfr - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, bgr = cap.read()
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()

    crops, manifest = [], {}
    for name in names:
        best = (None, 0.0, None)                              # (rgb, score, mask)
        for rgb in frames:
            if rgb is None:
                continue
            with torch.inference_mode(), ac:
                st = proc.set_image(Image.fromarray(rgb))
                out = proc.set_text_prompt(state=st, prompt=name)
            m = _np(out["masks"]); sc = _np(out["scores"]).ravel()
            if m.ndim == 4:
                m = m[:, 0]
            for j in range(len(m) if m.ndim == 3 else 0):
                mm = m[j] > 0.5; s = float(sc[j])
                if mm.sum() > 0 and s > best[1] and mm.mean() < 0.30:   # clearest, object-scale
                    best = (rgb, s, mm)
        if best[0] is None:
            print(f"  {name}: no clear view found"); continue
        rgb, score, mm = best
        ys, xs = np.where(mm); pad = 24
        y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
        crop = rgb[y0:y1, x0:x1].copy()
        cont, _ = cv2.findContours(mm[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cont, -1, (0, 255, 0), 2)
        crops.append((name, crop)); manifest[name] = round(score, 3)
        print(f"  {name}: clearest score {score:.2f}", flush=True)

    # assemble a labeled reference SHEET (one image, each object cropped + named)
    if crops:
        cell = 220
        cols = min(4, len(crops)); rows = (len(crops) + cols - 1) // cols
        sheet = np.full((rows * (cell + 26), cols * cell, 3), 30, np.uint8)
        for i, (name, crop) in enumerate(crops):
            h, w = crop.shape[:2]; sc = min(cell / w, cell / h)
            rz = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))))
            r, c = divmod(i, cols); oy, ox = r * (cell + 26) + 26, c * cell
            sheet[oy:oy + rz.shape[0], ox:ox + rz.shape[1]] = cv2.cvtColor(rz, cv2.COLOR_RGB2BGR)
            cv2.putText(sheet, name[:30], (ox + 2, r * (cell + 26) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, f"_refsheet_{tag}.png"), sheet)
    json.dump(manifest, open(os.path.join(outdir, f"_refs_{tag}.json"), "w"), indent=2)
    print(f"{tag}: reference sheet for {len(crops)}/{len(names)} objects -> {outdir}/_refsheet_{tag}.png")


if __name__ == "__main__":
    main()
