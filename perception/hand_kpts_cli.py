#!/usr/bin/env python3
"""hand_kpts_cli.py (3.10 venv) — run the handpose detector on every *.jpg in a dir, dump 21-keypoint hands
per frame to a JSON keyed by basename. Used by build_inventory_grasp.py (which runs under sam3py and can't
import ultralytics). Run: hand_kpts_cli.py <frames_dir> <out.json>"""
import sys, glob, os, json
sys.path.insert(0, "/home/ubuntu/local/yolo_hands/yolo_bundle")
import cv2
from ultralytics import YOLO
from hand_yolo_infer import detect_hands

framedir, outp = sys.argv[1], sys.argv[2]
det = YOLO("/home/ubuntu/local/yolo_hands/yolo_bundle/hand_yolo_detector@20260314.pt")
res = {}
for f in sorted(glob.glob(os.path.join(framedir, "*.jpg"))):
    bgr = cv2.imread(f)
    if bgr is None:
        continue
    hands = detect_hands(det, bgr, conf=0.3)
    if hands:
        res[os.path.basename(f)] = [
            {"cls": h["cls"], "conf": round(h["conf"], 3),
             "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in h["kpts"]]}
            for h in hands if h["kpts"] is not None]
json.dump(res, open(outp, "w"))
print(f"handpose: {len(res)} frames with hands -> {outp}")
