#!/usr/bin/env python3
"""emit_prompts_simple.py (3.10 venv) — grasp prompts straight from per-frame handpose (NO motion-segmentation;
works on proxy/stitched videos). Samples the video, runs the hand detector, and for each detected hand emits a
prompt: grip = fingertip centroid (kp 4,8,12,16,20), box = hand box, thumb=kp4, index=kp8. Capped to ~MAX_PROMPTS
evenly-spaced frames so the grounder stays cheap. Output = ground_simple's prompts.json format.

Run: emit_prompts_simple.py <video> <out.json> [tag]   env: MAX_PROMPTS(8) SAMPLE_FPS(2)
"""
import sys, os, json
sys.path.insert(0, "/home/ubuntu/local/yolo_hands/yolo_bundle")
import cv2, numpy as np
from ultralytics import YOLO
from hand_yolo_infer import detect_hands

video, out = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "clip"
MAXP = int(os.environ.get("MAX_PROMPTS", "8"))
SFPS = float(os.environ.get("SAMPLE_FPS", "2"))
det = YOLO("/home/ubuntu/local/yolo_hands/yolo_bundle/hand_yolo_detector@20260314.pt")
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 6.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
step = max(1, int(round(fps / SFPS)))
cands = []                                          # (frame_idx, hand_dict)
for fi in range(0, n, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, bgr = cap.read()
    if not ok:
        continue
    for h in detect_hands(det, bgr, conf=0.35):
        if h.get("kpts") is not None:
            cands.append((fi, h))
cap.release()
# keep frames with the most hands first, then spread evenly, cap to MAXP
by_frame = {}
for fi, h in cands:
    by_frame.setdefault(fi, []).append(h)
frames = sorted(by_frame)
if len(frames) > MAXP:
    idx = np.linspace(0, len(frames) - 1, MAXP).astype(int)
    frames = [frames[i] for i in idx]
prompts = []
for fi in frames:
    t = fi / fps
    for h in by_frame[fi]:
        kp = h["kpts"]
        tips = [kp[j] for j in (4, 8, 12, 16, 20) if kp[j][2] > 0.25]
        if not tips:
            continue
        gx = int(np.mean([p[0] for p in tips])); gy = int(np.mean([p[1] for p in tips]))
        x0, y0, x1, y1 = [int(v) for v in h["box"]]
        prompts.append({"t": round(t, 3), "hand": "L" if h["cls"] == 0 else "R", "x_px": gx, "y_px": gy,
                        "box_px": [x0, y0, x1, y1],
                        "thumb_px": ([int(kp[4][0]), int(kp[4][1])] if kp[4][2] > 0.25 else None),
                        "index_px": ([int(kp[8][0]), int(kp[8][1])] if kp[8][2] > 0.25 else None)})
json.dump({"tag": tag, "video": video, "W": W, "H": H, "n_segs": 0, "prompts": prompts}, open(out, "w"), indent=2)
print(f"{tag}: {len(prompts)} grasp prompts ({len(frames)} frames, {n} total @ {fps:.0f}fps)")
