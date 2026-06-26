#!/usr/bin/env python3
"""make_clip_video.py — build a short proxy mp4 from dataset frames for the grounding pipeline.
  meta:  <clip> 120 frames @6fps -> ~20s mp4
  visor: <clip> sparse annotated frames in the DENSEST 1-minute real-time window (50fps -> 3000 frame-nums),
         stitched @6fps (a proxy — frames aren't temporally continuous).
Run: make_clip_video.py <meta|visor> <clip> <out.mp4>
"""
import sys, glob, os, re
import cv2

ds, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = 6.0
if ds == "meta":
    fs = sorted(glob.glob(f"/home/ubuntu/local/saco_sg/extract/**/{clip}/*.jpg", recursive=True))
else:
    fs_all = sorted(glob.glob(f"/home/ubuntu/local/visor/frames/**/{clip}*.jpg", recursive=True),
                    key=lambda p: int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)))
    nums = [int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)) for p in fs_all]
    span = 3000; best = (0, 0)                       # densest 1-min window (50fps)
    for i, nn in enumerate(nums):
        j = i
        while j < len(nums) and nums[j] - nn <= span:
            j += 1
        if j - i > best[1]:
            best = (i, j - i)
    fs = fs_all[best[0]:best[0] + best[1]]
if not fs:
    print("NO FRAMES"); sys.exit(1)
h, w = cv2.imread(fs[0]).shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for f in fs:
    im = cv2.imread(f)
    if im is not None:
        vw.write(im)
vw.release()
print(f"{ds}:{clip} -> {out}  ({len(fs)} frames @ {FPS}fps = {len(fs)/FPS:.1f}s, {w}x{h})")
