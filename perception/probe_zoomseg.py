#!/usr/bin/env python3
"""PROBE (sam3py): does text-seg of a held-object name on a ZOOMED grasp crop return a FULL mask
(covering the whole object) vs a fingertip speck? Per flagged pen frame, zoom into the grasp,
text-seg EACH manipulable name, report best-mask coverage (area + bbox), and save a viz overlay.
Read-only experiment — touches no pipeline state. Run: sam3py perception/probe_zoomseg.py"""
import json, os
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = "/home/ubuntu/local/factsfirst"
inv = json.load(open(f"{ROOT}/out/v2_grounded/_inventory_4fddf957.json"))
names = [o["name"] for o in inv["objects"] if o.get("role") != "fixture"]
pj = json.load(open(f"{ROOT}/out/v2_grounded/batch/4fddf957/_grasp_4fddf957.json"))
W, H, video = pj["W"], pj["H"], pj["video"]
OUT = f"{ROOT}/out/v2_grounded/probe_zoomseg_4fddf957"; os.makedirs(OUT, exist_ok=True)
TARGET = [(7.7, "R"), (29.9, "R"), (84.6, "R"), (7.7, "L"), (29.9, "L")]
COLORS = [(80,255,80),(80,160,255),(255,255,60),(255,80,255),(80,255,255),(255,160,80),(200,200,200)]

def npx(x): return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
def pick(t, hand):
    for p in pj["prompts"]:
        if p["hand"] == hand and abs(p["t"] - t) < 0.3:
            return p
    return None

model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16); cap = cv2.VideoCapture(video)

for t, hand in TARGET:
    p = pick(t, hand)
    if not p: print(f"!! no prompt {t}{hand}"); continue
    cap.set(cv2.CAP_PROP_POS_MSEC, p["t"] * 1000.0); ok, bgr = cap.read()
    if not ok: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gx, gy = int(p["x_px"]), int(p["y_px"])
    tpx, ipx = p.get("thumb_px"), p.get("index_px")
    ax, ay = ((tpx[0]+ipx[0])/2, (tpx[1]+ipx[1])/2) if (tpx and ipx) else (gx, gy)
    rad = int(0.12 * W)
    cx0, cy0 = max(0, int(ax)-rad), max(0, int(ay)-rad); cx1, cy1 = min(W, int(ax)+rad), min(H, int(ay)+rad)
    crop = rgb[cy0:cy1, cx0:cx1]
    f = max(1, int(512 / max(crop.shape[:2])))
    up = cv2.resize(crop, (crop.shape[1]*f, crop.shape[0]*f), interpolation=cv2.INTER_CUBIC)
    Hu, Wu = up.shape[:2]; acx, acy = (ax-cx0)*f, (ay-cy0)*f
    res = {}
    with torch.inference_mode(), ac:
        st = proc.set_image(Image.fromarray(up))
        for nm in names:
            out = proc.set_text_prompt(state=st, prompt=nm.split(" with ")[0].strip())
            tm = npx(out["masks"]); ts = npx(out["scores"]).ravel()
            if tm.ndim == 4: tm = tm[:, 0]
            best = None
            for k in range(len(tm) if tm.ndim == 3 else 0):
                mk = tm[k] > 0.5
                if mk.ndim > 2: mk = mk.squeeze()
                if mk.shape != (Hu, Wu) or not mk.any(): continue
                ys, xs = np.nonzero(mk)
                if best is None or ts[k] > best["score"]:
                    best = {"score": float(ts[k]), "area": float(mk.mean()),
                            "bw": (xs.max()-xs.min())/Wu, "bh": (ys.max()-ys.min())/Hu, "mask": mk}
            res[nm] = best
    viz = up.copy(); rep = []
    for j, nm in enumerate(names):
        b = res[nm]
        if b is None: rep.append(f"{nm:<20} NONE"); continue
        col = COLORS[j % len(COLORS)]
        cont, _ = cv2.findContours(b["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, cont, -1, col, 2)
        rep.append(f"{nm:<20} score={b['score']:.2f} area={b['area']:.3f} bbox={b['bw']:.2f}x{b['bh']:.2f}")
    cv2.circle(viz, (int(acx), int(acy)), 6, (255, 140, 0), -1)
    cv2.imwrite(f"{OUT}/probe_{t}_{hand}.png", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    print(f"\n=== t={t} {hand}  crop {crop.shape[1]}x{crop.shape[0]} up x{f} (area/bbox = frac of crop) ===")
    for r in rep: print("  ", r)
print("\nwrote viz ->", OUT)
