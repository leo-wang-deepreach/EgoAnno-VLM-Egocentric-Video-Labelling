#!/usr/bin/env python3
"""ablation_label_zoom.py — ZOOMED variants of the SAM-highlight labeling conditions (sam3py; VLM=Gemini).
Production "focus" trick: crop to the grasp region (rad=0.12*W around the grip, 2x upscale) so background
look-alikes (rack cups) fall outside. Per hand, with inventory:
  point_inv_zoom : zoom crop + the point-seg held-object outline -> name
  text_inv_zoom  : zoom crop + text-seg inventory-object outlines (top instance/object) -> name
Reads the existing _label_ablation.json, ADDS these two fields per row, writes back.

Run (sam3py): ablation_label_zoom.py <video> <dense_prompts.json> <outdir>
"""
import base64, io, json, os, sys
sys.path.insert(0, "/home/ubuntu/local/factsfirst")
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import models  # noqa

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-pro-preview")
video, prompts_f, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
INV = ["clear plastic cup", "clear plastic jar", "yellow plastic scoop", "clear plastic lid"]
CAND_COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60)]
NA = ("Answer 'N/A' if the hand is empty, resting, only reaching/touching but not yet holding, or not visible.")
SCH1 = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


def sys_pt():
    return ("This is a ZOOMED-IN crop around ONE hand doing a tabletop task; the hand's grip point is the GREEN "
            "dot and the object SAM segmented at the grip is OUTLINED in green. Name the SINGLE object this hand "
            "is actively holding/using, choosing ONLY from: " + ", ".join(INV) + ". " + NA +
            " Output strict JSON {name}.")


def sys_tx():
    return ("This is a ZOOMED-IN crop around ONE hand doing a tabletop task; the hand's grip point is the GREEN "
            "dot. SAM has text-segmented and OUTLINED + LABELLED the inventory objects visible in this region. "
            "Name the SINGLE object this hand is actively holding/using, choosing ONLY from: " + ", ".join(INV) +
            ". " + NA + " Output strict JSON {name}.")


def b64(rgb):
    im = Image.fromarray(rgb)
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _cl(v, lo, hi):
    return max(lo, min(hi, int(v)))


pj = json.load(open(prompts_f)); W, H = pj["W"], pj["H"]
byt = {}
for p in pj["prompts"]:
    byt.setdefault(round(p["t"], 3), {})[p["hand"]] = p
times = sorted(byt)
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)
RAD = int(0.12 * W)


def seg_held(state, p, oth):
    ax, ay = _cl(p["x_px"], 0, W - 1), _cl(p["y_px"], 0, H - 1); negs = []
    b = p.get("box_px")
    if b:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        negs.append([_cl(2 * cx - ax, 0, W - 1), _cl(2 * cy - ay, 0, H - 1)])
    if oth:
        negs.append([_cl(oth["x_px"], 0, W - 1), _cl(oth["y_px"], 0, H - 1)])
    pts = np.array([[ax, ay]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:
        mm, ss, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    mm = _np(mm); ss = _np(ss).ravel(); best = None
    for k in range(len(ss) if mm.ndim == 3 else 0):
        m = mm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.35 and (best is None or float(ss[k]) > best[1]):
            best = (m, float(ss[k]))
    return best[0] if best else None


def seg_text_best(state, name):
    with torch.inference_mode(), ac:
        out = proc.set_text_prompt(state=state, prompt=name)
    tm = _np(out["masks"]); ts = _np(out["scores"]).ravel(); best = None
    if tm.ndim == 4:
        tm = tm[:, 0]
    for k in range(len(ts) if tm.ndim == 3 else 0):
        m = tm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.35 and (best is None or float(ts[k]) > best[1]):
            best = (m, float(ts[k]))
    return best[0] if best else None


def crop_zoom(rgb, p, mask_cols, labels=None):
    gx, gy = _cl(p["x_px"], 0, W - 1), _cl(p["y_px"], 0, H - 1)
    x0, y0 = max(0, gx - RAD), max(0, gy - RAD); x1, y1 = min(W, gx + RAD), min(H, gy + RAD)
    crop = rgb[y0:y1, x0:x1].copy()
    for i, (m, col) in enumerate(mask_cols):
        if m is None:
            continue
        mc = m[y0:y1, x0:x1]
        if mc.any():
            c, _ = cv2.findContours(mc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, c, -1, col, 2)
            if labels:
                ys, xs = np.nonzero(mc)
                cv2.putText(crop, labels[i], (int(xs.mean()) - 15, max(10, int(ys.min()) - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.circle(crop, (gx - x0, gy - y0), 6, (0, 255, 0), -1)
    return cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2))


def ask1(sys_txt, img):
    try:
        r = models.vlm_call("Name the object this hand is manipulating, or N/A.", [b64(img)], sys_txt, SCH1, max_tokens=300)
        return (r.get("name") or "N/A").strip()
    except Exception as e:
        print("  ask err", e); return "N/A"


rows = json.load(open(f"{outdir}/_label_ablation.json"))
by_frame = {r["frame"]: r for r in rows["rows"]}
cap = cv2.VideoCapture(video)
for ti, t in enumerate(times):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, bgr = cap.read()
    if not ok or ti not in by_frame:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), ac:
        state = proc.set_image(Image.fromarray(rgb))
    L, R = byt[t].get("L"), byt[t].get("R")
    txt_masks = [(seg_text_best(state, nm), CAND_COLORS[i]) for i, nm in enumerate(INV)]
    txt_labels = [nm.replace("clear plastic ", "").replace("yellow plastic ", "") for nm in INV]
    pz = {"left": "N/A", "right": "N/A"}; tz = {"left": "N/A", "right": "N/A"}
    for hand, p, oth, key in (("L", L, R, "left"), ("R", R, L, "right")):
        if not p:
            continue
        hm = seg_held(state, p, oth)
        pz[key] = ask1(sys_pt(), crop_zoom(rgb, p, [(hm, (0, 255, 0))]))
        tz[key] = ask1(sys_tx(), crop_zoom(rgb, p, txt_masks, txt_labels))
    by_frame[ti]["point_inv_zoom"] = pz
    by_frame[ti]["text_inv_zoom"] = tz
    if ti % 10 == 0:
        print(f"frame {ti}/{len(times)} t={t:.0f}: ptZ={pz} txZ={tz}", flush=True)
cap.release()
json.dump(rows, open(f"{outdir}/_label_ablation.json", "w"), indent=2)
print(f"\nmerged zoom fields into {outdir}/_label_ablation.json", flush=True)
