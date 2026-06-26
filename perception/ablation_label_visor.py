#!/usr/bin/env python3
"""ablation_label_visor.py — run the 7-condition labeling ablation on a VISOR (EPIC-Kitchens) clip, graded
against VISOR's native hand->active-object GT. (sam3py; VLM = Gemini 3.1 Pro)

VISOR gives, per annotated frame, hand masks + each hand's `in_contact_object` (the manipulated object's
name + polygon mask). We:
  * derive a grip point per hand (contact-region centroid, else hand-mask centroid),
  * use the clip's distinct manipulated-object names as the "given inventory" (option-C analog),
  * run: baseline(no-SAM+inv), 2 point+inv, 2z point+inv ZOOM, 2b text+inv top, 2bz text+inv ZOOM,
    4 text+inv all, 3 point no-inv,
  * grade object-ID + N/A per hand vs GT.

Run (sam3py): ablation_label_visor.py <visor_ann.json> <frames_dir> <outdir>   [env MAXF=limit frames]
"""
import base64, io, json, glob, os, sys
sys.path.insert(0, "/home/ubuntu/local/factsfirst")
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import models  # noqa

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-pro-preview")
ann_f, frames_dir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
MAXF = int(os.environ["MAXF"]) if os.environ.get("MAXF") else None
os.makedirs(outdir, exist_ok=True)
SCH = {"type": "object", "properties": {"left": {"type": "string"}, "right": {"type": "string"}}, "required": ["left", "right"]}
SCH1 = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
NA = "Answer 'N/A' for a hand that is empty, resting, only reaching/touching but not holding, or not visible."
GREEN, BLUE = (0, 220, 0), (40, 120, 255)
CAND = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60), (200, 80, 255), (80, 220, 220),
        (255, 140, 0), (180, 180, 80)]


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


jpgs = {os.path.basename(p): p for p in glob.glob(f"{frames_dir}/**/*.jpg", recursive=True)}
va = json.load(open(ann_f))["video_annotations"]
if MAXF:
    va = va[::max(1, len(va) // MAXF)][:MAXF]
H, W = None, None


def mask_of(segs, h, w):
    m = np.zeros((h, w), np.uint8)
    for poly in segs:
        cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


# ---- inventory = distinct manipulated-object names across the clip ----
inv = []
for fr in va:
    ents = {e["id"]: e for e in fr["annotations"]}
    for e in fr["annotations"]:
        ico = e.get("in_contact_object")
        if "hand" in e["name"] and ico in ents and ents[ico]["name"] not in inv:
            inv.append(ents[ico]["name"])
INV = inv
print(f"clip inventory ({len(INV)}): {INV}", flush=True)


def sys_prompt(with_inv, with_sam):
    s = ("You label what each HAND is manipulating in this egocentric kitchen frame. The LEFT hand's grip is a "
         "GREEN dot, the RIGHT hand's a BLUE dot. ")
    if with_sam:
        s += "The object SAM segmented at each hand is OUTLINED (green=left, blue=right) to guide you. "
    s += "For EACH hand, name the SINGLE object it is actively holding/using "
    s += ("choosing ONLY from this inventory: " + ", ".join(INV) + ". " if with_inv
          else "with a short canonical name (1-3 words). ")
    return s + NA + " Output strict JSON {left, right}."


def sys_allcand():
    return ("You label what each HAND manipulates. LEFT grip=GREEN dot, RIGHT grip=BLUE dot. SAM has text-"
            "segmented and OUTLINED EVERY matching instance of each inventory object (multiple outlines per "
            "object possible), coloured + labelled. For EACH hand choose which inventory object it holds: "
            + ", ".join(INV) + ". " + NA + " Output strict JSON {left, right}.")


def sys_zoom(with_inv):
    s = ("ZOOMED-IN crop around ONE hand in an egocentric kitchen; the grip is the GREEN dot and the object "
         "SAM segmented at the grip is OUTLINED. Name the SINGLE object this hand holds/uses ")
    s += ("from: " + ", ".join(INV) + ". " if with_inv else "with a short canonical name (1-3 words). ")
    return s + NA + " Output strict JSON {name}."


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def seg_held(state, gx, gy, negs):
    pts = np.array([[gx, gy]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:
        mm, ss, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    mm = _np(mm); ss = _np(ss).ravel(); best = None
    for k in range(len(ss) if mm.ndim == 3 else 0):
        m = mm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.4 and (best is None or float(ss[k]) > best[1]):
            best = (m, float(ss[k]))
    return best[0] if best else None


def seg_text_all(state, name):
    with torch.inference_mode(), ac:
        out = proc.set_text_prompt(state=state, prompt=name)
    tm = _np(out["masks"]); ts = _np(out["scores"]).ravel(); res = []
    if tm.ndim == 4:
        tm = tm[:, 0]
    for k in range(len(ts) if tm.ndim == 3 else 0):
        m = tm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.4:
            res.append((m, float(ts[k])))
    res.sort(key=lambda x: -x[1]); return res


def outline(img, m, col):
    if m is not None:
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, c, -1, col, 3)


def b64(rgb):
    im = Image.fromarray(rgb)
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


def ask(sys_txt, img, schema=SCH):
    try:
        r = models.vlm_call("Name what each hand is manipulating (or N/A).", [b64(img)], sys_txt, schema, max_tokens=400)
        if schema is SCH1:
            return (r.get("name") or "N/A").strip()
        return (r.get("left") or "N/A").strip(), (r.get("right") or "N/A").strip()
    except Exception as e:
        print("  ask err", e); return ("N/A" if schema is SCH1 else ("N/A", "N/A"))


rows = []
for fi, fr in enumerate(va):
    name = fr["image"]["name"]
    if name not in jpgs:
        continue
    bgr = cv2.imread(jpgs[name]); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    ents = {e["id"]: e for e in fr["annotations"]}
    hands = {}  # side -> {grip,box,gt}
    for e in fr["annotations"]:
        if "hand" not in e["name"]:
            continue
        side = "L" if "left" in e["name"] else "R"
        hm = mask_of(e["segments"], H, W)
        ico = e.get("in_contact_object")
        gt = ents[ico]["name"] if (ico in ents) else "N/A"
        om = None
        if ico in ents:                                  # grip = contact-region centroid; om = GT object mask
            om = mask_of(ents[ico]["segments"], H, W)
            inter = np.logical_and(hm, cv2.dilate(om.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool))
            ys, xs = np.nonzero(inter if inter.any() else hm)
        else:
            ys, xs = np.nonzero(hm)
        ys2, xs2 = np.nonzero(hm)
        hands[side] = {"grip": (int(xs.mean()), int(ys.mean())),
                       "box": (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())),
                       "gt": gt, "gtmask": om}
    with torch.inference_mode(), ac:
        state = proc.set_image(Image.fromarray(rgb))
    # point-seg held mask per hand
    pm = {}
    for s in hands:
        gx, gy = hands[s]["grip"]; oth = hands["R" if s == "L" else "L"]["grip"] if ("R" if s == "L" else "L") in hands else None
        pm[s] = seg_held(state, gx, gy, [list(oth)] if oth else [])
    # text-seg all inventory instances
    tinst = {nm: seg_text_all(state, nm) for nm in INV}
    # build images
    img_dots = rgb.copy()
    for s, col in (("L", GREEN), ("R", BLUE)):
        if s in hands:
            cv2.circle(img_dots, hands[s]["grip"], 12, col, -1); cv2.circle(img_dots, hands[s]["grip"], 12, (255, 255, 255), 2)
    img_sam = img_dots.copy(); outline(img_sam, pm.get("L"), GREEN); outline(img_sam, pm.get("R"), BLUE)
    img_best = img_dots.copy(); img_all = img_dots.copy()
    for ci, nm in enumerate(INV):
        col = CAND[ci % len(CAND)]; insts = tinst[nm]
        if insts:
            outline(img_best, insts[0][0], col)
            for m, _s in insts:
                outline(img_all, m, col)
    # zoom crop helper
    RAD = int(0.12 * W)

    def zoom(s, masks):
        gx, gy = hands[s]["grip"]; x0, y0 = max(0, gx - RAD), max(0, gy - RAD); x1, y1 = min(W, gx + RAD), min(H, gy + RAD)
        c = rgb[y0:y1, x0:x1].copy()
        for m, col in masks:
            if m is not None:
                outline(c, m[y0:y1, x0:x1], col)
        cv2.circle(c, (gx - x0, gy - y0), 7, (0, 255, 0), -1)
        return cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2))

    rec = {"frame": fi, "name": name, "gt": {s: hands[s]["gt"] for s in hands}}
    base = ask(sys_prompt(True, False), img_dots); pti = ask(sys_prompt(True, True), img_sam)
    txi = ask(sys_allcand(), img_best); txa = ask(sys_allcand(), img_all); ptn = ask(sys_prompt(False, True), img_sam)
    rec["noSAM_inv"] = {"left": base[0], "right": base[1]}
    rec["point_inv"] = {"left": pti[0], "right": pti[1]}
    rec["text_inv"] = {"left": txi[0], "right": txi[1]}
    rec["text_allmasks"] = {"left": txa[0], "right": txa[1]}
    rec["point_noinv"] = {"left": ptn[0], "right": ptn[1]}
    pz = {"left": "N/A", "right": "N/A"}; pzz = {"left": "N/A", "right": "N/A"}
    for s, key in (("L", "left"), ("R", "right")):
        if s in hands:
            pz[key] = ask(sys_zoom(True), zoom(s, [(pm.get(s), (0, 255, 0))]), SCH1)
            allm = [(tinst[nm][0][0] if tinst[nm] else None, CAND[ci % len(CAND)]) for ci, nm in enumerate(INV)]
            pzz[key] = ask(sys_zoom(True), zoom(s, allm), SCH1)
    rec["point_inv_zoom"] = pz; rec["text_inv_zoom"] = pzz

    # ---- mask IoU vs GT object mask (per hand, where GT has an object) ----
    def _iou(a, b):
        if a is None or b is None:
            return None
        u = int(np.logical_or(a, b).sum())
        return round(float(np.logical_and(a, b).sum()) / u, 3) if u else None

    def _tmask(nm):                                   # text-seg mask of the named inventory object
        for k in INV:
            if k.lower() == (nm or "").strip().lower() and tinst[k]:
                return tinst[k][0][0]
        return None
    miou = {}
    for cond, src in (("point_inv", "pt"), ("point_inv_zoom", "pt"), ("point_noinv", "pt"),
                      ("text_inv", "tx"), ("text_inv_zoom", "tx"), ("text_allmasks", "tx")):
        miou[cond] = {}
        for s, hk in (("L", "left"), ("R", "right")):
            if s not in hands or hands[s]["gtmask"] is None:
                miou[cond][hk] = None
            else:
                pred_mask = pm.get(s) if src == "pt" else _tmask(rec[cond][hk])
                miou[cond][hk] = _iou(pred_mask, hands[s]["gtmask"])
    rec["mask_iou"] = miou
    rows.append(rec)
    if fi % 10 == 0:
        print(f"frame {fi}/{len(va)} {name}: gt={rec['gt']} base={base} pt={pti}", flush=True)
json.dump({"video": ann_f, "inventory": INV, "rows": rows}, open(f"{outdir}/_visor_label.json", "w"), indent=2)
print(f"\nwrote {outdir}/_visor_label.json ({len(rows)} frames)", flush=True)
