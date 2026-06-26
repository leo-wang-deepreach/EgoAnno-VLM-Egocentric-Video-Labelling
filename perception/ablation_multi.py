#!/usr/bin/env python3
"""ablation_multi.py — 7-condition labeling ablation over MANY egocentric clips.

Datasets:
  * VISOR (EPIC-Kitchens): native hand->in_contact_object GT (per-hand manipulated object + polygon mask).
  * SA-Co SmartGlasses: objects + hands labelled separately (NO link), so we DERIVE the hand->object GT by
    mask overlap (the object whose mask overlaps a hand most = that hand's manipulated object). Approximate.

Per frame we derive, per hand: grip point, GT object name, GT object mask; the clip's distinct object names
are the "given inventory" (option-C analog). Conditions: baseline(no-SAM+inv), point+inv, point+inv ZOOM,
text+inv top, text+inv ZOOM, text+inv all-masks, point no-inv. Grades object-ID + N/A + mask IoU per hand.

The 9 VLM calls per frame run in PARALLEL (ThreadPool); SAM stays on the main thread.

Run (sam3py): ablation_multi.py <manifest.json> <outdir>   [env MAXF=10]
manifest: [{"dataset":"visor","name":"P09_02","ann":".../P09_02.json","frames":".../frames/P09_02"},
           {"dataset":"meta","name":"saco_sg_000451","split":"val","frames":".../extract"}, ...]
"""
import base64, io, json, glob, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/ubuntu/local/factsfirst")
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import models  # noqa
from pycocotools import mask as MUTIL

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-pro-preview")
MANIFEST, OUTDIR = sys.argv[1], sys.argv[2]
MAXF = int(os.environ.get("MAXF", "10"))
STAGE_DUMP = os.environ.get("STAGE_DUMP")
GRIP_BLIND = os.environ.get("GRIP_BLIND")   # grip = hand-mask centroid only (ignore GT object) — no GT leak into grip
METHOD_C = os.environ.get("METHOD_C")       # build the inventory with method C (VLM->SAM->VLM), NOT from GT
INV_DIAG = os.environ.get("INV_DIAG")       # only run inventory generation, dump pass1/pass2 vs GT, skip labeling
KPTS_JSON = os.environ.get("KPTS_JSON")     # if set -> GEOMETRIC grasp-region filter (no 2nd VLM); handpose kpts
GRASP_DILATE = float(os.environ.get("GRASP_DILATE", "1.0"))   # dilation factor for the grasp region
GRASP_EXPAND = float(os.environ.get("GRASP_EXPAND", "0.0"))   # extend non-wrist kpts outward from wrist by this fraction (0.2 = +20%)
REGION_MODE = os.environ.get("REGION_MODE", "hand")           # "hand" = convex hull of all 21 kpts; "pinch" = thumb+index
RENDER = os.environ.get("RENDER")                             # save per-frame overlays (grasp region + kept/removed masks)
KPTS = json.load(open(KPTS_JSON)) if KPTS_JSON else {}
GRASP = bool(KPTS_JSON)
META_ANN = {"val": "/home/ubuntu/local/saco_sg/annotation/smartglasses_val.json",
            "test": "/home/ubuntu/local/saco_sg/annotation/smartglasses_test.json"}
os.makedirs(OUTDIR, exist_ok=True)
THUMBS = os.path.join(OUTDIR, "thumbs"); os.makedirs(THUMBS, exist_ok=True)
SCH = {"type": "object", "properties": {"left": {"type": "string"}, "right": {"type": "string"}}, "required": ["left", "right"]}
SCH1 = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
NA = "Answer 'N/A' for a hand that is empty, resting, only reaching/touching but not holding, or not visible."
# ---- method-C inventory prompts (GT-blind: built from the frames, not from GT) ----
INV_SYS_M = (
    "This is the PROPOSAL stage of a multi-stage pipeline. Produce candidate objects that the person may be "
    "manipulating with their hands in this egocentric frame (holding, carrying, moving, using, or operating). "
    "Brainstorm generously: MISSING A REAL OBJECT IS WORSE THAN PROPOSING AN EXTRA ONE — a later per-object "
    "verification stage will remove wrong proposals. Use a short canonical name (colour/material/form when "
    "appropriate, <=4 words). Exclude hands, arms, body parts, clothing, worn items, and the passive "
    "supporting surface. List each candidate at most once. Output strict JSON.")
INV_SCH = {"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"]}}}, "required": ["objects"]}
PEROBJ_SYS = (
    "You are performing inventory consolidation, not object detection. ONE candidate object from a proposal "
    "list is under review. You are shown ALL the clip's sampled frames and (as the LAST image, when available) "
    "this candidate's best SAM crop, plus its SAM confidence and how many frames proposed it. SAM masks are "
    "often inaccurate, so the confidence is SUPPORTING evidence only — a low score is NOT by itself a reason "
    "to remove. Decide whether this candidate is a real physical object that a hand acts upon somewhere in "
    "the clip (holds/carries/moves/uses/operates). Return exactly one decision: KEEP (real and acted upon), "
    "RENAME (real and acted upon, but give the correct canonical name), or REMOVE (clearly NOT a real "
    "acted-upon object — a body part, the surface/background, or a hallucination). When the evidence is "
    "ambiguous, PREFER KEEP over REMOVE. "
    "Also fill a `feedback` field: in one or two sentences give the reason for your decision AND flag anything "
    "about this task, these instructions, or the evidence that was ambiguous, underspecified, or made you "
    "unsure — including any clarification that would have helped you decide. If nothing was unclear, write "
    "'clear'. Output strict JSON {decision, name, feedback}.")
PEROBJ_SCH = {"type": "object", "properties": {"decision": {"type": "string"}, "name": {"type": "string"},
              "feedback": {"type": "string"}}, "required": ["decision", "name", "feedback"]}
GREEN, BLUE = (0, 220, 0), (40, 120, 255)
CAND = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60), (200, 80, 255), (80, 220, 220),
        (255, 140, 0), (180, 180, 80)]
H, W = None, None


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# ----------------------------- prompts (INV passed in) -----------------------------
def sys_prompt(INV, with_inv, with_sam):
    s = ("You label what each HAND is manipulating in this egocentric frame. The LEFT hand's grip is a "
         "GREEN dot, the RIGHT hand's a BLUE dot. ")
    if with_sam:
        s += "The object SAM segmented at each hand is OUTLINED (green=left, blue=right) to guide you. "
    s += "For EACH hand, name the SINGLE object it is actively holding/using "
    s += ("choosing ONLY from this inventory: " + ", ".join(INV) + ". " if with_inv
          else "with a short canonical name (1-3 words). ")
    return s + NA + " Output strict JSON {left, right}."


def sys_allcand(INV):
    return ("You label what each HAND manipulates. LEFT grip=GREEN dot, RIGHT grip=BLUE dot. SAM has text-"
            "segmented and OUTLINED EVERY matching instance of each inventory object (multiple outlines per "
            "object possible), coloured + labelled. For EACH hand choose which inventory object it holds: "
            + ", ".join(INV) + ". " + NA + " Output strict JSON {left, right}.")


def sys_zoom(INV, with_inv):
    s = ("ZOOMED-IN crop around ONE hand in an egocentric frame; the grip is the GREEN dot and the object "
         "SAM segmented at the grip is OUTLINED. Name the SINGLE object this hand holds/uses ")
    s += ("from: " + ", ".join(INV) + ". " if with_inv else "with a short canonical name (1-3 words). ")
    return s + NA + " Output strict JSON {name}."


# ----------------------------- SAM model -----------------------------
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


# ----------------------------- GT builders (per dataset) -----------------------------
def _poly_mask(segs, h, w):
    m = np.zeros((h, w), np.uint8)
    for poly in segs:
        cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def _rle_mask(seg):
    if not seg:
        return None
    s = seg
    if isinstance(s.get("counts"), str):
        s = {"size": s["size"], "counts": s["counts"].encode()}
    return MUTIL.decode(s).astype(bool)


def _grip_box_gt(hm, om):
    """grip = centroid of hand∩dilated-object (else hand centroid); box = hand bbox."""
    ys2, xs2 = np.nonzero(hm)
    if om is not None:
        inter = np.logical_and(hm, cv2.dilate(om.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool))
        ys, xs = np.nonzero(inter if inter.any() else hm)
    else:
        ys, xs = np.nonzero(hm)
    return ((int(xs.mean()), int(ys.mean())),
            (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())))


def build_visor(ann, frames_dir):
    jpgs = {os.path.basename(p): p for p in glob.glob(f"{frames_dir}/**/*.jpg", recursive=True)}
    va = json.load(open(ann))["video_annotations"]
    # prefer frames where some hand has an in_contact_object
    def manip(fr):
        ents = {e["id"]: e for e in fr["annotations"]}
        return any("hand" in e["name"] and e.get("in_contact_object") in ents for e in fr["annotations"])
    pool = [fr for fr in va if fr["image"]["name"] in jpgs and manip(fr)] or [fr for fr in va if fr["image"]["name"] in jpgs]
    sel = pool[:: max(1, len(pool) // MAXF)][:MAXF]
    inv = []
    for fr in sel:
        ents = {e["id"]: e for e in fr["annotations"]}
        for e in fr["annotations"]:
            ico = e.get("in_contact_object")
            if "hand" in e["name"] and ico in ents and ents[ico]["name"] not in inv:
                inv.append(ents[ico]["name"])
    out = []
    for fi, fr in enumerate(sel):
        bgr = cv2.imread(jpgs[fr["image"]["name"]]); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]; ents = {e["id"]: e for e in fr["annotations"]}
        hands = {}
        for e in fr["annotations"]:
            if "hand" not in e["name"]:
                continue
            side = "L" if "left" in e["name"] else "R"
            hm = _poly_mask(e["segments"], h, w); ico = e.get("in_contact_object")
            om = _poly_mask(ents[ico]["segments"], h, w) if ico in ents else None
            gt = ents[ico]["name"] if ico in ents else "N/A"
            grip, box = _grip_box_gt(hm, None if GRIP_BLIND else om)
            hands[side] = {"grip": grip, "box": box, "gt": gt, "gtmask": om}
        out.append({"fi": fi, "name": fr["image"]["name"], "rgb": rgb, "hands": hands})
    return inv, out


def build_meta(split, video_name, frames_dir):
    d = json.load(open(META_ANN[split]))
    vid = next(v for v in d["videos"] if v["video_name"] == video_name)
    files = vid["file_names"]
    jpgs = {os.path.basename(p): p for p in glob.glob(f"{frames_dir}/**/{video_name}/*.jpg", recursive=True)}
    anns = [a for a in d["annotations"] if a["video_id"] == vid["id"]]
    hands = [a for a in anns if "hand" in (a.get("noun_phrase") or "").lower()]
    objs = [a for a in anns if "hand" not in (a.get("noun_phrase") or "").lower()]
    inv = []
    for o in objs:
        if o["noun_phrase"] not in inv:
            inv.append(o["noun_phrase"])
    # candidate frames = those where a hand mask overlaps an object mask
    cand = []
    for fi in range(len(files)):
        hm = [(_rle_mask(h["segmentations"][fi]) if fi < len(h["segmentations"]) else None) for h in hands]
        hm = [m for m in hm if m is not None and m.any()]
        if not hm:
            continue
        om = [(_rle_mask(o["segmentations"][fi]) if fi < len(o["segmentations"]) else None) for o in objs]
        if any(m is not None and m.any() for m in om):
            cand.append(fi)
    cand = cand[:: max(1, len(cand) // MAXF)][:MAXF] if cand else []
    out = []
    for k, fi in enumerate(cand):
        base = os.path.basename(files[fi])
        if base not in jpgs:
            continue
        bgr = cv2.imread(jpgs[base]); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # collect present hands, assign L/R by x-centroid
        hmasks = []
        for hh in hands:
            m = _rle_mask(hh["segmentations"][fi]) if fi < len(hh["segmentations"]) else None
            if m is not None and m.any():
                xs = np.nonzero(m)[1]; hmasks.append((float(xs.mean()), m))
        hmasks.sort(key=lambda x: x[0])
        omasks = []
        for oo in objs:
            m = _rle_mask(oo["segmentations"][fi]) if fi < len(oo["segmentations"]) else None
            if m is not None and m.any():
                omasks.append((oo["noun_phrase"], m))
        hd = {}
        sides = (["L", "R"] if len(hmasks) >= 2 else
                 (["L"] if (hmasks and hmasks[0][0] < w / 2) else ["R"]))
        for side, (_cx, hm) in zip(sides, hmasks):
            # GT object = object whose mask overlaps this hand most (dilated)
            best = None
            hd_dil = cv2.dilate(hm.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
            for nm, om in omasks:
                ov = int(np.logical_and(hd_dil, om).sum())
                if ov > 0 and (best is None or ov > best[2]):
                    best = (nm, om, ov)
            om = best[1] if best else None; gt = best[0] if best else "N/A"
            grip, box = _grip_box_gt(hm, None if GRIP_BLIND else om)
            hd[side] = {"grip": grip, "box": box, "gt": gt, "gtmask": om}
        out.append({"fi": k, "name": base, "rgb": rgb, "hands": hd})
    return inv, out


# ----------------------------- per-frame processing (parallel VLM) -----------------------------
def process_frame(frec, INV):
    global H, W
    rgb = frec["rgb"]; H, W = rgb.shape[:2]; hands = frec["hands"]
    with torch.inference_mode(), ac:
        state = proc.set_image(Image.fromarray(rgb))
    pm = {}
    for s in hands:
        oth = ("R" if s == "L" else "L")
        og = hands[oth]["grip"] if oth in hands else None
        pm[s] = seg_held(state, hands[s]["grip"][0], hands[s]["grip"][1], [list(og)] if og else [])
    tinst = {nm: seg_text_all(state, nm) for nm in INV}
    img_dots = rgb.copy()
    for s, col in (("L", GREEN), ("R", BLUE)):
        if s in hands:
            cv2.circle(img_dots, hands[s]["grip"], 12, col, -1); cv2.circle(img_dots, hands[s]["grip"], 12, (255, 255, 255), 2)
    img_sam = img_dots.copy(); outline(img_sam, pm.get("L"), GREEN); outline(img_sam, pm.get("R"), BLUE)
    # thumbnail for the website: grip dots + GT object outlined white
    thumb = img_dots.copy()
    for s in hands:
        outline(thumb, hands[s]["gtmask"], (255, 255, 255))
    th = cv2.resize(thumb, (480, int(480 * H / W)))
    cv2.imwrite(f"{THUMBS}/{frec.get('video', 'v')}_{frec['fi']:02d}.jpg",
                cv2.cvtColor(th, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    img_best = img_dots.copy(); img_all = img_dots.copy()
    for ci, nm in enumerate(INV):
        col = CAND[ci % len(CAND)]
        if tinst[nm]:
            outline(img_best, tinst[nm][0][0], col)
            for m, _s in tinst[nm]:
                outline(img_all, m, col)
    RAD = int(0.12 * W)

    def zoom(s, masks):
        gx, gy = hands[s]["grip"]; x0, y0 = max(0, gx - RAD), max(0, gy - RAD); x1, y1 = min(W, gx + RAD), min(H, gy + RAD)
        c = rgb[y0:y1, x0:x1].copy()
        for m, col in masks:
            if m is not None:
                outline(c, m[y0:y1, x0:x1], col)
        cv2.circle(c, (gx - x0, gy - y0), 7, (0, 255, 0), -1)
        return cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2))

    # build the parallel task list (key, sys, img, schema)
    tasks = [("noSAM_inv", sys_prompt(INV, True, False), img_dots, SCH),
             ("vlm_e2e", sys_prompt(INV, False, False), img_dots, SCH),   # pure VLM end-to-end: no inv, no SAM
             ("point_inv", sys_prompt(INV, True, True), img_sam, SCH),
             ("text_inv", sys_allcand(INV), img_best, SCH),
             ("text_allmasks", sys_allcand(INV), img_all, SCH),
             ("point_noinv", sys_prompt(INV, False, True), img_sam, SCH)]
    zimgs = {}
    for s, key in (("L", "left"), ("R", "right")):
        if s in hands:
            zp = zoom(s, [(pm.get(s), (0, 255, 0))])
            allm = [(tinst[nm][0][0] if tinst[nm] else None, CAND[ci % len(CAND)]) for ci, nm in enumerate(INV)]
            za = zoom(s, allm)
            zimgs[f"pz_{s}"] = zp; zimgs[f"pzz_{s}"] = za
            tasks.append((f"pz_{s}", sys_zoom(INV, True), zp, SCH1))
            tasks.append((f"pzz_{s}", sys_zoom(INV, True), za, SCH1))
    if STAGE_DUMP:                                   # dump the EXACT images each stage sends to the VLM
        sd = f"{OUTDIR}/stages/{frec.get('video', 'v')}_{frec['fi']:02d}"; os.makedirs(sd, exist_ok=True)
        def _save(nm, rgbimg):
            cv2.imwrite(f"{sd}/{nm}.jpg", cv2.cvtColor(rgbimg, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        _save("0_frame", rgb); _save("1_grip_noSAM", img_dots); _save("2_pointseg", img_sam)
        _save("3_textseg_top", img_best); _save("4_textseg_all", img_all); _save("gt_white", thumb)
        for k, zi in zimgs.items():
            _save(k, zi)
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = {k: ex.submit(ask, sysx, img, sch) for k, sysx, img, sch in tasks}
        res = {k: f.result() for k, f in futs.items()}

    rec = {"frame": frec["fi"], "name": frec["name"], "gt": {s: hands[s]["gt"] for s in hands}}
    rec["noSAM_inv"] = {"left": res["noSAM_inv"][0], "right": res["noSAM_inv"][1]}
    rec["vlm_e2e"] = {"left": res["vlm_e2e"][0], "right": res["vlm_e2e"][1]}
    rec["point_inv"] = {"left": res["point_inv"][0], "right": res["point_inv"][1]}
    rec["text_inv"] = {"left": res["text_inv"][0], "right": res["text_inv"][1]}
    rec["text_allmasks"] = {"left": res["text_allmasks"][0], "right": res["text_allmasks"][1]}
    rec["point_noinv"] = {"left": res["point_noinv"][0], "right": res["point_noinv"][1]}
    rec["point_inv_zoom"] = {"left": res.get("pz_L", "N/A"), "right": res.get("pz_R", "N/A")}
    rec["text_inv_zoom"] = {"left": res.get("pzz_L", "N/A"), "right": res.get("pzz_R", "N/A")}

    def _iou(a, b):
        if a is None or b is None:
            return None
        u = int(np.logical_or(a, b).sum())
        return round(float(np.logical_and(a, b).sum()) / u, 3) if u else None

    def _tmask(nm):
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
    return rec


# ----------------------------- method-C inventory (VLM->SAM->VLM, GT-blind) -----------------------------
def pinch_region(hands, H, W):
    """Grasp region from hand keypoints, dilated. REGION_MODE 'hand' = convex hull of ALL 21 keypoints
    (whole-hand grasp footprint); 'pinch' = thumb (1-4) + index (5-8) only. bool mask (H,W) or None."""
    if not hands:
        return None
    idxs = range(21) if REGION_MODE == "hand" else (1, 2, 3, 4, 5, 6, 7, 8)
    reg = np.zeros((H, W), np.uint8); got = False
    for h in hands:
        kp = h.get("kpts") or []
        valid = [(j, float(kp[j][0]), float(kp[j][1])) for j in idxs if len(kp) > j and kp[j][2] > 0.25]
        if len(valid) < 3:
            continue
        # anchor = wrist (kp0) if confident, else centroid; non-wrist points extend OUTWARD from anchor by GRASP_EXPAND
        if len(kp) > 0 and kp[0][2] > 0.25:
            ax, ay = float(kp[0][0]), float(kp[0][1])
        else:
            ax = sum(v[1] for v in valid) / len(valid); ay = sum(v[2] for v in valid) / len(valid)
        pts = []
        for j, x, y in valid:
            if GRASP_EXPAND > 0 and j != 0:
                x = ax + (1.0 + GRASP_EXPAND) * (x - ax); y = ay + (1.0 + GRASP_EXPAND) * (y - ay)
            pts.append([int(round(x)), int(round(y))])
        hull = cv2.convexHull(np.array(pts, np.int32))
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, hull, 1)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5      # hand size in this frame
        ksz = max(25, int(GRASP_DILATE * 0.15 * diag))                            # catch grips just past the hand
        reg |= cv2.dilate(m, np.ones((ksz, ksz), np.uint8)); got = True
    return reg.astype(bool) if got else None


def build_inventory_C(frames_rgb, frame_kpts=None):
    global H, W
    frame_kpts = frame_kpts or [[] for _ in frames_rgb]
    # STAGE 1: per-frame VLM proposal -> union with votes (parallel)
    def _propose(rgb):
        try:
            r = models.vlm_call("Propose candidate objects the hands may be manipulating in this frame.",
                                [b64(rgb)], INV_SYS_M, INV_SCH, max_tokens=1200)
            seen, out = set(), []
            for o in r.get("objects", []):
                n = (o.get("name") or "").strip()
                if n and n.lower() not in seen:
                    seen.add(n.lower()); out.append(n)
            return out
        except Exception as e:
            print("  invC s1 err", e); return []
    votes, disp = {}, {}
    with ThreadPoolExecutor(max_workers=min(8, len(frames_rgb))) as ex:
        for names in ex.map(_propose, frames_rgb):
            for n in names:
                votes[n.lower()] = votes.get(n.lower(), 0) + 1; disp.setdefault(n.lower(), n)
    raw = sorted(votes, key=lambda k: -votes[k])
    if not raw:
        return [], {"pass1": [], "sam_conf": {}, "votes": {}}
    # STAGE 2: SAM evidence — per-object text-seg across frames: best crop + confidence, AND (if GRASP) whether
    # the object's mask lands inside the thumb-index pinch region in any frame.
    states, regions = [], []
    for rgb, hk in zip(frames_rgb, frame_kpts):
        H, W = rgb.shape[:2]
        with torch.inference_mode(), ac:
            states.append(proc.set_image(Image.fromarray(rgb)))
        regions.append(pinch_region(hk, H, W) if GRASP else None)
    framemasks = [dict() for _ in frames_rgb] if RENDER else None
    frame_confirmed = [[] for _ in frames_rgb] if RENDER else None   # per-frame: objects whose mask hit the grasp region
    sam_score, best, hit_frame = {}, {}, {}
    for k in raw:
        bsc, bcrop, hit = 0.0, None, None
        for i, (rgb, st) in enumerate(zip(frames_rgb, states)):
            h, w = rgb.shape[:2]
            with torch.inference_mode(), ac:
                out = proc.set_text_prompt(state=st, prompt=disp[k])
            sc = _np(out["scores"]).ravel(); mk = _np(out["masks"])
            if mk.ndim == 4:
                mk = mk[:, 0]
            fbest_m, fbest_sc = None, 0.0
            for j in range(len(sc)):
                m = (mk[j] > 0.5)
                if m.ndim > 2:
                    m = m.squeeze()
                if m.shape != (h, w) or not (0.0004 < float(m.mean()) < 0.4):
                    continue
                if float(sc[j]) > bsc:
                    ys, xs = np.nonzero(m); pad = 20
                    bcrop = rgb[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad].copy()
                    bsc = float(sc[j])
                if float(sc[j]) > fbest_sc:
                    fbest_sc = float(sc[j]); fbest_m = m
                if regions[i] is not None and np.logical_and(m, regions[i]).any():
                    if hit is None:
                        hit = i
                    if frame_confirmed is not None and disp[k] not in frame_confirmed[i]:
                        frame_confirmed[i].append(disp[k])
            if framemasks is not None and fbest_m is not None:
                framemasks[i][disp[k]] = fbest_m
        sam_score[k] = round(bsc, 3); best[k] = bcrop; hit_frame[k] = hit

    if GRASP:
        # GEOMETRIC FILTER (no 2nd VLM): keep a candidate iff its SAM mask was inside a thumb-index pinch
        # region in >=1 frame. If NO hand was detected anywhere in the clip we cannot filter -> keep all.
        any_region = any(r is not None for r in regions)
        verify = []
        for k in raw:
            kept = (hit_frame[k] is not None) or (not any_region)
            fb = (f"inside {REGION_MODE} grasp region at frame {hit_frame[k]}" if hit_frame[k] is not None
                  else ("no hand detected in clip — kept (cannot filter)" if not any_region
                        else f"SAM mask never inside the {REGION_MODE} grasp region"))
            verify.append({"candidate": disp[k], "decision": "KEEP" if kept else "REMOVE", "name": disp[k],
                           "sam_conf": sam_score[k], "votes": votes[k], "feedback": fb,
                           "kept": disp[k] if kept else None})
        final, seen = [], set()
        for v in verify:
            if v["kept"] and v["kept"].lower() not in seen:
                seen.add(v["kept"].lower()); final.append(v["kept"])
        info = {"pass1": [disp[k] for k in raw], "sam_conf": {disp[k]: sam_score[k] for k in raw},
                "votes": {disp[k]: votes[k] for k in raw}, "verify": verify}
        if RENDER:
            info["framemasks"] = framemasks; info["regions"] = regions; info["final"] = final
            info["frame_confirmed"] = frame_confirmed
        return (final or [disp[k] for k in raw]), info

    # PASS 2 (fallback, no kpts): PER-OBJECT VLM verify — each call sees ALL sampled frames + the candidate's crop.
    base = [b64(f) for f in frames_rgb]

    def _verify(k):
        crop = best[k]; has = crop is not None and crop.size
        imgs = base + ([b64(crop)] if has else [])
        msg = (f"Candidate object:\nName: {disp[k]}\nProposed in {votes[k]}/{len(frames_rgb)} frames\n"
               f"Best SAM confidence: {sam_score[k]:.2f}\n"
               + ("The LAST image is this candidate's best SAM crop." if has
                  else "SAM could not segment this candidate (no crop) — judge from the frames."))
        try:
            r = models.vlm_call(msg, imgs, PEROBJ_SYS, PEROBJ_SCH, max_tokens=400)
            dec = (r.get("decision") or "").strip().upper()
            nm = (r.get("name") or "").strip() or disp[k]
            fb = (r.get("feedback") or "").strip()
        except Exception as e:
            print("  perobj err", e); dec, nm, fb = "KEEP(err)", disp[k], ""
        return {"candidate": disp[k], "decision": dec, "name": nm, "sam_conf": sam_score[k],
                "votes": votes[k], "feedback": fb, "kept": None if dec.startswith("REMOVE") else nm}
    with ThreadPoolExecutor(max_workers=min(8, len(raw))) as ex:
        vres = list(ex.map(_verify, raw))
    final, seen = [], set()
    for v in vres:
        nm = v["kept"]
        if nm and nm.lower() not in seen:
            seen.add(nm.lower()); final.append(nm)
    info = {"pass1": [disp[k] for k in raw], "sam_conf": {disp[k]: sam_score[k] for k in raw},
            "votes": {disp[k]: votes[k] for k in raw}, "verify": vres}
    return (final or [disp[k] for k in raw]), info


# ----------------------------- driver -----------------------------
manifest = json.load(open(MANIFEST))
all_rows = []
diag = []
for v in manifest:
    name, ds = v["name"], v["dataset"]
    print(f"\n=== {ds}:{name} ===", flush=True)
    try:
        if ds == "visor":
            INV, frames = build_visor(v["ann"], v["frames"])
        else:
            INV, frames = build_meta(v.get("split", "val"), name, v["frames"])
    except Exception as e:
        print("  BUILD FAIL", repr(e)); continue
    gt_inv = INV
    if METHOD_C or INV_DIAG:                          # replace GT inventory with method-C generated one
        fk = [KPTS.get(f"{name}/{fr['name']}", []) for fr in frames]   # handpose kpts for the geometric filter
        INV, invinfo = build_inventory_C([f["rgb"] for f in frames], fk)
        print(f"  pass1 ({len(invinfo['pass1'])}): {invinfo['pass1']}", flush=True)
        print(f"  pass2/final ({len(INV)}): {INV}", flush=True)
        print(f"  (GT inventory was ({len(gt_inv)}): {gt_inv})", flush=True)
        if RENDER and "framemasks" in invinfo:    # per-frame overlay: grasp region + kept(green)/removed(red) masks
            gdir = os.path.join(OUTDIR, "gallery_frames"); os.makedirs(gdir, exist_ok=True)
            fc = invinfo.get("frame_confirmed") or [[] for _ in frames]
            for i, frec in enumerate(frames):
                ov = cv2.cvtColor(frec["rgb"], cv2.COLOR_RGB2BGR).copy()
                reg = invinfo["regions"][i]
                if reg is not None:
                    cs, _ = cv2.findContours(reg.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(ov, cs, -1, (190, 190, 190), 2)
                conf_i = set(fc[i])                  # objects CONFIRMED in THIS frame (mask in grasp region here)
                for nm, m in invinfo["framemasks"][i].items():
                    kept = nm in conf_i
                    col = (70, 210, 70) if kept else (70, 70, 225)
                    cs, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(ov, cs, -1, col, 3 if kept else 1)
                    if kept:
                        ys, xs = np.nonzero(m)
                        cv2.putText(ov, nm[:20], (max(2, int(xs.mean()) - 45), max(15, int(ys.min()) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(ov, nm[:20], (max(2, int(xs.mean()) - 45), max(15, int(ys.min()) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
                hh, ww = ov.shape[:2]
                cv2.imwrite(f"{gdir}/{name}_{i:02d}.jpg", cv2.resize(ov, (760, int(760 * hh / ww))),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        if INV_DIAG:                                  # inventory diagnostic only — skip labeling
            diag.append({"dataset": ds, "video": name, "gt": gt_inv, "pass1": invinfo["pass1"],
                         "pass2": INV, "sam_conf": invinfo["sam_conf"], "votes": invinfo["votes"],
                         "verify": invinfo.get("verify", []), "frame_confirmed": invinfo.get("frame_confirmed")})
            continue
    else:
        print(f"  inventory ({len(INV)}): {INV}  | frames: {len(frames)}", flush=True)
    vid_rows = []
    for frec in frames:
        frec["video"] = name
        rec = process_frame(frec, INV); rec["video"] = name; rec["dataset"] = ds
        vid_rows.append(rec); all_rows.append(rec)
        print(f"  {name} f{rec['frame']} {rec['name']}: gt={rec['gt']} base={rec['noSAM_inv']} txt+inv={rec['text_inv']}", flush=True)
    json.dump({"video": name, "dataset": ds, "inventory": INV, "gt_inventory": gt_inv, "rows": vid_rows},
              open(f"{OUTDIR}/{name}_label.json", "w"), indent=2)
if INV_DIAG:
    json.dump({"clips": diag}, open(f"{OUTDIR}/_inv_diag.json", "w"), indent=2)
    print(f"\nINV_DIAG done: {len(diag)} clips -> {OUTDIR}/_inv_diag.json", flush=True)
else:
    json.dump({"rows": all_rows}, open(f"{OUTDIR}/_all.json", "w"), indent=2)
    print(f"\nDONE {len(all_rows)} frames across {len(manifest)} videos -> {OUTDIR}/_all.json", flush=True)
