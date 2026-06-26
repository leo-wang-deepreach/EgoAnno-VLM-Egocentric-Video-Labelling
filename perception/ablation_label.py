#!/usr/bin/env python3
"""ablation_label.py — ABLATION 2: is SAM useful in the per-frame LABELING step? (sam3py; VLM = Gemini)
Per frame, per hand, name the manipulated object (or N/A), under 3 conditions; condition 4 (current pipeline,
ground_simple) is scored separately from the dense run.
  1) VLM + inventory, NO SAM            (frame + L/R grip dots + inventory list)
  2) VLM + inventory + SAM highlight    (+ SAM's held-object mask outlined per hand)
  3) VLM + SAM highlight, NO inventory  (free naming)
Always allows N/A. Writes per-frame labels -> <outdir>/_label_ablation.json

Run (sam3py): ablation_label.py <video> <dense_prompts.json> <outdir>
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
os.makedirs(outdir, exist_ok=True)
INV = ["clear plastic cup", "clear plastic jar", "yellow plastic scoop", "clear plastic lid"]   # option C
GREEN, BLUE = (0, 220, 0), (40, 120, 255)        # RGB: left, right
SCH = {"type": "object", "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
       "required": ["left", "right"]}
NA = ("Answer 'N/A' for a hand that is empty, resting, only reaching/touching but not yet holding, or not "
      "visible/unmarked.")


CAND_COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60)]   # RGB per inventory object


def sys_prompt(with_inv, with_sam):
    s = ("You label what each HAND is manipulating in this egocentric tabletop frame. The LEFT hand's grip "
         "point is a GREEN dot, the RIGHT hand's a BLUE dot. ")
    if with_sam:
        s += "The object SAM segmented at each hand is OUTLINED (green=left, blue=right) to guide you. "
    s += "For EACH hand, name the SINGLE object that hand is actively holding/using "
    s += ("choosing ONLY from this inventory: " + ", ".join(INV) + ". " if with_inv
          else "with a short canonical name (colour + material + form, <=4 words). ")
    return s + NA + " Output strict JSON {left, right}."


def sys_allcand():
    return ("You label what each HAND is manipulating in this egocentric tabletop frame. The LEFT hand's grip "
            "point is a GREEN dot, the RIGHT hand's a BLUE dot. SAM has localized EACH inventory object — every "
            "one is OUTLINED and LABELLED with its name in the image. For EACH hand, choose which of these "
            "inventory objects it is actively holding/using: " + ", ".join(INV) + ". " + NA +
            " Output strict JSON {left, right}.")


def sys_allmasks():
    return ("You label what each HAND is manipulating in this egocentric tabletop frame. The LEFT hand's grip "
            "point is a GREEN dot, the RIGHT hand's a BLUE dot. SAM has text-segmented the inventory objects and "
            "OUTLINED EVERY matching instance it found (there may be MULTIPLE outlines per object — e.g. several "
            "cups), each coloured + labelled by name. For EACH hand, choose which of these inventory objects it "
            "is actively holding/using: " + ", ".join(INV) + ". " + NA + " Output strict JSON {left, right}.")


def b64(rgb):
    im = Image.fromarray(rgb)
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


pj = json.load(open(prompts_f)); W, H = pj["W"], pj["H"]
byt = {}
for p in pj["prompts"]:
    byt.setdefault(round(p["t"], 3), {})[p["hand"]] = p
times = sorted(byt)
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def seg_held(state, p, oth):
    """point-seg the held object at the grasp anchor (+ wrist & other-hand negatives), cleaned to the anchor."""
    ax, ay = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
    negs = []
    b = p.get("box_px")
    if b:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        negs.append([_clamp(2 * cx - ax, 0, W - 1), _clamp(2 * cy - ay, 0, H - 1)])    # wrist (reflect)
    if oth:
        negs.append([_clamp(oth["x_px"], 0, W - 1), _clamp(oth["y_px"], 0, H - 1)])    # other hand
    pts = np.array([[ax, ay]] + negs, float); lab = np.array([1] + [0] * len(negs), int)
    with torch.inference_mode(), ac:
        mm, ss, _ = model.predict_inst(state, point_coords=pts, point_labels=lab, multimask_output=True)
    mm = _np(mm); ss = _np(ss).ravel(); best = None
    for k in range(len(ss) if mm.ndim == 3 else 0):
        m = mm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.35:
            mm2 = m.astype(np.uint8); n, labim, _, cents = cv2.connectedComponentsWithStats(mm2, 8)
            if n > 2:
                sl = int(labim[ay, ax]) or min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
                m = labim == sl
            if best is None or float(ss[k]) > best[1]:
                best = (m, float(ss[k]))
    return best[0] if best else None


def draw_dot(img, p, col):
    cv2.circle(img, (_clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)), 11, col, -1)
    cv2.circle(img, (_clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)), 11, (255, 255, 255), 2)


def outline(img, m, col):
    if m is not None:
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, c, -1, col, 3)


def seg_text_all(state, name):
    """ALL size-sane text-seg instance masks of `name`, best-score first."""
    with torch.inference_mode(), ac:
        out = proc.set_text_prompt(state=state, prompt=name)
    tm = _np(out["masks"]); ts = _np(out["scores"]).ravel(); res = []
    if tm.ndim == 4:
        tm = tm[:, 0]
    for k in range(len(ts) if tm.ndim == 3 else 0):
        m = tm[k] > 0.5
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape == (H, W) and 0.0004 < float(m.mean()) < 0.35:
            res.append((m, float(ts[k])))
    res.sort(key=lambda x: -x[1])
    return res


cap = cv2.VideoCapture(video); rows = []
for ti, t in enumerate(times):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, bgr = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    L, R = byt[t].get("L"), byt[t].get("R")
    with torch.inference_mode(), ac:
        state = proc.set_image(Image.fromarray(rgb))
    mL = seg_held(state, L, R) if L else None
    mR = seg_held(state, R, L) if R else None
    # image with just grip dots (cond 1)
    img_dots = rgb.copy()
    if L:
        draw_dot(img_dots, L, GREEN)
    if R:
        draw_dot(img_dots, R, BLUE)
    # image with dots + per-hand held-object SAM masks (cond 2,3)
    img_sam = img_dots.copy(); outline(img_sam, mL, GREEN); outline(img_sam, mR, BLUE)
    # text-seg each inventory object ONCE -> BEST-instance overlay (2b) + ALL-instances overlay (4)
    img_best = img_dots.copy(); img_allm = img_dots.copy()
    for ci, nm in enumerate(INV):
        insts = seg_text_all(state, nm); col = CAND_COLORS[ci % len(CAND_COLORS)]
        lab = nm.replace("clear plastic ", "").replace("yellow plastic ", "")
        if insts:
            bm = insts[0][0]; outline(img_best, bm, col)          # 2b: best instance only
            ys, xs = np.nonzero(bm)
            cv2.putText(img_best, lab, (int(xs.mean()) - 20, int(ys.min()) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
            for (m, _s) in insts:                                  # 4: every matching instance
                outline(img_allm, m, col)
            ys, xs = np.nonzero(bm)
            cv2.putText(img_allm, lab, (int(xs.mean()) - 20, int(ys.min()) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    b_dots, b_sam, b_best, b_allm = b64(img_dots), b64(img_sam), b64(img_best), b64(img_allm)
    user = "Name the object the LEFT hand and the RIGHT hand are each manipulating (or N/A)."

    def ask(sys_txt, img):
        try:
            r = models.vlm_call(user, [img], sys_txt, SCH, max_tokens=400)
            return (r.get("left") or "N/A").strip(), (r.get("right") or "N/A").strip()
        except Exception as e:
            print("  ask err", e); return "N/A", "N/A"
    c_nosam = ask(sys_prompt(True, False), b_dots)    # baseline: inventory, no SAM
    c_pt_i = ask(sys_prompt(True, True), b_sam)       # 2: point-seg + inventory
    c_tx_i = ask(sys_allcand(), b_best)               # 2b: text-seg + inventory (best instance per object)
    c_pt_n = ask(sys_prompt(False, True), b_sam)      # 3: point-seg, no inventory
    c_tx_all = ask(sys_allmasks(), b_allm)            # 4: text-seg + inventory, ALL instance masks shown
    rows.append({"frame": ti, "t": round(t, 2),
                 "noSAM_inv": {"left": c_nosam[0], "right": c_nosam[1]},
                 "point_inv": {"left": c_pt_i[0], "right": c_pt_i[1]},
                 "text_inv": {"left": c_tx_i[0], "right": c_tx_i[1]},
                 "point_noinv": {"left": c_pt_n[0], "right": c_pt_n[1]},
                 "text_allmasks": {"left": c_tx_all[0], "right": c_tx_all[1]},
                 "sam_held": {"left": bool(mL is not None), "right": bool(mR is not None)}})
    if ti % 10 == 0:
        print(f"frame {ti}/{len(times)} t={t:.0f}: pt+i={c_pt_i} tx+i={c_tx_i} pt-n={c_pt_n} tx-all={c_tx_all}", flush=True)
cap.release()
json.dump({"video": video, "inventory_C": INV, "model": os.environ["GEMINI_MODEL"], "rows": rows},
          open(f"{outdir}/_label_ablation.json", "w"), indent=2)
print(f"\nwrote {outdir}/_label_ablation.json ({len(rows)} frames)", flush=True)
