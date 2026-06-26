#!/usr/bin/env python3
"""ablation_inventory.py — ABLATION 1: SAM3 as a VERIFIER of the VLM inventory (sam3py; VLM = Gemini via models).
Flow:
  1. sample N frames across a window -> Gemini lists objects per frame -> UNION = raw inventory (with vote counts)
  2. SAM3: (a) text-seg EACH raw object across the frames (best view + score), (b) point-grid "segment-everything"
     on a rich frame (approx, since SAM3 has no AMG)
  3. Gemini VERIFY: shown the clean frame, the segment-everything overlay, and a labeled per-object crop montage
     -> outputs the FINAL corrected inventory (drop unconfirmable, add clearly-missed)
Writes raw + final inventories + the review images to <outdir>.

Run (sam3py): ablation_inventory.py <video> <t0> <t1> <outdir> [n_frames=8]
"""
import base64, io, json, os, sys
sys.path.insert(0, "/home/ubuntu/local/factsfirst")
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import models  # noqa: E402  (pure-HTTP; vlm_call routes to Gemini)

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-pro-preview")
video, t0, t1, outdir = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 8
os.makedirs(outdir, exist_ok=True)

INV_SYS = (
    "Your job is to inventory the objects the person MANIPULATES WITH THEIR HANDS in this egocentric tabletop "
    "video frame — i.e. objects a hand PICKS UP, HOLDS, MOVES, POURS, or USES AS A TOOL. Give each a canonical "
    "name = colour + material + form (<=4 words), e.g. 'clear plastic cup'. INCLUDE caps/lids and handheld "
    "devices that are actually picked up, even if partly occluded. EXCLUDE: hands/arms/body/clothing/worn items; "
    "the work surface/table; FIXED containers, trays, bins, racks, holders, or a weighing scale that the task "
    "happens ON or IN but that are NOT themselves picked up; and loose material/contents (e.g. granular beans) "
    "sitting inside a container. List each DISTINCT manipulated object AT MOST ONCE — no duplicates/synonyms "
    "(typically 3-6 objects). Output strict JSON.")
INV_SCHEMA = {"type": "object", "properties": {"objects": {"type": "array",
              "items": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
              "required": ["objects"]}
# Condition B: a 2nd VLM pass, SAME frames, NO SAM evidence
VERIFY_B_SYS = (
    "You proposed an inventory of the objects the person MANIPULATES WITH THEIR HANDS (picks up/holds/moves/"
    "pours/uses as a tool) in this egocentric tabletop video. Re-examine the frames CAREFULLY and output the "
    "FINAL corrected inventory of MANIPULATED objects: KEEP only objects a hand actually picks up/holds/uses; "
    "REMOVE fixtures (work surface, fixed trays/bins/racks/holders, weighing scale), worn items, loose contents "
    "(e.g. beans inside a container), hallucinations, and MERGE duplicates/synonyms that refer to the SAME "
    "physical object (e.g. spoon vs scoop, bin vs tray, container vs cup/jar); ADD any clearly-manipulated "
    "object you missed. Canonical colour+material+form names (<=4 words). Output strict JSON.")
# Condition C: a 2nd VLM pass with SAM evidence (segment-everything + per-object seg + confidence)
VERIFY_C_SYS = (
    "You are VERIFYING a proposed inventory of the objects the person MANIPULATES WITH THEIR HANDS (picks up/"
    "holds/moves/pours/uses) in an egocentric tabletop task. You are shown the same frames PLUS: a "
    "SEGMENT-EVERYTHING overlay (coloured outlines of every region a segmenter found) and a montage of "
    "per-object SAM segmentation crops, each titled with the proposed NAME and a SAM confidence (0-1: low "
    "confidence means SAM could not clearly segment that named object -> likely a hallucination). Output the "
    "FINAL corrected inventory of MANIPULATED objects: KEEP only objects a hand actually picks up/holds/uses; "
    "REMOVE fixtures (surface, fixed trays/bins/racks, scale), worn items, loose contents, and anything the "
    "segmentations do not support (low SAM confidence AND not clearly held); MERGE duplicates/synonyms for the "
    "same physical object; ADD any clearly-manipulated object missed (use the overlay to spot misses). "
    "Canonical colour+material+form names. Output strict JSON.")
VERIFY_SCHEMA = {"type": "object", "properties": {"objects": {"type": "array",
                 "items": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
                 "reason": {"type": "string"}}, "required": ["objects"]}


def b64(rgb, q=90):
    im = Image.fromarray(rgb)
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q); return base64.b64encode(buf.getvalue()).decode()


def norm(s):
    return " ".join(str(s).strip().lower().split())


# ---- sample frames ----
cap = cv2.VideoCapture(video); fps = cap.get(5) or 30
idxs = [int(round(t * fps)) for t in np.linspace(t0, t1, N)]
frames = []
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, bgr = cap.read()
    if ok:
        frames.append((fi / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
cap.release()
print(f"sampled {len(frames)} frames over [{t0},{t1}]s", flush=True)

# ---- STAGE 1: Gemini per-frame inventory -> union ----
votes, disp = {}, {}
perframe = []
for t, rgb in frames:
    try:
        r = models.vlm_call("List every distinct task-relevant object in this frame.", [b64(rgb)], INV_SYS, INV_SCHEMA, max_tokens=1400)
        seen, names = set(), []
        for o in r.get("objects", []):          # dedupe within a frame (guard against repetition loops)
            n = o.get("name")
            if n and norm(n) not in seen:
                seen.add(norm(n)); names.append(n)
    except Exception as e:
        names = []; print("inv call err", e)
    perframe.append({"t": round(t, 1), "objects": names})
    for n in names:
        votes[norm(n)] = votes.get(norm(n), 0) + 1; disp.setdefault(norm(n), n)
    print(f"  t={t:.0f}s: {names}", flush=True)
raw = sorted(votes, key=lambda k: -votes[k])
print(f"\nRAW union inventory ({len(raw)}): " + ", ".join(f"{disp[k]}({votes[k]})" for k in raw), flush=True)

# ---- STAGE 2: SAM verify ----
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# (a) per-object best text-seg across frames
best = {k: (0.0, None, None) for k in raw}     # name -> (score, crop_rgb, mask)
states = []
for t, rgb in frames:
    with torch.inference_mode(), ac:
        states.append(proc.set_image(Image.fromarray(rgb)))
for k in raw:
    for (t, rgb), st in zip(frames, states):
        with torch.inference_mode(), ac:
            out = proc.set_text_prompt(state=st, prompt=disp[k])
        sc = _np(out["scores"]).ravel(); mk = _np(out["masks"])
        if mk.ndim == 4:
            mk = mk[:, 0]
        if len(sc) == 0:
            continue
        j = int(np.argmax(sc))
        if float(sc[j]) > best[k][0]:
            m = (mk[j] > 0.5)
            if m.ndim > 2:
                m = m.squeeze()
            if m.shape == rgb.shape[:2] and m.any():
                ys, xs = np.nonzero(m); pad = 20
                y0, y1c = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
                x0, x1c = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
                crop = rgb[y0:y1c, x0:x1c].copy()
                best[k] = (float(sc[j]), crop, None)
sam_score = {k: round(best[k][0], 3) for k in raw}
print("SAM per-object best text-seg score:", sam_score, flush=True)

# (b) point-grid "segment-everything" on the richest frame (most raw objects detected)
rich_t, rich_rgb = frames[len(frames) // 2]
st = states[len(frames) // 2]; H, W = rich_rgb.shape[:2]
ev = rich_rgb.copy(); masks = []; G = 8
for gy in np.linspace(0.12, 0.88, G):
    for gx in np.linspace(0.12, 0.88, G):
        with torch.inference_mode(), ac:
            mm, ss, _ = model.predict_inst(st, point_coords=np.array([[gx * W, gy * H]], float),
                                           point_labels=np.array([1]), multimask_output=True)
        mm = _np(mm); ss = _np(ss).ravel()
        if len(ss) == 0:
            continue
        m = (mm[int(np.argmax(ss))] > 0.5)
        if m.ndim > 2:
            m = m.squeeze()
        a = float(m.mean())
        if m.shape == (H, W) and 0.002 < a < 0.4 and all(np.logical_and(m, q).sum() < 0.7 * min(m.sum(), q.sum()) for q in masks):
            masks.append(m)
rng = np.random.RandomState(0)
for m in masks:
    col = rng.randint(60, 256, 3)
    c, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ev, c, -1, col.tolist(), 2)
cv2.imwrite(f"{outdir}/_segment_everything.png", cv2.cvtColor(ev, cv2.COLOR_RGB2BGR))
print(f"segment-everything: {len(masks)} regions on t={rich_t:.0f}s", flush=True)

# per-object crop montage (labeled name + sam score)
TILE = 220; cols = 4; rowsN = (len(raw) + cols - 1) // cols
mont = np.full((rowsN * (TILE + 26), cols * TILE, 3), 30, np.uint8)
for i, k in enumerate(raw):
    r, c = divmod(i, cols); crop = best[k][1]
    tile = np.full((TILE, TILE, 3), 50, np.uint8)
    if crop is not None and crop.size:
        ch, cw = crop.shape[:2]; s = min(TILE / cw, TILE / ch)
        rs = cv2.resize(crop, (int(cw * s), int(ch * s)))
        tile[:rs.shape[0], :rs.shape[1]] = cv2.cvtColor(rs, cv2.COLOR_RGB2BGR)
    y = r * (TILE + 26); mont[y + TILE:y + TILE + 26, c * TILE:(c + 1) * TILE] = 20
    mont[y:y + TILE, c * TILE:c * TILE + tile.shape[1]] = tile
    cv2.putText(mont, f"{disp[k][:20]} ({sam_score[k]:.2f})", (c * TILE + 3, y + TILE + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(f"{outdir}/_object_montage.png", mont)

# ---- shared base frames for the 2nd pass (first, middle, last of the sample) ----
base_idx = sorted(set([0, len(frames) // 2, len(frames) - 1]))
base_frames = [b64(frames[i][1]) for i in base_idx]
list_plain = "\n".join(f"- {disp[k]}  (seen in {votes[k]}/{len(frames)} frames)" for k in raw)
list_sam = "\n".join(f"- {disp[k]}  (SAM conf {sam_score[k]:.2f}, seen in {votes[k]}/{len(frames)} frames)" for k in raw)


def run_verify(sys_txt, imgs, listing):
    try:
        fr = models.vlm_call("Proposed inventory:\n" + listing + "\n\nReturn the FINAL corrected inventory.",
                             imgs, sys_txt, VERIFY_SCHEMA, max_tokens=900)
        return [o["name"] for o in fr.get("objects", []) if o.get("name")]
    except Exception as e:
        print("verify err", e); return list(disp[k] for k in raw)


cond = {"A_vlm_1pass": [disp[k] for k in raw]}                                   # A: raw union, no verify
cond["B_vlm_2pass"] = run_verify(VERIFY_B_SYS, base_frames, list_plain)           # B: 2nd VLM pass, no SAM
seg_ev = b64(cv2.cvtColor(cv2.imread(f"{outdir}/_segment_everything.png"), cv2.COLOR_BGR2RGB))
mont = b64(cv2.cvtColor(cv2.imread(f"{outdir}/_object_montage.png"), cv2.COLOR_BGR2RGB))
cond["C_vlm_sam_vlm"] = run_verify(VERIFY_C_SYS, base_frames + [seg_ev, mont], list_sam)  # C: with SAM evidence

print("\n=== CONDITIONS ===")
for c, lst in cond.items():
    print(f"  {c} ({len(lst)}): {lst}", flush=True)
json.dump({"window": [t0, t1], "n_frames": len(frames), "model": os.environ["GEMINI_MODEL"],
           "perframe": perframe, "sam_conf": sam_score,
           "raw_inventory": [{"name": disp[k], "votes": votes[k], "sam_conf": sam_score[k]} for k in raw],
           "conditions": cond}, open(f"{outdir}/_inventory_ablation.json", "w"), indent=2)
print("\nwrote", f"{outdir}/_inventory_ablation.json", flush=True)
