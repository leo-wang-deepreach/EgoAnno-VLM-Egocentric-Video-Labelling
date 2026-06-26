#!/usr/bin/env python3
"""build_inventory_grasp.py — OUR geometric grasp-filter inventory for a VIDEO, in the EXACT format
ground_simple.py consumes ({tag, video, objects:[{name, role}]}). Recall-first pool:
  pass-1  Gemini 3.1 pro preview proposes candidate manipulated objects per sampled frame -> union
  SAM3    text-segments each candidate per frame
  filter  keep a candidate iff its mask falls inside the hand's whole-hand grasp region (21-kpt convex hull,
          non-wrist points expanded GRASP_EXPAND outward from the wrist), dilated, in >=1 frame
Handpose runs via a 3.10 subprocess (hand_kpts_cli.py). No 2nd VLM, no Opus.

Run (sam3py): build_inventory_grasp.py <tag> <video> [t0] [t1]
  env: N_FRAMES(12) GRASP_EXPAND(0.2) GRASP_DILATE(1.0) OUTDIR(out/v2_grounded)
"""
import base64, io, json, os, sys, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/ubuntu/local/factsfirst")
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import models  # noqa

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-pro-preview")
PY310 = "/home/ubuntu/local/.venv/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N_FRAMES", "12"))
EXPAND = float(os.environ.get("GRASP_EXPAND", "0.2"))
DIL = float(os.environ.get("GRASP_DILATE", "1.0"))
OUTDIR = os.environ.get("OUTDIR", "out/v2_grounded")
tag, video = sys.argv[1], sys.argv[2]
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else None
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None

INV_SYS_M = (
    "This is the PROPOSAL stage of a multi-stage pipeline. Produce candidate objects that the person may be "
    "manipulating with their hands in this egocentric frame (holding, carrying, moving, using, or operating). "
    "Brainstorm generously: MISSING A REAL OBJECT IS WORSE THAN PROPOSING AN EXTRA ONE — a later stage removes "
    "wrong proposals. Use a short canonical name (colour/material/form when appropriate, <=4 words). Exclude "
    "hands, arms, body parts, clothing, worn items, and the passive supporting surface. List each candidate at "
    "most once. Output strict JSON.")
INV_SCH = {"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"]}}}, "required": ["objects"]}


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def b64(rgb):
    im = Image.fromarray(rgb)
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


def pinch_region(hands, H, W):
    if not hands:
        return None
    reg = np.zeros((H, W), np.uint8); got = False
    for h in hands:
        kp = h.get("kpts") or []
        valid = [(j, float(kp[j][0]), float(kp[j][1])) for j in range(21) if len(kp) > j and kp[j][2] > 0.25]
        if len(valid) < 3:
            continue
        if len(kp) > 0 and kp[0][2] > 0.25:
            ax, ay = float(kp[0][0]), float(kp[0][1])
        else:
            ax = sum(v[1] for v in valid) / len(valid); ay = sum(v[2] for v in valid) / len(valid)
        pts = []
        for j, x, y in valid:
            if EXPAND > 0 and j != 0:
                x = ax + (1.0 + EXPAND) * (x - ax); y = ay + (1.0 + EXPAND) * (y - ay)
            pts.append([int(round(x)), int(round(y))])
        hull = cv2.convexHull(np.array(pts, np.int32))
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, hull, 1)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        reg |= cv2.dilate(m, np.ones((max(25, int(DIL * 0.15 * diag)),) * 2, np.uint8)); got = True
    return reg.astype(bool) if got else None


# ---- 1. sample N frames from the video ----
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
lo = 0 if t0 is None else max(0, int(t0 * fps)); hi = (n - 1) if t1 is None else min(n - 1, int(t1 * fps))
tmpd = tempfile.mkdtemp(); frames, paths = [], []
for k, i in enumerate(np.linspace(lo, hi, N).astype(int)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, bgr = cap.read()
    if not ok:
        continue
    p = os.path.join(tmpd, f"f{k:03d}.jpg"); cv2.imwrite(p, bgr); paths.append(p)
    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
cap.release()
print(f"{tag}: sampled {len(frames)} frames", flush=True)

# ---- 2. handpose (3.10 subprocess) ----
kf = os.path.join(tmpd, "kpts.json")
subprocess.run([PY310, os.path.join(HERE, "hand_kpts_cli.py"), tmpd, kf], check=True)
KP = json.load(open(kf))
frame_kpts = [KP.get(os.path.basename(p), []) for p in paths]

# ---- 3. SAM + Gemini geometric inventory ----
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
ac = torch.autocast("cuda", dtype=torch.bfloat16)


def propose(rgb):
    try:
        r = models.vlm_call("Propose candidate objects the hands may be manipulating in this frame.",
                            [b64(rgb)], INV_SYS_M, INV_SCH, max_tokens=1200)
        seen, out = set(), []
        for o in r.get("objects", []):
            nm = (o.get("name") or "").strip()
            if nm and nm.lower() not in seen:
                seen.add(nm.lower()); out.append(nm)
        return out
    except Exception as e:
        print("  propose err", e); return []


votes, disp = {}, {}
with ThreadPoolExecutor(max_workers=8) as ex:
    for names in ex.map(propose, frames):
        for nm in names:
            votes[nm.lower()] = votes.get(nm.lower(), 0) + 1; disp.setdefault(nm.lower(), nm)
raw = sorted(votes, key=lambda k: -votes[k])
print(f"  pass-1 proposed ({len(raw)}): {[disp[k] for k in raw]}", flush=True)

states, regions = [], []
for rgb, hk in zip(frames, frame_kpts):
    H, W = rgb.shape[:2]
    with torch.inference_mode(), ac:
        states.append(proc.set_image(Image.fromarray(rgb)))
    regions.append(pinch_region(hk, H, W))
any_region = any(r is not None for r in regions)

kept = []
for k in raw:
    hit = False
    for i, (rgb, st) in enumerate(zip(frames, states)):
        if regions[i] is None:
            continue
        h, w = rgb.shape[:2]
        with torch.inference_mode(), ac:
            out = proc.set_text_prompt(state=st, prompt=disp[k])
        sc = _np(out["scores"]).ravel(); mk = _np(out["masks"])
        if mk.ndim == 4:
            mk = mk[:, 0]
        for j in range(len(sc)):
            m = (mk[j] > 0.5)
            if m.ndim > 2:
                m = m.squeeze()
            if m.shape != (h, w) or not (0.0004 < float(m.mean()) < 0.4):
                continue
            if np.logical_and(m, regions[i]).any():
                hit = True; break
        if hit:
            break
    if hit:
        kept.append(disp[k])
final = kept if any_region else [disp[k] for k in raw]   # no hand detected anywhere -> can't filter, keep all

# ---- 4. write inventory.json in ground_simple's format ----
os.makedirs(OUTDIR, exist_ok=True)
objs = [{"name": nm, "role": "manipulable"} for nm in final]
outp = os.path.join(OUTDIR, f"_inventory_{tag}.json")
json.dump({"tag": tag, "video": video, "objects": objs}, open(outp, "w"), indent=2)
print(f"\n{tag} grasp inventory ({len(final)}): {final}\nwrote {outp}", flush=True)
