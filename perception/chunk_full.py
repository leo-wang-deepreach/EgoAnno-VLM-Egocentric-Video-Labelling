#!/usr/bin/env python3
"""chunk_full.py — run the WHOLE object-manipulation pipeline on ONE chunk and render it segmented:
  (1) DENSE-track the SAM3-verified objects on the chunk,
  (2) use the grounding's per-segment MANIPULATED-object identity (which object each hand holds),
  (3) draw each hand's identified object mask (one circle when both hands share it; nothing when N/A).
Lets us validate the full flow on a single chunk before scaling to all chunks.

Run (sam3py): chunk_full.py <chunk.mp4> <chunk_start_sec> <grasp.json> <grounding_index.json> <refs.json> <inventory.json> <out.mp4>
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/local/factsfirst/perception")
import perception  # noqa: E402
from sam3.model_builder import build_sam3_video_predictor  # noqa: E402

chunk, cstart, grasp_f, idx_f, refs_f, inv_f, out_mp4 = (
    sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
GREEN, BLUE, YELLOW = (80, 230, 80), (90, 160, 255), (70, 220, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

verified = set(json.load(open(refs_f)).keys())
inv = json.load(open(inv_f))
names = [o["name"] for o in inv["objects"] if o.get("role") != "fixture" and o["name"] in verified]
_only = [s.strip() for s in os.environ.get("TRACK_OBJS", "").split(",") if s.strip()]
if _only:                                                  # track ONLY the objects this chunk actually needs
    names = [n for n in names if n in _only]
cap = cv2.VideoCapture(chunk); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30
cn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
print(f"chunk {chunk} ({cn}f {W}x{H} @{fps:.0f}fps), start {cstart}s; tracking {names}", flush=True)


def union(out):
    mks = out.get("out_binary_masks")
    if mks is None or len(mks) == 0:
        return None
    u = None
    for m in mks:
        m = np.asarray(m)
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m.astype(bool)
        if m.any():
            u = m if u is None else (u | m)
    return u


# (1) dense-track each verified object on the chunk (cache so a render failure never re-tracks)
import pickle  # noqa: E402
TCACHE = os.environ.get("TRACK_CACHE", "")
if TCACHE and os.path.exists(TCACHE):
    tracks = pickle.load(open(TCACHE, "rb"))
    print(f"  loaded tracks from cache: {[ (k, len(v)) for k, v in tracks.items() ]}", flush=True)
else:
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    pred = build_sam3_video_predictor()
    sid = pred.handle_request(dict(type="start_session", resource_path=chunk,
                                   offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
    tracks = {}
    for nm in names:
        seed = None
        for sf in range(0, cn, max(1, cn // 8)):
            pred.handle_request(dict(type="reset_session", session_id=sid))
            r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=sf, text=nm))
            if union(r["outputs"]) is not None:
                seed = sf; break
        d = {}
        if seed is not None:
            for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                    start_frame_index=seed, propagation_direction="both")):
                u = union(resp["outputs"])
                if u is not None and u.any():
                    d[resp["frame_index"]] = u
        tracks[nm] = d
        print(f"  tracked {nm}: {len(d)}/{cn} frames", flush=True)
    pred.handle_request(dict(type="close_session", session_id=sid)); torch.cuda.empty_cache()
    if TCACHE:
        pickle.dump(tracks, open(TCACHE, "wb")); print(f"  cached tracks -> {TCACHE}", flush=True)

# (2) grounding identity: per segment (global span) -> L/R manipulated object
g = json.load(open(grasp_f)); segs = {}
for p in g["prompts"]:
    if p.get("primary", True):
        segs.setdefault(p["seg_id"][:-1], {"span": p["seg"], "L": None, "R": None})
for r in json.load(open(idx_f)):
    num, h = r["seg_id"][:-1], r["seg_id"][-1]
    if num in segs:
        segs[num][h] = r["name"] if r["obj"] else None
seglist = sorted(segs.values(), key=lambda s: s["span"][0])

# per-frame hand anchors on the chunk — handpose (ultralytics) lives in the 3.10 venv, NOT sam3py, so
# load a precomputed pickle when running under sam3py; fall back to extract if available.
hp = os.environ.get("HANDPOSE_PKL", "")
if hp and os.path.exists(hp):
    sig = pickle.load(open(hp, "rb"))
else:
    sig = perception.extract(chunk, fps_target=fps)
T = sig["t"]
cap = cv2.VideoCapture(chunk); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()


def nearest_blob(mask, ax, ay):
    m = mask.astype(np.uint8); n, lab, _, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return mask
    ai = min(max(int(ay), 0), m.shape[0] - 1); aj = min(max(int(ax), 0), m.shape[1] - 1); sl = int(lab[ai, aj])
    if sl == 0:
        sl = min(range(1, n), key=lambda i: (cents[i][0] - ax) ** 2 + (cents[i][1] - ay) ** 2)
    return lab == sl


def short(n):
    return n.split(" with ")[0] if n else n


def draw(img, mask, col, lbl):
    cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return
    f = img.astype(np.float32); f[mask] = 0.55 * f[mask] + 0.45 * np.array(col, np.float32)
    img[:] = f.clip(0, 255).astype(np.uint8)
    cv2.drawContours(img, cont, -1, col, 2)
    ys, xs = np.nonzero(mask); cx, cy = int(xs.mean()), int(ys.mean())
    r = int(0.6 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 8
    cv2.circle(img, (cx, cy), r, col, 2)
    x, y = cx, max(16, cy - r - 6); (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.5, 1)
    x = min(max(2, x - tw // 2), img.shape[1] - tw - 2)
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(img, lbl, (x, y), FONT, 0.5, col, 1, cv2.LINE_AA)


def active(tg):
    return next((s for s in seglist if s["span"][0] <= tg <= s["span"][1]), None)


def omask(nm, fi):
    d = tracks.get(nm)
    if not d:
        return None
    k = min(d, key=lambda kk: abs(kk - fi))
    return d[k] if abs(k - fi) <= 3 else None


# (3) render the chunk segmented
outd = tempfile.mkdtemp(); drawn = 0
for i, f in enumerate(frames):
    if i >= len(T):
        break
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); tl = float(T[i]); tg = cstart + tl; s = active(tg)
    if s:
        gl = sig["grasp"]["L"][i]; gr = sig["grasp"]["R"][i]
        la = (gl[0] * W, gl[1] * H) if np.isfinite(gl).all() else None
        ra = (gr[0] * W, gr[1] * H) if np.isfinite(gr).all() else None
        ml = nearest_blob(omask(s["L"], i), *la) if (s["L"] and la and omask(s["L"], i) is not None) else None
        mr = nearest_blob(omask(s["R"], i), *ra) if (s["R"] and ra and omask(s["R"], i) is not None) else None
        same = (s["L"] and s["L"] == s["R"] and ml is not None and mr is not None
                and np.logical_and(ml, mr).sum() > 0.2 * min(ml.sum(), mr.sum()))
        if same:
            draw(img, np.logical_or(ml, mr), YELLOW, short(s["L"])); drawn += 1
        else:
            if ml is not None:
                draw(img, ml, GREEN, "L: " + short(s["L"])); drawn += 1
            if mr is not None:
                draw(img, mr, BLUE, "R: " + short(s["R"])); drawn += 1
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"t={tg:5.1f}s", (8, 17), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outd, f"o{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"rendered {min(len(frames), len(T))} frames, {drawn} mask draws", flush=True)
enc = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
       "-i", os.path.join(outd, "o%04d.png"), "-pix_fmt", "yuv420p"]
try:
    subprocess.run(enc + ["-c:v", "h264_nvenc", out_mp4], check=True)
except subprocess.CalledProcessError:
    subprocess.run(enc + ["-c:v", "libx264", out_mp4], check=True)
print("wrote", out_mp4, flush=True)
