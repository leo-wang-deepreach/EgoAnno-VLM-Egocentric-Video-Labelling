#!/usr/bin/env python3
"""track_mux.py — SAM 3.1 MULTIPLEX video tracking (sam3py). Follows the shortened inventory through the
whole clip so each object has ONE consistent identity + a mask carried into the hard (transparent / occluded)
grasp frames.

Per the SAM3 docs (facebookresearch/sam3 issue #206) multiple DISTINCT text categories CANNOT share one
session — adding "cup" then "scoop" breaks tracking. So we track ONE category per session (reset between),
on the faster 3.1 MULTIPLEX predictor which tracks ALL instances of that one category JOINTLY via shared
memory (so a rack of look-alike jars is handled in a single pass). For each verified inventory name: find its
clearest seed frame, text-seed there, propagate BOTH directions, and save the UNION of its instance masks at
every grasp time. The grounder's _clean then localizes that union to the at-grasp instance.

Run (sam3py): track_mux.py <tag> <inventory.json> <prompts.json> <outdir>
  env: REFS_MANIFEST (verified vocab json), TRACK_FPS (3), TRACK_ONLY (csv), USE_FA3 (0), N_PROBE (16)
Writes: <outdir>/<tag>_t<t>.npz  {object_name: bool mask}
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_multiplex_video_predictor


def _union_at(out, W, H):
    """Union of ALL instance masks in a multiplex output (every obj_id of the one prompted category)."""
    mks = out.get("out_binary_masks")
    if mks is None or len(mks) == 0:
        return None
    u = None
    for m in mks:
        m = np.asarray(m)
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (H, W):
            if not m.any():
                continue
            m = cv2.resize(m.astype(np.uint8), (W, H)).astype(bool)
        m = m.astype(bool)
        if m.any():
            u = m if u is None else (u | m)
    return u


def main():
    tag, inv_file, prompts_file, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    inv = json.load(open(inv_file))
    names = [o["name"] if isinstance(o, dict) else o for o in inv["objects"]
             if not (isinstance(o, dict) and o.get("role") == "fixture")]
    refman = os.environ.get("REFS_MANIFEST", "")
    if refman and os.path.exists(refman):
        verified = set(json.load(open(refman)).keys())
        names = [n for n in names if n in verified]
    only = [s.strip() for s in os.environ.get("TRACK_ONLY", "").split(",") if s.strip()]
    if only:
        names = [n for n in names if n in only]
    pj = json.load(open(prompts_file)); video = pj["video"]; W, H = pj["W"], pj["H"]
    grasp_times = sorted({round(p["t"], 3) for p in pj["prompts"]})
    fps_t = float(os.environ.get("TRACK_FPS", "3"))
    n_probe = int(os.environ.get("N_PROBE", "16"))
    dense_on = bool(os.environ.get("DENSE", ""))           # also save every propagated frame (downscaled) for video
    DS = int(os.environ.get("DENSE_W", "480"))             # dense mask width
    window_sec = float(os.environ.get("WINDOW_SEC", "0"))  # >0 -> windowed propagation (bounds GPU memory at high fps)
    sub_w = int(os.environ.get("SUB_W", "0"))              # >0 -> DOWNSCALE the tracking sub (the session loads ALL
    #   frames into CPU RAM; at 30fps full-res that's ~35GB -> OOM-kill. Masks are upscaled back, so a small sub is
    #   fine for tracking — the grasp-time grounding uses the separate full-res 1fps tracks.)
    sub = os.path.join(outdir, f"_sub_{tag}.mp4")
    if not os.path.exists(sub):
        vf = f"fps={fps_t}" + (f",scale={sub_w}:-2" if sub_w > 0 else "")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", vf, "-an", sub], check=True)
    cap = cv2.VideoCapture(sub); sfps = cap.get(cv2.CAP_PROP_FPS) or fps_t
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    gframes = {min(nfr - 1, max(0, round(t * sfps))): t for t in grasp_times}
    print(f"{tag}: {len(names)} categories, {nfr} frames @ {sfps:.1f}fps, {len(grasp_times)} grasp times", flush=True)

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    ver = os.environ.get("SAM_VER", "3.0")
    if ver == "3.1":
        # SAM 3.1 MULTIPLEX: faster + tracks instances jointly, BUT the checkpoint fixes multiplex_count=16 and
        # (without FlashAttention-3 installed) full-video propagation OOMs a 22 GB L4. Needs FA3 or a bigger GPU.
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        pred = build_sam3_multiplex_video_predictor(use_fa3=os.environ.get("USE_FA3", "0") != "0", compile=False)
        _orig = type(pred.model).init_state           # base start_session injects offload_state_to_cpu; 3.1 rejects
        def _shim(self, *a, **k):
            k.pop("offload_state_to_cpu", None); return _orig(self, *a, **k)
        type(pred.model).init_state = _shim
        ss = dict(type="start_session", resource_path=sub, offload_video_to_cpu=True)
    else:
        # SAM 3.0 video predictor — FITS the L4 via offload_state_to_cpu (the 3.1 multiplex build can't offload it).
        from sam3.model_builder import build_sam3_video_predictor
        pred = build_sam3_video_predictor()
        ss = dict(type="start_session", resource_path=sub, offload_video_to_cpu=True, offload_state_to_cpu=True)
    sid = pred.handle_request(ss)["session_id"]
    by_time = {t: {} for t in grasp_times}
    dsh = int(round(H * DS / W))

    def _collect(resp, dense, name):
        fi = resp["frame_index"]; u = _union_at(resp["outputs"], W, H)
        if u is None or not u.any():
            return
        if fi in gframes:
            by_time[gframes[fi]][name] = u                 # grasp-time tracks (for the grounder)
        if dense_on:
            dense[fi] = cv2.resize(u.astype(np.uint8), (DS, dsh), interpolation=cv2.INTER_NEAREST).astype(bool)

    for name in names:
        dense = {}                                         # frame_index -> downscaled mask (for video render)
        if window_sec > 0:
            # WINDOWED: re-seed + propagate within bounded ~window_sec spans so the GPU memory bank stays small
            # even at high fps. Re-detection each window (text-seg) — fine for a per-frame mask; identity carried
            # by the name. Seed at the window start; if the object isn't there yet, scan forward for first hit.
            wf = max(2, int(window_sec * sfps)); seeded_any = False
            print(f"  [win] {name}: {len(range(0, nfr, wf))} windows of {wf} frames", flush=True)
            for ws in range(0, nfr, wf):
                seed = None
                for cand in range(ws, min(nfr, ws + wf), max(1, wf // 6)):
                    pred.handle_request(dict(type="reset_session", session_id=sid))
                    r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=cand, text=name))
                    if _union_at(r["outputs"], W, H) is not None:
                        seed = cand; break
                if seed is None:
                    print(f"  [win] ws={ws}: no seed", flush=True); continue
                seeded_any = True
                got = 0
                for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                        start_frame_index=seed, max_frame_num_to_track=ws + wf - seed,
                        propagation_direction="forward")):
                    if resp["frame_index"] >= ws + wf:
                        break
                    _collect(resp, dense, name); got += 1
                print(f"  [win] ws={ws} seed={seed} -> {got} frames (dense={len(dense)})", flush=True)
            if not seeded_any:
                print(f"  {name}: never segmented — skip", flush=True); continue
            best_sf = "windowed"
        else:
            pred.handle_request(dict(type="reset_session", session_id=sid))
            best_sf, best_a = None, 0                          # PHASE A: clearest seed = largest text mask
            for sf in range(0, nfr, max(1, nfr // n_probe)):
                r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=sf, text=name))
                u = _union_at(r["outputs"], W, H); a = int(u.sum()) if u is not None else 0
                if a > best_a:
                    best_sf, best_a = sf, a
                pred.handle_request(dict(type="reset_session", session_id=sid))
            if best_sf is None:
                print(f"  {name}: never segmented — skip", flush=True); continue
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=best_sf, text=name))
            for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                _collect(resp, dense, name)
        cov = sum(1 for d in by_time.values() if name in d)
        print(f"  {name}: seed={best_sf} -> tracked at {cov}/{len(grasp_times)} grasp times "
              f"({len(dense)} dense)", flush=True)
        if dense_on and dense:
            slug = name.replace(" ", "_")
            meta = json.dumps({"name": name, "sfps": sfps, "W": W, "H": H, "DS": DS, "nfr": nfr})
            np.savez_compressed(os.path.join(outdir, f"_dense_{tag}_{slug}.npz"),
                                _meta=np.array(meta), **{f"f{fi}": m for fi, m in dense.items()})
    pred.handle_request(dict(type="close_session", session_id=sid))

    saved = 0
    for t, d in by_time.items():
        if not d:
            continue
        path = os.path.join(outdir, f"{tag}_t{t:05.1f}.npz")
        if os.path.exists(path):                               # MERGE: accumulate categories across the
            z = np.load(path, allow_pickle=True)               # per-category subprocess runs (one process per
            merged = {k: z[k] for k in z.files}                # category so the OS frees the ~20 GB GPU state
            merged.update(d); d = merged                       # between categories — reset_session does NOT)
        np.savez_compressed(path, **d); saved += 1
    counts = {n: sum(1 for d in by_time.values() if n in d) for n in names}
    print(f"{tag}: tracked {len(names)} categories (SAM {ver}) -> tracks at {saved}/{len(grasp_times)} "
          f"grasp times -> {outdir}\n  per-object coverage: {counts}")


if __name__ == "__main__":
    main()
