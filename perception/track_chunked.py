#!/usr/bin/env python3
"""track_chunked.py — DENSE SAM3-video tracking at high fps on a memory-limited box (sam3py).

SAM3's start_session ingests the WHOLE video up front, so a 30fps full clip (~5700 frames) OOMs CPU RAM
before tracking even starts. Fix: cut the clip into short chunks (~CHUNK_SEC) and track each in its OWN
fresh session (~CHUNK_SEC*fps frames loaded), then stitch by global frame index. Model is loaded once;
each chunk gets a fresh session (close + empty_cache between) so GPU/CPU stay bounded.

Run (sam3py): track_chunked.py <tag> <inventory.json> <video> <outdir>
  env: FPS(30), CHUNK_SEC(8), SUB_W(640), REFS_MANIFEST, TRACK_ONLY(csv), DENSE_W(480)
Writes: <outdir>/_dense_<tag>_<slug>.npz  {f<global_frame>: mask(DS), _meta}
"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor


def _union(out, w, h):
    mks = out.get("out_binary_masks")
    if mks is None or len(mks) == 0:
        return None
    u = None
    for m in mks:
        m = np.asarray(m)
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m.astype(bool)
        if m.any():
            u = m if u is None else (u | m)
    return u


def main():
    tag, inv_file, video, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    fps = float(os.environ.get("FPS", "30"))
    chunk_sec = float(os.environ.get("CHUNK_SEC", "8"))
    sub_w = int(os.environ.get("SUB_W", "640"))
    DS = int(os.environ.get("DENSE_W", "480"))
    inv = json.load(open(inv_file))
    names = [o["name"] if isinstance(o, dict) else o for o in inv["objects"]
             if not (isinstance(o, dict) and o.get("role") == "fixture")]
    refman = os.environ.get("REFS_MANIFEST", "")
    if refman and os.path.exists(refman):
        verified = set(json.load(open(refman)).keys()); names = [n for n in names if n in verified]
    only = [s.strip() for s in os.environ.get("TRACK_ONLY", "").split(",") if s.strip()]
    if only:
        names = [n for n in names if n in only]

    # 1) segment the clip into chunk sub-videos (downscaled) — ONE ffmpeg pass
    cdir = os.path.join(outdir, "_chunks"); os.makedirs(cdir, exist_ok=True)
    if not glob.glob(os.path.join(cdir, "c*.mp4")):
        # force a keyframe every chunk_sec so the segment muxer actually cuts at chunk_sec (not the source GOP)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        "-vf", f"fps={fps},scale={sub_w}:-2", "-an",
                        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_sec})", "-f", "segment",
                        "-segment_time", str(chunk_sec), "-reset_timestamps", "1",
                        os.path.join(cdir, "c%03d.mp4")], check=True)
    chunks = sorted(glob.glob(os.path.join(cdir, "c*.mp4")))
    # global frame offset per chunk (actual frame counts)
    offs, tot = [], 0
    sw = sh = 0
    for c in chunks:
        cap = cv2.VideoCapture(c); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        offs.append(tot); tot += n
    print(f"{tag}: {len(chunks)} chunks, {tot} frames @ {fps}fps, sub {sw}x{sh}; names={names}", flush=True)

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    pred = build_sam3_video_predictor()
    dsh = int(round(sh * DS / sw))

    for name in names:
        dense = {}
        for ci, c in enumerate(chunks):
            sid = pred.handle_request(dict(type="start_session", resource_path=c,
                                           offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
            cap = cv2.VideoCapture(c); cn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            seed = None                                    # find first frame in the chunk where the object appears
            for sf in range(0, cn, max(1, cn // 6)):
                pred.handle_request(dict(type="reset_session", session_id=sid))
                r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=sf, text=name))
                if _union(r["outputs"], sw, sh) is not None:
                    seed = sf; break
            if seed is not None:
                for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                        start_frame_index=seed, propagation_direction="both")):
                    u = _union(resp["outputs"], sw, sh)
                    if u is None or not u.any():
                        continue
                    gfi = offs[ci] + resp["frame_index"]
                    dense[gfi] = cv2.resize(u.astype(np.uint8), (DS, dsh), interpolation=cv2.INTER_NEAREST).astype(bool)
            pred.handle_request(dict(type="close_session", session_id=sid)); torch.cuda.empty_cache()
            if ci % 5 == 0:
                fr = torch.cuda.mem_get_info()[0] / 1e9
                print(f"  {name}: chunk {ci+1}/{len(chunks)} done, {len(dense)} dense frames (GPU free {fr:.1f}GB)", flush=True)
        slug = name.replace(" ", "_")
        meta = json.dumps({"name": name, "sfps": fps, "W": sw, "H": sh, "DS": DS, "nfr": tot})
        np.savez_compressed(os.path.join(outdir, f"_dense_{tag}_{slug}.npz"),
                            _meta=np.array(meta), **{f"f{fi}": m for fi, m in dense.items()})
        print(f"{name}: {len(dense)}/{tot} dense frames -> _dense_{tag}_{slug}.npz", flush=True)


if __name__ == "__main__":
    main()
