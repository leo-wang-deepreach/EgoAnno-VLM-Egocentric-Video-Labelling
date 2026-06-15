#!/usr/bin/env python3
"""to_viewer.py — publish a facts-first episode into the existing egoanno viewer.

The viewer's manifest builder pairs each episode JSON with a source video by the
JSON filename stem, and renders `subtasks` + per-hand lanes. This adapter takes a
facts-first episode (old or new export shape) and writes a legacy-shaped JSON named
<video_stem>.json into out/episodes_factsfirst/, then rebuilds the manifest.

  python to_viewer.py out/76a67a82_v2.json [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EGO = Path(__file__).resolve().parent.parent           # egoanno root
OUTDIR = EGO / "out" / "episodes_factsfirst"
VIDEO_DIRS = ["videos", "not_for_testing_videos", "not_for_testing_videos_2",
              "not_for_testing_videos_3", "not_for_testing_videos_4"]


def find_video(stem: str) -> str | None:
    """Prefer the CLOCKED video the pipeline actually saw (burned-in µs clock) so the
    viewer's timeline lines up with the timestamps; copy it to a served path. Fall
    back to the 720p proxy, then the raw source."""
    # 1. clocked video from the most recent run workdir (workdirs are named by the
    # short clip id, e.g. logs/76a67a82_v4/clocked.mp4)
    short = stem.split("-")[0]
    cand = sorted(list((EGO / "factsfirst" / "logs").glob(f"{short}*/clocked.mp4"))
                  + list((EGO / "factsfirst" / "logs").glob(f"*{stem}*/clocked.mp4")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        dst = OUTDIR / f"{stem}.clocked.mp4"
        if (not dst.exists()) or dst.stat().st_mtime < cand[0].stat().st_mtime:
            shutil.copy2(cand[0], dst)
        return f"/out/episodes_factsfirst/{stem}.clocked.mp4"
    # 2. proxy / source
    for d in VIDEO_DIRS:
        if (EGO / d / f"{stem}.mp4").exists():
            px = EGO / "out" / "proxies" / f"{d}_{stem}.mp4"
            return f"/out/proxies/{d}_{stem}.mp4" if px.exists() else f"/{d}/{stem}.mp4"
    return None


def write_factsfirst_manifest():
    """Scan out/episodes_factsfirst and write viewer/factsfirst_manifest.json so the
    dashboard can list clips. Paths are root-absolute (served from the egoanno root)."""
    clips = []
    for ep_path in sorted(OUTDIR.glob("*.json")):
        try:
            ep = json.loads(ep_path.read_text())
        except Exception:
            continue
        stem = ep_path.stem
        clips.append({"clip": stem.split("-")[0] + "…",
                      "stem": stem,
                      "episode": f"/out/episodes_factsfirst/{stem}.json",
                      "video": find_video(stem) or ""})
    out = EGO / "viewer" / "factsfirst_manifest.json"
    out.write_text(json.dumps({"clips": clips}, ensure_ascii=False, indent=2))
    print(f"factsfirst manifest -> {out} ({len(clips)} clips)")


def _lane(segs, hand):
    """Per-hand lane in the format build_viewer_manifest expects
    ({start_sec, end_sec, action, needs_review}); it converts to s/e/a/rv itself."""
    lane = []
    for s in segs:
        a = s.get(hand) or "N/A"
        if lane and lane[-1]["action"].strip().lower() == a.strip().lower():
            lane[-1]["end_sec"] = round(s["end_sec"], 2)
        else:
            lane.append({"start_sec": round(s["start_sec"], 2),
                         "end_sec": round(s["end_sec"], 2),
                         "action": a, "needs_review": False})
    return lane


def to_legacy(ep: dict) -> dict:
    segs = ep["segments"]
    qa = ep.get("_qa")
    qa_list = qa.get("violations", []) if isinstance(qa, dict) else (qa or [])
    flagged = {f["seg"] - 1 for f in ep.get("_flags", [])}
    qa_segs = {v["seg"] - 1 for v in qa_list if v.get("seg")}
    subtasks = [{"start_sec": s["start_sec"], "end_sec": s["end_sec"],
                 "left": s.get("left", ""), "right": s.get("right", ""),
                 "needs_review": (i in flagged or i in qa_segs)}
                for i, s in enumerate(segs)]
    dur = ep.get("duration_sec") or (ep.get("meta") or {}).get("duration_sec")
    return {
        **ep,
        "instruction": ep.get("goal", ""),
        "environment": ep.get("environment") or {"category": ""},
        "meta": ep.get("meta") or {"duration_sec": dur,
                                   "model": "facts-first (gemini+gpt+claude)"},
        "subtasks": subtasks,
        "left_timeline": _lane(segs, "left"),
        "right_timeline": _lane(segs, "right"),
        "_qa": {"violations": qa_list},
        "_trace": {},                          # manifest expects a dict here
        "_seg_trace": ep.get("_trace") if isinstance(ep.get("_trace"), list)
                      else ep.get("_seg_trace", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--rebuild", action="store_true", default=True)
    args = ap.parse_args()
    ep = json.loads(Path(args.episode).read_text())
    stem = ep.get("clip") or Path(args.episode).stem
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / f"{stem}.json"
    dst.write_text(json.dumps(to_legacy(ep), ensure_ascii=False, indent=2))
    print(f"published -> {dst}")
    write_factsfirst_manifest()                  # for the dedicated facts-first dashboard
    if args.rebuild:
        r = subprocess.run([sys.executable, "build_viewer_manifest.py"], cwd=EGO,
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip()[:400])


if __name__ == "__main__":
    main()
