#!/usr/bin/env python3
"""to_viewer.py — publish a facts-first episode into the egoanno viewer, VERSION-AWARE.

Each episode is published as `<clip>__<ver>.json` so multiple versions of the same clip
coexist and the dashboard can compare them (v18 vs v19 vs v20…). The version is inferred
from the episode's parent dir (out/v20 -> "v20", out/batch -> "v18b") or an explicit
--ver. The exact clocked video the run saw (ep["_clocked"]) is copied to a served path.

  python to_viewer.py out/v20/76a67a82.json            # ver inferred = "v20"
  python to_viewer.py out/batch/X.json --ver v18b
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

EGO = Path(__file__).resolve().parent.parent           # egoanno root
OUTDIR = EGO / "out" / "episodes_factsfirst"
VIDEO_DIRS = ["videos", "not_for_testing_videos", "not_for_testing_videos_2",
              "not_for_testing_videos_3", "not_for_testing_videos_4"]


def _infer_ver(ep_path: Path) -> str:
    parent = ep_path.parent.name
    if parent == "batch":
        return "v18b"
    if re.fullmatch(r"v\d+\w*", parent):               # out/v19, out/v20
        return parent
    m = re.search(r"_(v\d+\w*)$", ep_path.stem)         # out/76a67a82_v17.json
    if m:
        return m.group(1)
    return "v?"


def _copy_clocked(key: str, clocked_src: str | None) -> str | None:
    if clocked_src and Path(clocked_src).exists():
        dst = OUTDIR / f"{key}.clocked.mp4"
        if (not dst.exists()) or dst.stat().st_mtime < Path(clocked_src).stat().st_mtime:
            shutil.copy2(clocked_src, dst)
        return f"/out/episodes_factsfirst/{key}.clocked.mp4"
    return None


def _video_for(key: str, stem: str) -> str:
    """Served video path for a published key: prefer the copied per-version clocked
    video, else any clocked workdir for this clip, else proxy/source."""
    dst = OUTDIR / f"{key}.clocked.mp4"
    if dst.exists():
        return f"/out/episodes_factsfirst/{key}.clocked.mp4"
    short = stem.split("-")[0]
    cand = sorted((EGO / "factsfirst" / "logs").glob(f"{short}*/clocked.mp4"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        shutil.copy2(cand[0], dst)
        return f"/out/episodes_factsfirst/{key}.clocked.mp4"
    for d in VIDEO_DIRS:
        if (EGO / d / f"{stem}.mp4").exists():
            return f"/{d}/{stem}.mp4"
    return ""


def write_factsfirst_manifest():
    """Scan out/episodes_factsfirst and write viewer/factsfirst_manifest.json grouped by
    clip, each with its versions sorted, so the dashboard offers a version selector."""
    groups: dict[str, list] = {}
    for ep_path in sorted(OUTDIR.glob("*.json")):
        key = ep_path.stem
        if "__" in key:
            stem, ver = key.rsplit("__", 1)
        else:
            stem, ver = key, "v?"
        groups.setdefault(stem, []).append({
            "ver": ver,
            "episode": f"/out/episodes_factsfirst/{key}.json",
            "video": _video_for(key, stem)})
    clips = []
    for stem, vers in sorted(groups.items()):
        vers.sort(key=lambda v: v["ver"])
        clips.append({"clip": stem.split("-")[0] + "…", "stem": stem, "versions": vers})
    out = EGO / "viewer" / "factsfirst_manifest.json"
    out.write_text(json.dumps({"clips": clips}, ensure_ascii=False, indent=2))
    nver = sum(len(c["versions"]) for c in clips)
    print(f"factsfirst manifest -> {out} ({len(clips)} clips, {nver} versions)")


def _lane(segs, hand):
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
                                   "model": "facts-first"},
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
    ap.add_argument("--ver", default="")
    ap.add_argument("--rebuild", action="store_true", default=True)
    args = ap.parse_args()
    ep_path = Path(args.episode)
    ep = json.loads(ep_path.read_text())
    stem = ep.get("clip") or ep_path.stem
    ver = args.ver or _infer_ver(ep_path)
    key = f"{stem}__{ver}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _copy_clocked(key, ep.get("_clocked"))
    dst = OUTDIR / f"{key}.json"
    dst.write_text(json.dumps(to_legacy(ep), ensure_ascii=False, indent=2))
    print(f"published -> {dst} (ver {ver})")
    write_factsfirst_manifest()
    if args.rebuild:
        r = subprocess.run([sys.executable, "build_viewer_manifest.py"], cwd=EGO,
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip()[:400])


if __name__ == "__main__":
    main()
