#!/usr/bin/env python3
"""er16_pipeline.py — ONE-PASS egocentric segment+annotate with Gemini Robotics-ER 1.6.

A clean third path alongside pipeline.py (VLM-heavy) and perception/pipeline2.py
(measured spine). The contract here is strict and simple:

    EXACTLY ONE model inference — a single generateContent to
    gemini-robotics-er-1.6-preview that BOTH segments the clip and names each
    hand's atomic action (and the goal + direction) in one shot.

Everything else is deterministic, non-model code:
    burn_clock (timestamps + audio strip)  ->  [1x ER 1.6 call]  ->  contract
    validation (contiguity/overlap/gap/tiny)  ->  episode.json  ->  :8800 viewer

The single pass is provable: every model call goes through _one_pass() which
increments VLM_CALLS; the run asserts VLM_CALLS == 1 before it exports.

Usage:
    .venv/bin/python er16_pipeline.py <video> [--out out/er01/<clip>.json]
        [--fps 2.0] [--ver er01] [--workdir DIR] [--no-viewer]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import GeminiVideo, USAGE                       # noqa: E402
from media import burn_clock, probe_duration               # noqa: E402
from clipstate import ClipState, Segment, derive_track_from_labels  # noqa: E402
import qa as QA                                             # noqa: E402

MODEL = "gemini-robotics-er-1.6-preview"
# media resolution = how many tokens Gemini spends PER FRAME (detail), NOT video pixels.
# "default" keeps original frame detail (~258 tok/frame); "low" compresses (~79) to fit
# more video in the window. Default here is ORIGINAL detail.
MEDIA_RES_MAP = {"default": None, "low": "MEDIA_RESOLUTION_LOW",
                 "medium": "MEDIA_RESOLUTION_MEDIUM", "high": "MEDIA_RESOLUTION_HIGH"}
TOK_PER_FRAME = {"default": 258, "low": 79, "medium": 130, "high": 258}  # for the budget guard
SAFE_VIDEO_BUDGET = 115_000                 # leave ~16K of the 131K window for prompt + output

# --- the ONE-PASS schema: goal + direction + the per-hand timeline, all in one object.
# (no `think` field — ER 1.6 keeps its reasoning in hidden thought tokens, so a visible
# think string comes back empty and only wastes output budget.)
ONEPASS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "goal": {"type": "STRING"},
        "direction": {"type": "STRING",
                      "enum": ["assembly", "disassembly", "transfer_or_portioning",
                               "mixed_or_alternating", "other_or_ambiguous"]},
        "segments": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "start_sec": {"type": "NUMBER"},
                "end_sec": {"type": "NUMBER"},
                "left": {"type": "STRING"},
                "right": {"type": "STRING"}},
            "required": ["start_sec", "end_sec", "left", "right"]}},
    },
    "required": ["goal", "direction", "segments"],
}

SYSTEM = (
    "You are an expert annotator of first-person (egocentric, head-mounted camera) "
    "two-handed manipulation videos for robot learning. From the camera wearer's view, "
    "the LEFT hand enters from the left side of the frame and the RIGHT hand from the "
    "right. A yellow timestamp in seconds is burned into the top-right of every frame; "
    "read it to set precise boundaries. In ONE pass over the whole clip you produce an "
    "atomic per-hand action timeline plus the overall goal and direction."
)

PROMPT = (
    "Watch the entire clip once and segment it into contiguous, non-overlapping time "
    "spans, cutting a new segment whenever EITHER hand starts a new atomic action "
    "(grasp, move, place, turn, press, hand-off, release). For each segment give "
    "start_sec and end_sec (use the burned-in clock for exact seconds) and, for EACH "
    "hand, a short verb-object phrase for that hand's atomic action during the span — "
    "e.g. 'pick up <part>', 'hold <part> steady', 'turn <tool> clockwise', "
    "'place <part> on <surface>', 'press <part> into <counterpart>'. Use exactly 'N/A' "
    "for a hand that is idle, resting, or out of frame. The segments together must cover "
    "the whole clip with no gaps and no overlaps.\n"
    "Also decide the overall 'goal' (one short phrase) and 'direction': 'assembly' if "
    "parts are being joined, 'disassembly' if a unit is being taken apart, "
    "'transfer_or_portioning' for moving/dividing material, 'mixed_or_alternating', or "
    "'other_or_ambiguous'. Return ONLY the JSON object."
)

# ----- one-pass guard: every model call goes through here ------------------- #
VLM_CALLS = 0


def _one_pass(gv: GeminiVideo, fps: float, media_res: str | None) -> dict:
    """The single, only model inference of the whole pipeline."""
    global VLM_CALLS
    VLM_CALLS += 1
    return gv.watch(PROMPT, SYSTEM, ONEPASS_SCHEMA,
                    fps=fps, media_resolution=media_res,
                    max_tokens=60_000,        # generous so a truncation-resend is never needed
                    temperature=0.2, retries=3)


# ----- deterministic contract validation (NO model) ------------------------- #
def _na(s) -> bool:
    return str(s or "N/A").strip().upper() in ("N/A", "NA", "NONE", "")


def clean_timeline(raw: list[dict], duration: float,
                   min_dur: float = 0.5, eps: float = 0.05) -> list[dict]:
    """Turn the model's raw segments into a strictly valid timeline: in-bounds,
    sorted, no overlaps, no gaps (bridged with N/A), adjacent-identical merged,
    sub-min_dur slivers absorbed. Pure code — this is the structural guarantee the
    single VLM pass cannot make for itself."""
    segs = []
    for s in raw or []:
        try:
            a, b = float(s.get("start_sec")), float(s.get("end_sec"))
        except (TypeError, ValueError):
            continue
        a, b = max(0.0, min(a, duration)), max(0.0, min(b, duration))
        if b - a <= eps:
            continue
        segs.append({"start_sec": a, "end_sec": b,
                     "left": str(s.get("left") or "N/A").strip() or "N/A",
                     "right": str(s.get("right") or "N/A").strip() or "N/A"})
    segs.sort(key=lambda s: s["start_sec"])

    # de-overlap: push each segment's start up to the previous end; drop if it collapses
    fixed = []
    for s in segs:
        if fixed and s["start_sec"] < fixed[-1]["end_sec"]:
            s["start_sec"] = fixed[-1]["end_sec"]
        if s["end_sec"] - s["start_sec"] > eps:
            fixed.append(s)
    segs = fixed

    # bridge gaps (incl. leading/trailing) with N/A so coverage is total
    bridged, prev_end = [], 0.0
    for s in segs:
        if s["start_sec"] - prev_end > eps:
            bridged.append({"start_sec": prev_end, "end_sec": s["start_sec"],
                            "left": "N/A", "right": "N/A"})
        bridged.append(s)
        prev_end = s["end_sec"]
    if duration - prev_end > eps:
        bridged.append({"start_sec": prev_end, "end_sec": duration,
                        "left": "N/A", "right": "N/A"})
    segs = bridged or [{"start_sec": 0.0, "end_sec": duration,
                        "left": "N/A", "right": "N/A"}]

    segs = _merge_identical(segs)

    # absorb sub-min_dur slivers into a neighbour (keeps contiguity), then re-merge
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s["end_sec"] - s["start_sec"] >= min_dur:
                continue
            if i > 0:
                segs[i - 1]["end_sec"] = s["end_sec"]
            else:
                segs[i + 1]["start_sec"] = s["start_sec"]
            segs.pop(i)
            changed = True
            break
    segs = _merge_identical(segs)

    for s in segs:                                  # round last, then snap contiguity
        s["start_sec"], s["end_sec"] = round(s["start_sec"], 1), round(s["end_sec"], 1)
    for i in range(1, len(segs)):
        segs[i]["start_sec"] = segs[i - 1]["end_sec"]
    return segs


def _merge_identical(segs: list[dict]) -> list[dict]:
    out = []
    for s in segs:
        same = (out and _na(out[-1]["left"]) == _na(s["left"])
                and out[-1]["left"].strip().lower() == s["left"].strip().lower()
                and out[-1]["right"].strip().lower() == s["right"].strip().lower())
        if same:
            out[-1]["end_sec"] = s["end_sec"]
        else:
            out.append(dict(s))
    return out


# ----- orchestration -------------------------------------------------------- #
def run(video: str, out_path: str, ver: str, fps: float, res: str,
        workdir: str | None, publish: bool, overlay: bool = True) -> dict:
    video = str(Path(video).resolve())
    media_res = MEDIA_RES_MAP[res]
    dur = probe_duration(video)
    wd = Path(workdir or f"logs/{Path(video).stem}_{ver}")
    wd.mkdir(parents=True, exist_ok=True)
    print(f"[er16] clip={Path(video).name}  dur={dur:.1f}s  fps={fps}  "
          f"res={res} ({'original detail' if media_res is None else media_res})")

    # budget guard (user chose fixed fps -> warn but proceed)
    est = TOK_PER_FRAME[res] * fps * dur
    print(f"[er16] est video tokens ~= {est:,.0f} / {SAFE_VIDEO_BUDGET:,} budget"
          f"  (~{TOK_PER_FRAME[res]} tok/frame)")
    if est > SAFE_VIDEO_BUDGET:
        rec = SAFE_VIDEO_BUDGET / (TOK_PER_FRAME[res] * dur)
        print(f"[er16] ⚠ clip likely OVERFLOWS the 131K window at fps={fps}, res={res}. "
              f"Recommend --fps {rec:.2f} (or --media-res low) for a clean single pass. "
              f"Proceeding anyway.")

    # 1) deterministic preprocess: burn clock + strip audio -> the one video the model sees
    clocked = str(wd / "clocked.mp4")
    t0 = time.time()
    burn_clock(video, clocked)
    print(f"[er16] clock-burned -> {clocked}  ({time.time()-t0:.0f}s)")

    # 2) THE ONE PASS
    gv = GeminiVideo(model=MODEL)
    uri = gv.upload(clocked)
    print(f"[er16] uploaded -> {uri}")
    t1 = time.time()
    result = _one_pass(gv, fps, media_res)
    print(f"[er16] ER 1.6 one-pass done ({time.time()-t1:.0f}s)  VLM_CALLS={VLM_CALLS}")
    assert VLM_CALLS == 1, f"one-pass violated: {VLM_CALLS} model calls"

    # 3) deterministic contract validation
    segs = clean_timeline(result.get("segments", []), dur)
    cover = sum(s["end_sec"] - s["start_sec"] for s in segs)
    print(f"[er16] {len(segs)} segments after contract; coverage {cover:.1f}/{dur:.1f}s")

    # 4) assemble ClipState + export episode.json
    state = ClipState(video=video)
    state.clocked = clocked
    state.duration = dur
    state.goal = str(result.get("goal", "")).strip()
    state.direction = str(result.get("direction", "")).strip()
    state.segments = [Segment(start=s["start_sec"], end=s["end_sec"],
                              left=s["left"], right=s["right"],
                              boundary_provenance="er16_onepass", confidence=0.6)
                      for s in segs]
    state.track = derive_track_from_labels(state.segments)
    episode = QA.export_episode(state, out_path)

    # stamp the real model + pass count into meta (export hardcodes facts-first)
    episode["meta"]["model"] = f"er16-onepass ({MODEL}, fps={fps}, media-res={res})"
    episode["meta"]["vlm_passes"] = VLM_CALLS

    # 5) burn the per-hand labels onto the video — the website serves THIS one
    if overlay:
        from er16_overlay import overlay_video
        overlaid = str(wd / "overlaid.mp4")
        t2 = time.time()
        overlay_video(clocked, segs, state.goal, overlaid)
        print(f"[er16] label overlay built ({time.time()-t2:.0f}s) -> {overlaid}")
        episode["_clocked_plain"] = clocked     # provenance: the exact video the model saw
        episode["_clocked"] = overlaid          # what to_viewer copies + the site plays
    Path(out_path).write_text(json.dumps(episode, indent=2))

    print(f"[er16] wrote {out_path}")
    print(f"[er16] goal: {state.goal}  | direction: {state.direction}")
    print(f"[er16] tokens: {USAGE.summary()}")

    # 6) publish to the :8800 viewer
    if publish:
        r = subprocess.run([sys.executable, str(HERE / "to_viewer.py"), out_path,
                            "--ver", ver], capture_output=True, text=True)
        print(f"[er16] viewer publish rc={r.returncode}"
              + (f"  {r.stderr.strip()[:200]}" if r.returncode else ""))
    return episode


def main():
    ap = argparse.ArgumentParser(description="One-pass ER 1.6 segment+annotate.")
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--ver", default="er01")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--media-res", choices=list(MEDIA_RES_MAP), default="default",
                    help="per-frame detail; 'default' = original (recommended)")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip burning labels onto the video")
    a = ap.parse_args()
    out = a.out or f"out/{a.ver}/{Path(a.video).stem}.json"
    run(a.video, out, a.ver, a.fps, a.media_res, a.workdir or None,
        publish=not a.no_viewer, overlay=not a.no_overlay)


if __name__ == "__main__":
    main()
