#!/usr/bin/env python3
"""qa.py — deterministic QA gate + export.

These checks compare the labels (from GPT) against the possession track (from the
fact layer) — two INDEPENDENT elicitations, so wherever they disagree at least one
is wrong and the segment is routed to needs_review. Nothing here calls a model.

Human override: out/overrides/<id>.yaml outranks every model AT EXPORT — but it is
NEVER fed to any model (leak rule) and is OFF by default so a test run shows raw
pipeline quality.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REL_VERBS = ("drop", "place", "put", "puts", "insert", "release", "deposit",
              "set ", "sets ", "lower", "stack")
_PICK_VERBS = ("pick", "grasp", "grab", "take", "takes", "lift", "retrieve")
_STOP = {"the", "a", "an", "of", "to", "into", "onto", "from", "out", "off", "on",
         "in", "and", "with", "at", "its", "it", "down", "up", "over", "another",
         "next", "new", "one", "two", "hand", "left", "right", "side", "small",
         "large", "back", "front", "top", "this", "that"}


def _obj_tokens(label: str) -> set[str]:
    toks = re.split(r"[^a-z0-9]+", (label or "").lower())
    out = set()
    for t in toks:
        if not t or t in _STOP or len(t) < 3:
            continue
        out.add(t[:-1] if t.endswith("s") and len(t) > 4 else t)   # crude singularize
    return out


def _ivs(track: dict, hand: str) -> list[dict]:
    return track.get(hand, []) or []


# --------------------------------------------------------------------------- #
#  contact / N-A audit                                                         #
# --------------------------------------------------------------------------- #
def contact_track_qa(track: dict, segments: list[dict]) -> list[dict]:
    out = []
    for i, s in enumerate(segments):
        for hand in ("left", "right"):
            lab = str(s.get(hand, "")).strip().upper()
            sdur = s["end_sec"] - s["start_sec"]
            ivs = [iv for iv in _ivs(track, hand)
                   if iv.get("end_sec", 0) > s["start_sec"] + 0.1
                   and iv.get("start_sec", 0) < s["end_sec"] - 0.1
                   and str(iv.get("interacting_with", "")).lower().strip()
                   not in ("", "none", "empty", "out of frame", "nothing")]
            cover = sum(min(iv["end_sec"], s["end_sec"]) - max(iv["start_sec"], s["start_sec"])
                        for iv in ivs)
            span = [round(s["start_sec"], 2), round(s["end_sec"], 2)]
            if lab == "N/A" and cover > 0.6 * sdur and sdur > 0.6:
                out.append({"seg": i + 1, "code": "NA-WHILE-TRACKED", "hand": hand,
                            "span": span,
                            "detail": f"track holds {str(ivs[0].get('interacting_with',''))[:40]} "
                                      f"for {cover:.1f}s of {sdur:.1f}s but label N/A"})
            elif lab not in ("N/A", "") and not ivs and sdur > 1.0:
                out.append({"seg": i + 1, "code": "ACTION-WHILE-UNTRACKED", "hand": hand,
                            "span": span, "detail": str(s.get(hand, ""))[:50]})
    return out


def track_consistency(track: dict, segments: list[dict]) -> list[dict]:
    """A drop/place hand must have HELD the object; a pick hand must hold it right
    after. High-precision only."""
    flags = []
    for i, s in enumerate(segments):
        for hand in ("left", "right"):
            label = (s.get(hand) or "").lower()
            if not label or label == "n/a":
                continue
            ivs = _ivs(track, hand)
            lbl_obj = _obj_tokens(label)
            if not lbl_obj:
                continue
            if any(label.startswith(v) or f" {v}" in label for v in _REL_VERBS):
                near = [iv for iv in ivs
                        if iv["start_sec"] - 0.8 <= s["start_sec"] <= iv["end_sec"] + 0.8
                        and str(iv.get("interacting_with", "")) not in
                        ("none", "empty", "out of frame", "")]
                if near and not any(_obj_tokens(iv["interacting_with"]) & lbl_obj for iv in near):
                    flags.append({"seg": i + 1, "code": "TRACK-MISMATCH", "hand": hand,
                                  "detail": f"releases '{label[:36]}' but track holds "
                                            f"'{near[0].get('interacting_with','')[:30]}'"})
            elif any(label.startswith(v) or f" {v}" in label for v in _PICK_VERBS):
                after = [iv for iv in ivs
                         if iv["start_sec"] - 0.5 <= s["end_sec"] <= iv["end_sec"] + 0.5
                         and str(iv.get("interacting_with", "")) not in
                         ("none", "empty", "out of frame", "")]
                if after and not any(_obj_tokens(iv["interacting_with"]) & lbl_obj for iv in after):
                    flags.append({"seg": i + 1, "code": "TRACK-MISMATCH", "hand": hand,
                                  "detail": f"picks '{label[:36]}' but track then holds "
                                            f"'{after[0].get('interacting_with','')[:30]}'"})
    return flags


def pair_exclusivity(segments: list[dict]) -> list[dict]:
    """ANTI-MIRROR, deterministic: both hands given the SAME active fine verb on the
    SAME object in one segment is almost always a mirror error."""
    out = []
    fine = ("twist", "screw", "unscrew", "turn", "thread", "tighten", "loosen",
            "rotate", "press", "insert", "pull", "push")
    for i, s in enumerate(segments):
        l, r = (s.get("left") or "").lower(), (s.get("right") or "").lower()
        if l in ("", "n/a") or r in ("", "n/a"):
            continue
        lv = next((v for v in fine if l.startswith(v) or f" {v}" in l), None)
        rv = next((v for v in fine if r.startswith(v) or f" {v}" in r), None)
        if lv and lv == rv and (_obj_tokens(l) & _obj_tokens(r)):
            out.append({"seg": i + 1, "code": "PAIR-MIRROR", "hand": "both",
                        "detail": f"both hands '{lv}' the same object — verify roles"})
    return out


def direction_label_conflict(direction: str, segments: list[dict]) -> list[dict]:
    """If the decided direction is single (assembly|disassembly) but a clear
    majority of decisive labels point the other way, flag for the gate."""
    if direction not in ("assembly", "disassembly"):
        return []
    asm = ("screw on", "thread on", "thread onto", "attach", "insert", "join",
           "tighten", "assemble", "fit ", "press ")
    dis = ("unscrew", "screw off", "detach", "remove", "pull out", "extract",
           "separate", "loosen", "take apart")
    a = d = 0
    for s in segments:
        for hand in ("left", "right"):
            t = (s.get(hand) or "").lower()
            a += any(k in t for k in asm)
            d += any(k in t for k in dis)
    out = []
    if direction == "assembly" and d > a and d >= 3:
        out.append({"seg": 0, "code": "DIRECTION-CONFLICT", "hand": "both",
                    "detail": f"direction=assembly but {d} disassembly-leaning labels vs {a}"})
    if direction == "disassembly" and a > d and a >= 3:
        out.append({"seg": 0, "code": "DIRECTION-CONFLICT", "hand": "both",
                    "detail": f"direction=disassembly but {a} assembly-leaning labels vs {d}"})
    return out


def run_qa(track: dict, segments: list[dict], direction: str) -> list[dict]:
    qa = []
    qa += contact_track_qa(track, segments)
    qa += track_consistency(track, segments)
    qa += pair_exclusivity(segments)
    qa += direction_label_conflict(direction, segments)
    return qa


# --------------------------------------------------------------------------- #
#  human override (export-time only; NEVER fed to a model)                     #
# --------------------------------------------------------------------------- #
def clip_id(video: str) -> str:
    return Path(video).stem


def load_override(video: str, root: str = "out/overrides") -> dict | None:
    p = Path(root) / f"{clip_id(video)}.yaml"
    if not p.exists():
        return None
    try:
        import yaml  # optional
        return yaml.safe_load(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  export                                                                      #
# --------------------------------------------------------------------------- #
def _derive_lane(segs: list[dict], hand: str) -> list[dict]:
    """Merge consecutive segments with the same label for one hand into per-hand
    timeline intervals. Keys match what build_viewer_manifest expects
    ({start_sec, end_sec, action, needs_review}); it converts to s/e/a/rv itself."""
    lane = []
    for s in segs:
        a = (s.get(hand) or "N/A")
        if lane and lane[-1]["action"].strip().lower() == a.strip().lower():
            lane[-1]["end_sec"] = round(s["end_sec"], 2)
        else:
            lane.append({"start_sec": round(s["start_sec"], 2),
                         "end_sec": round(s["end_sec"], 2),
                         "action": a, "needs_review": False})
    return lane


def export_episode(state, out_path: str, apply_overrides: bool = False) -> dict:
    segs = [{"start_sec": round(s.start, 2), "end_sec": round(s.end, 2),
             "left": s.left, "right": s.right,
             "boundary_provenance": s.boundary_provenance,
             "confidence": round(s.confidence, 2)} for s in state.segments]
    qa = run_qa(state.track, segs, state.direction)
    # drop stale flags whose seg index no longer exists (a flag raised on a pre-merge
    # timeline whose segment was later merged away) — avoids "seg#7" with 5 segments.
    state.flags = [f for f in state.flags if 0 <= f.seg < len(segs)]
    flagged = {f.seg for f in state.flags}
    qa_segs = {v["seg"] - 1 for v in qa if v.get("seg")}
    # legacy-viewer-compatible subtasks (needs_review where a flag/qa lands)
    subtasks = [{**sg, "needs_review": (i in flagged or i in qa_segs)}
                for i, sg in enumerate(segs)]
    episode = {
        "clip": clip_id(state.video),
        "duration_sec": round(state.duration, 2),
        "goal": state.goal,
        "instruction": state.goal,                 # viewer reads ep.instruction
        "direction": state.direction,
        "phase_switch_sec": state.phase_switch_sec,
        "objects": state.objects,
        "environment": {"category": ""},
        "meta": {"duration_sec": round(state.duration, 2),
                 "model": "facts-first (gemini-3.1-pro + claude-opus-4-8)"},
        "phase_boundaries": state.phase_boundaries,
        "_transitions": state.transitions,
        "_contact_frames": state.contact_frames,
        "segments": segs,
        "subtasks": subtasks,                      # viewer renders these
        "left_timeline": _derive_lane(segs, "left"),
        "right_timeline": _derive_lane(segs, "right"),
        "_track": state.track,
        "_bursts_reduced": state.bursts_reduced,
        "_direction_burst_summary": state.direction_burst_summary,
        "_flags": [{"seg": f.seg + 1, "type": f.type, "raised_by": f.raised_by,
                    "detail": f.detail} for f in state.flags],
        "_seg_trace": [s.draft for s in state.segments],
        "_trace": {},                              # manifest's _coord_for expects a dict
        "_qa": {"violations": qa},                 # viewer reads _qa.violations
        "_gate_findings": state.gate_findings,
        "_purpose_verdict": state.purpose_verdict,
        "override_applied": False,
    }
    if apply_overrides:
        ov = load_override(state.video)
        if ov:
            episode["_pipeline_segments"] = segs
            episode["segments"] = ov.get("segments", segs)
            episode["override_applied"] = True
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(episode, indent=2))
    return episode
