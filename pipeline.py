#!/usr/bin/env python3
"""pipeline.py — egocentric two-handed manipulation -> per-hand action timeline.

ONE pipeline, no modes/flags. The segment->verify block runs in a keep-best rerun loop
(<= max_attempts); the verifier's feedback picks the re-entry stage (relabel by default,
full re-segment on a direction flip).

  raw video
   -> P0  hand overlay   (yolo handpose) burn GREEN=L / BLUE=R circles into the video
   -> P0b clock burn      timestamp + audio strip -> the one video every stage reads
   -> P1  bursts          (opus, frames) rotation/role/colour -> deterministic reduce
   -> P2  direction       (Gemini 10fps whole) direction + derived goal
   -> LOOP <= max_attempts, keep best by (verdict, -#segs):
        S   segment        (Gemini 10fps)         cut times only
        S+  seg-reconcile  (Gemini, per long seg) split swallowed multi-action cycles
        A   label          (Gemini 10fps, per-seg)per-hand labels on the fixed spans
        R   atomic-contract(deterministic)        fuzzy same-action merge + chained flag
        S2  merge-critic   (opus, frames)         delete-only merge of over-split fragments
        PP  pick<->place   (Gemini 10fps whole)   completeness chain: every pick has a
                                                  matching place / handoff, and vice versa
        V   verifier       (opus, frames)         verdict + corrections + rerun feedback
   -> W   global audit     (GPT-5.5, frames)      3rd-family whole-clip corrected timeline
  export -> episode.json (+ _stages trace, _qa) -> dashboard

Long clips (> CHUNK_SEC) are split into <= CHUNK_SEC parts, each run through the SAME
pipeline in parallel, then merged and every seam re-flowed.

Retired v49 top-down path lives under unused/; the earlier facts-first (1A/1C) orchestrator
is preserved in git history (tags <= v43).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import analysis as AN
import bursts as B
import qa as QA
import schemas as SC
from clipstate import (ClipState, Flag, Segment, build_track,
                       derive_track_from_labels, track_possession_changes)
from media import burn_clock, probe_duration, render_labeled, render_strip
from models import (CLAUDE_GATE, GEMINI_NATIVE, GPT_MODEL, USAGE,
                    GeminiFrames, GeminiVideo, claude_call, gpt_call)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

# frame budgets (facts-first; refine-heavy then label-heavy)
FPS_CONTACT = 10.0
CHUNK_SEC = 300.0         # long videos are split into even parts of <= this many seconds, each
                          # annotated independently then merged on a global timeline (hard edge
                          # at each seam). Keeps every per-stage frame count under Gemini's cap.
FPS_TRANSITION = 10.0     # dense whole-clip read to catch brief place/pickup/handoff
FPS_DIRECTION = 10.0      # user override: dense whole-clip read for direction
FPS_SEGMENT = 10.0        # v49 segmentation, dense so brief pick/place are visible
FPS_LABEL = 10.0          # per-segment NATIVE labeling / focused refine
FPS_EDGE = 30.0
FPS_GATE = 2.0
FPS_FRESH = 4.0           # frames for the context-free fresh-eye overlay review
CONTACT_WIN = 20.0        # seconds per 10fps contact-track window
EDGE_HALF = 0.6           # edge verifier half-window
LABEL_CTX = 1.0           # neighbor overlap for the labeler
MIN_SEG = 0.5             # drop/merge segments shorter than this


def _p(name: str, **kw) -> str:
    t = (PROMPTS / name).read_text()
    for k, v in kw.items():
        t = t.replace(f"<<{k}>>", str(v))
    return t


def _log(state_dir: Path, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(state_dir / "run.log", "a") as f:
        f.write(line + "\n")


def plan_windows(duration: float, win: float) -> list[tuple[float, float]]:
    n = max(1, int((duration + win - 0.01) // win))
    step = duration / n
    return [(round(k * step, 2), round((k + 1) * step, 2)) for k in range(n)]


def _spans(bounds: list[float]) -> list[tuple[float, float]]:
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _parallel(fn, items, levels=(8, 6, 4), wd: Path | None = None, tag: str = ""):
    """Run fn(item) concurrently, STARTING at levels[0] workers; any item whose fn raises
    is retried at the next-lower worker level (graceful rate-limit fallback 8 -> 6 -> 4).
    Returns results in input order; an item that fails at every level becomes None."""
    items = list(items)
    results: dict = {}
    pending = list(enumerate(items))
    for li, lvl in enumerate(levels):
        if not pending:
            break
        nxt = []
        with cf.ThreadPoolExecutor(max_workers=max(1, min(lvl, len(pending)))) as ex:
            futs = {ex.submit(fn, it): idx for idx, it in pending}
            for fu in cf.as_completed(futs):
                idx = futs[fu]
                try:
                    results[idx] = fu.result()
                except Exception:
                    nxt.append((idx, items[idx]))
        if nxt and wd is not None:
            nextlvl = levels[li + 1] if li + 1 < len(levels) else None
            _log(wd, f"{tag}: {len(nxt)} call(s) failed at {lvl} workers"
                    + (f" -> retry at {nextlvl}" if nextlvl else " -> gave up"))
        pending = nxt
    return [results.get(i) for i in range(len(items))]


def _snap(s: ClipState, stage: str, wd: Path | None = None):
    """Record the full timeline after a transform, so the dashboard can show the label
    EVOLVE stage by stage and the user can see exactly which step changed/broke it."""
    segs = [{"start_sec": round(x.start, 2), "end_sec": round(x.end, 2),
             "left": x.left, "right": x.right, "prov": x.boundary_provenance}
            for x in s.segments]
    s.stage_snapshots.append({"stage": stage, "n": len(segs), "segs": segs})
    if wd is not None:
        # one segment per line WITH its [start-end] span, so the console shows exactly
        # what each stage produced and when (times come straight from the track/segments).
        lines = [f"TRACE [{stage}] ({len(segs)} segs)"]
        for i, x in enumerate(segs):
            lines.append(f"    #{i+1:<2} [{x['start_sec']:5.1f}-{x['end_sec']:5.1f}s]  "
                         f"L: {x['left']:<24} R: {x['right']}")
        _log(wd, "\n".join(lines))


# --------------------------------------------------------------------------- #
#  THE TIMELINE — one structure, progressively refined                        #
# --------------------------------------------------------------------------- #


def _transitions_text(s: ClipState) -> str:
    ev = [e for e in s.transitions if e.get("t") is not None]
    if not ev:
        return "(none detected)"
    return "\n".join(f"{float(e['t']):.1f}s {e.get('hand', '?')} {e.get('kind', '?')} "
                     f"of {e.get('object', '?')}" for e in ev)


def _locked_transitions(s: ClipState) -> list[float]:
    """The detected place/pickup/throw/handoff times that NO stage may merge across
    (the global invariant). One definition, used by label+collapse, the verifier, the
    gate and fresh-eye, so every stage protects the exact same boundaries."""
    return sorted({round(float(e["t"]), 2) for e in s.transitions
                   if e.get("t") is not None and 0.3 < float(e["t"]) < s.duration - 0.3})


def _required_cuts_text(s: ClipState) -> str:
    """Hand-EXPLICIT required-cut hints for the labeler: each detected transition as
    «KIND-HAND» at its time. The acting hand (the one that places / picks) SWAPS cycle to
    cycle, so naming it per event stops the labeler freezing one hand's role across the clip
    and lets it put the action on the right hand with N/A on the other."""
    out = []
    for e in s.transitions:
        t = e.get("t")
        if t is None or not (0.3 < float(t) < s.duration - 0.3):
            continue
        kind = str(e.get("kind", "")).upper()
        hand = str(e.get("hand", "")).lower()
        tag = f"«{kind}-{hand.upper()}»" if hand in ("left", "right") else f"«{kind}»"
        out.append(f"{tag} at {float(t):.1f}s")
    return ", ".join(out) or "(none)"


def _enforce_cuts(s: ClipState, cut_times: list[float], wd: Path):
    """A detected transition may NEVER be swallowed: split any segment that bridges a
    cut time. Structural only (no hardcoded verbs) — the split piece inherits the parent
    label and the gate/verifier relabel it from frames. Guarantees no mega-segment."""
    changed = 0
    for t in cut_times:
        for i, g in enumerate(s.segments):
            if g.start + 0.25 < t < g.end - 0.25:        # t well inside -> split there
                new = Segment(start=round(t, 2), end=g.end, left=g.left, right=g.right)
                new.boundary_provenance = "transition_cut"
                g.end = round(t, 2)
                s.segments.insert(i + 1, new)
                changed += 1
                break
    if changed:
        s.track = derive_track_from_labels(s.segments)
        _log(wd, f"P4 enforce: split {changed} swallowed transition(s) at {cut_times}")


# =========================================================================== #
#  PHASE 1 — the fact layer                                                    #
# =========================================================================== #
_EMPTY_NAMES = {"", "empty", "none", "out of frame", "n/a", "null", "-", "nothing"}


# --- Stage-1A two-pass self-review ------------------------------------------- #
# A single pass is unstable, so each window is tracked TWICE: a draft viewing (the model may
# leave itself notes about uncertainties), then a final viewing where it re-watches with its
# draft + notes + the prompt's CHECKLIST and corrects it — above all fixing left/right N/A
# placement. 1A gets NO separate system prompt; everything it needs is in contact_track.txt.
_DRAFT_SUFFIX = ("\n\n[FIRST VIEWING] Give your best track now. Use the \"notes\" field to "
                 "record anything you are unsure about — which hand holds what, an object's "
                 "identity, or a boundary time — to resolve on a second viewing.")
_FINAL_SUFFIX = ("\n\n[FINAL VIEWING — your last look; make it correct] Your first-pass "
                 "draft:\n{draft}{notes}\nRe-watch the window and run the CHECKLIST on your "
                 "draft. Fix every problem; above all, re-find each forearm from the side it "
                 "enters and confirm each object is on the CORRECT hand, with NO empty/N/A "
                 "interval. Output the corrected final track.")


def _window_track(gv: GeminiVideo, a: float, b: float):
    """Two-pass contact track for one window: draft (+optional notes) -> self-reviewed final."""
    base = _p("contact_track.txt", A=round(a, 1), B=round(b, 1))
    draft = gv.watch(base + _DRAFT_SUFFIX, "", SC.CONTACT_TRACK_DRAFT,
                     a=a, b=b, fps=FPS_CONTACT, max_tokens=8000)
    if not draft:
        return None
    dtxt = json.dumps({"left": draft.get("left", []), "right": draft.get("right", [])},
                      ensure_ascii=False)
    notes = (draft.get("notes") or "").strip()
    suffix = _FINAL_SUFFIX.format(draft=dtxt, notes=("\nYour notes: " + notes if notes else ""))
    final = gv.watch(base + suffix, "", SC.CONTACT_TRACK,
                     a=a, b=b, fps=FPS_CONTACT, max_tokens=8000)
    return final or draft


TRANS_WIN = 3.5           # sliding window length (s)
TRANS_STRIDE = 2.0        # window stride (s)


# =========================================================================== #
#  PHASE 2 — bursts                                                            #
# =========================================================================== #
def phase2_bursts(s: ClipState, system: str, wd: Path):
    reqs = []
    # seed 2-3 direction probes across the clip (where rotation shows). (No cycle stage:
    # it was vestigial/unstable and never drove anything — removed.)
    spans = plan_windows(s.duration, max(4.0, s.duration / 4))   # already (a,b) pairs
    # HIGH ABSTENTION BAR: a nut spinning on a thread looks near-identical assembling vs
    # disassembling, so a probe must answer a direction ONLY when the direction of change is
    # unmistakable; otherwise 'cannot determine' (dropped by burst_reduce). This kills the
    # assembly-default that let one ambiguous burst set a whole clip's (wrong) direction.
    qdir = ("Watching ONLY the visible rotation/motion of the part being worked: can you tell "
            "UNMISTAKABLY which way it is going? ASSEMBLY = parts being joined / a fastener "
            "screwed ON / inserted / engagement INCREASING. DISASSEMBLY = parts being separated "
            "/ a fastener unscrewed OFF / removed / engagement DECREASING. A fastener turning on "
            "a thread looks almost the same either way — answer 'assembly' or 'disassembly' ONLY "
            "if the direction of change is CLEAR in these frames; if you cannot be sure which "
            "way it is progressing, answer 'cannot determine'. Verdict exactly 'assembly', "
            "'disassembly', or 'cannot determine'.")
    # Sample MANY direction probes spread across the WHOLE clip (~one per 6s) — a single burst
    # must never be able to commit a clip's direction (the assembly-bias fix).
    n_dir = max(4, int(round(s.duration / 6.0)))
    dir_spans = plan_windows(s.duration, s.duration / n_dir)
    for (a, b) in dir_spans:
        reqs.append({"t": round((a + b) / 2, 2), "kind": "direction", "question": qdir})
    # one role probe at the busiest moment
    if spans:
        a, b = spans[0]
        reqs.append({"t": round((a + b) / 2, 2), "kind": "role",
                     "question": ("Which forearm is the ACTOR doing the fine "
                                  "manipulation here? Verdict exactly 'left', "
                                  "'right', 'both', or 'cannot determine'.")})
    _log(wd, f"P2 burst sweep: {len(reqs)} requests @ 30fps")
    raw = B.run_bursts(reqs, s.clocked, str(wd), system=system, inv_names=s.object_names())
    red = B.burst_reduce(raw)
    s.bursts_raw = raw
    s.bursts_reduced = red["bursts_reduced"]
    s.direction_burst_summary = red["direction_burst_summary"]
    s.recheck_times = red["recheck_times"]
    # re-burst isolated dissenters ONCE
    if s.recheck_times:
        _log(wd, f"P2 recheck dissenters: {s.recheck_times}")
        rc = [{"t": t, "kind": "direction", "question": qdir} for t in s.recheck_times]
        raw2 = B.run_bursts(rc, s.clocked, str(wd), system=system, inv_names=s.object_names())
        red2 = B.burst_reduce(raw + raw2)
        s.bursts_raw = raw + raw2
        s.bursts_reduced = red2["bursts_reduced"]
        s.direction_burst_summary = red2["direction_burst_summary"]
        s.recheck_times = red2["recheck_times"]
    _log(wd, f"P2 -> {len(s.bursts_reduced)} verdicts; summary={s.direction_burst_summary}")


# =========================================================================== #
#  PHASE 3 — direction (also derives the goal)                                 #
# =========================================================================== #
def phase3_direction(s: ClipState, gv: GeminiVideo, system: str, wd: Path):
    r = gv.watch(_p("direction_decide.txt",
                    OBJECTS=s.objects_line(),
                    TRACK=s.track_lines(),
                    BURSTS="\n".join(s.bursts_reduced) or "(no decisive bursts)",
                    BURST_SUMMARY=s.direction_burst_summary),
                 system, SC.DIRECTION_DECIDE, a=0.0, b=s.duration,
                 fps=FPS_DIRECTION, max_tokens=4000)
    s.direction = r.get("direction", "other_or_ambiguous")
    s.goal = r.get("goal", "")
    s.phase_switch_sec = r.get("phase_switch_sec")
    # STABILIZER: direction_decide sometimes returns "other_or_ambiguous" on a noisy run
    # even when the rotation bursts have a clear single-direction majority. Don't accept
    # ambiguity when the deterministic burst summary already commits (the doctrine:
    # ambiguity must be COSTLY). Override from the burst majority.
    summ = s.direction_burst_summary or {}
    decisive = int(summ.get("assembly", 0)) + int(summ.get("disassembly", 0))
    # Bursts may override an "ambiguous" structural read ONLY with >=2 decisive verdicts.
    # A lone decisive burst must NOT commit a direction — that single-vote override is exactly
    # what set assembly on the two clips that were really disassembly. <2 decisive -> leave the
    # structural verdict (direction_decide already weighs end-state inventory) as-is.
    if s.direction in ("other_or_ambiguous", "") and summ.get("rule") == "single_direction_majority" \
            and summ.get("majority") in ("assembly", "disassembly") and decisive >= 2:
        _log(wd, f"P3 direction was '{s.direction}' but {decisive} decisive bursts commit "
                f"{summ['majority']} -> override")
        s.direction = summ["majority"]
    elif s.direction in ("other_or_ambiguous", "") and decisive < 2:
        _log(wd, f"P3 direction ambiguous and only {decisive} decisive burst(s) -> "
                f"NOT overriding from bursts (structure/end-state decides)")
    _log(wd, f"P3 direction={s.direction} switch={s.phase_switch_sec} | goal: {s.goal}")


# =========================================================================== #
#  PHASE 4 — rough -> refine                                                   #
# =========================================================================== #


def _fullness(seg: Segment) -> int:
    return sum(1 for x in (seg.left, seg.right) if x and x.upper() != "N/A")


def delete_only_critic(s: ClipState, system: str, wd: Path):
    """Delete-only merge critic (Claude opus, whole-clip 2fps frames): merge adjacent
    segments that are ONE over-split action. It can only DELETE boundaries — never add,
    move, or relabel — so it safely fixes over-segmentation without risk of over-cutting.
    Detected place/pickup/handoff boundaries are protected from merging. Runs ONCE
    unless re-requested on demand by a later stage (guarded by s.ran)."""
    if "merge_critic" in s.ran:
        return
    s.ran.add("merge_critic")
    frames, _ = render_strip(s.clocked, 0.0, s.duration, FPS_GATE, s.track, str(wd),
                             ctx=0.0, cap_frames=90)
    r = claude_call(_p("merge_critic.txt", DIRECTION=s.direction,
                       TIMELINE=s.timeline_text()),
                    frames, system, SC.MERGE_CRITIC, model=CLAUDE_GATE)
    protect = [float(e["t"]) for e in s.transitions if e.get("t") is not None]
    n = 0
    for m in sorted(r.get("merges", []), key=lambda x: -float(x.get("delete_t", 0))):
        dt = float(m.get("delete_t", -1))
        keep = m.get("keep", "after")
        if any(abs(dt - t) <= 0.5 for t in protect):     # never merge a real transition
            continue
        for i in range(len(s.segments) - 1):
            if abs(s.segments[i].end - dt) < 0.4:
                left, right = s.segments[i], s.segments[i + 1]
                src = right if keep == "after" else left
                left.end = right.end
                left.left, left.right = src.left, src.right
                left.boundary_provenance = "merge_critic"
                left.draft = {"labeled": True, "origin": "merge_critic",
                              "label": {"left": left.left, "right": left.right}}
                del s.segments[i + 1]
                n += 1
                break
    _log(wd, f"P4d delete-only critic: merged {n} boundaries -> {len(s.segments)} segments")


_GAP_KIND = {"missing_pick": "pickup", "missing_place": "place",
             "missing_handoff": "handoff"}


def completeness_check(s: ClipState, gv: GeminiVideo, system: str, wd: Path) -> int:
    """Completeness audit (Gemini native, 10fps whole clip): every pick needs a later
    place/hand-off (or still-in-hand at end), every place needs a prior pick. A
    confirmed gap IS a place/pickup transition the scan missed, so each one is FOLDED
    INTO s.transitions (with its object) — a final segmentation pass then turns it into
    a real labeled segment. Also flags it for the gate. Returns the count of NEW
    transitions added (boundaries not already near an existing one). Runs ONCE unless
    re-requested on demand (guarded by s.ran)."""
    if "completeness" in s.ran:
        return 0
    s.ran.add("completeness")
    r = gv.watch(_p("completeness.txt", DIRECTION=s.direction,
                    OBJECTS=s.objects_line(), TIMELINE=s.timeline_text()),
                 system, SC.COMPLETENESS, a=0.0, b=s.duration, fps=FPS_LABEL,
                 max_tokens=3000)
    gaps = r.get("gaps", []) or []
    have = [float(e["t"]) for e in s.transitions if e.get("t") is not None]
    added = 0
    for g in gaps:
        seg = int(g.get("in_seg", 0)) - 1
        seg = seg if 0 <= seg < len(s.segments) else 0
        s.flags.append(Flag(seg, str(g.get("kind", "incomplete")), "completeness",
                            f"{g.get('hand','')} @{g.get('t','?')}s: "
                            f"{str(g.get('evidence',''))[:90]}"))
        try:
            t = float(g.get("t"))
        except (TypeError, ValueError):
            continue
        if not (0.5 < t < s.duration - 0.5):
            continue
        if any(abs(t - ht) <= 0.6 for ht in have):   # already a transition there
            continue
        s.transitions.append({"t": round(t, 2), "hand": g.get("hand", "left"),
                              "kind": _GAP_KIND.get(g.get("kind"), "place"),
                              "object": g.get("object", ""),
                              "evidence": "completeness: " + str(g.get("evidence", ""))[:80]})
        have.append(t)
        added += 1
    s.transitions.sort(key=lambda e: float(e["t"]))
    _log(wd, f"P4e completeness: {len(gaps)} chain gaps "
            f"{[g.get('kind') for g in gaps]} -> +{added} new transitions")
    return added


# =========================================================================== #
#  PHASE 5 — gate + export                                                     #
# =========================================================================== #
def phase5_gate(s: ClipState, gv: GeminiVideo, system: str, wd: Path):
    frames, _ = render_strip(s.clocked, 0.0, s.duration, FPS_GATE, s.track, str(wd),
                             ctx=0.0, cap_frames=90)
    flagged = s.flags_text()
    r = claude_call(_p("opus_final.txt", DIRECTION=s.direction, GOAL=s.goal,
                       TRACK=s.track_lines(),
                       BURSTS="\n".join(s.bursts_reduced) or "(none)",
                       TIMELINE=s.timeline_text(), FLAGGED=flagged,
                       TRANSITIONS=_transitions_text(s),
                       ANALYSIS=s.analysis_report or ""),
                    frames, system, SC.GATE, model=CLAUDE_GATE)
    s.gate_findings = str(r.get("findings", ""))     # full findings (no truncation)
    s.purpose_verdict = r.get("purpose_verdict", "")
    if r.get("purpose_check"):
        s.goal = r["purpose_check"] if s.purpose_verdict == "corrected" else s.goal
    # DIRECTION CORRECTION: if the gate (seeing the frames) reads the opposite direction,
    # flip it and force a relabel pass so the whole timeline re-flows with correct verbs.
    _DIRS = {"assembly", "disassembly", "transfer_or_portioning",
             "mixed_or_alternating", "other_or_ambiguous"}
    newdir = r.get("direction")
    if newdir in _DIRS and newdir != s.direction:
        _log(wd, f"P5 gate CORRECTED direction: {s.direction} -> {newdir} -> relabel")
        s.direction = newdir
        r["quality_verdict"] = "needs_rerun"          # next attempt relabels with new direction
        r["rerun_feedback"] = (f"Direction corrected to '{newdir}'; relabel all verbs to match. "
                               + str(r.get("rerun_feedback", "")))
    # apply corrections to flagged segments only
    nc = 0
    for c in r.get("corrections", []):
        idx = int(c.get("seg", 0)) - 1
        if 0 <= idx < len(s.segments) and c.get("hand") in ("left", "right"):
            seg = s.segments[idx]
            old = getattr(seg, c["hand"])
            setattr(seg, c["hand"], c.get("label", old) or "N/A")
            seg.draft.setdefault("gate", {})[c["hand"]] = {"from": old, "to": c["label"]}
            nc += 1
    # window-seam merges (guarded by possession change AND detected transitions)
    nm = 0
    protect = [float(e["t"]) for e in s.transitions if e.get("t") is not None]
    for t in sorted(r.get("merge_at_sec", []), reverse=True):
        if any(abs(float(t) - pt) <= 0.5 for pt in protect):   # never merge a real transition
            continue
        for i in range(len(s.segments) - 1):
            if abs(s.segments[i].end - float(t)) < 0.25:
                left, right = s.segments[i], s.segments[i + 1]
                if not track_possession_changes(s.track, left.start, right.end):
                    fuller = left if _fullness(left) >= _fullness(right) else right
                    left.end = right.end
                    left.left, left.right = fuller.left, fuller.right
                    left.boundary_provenance = "gate_seam_merge"
                    del s.segments[i + 1]
                    nm += 1
                break
    # split_request dropped: in the one-timeline model the gate edits DIRECTLY (relabel +
    # merge); a needed split is the labeler/collapse's job, not an orphan flag nobody reads.
    # DETERMINISTIC BACKSTOP: the gate must not be able to rubber-stamp a code-detected
    # STRUCTURAL contradiction. frozen_across_transition?/dup_across_transition? are grounded
    # in the reliable P1c transitions (not the flaky 1A contact), so if any survive, FORCE
    # needs_rerun regardless of the model's verdict — a hand cannot hold the same thing across
    # its own place/pickup, and identical labels cannot straddle a real transition.
    _STRUCT = {"frozen_across_transition?", "dup_across_transition?"}
    struct = [f for f in s.flags if f.type in _STRUCT]
    if struct and r.get("quality_verdict") == "good":
        names = "; ".join(f.detail for f in struct[:3])
        _log(wd, f"P5 gate said good but {len(struct)} structural flag(s) -> FORCE needs_rerun")
        r["quality_verdict"] = "needs_rerun"
        r["rerun_feedback"] = (f"Structural contradiction(s) [code-detected, reliable]: {names}. "
                               + str(r.get("rerun_feedback", "")))
    _log(wd, f"P5 gate: purpose={s.purpose_verdict}, {nc} corrections, {nm} seam-merges, "
            f"quality={r.get('quality_verdict')}")
    # on-demand: the gate may invoke either critic specifically (each guarded to once)
    if r.get("request_merge_critic"):
        delete_only_critic(s, system, wd)
        s.track = derive_track_from_labels(s.segments)
    if r.get("request_completeness"):
        completeness_check(s, gv, system, wd)
    return r


# =========================================================================== #
#  PHASE 6 — fresh-eye final review (context-free)                            #
# =========================================================================== #


# =========================================================================== #
#  TOP-LEVEL                                                                   #
# =========================================================================== #
def _extract_clip(video: str, a: float, dur: float, out: str):
    """Cut [a, a+dur] from `video` into `out`, re-encoded so the part starts exactly at `a`
    (a fresh local-0 clock is burned per part downstream)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", video,
                    "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-an", out], capture_output=True)


def _shift(items, off, keys=("start_sec", "end_sec")):
    """Return a copy of a list of dicts with the given time keys offset by `off` seconds."""
    out = []
    for it in items or []:
        d = dict(it)
        for k in keys:
            if isinstance(d.get(k), (int, float)):
                d[k] = round(d[k] + off, 2)
        out.append(d)
    return out


def _shift_pb(pb, off):
    out = []
    for x in pb or []:
        if isinstance(x, (int, float)):
            out.append(round(x + off, 2))
        elif isinstance(x, dict):
            out.append(_shift([x], off, keys=("sec", "start_sec", "end_sec", "t"))[0])
        else:
            out.append(x)
    return out


def _merge_chunk_episodes(video: str, out_path: str, total_dur: float, parts: list) -> dict:
    """Stitch independently-annotated parts (each (offset_sec, episode_dict)) into ONE episode
    on a global timeline: time-bearing lists offset by the part's start, segment-index fields
    shifted by the running segment count, objects unioned. Hard boundary at each seam."""
    segs, subs, lt, rt = [], [], [], []
    trans, cframes, flags, qa, pbs = [], [], [], [], []
    ctrack = {"left": [], "right": []}
    objs, dirs, goals = {}, [], []
    seg_off = 0
    for off, ep in parts:
        es = _shift(ep.get("segments", []), off)
        segs += es
        subs += _shift(ep.get("subtasks", []), off)
        lt += _shift(ep.get("left_timeline", []), off)
        rt += _shift(ep.get("right_timeline", []), off)
        trans += _shift(ep.get("_transitions", []), off, keys=("t", "sec", "start_sec", "end_sec"))
        cframes += _shift(ep.get("_contact_frames", []), off, keys=("t",))
        pbs += _shift_pb(ep.get("phase_boundaries", []), off)
        for h in ("left", "right"):
            ctrack[h] += _shift((ep.get("_contact_track") or {}).get(h, []), off)
        for o in ep.get("objects", []) or []:
            nm = (o.get("name") if isinstance(o, dict) else o) or ""
            if nm and nm.lower() not in objs:
                objs[nm.lower()] = o
        for f in ep.get("_flags", []) or []:
            f2 = dict(f)
            if isinstance(f2.get("seg"), int):
                f2["seg"] += seg_off
            flags.append(f2)
        for v in (ep.get("_qa") or {}).get("violations", []) or []:
            v2 = dict(v)
            if isinstance(v2.get("seg"), int):
                v2["seg"] += seg_off
            qa.append(v2)
        dirs.append(ep.get("direction"))
        goals.append(ep.get("goal"))
        seg_off += len(es)

    uniq_dirs = [d for d in dict.fromkeys(dirs) if d]
    direction = uniq_dirs[0] if len(uniq_dirs) <= 1 else "mixed_or_alternating"
    goal = " | ".join(dict.fromkeys(g for g in goals if g))
    episode = {
        "clip": QA.clip_id(video),
        "duration_sec": round(total_dur, 2),
        "goal": goal, "instruction": goal,
        "direction": direction,
        "phase_switch_sec": None,
        "objects": list(objs.values()),
        "environment": {"category": ""},
        "meta": {"duration_sec": round(total_dur, 2),
                 "model": "facts-first (gemini-3.1-pro + claude-opus-4-8)",
                 "chunked": True, "n_chunks": len(parts), "chunk_sec": CHUNK_SEC,
                 "chunk_offsets": [round(o, 2) for o, _ in parts],
                 "chunk_directions": dirs, "chunk_goals": goals},
        "phase_boundaries": pbs,
        "_transitions": trans,
        "_contact_frames": cframes,
        "segments": segs,
        "subtasks": subs,
        "left_timeline": lt, "right_timeline": rt,
        "_track": {}, "_contact_track": ctrack,
        "_bursts_reduced": [], "_direction_burst_summary": "",
        "_flags": flags,
        "_seg_trace": [],
        "_trace": {},
        "_qa": {"violations": qa},
        "_gate_findings": [], "_purpose_verdict": {},
        "_analysis_report": {}, "_fresh_eye": "",
        "_stages": [], "_clocked": "",
        "override_applied": False,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(episode, indent=2))
    return episode


def _repair_seams(video: str, out_path: str, parts: list, merged: dict,
                  wd: Path, max_attempts: int) -> dict:
    """Re-flow every boundary so an action straddling a seam is NOT cut. For each seam, take
    the last segment before it and the first segment after it, re-extract that combined span
    from the ORIGINAL video, re-run full segmentation+annotation on it, and splice the result
    back in place of those two segments. Then rebuild the deliverable timelines."""
    segs = list(merged["segments"])
    offsets = [o for o, _ in parts]
    # 1) identify every seam region against the (unspliced) merged segments
    jobs = []
    for i in range(1, len(parts)):
        seam_t = offsets[i]
        left = [j for j, s in enumerate(segs) if s["end_sec"] <= seam_t + 0.25]
        right = [j for j, s in enumerate(segs) if s["start_sec"] >= seam_t - 0.25]
        if not left or not right:
            continue
        li = max(left, key=lambda j: segs[j]["end_sec"])
        ri = min(right, key=lambda j: segs[j]["start_sec"])
        if ri <= li:
            continue
        a, b = segs[li]["start_sec"], segs[ri]["end_sec"]
        if b - a >= 0.5:
            jobs.append((i, li, ri, a, b, seam_t))

    # 2) re-segment + re-annotate every seam span IN PARALLEL (each is independent)
    def _do_seam(job):
        i, li, ri, a, b, seam_t = job
        seam_clip = str(wd / f"seam{i}.mp4")
        _extract_clip(video, a, b - a, seam_clip)
        _log(wd, f"-- seam {i}: re-flow [{a:.1f}-{b:.1f}s] across boundary {seam_t:.1f}s")
        ep = _annotate(seam_clip, str(wd / f"seam{i}.json"),
                       str(wd / f"seam{i}"), max_attempts)
        return (li, ri, _shift(ep.get("segments", []), a))

    if jobs:
        _log(wd, f"-- re-flowing {len(jobs)} seam(s) IN PARALLEL")
        res = _parallel(_do_seam, jobs, levels=(len(jobs), 1), wd=wd, tag="seam re-flow")
        # 3) splice highest-index first so the lower indices stay valid (seams never overlap)
        for li, ri, new in sorted([r for r in res if r], key=lambda r: r[0], reverse=True):
            if new:
                segs = segs[:li] + new + segs[ri + 1:]

    segs.sort(key=lambda s: s["start_sec"])
    merged["segments"] = segs
    merged["subtasks"] = [{**s, "needs_review": False} for s in segs]
    merged["left_timeline"] = QA._derive_lane(segs, "left")
    merged["right_timeline"] = QA._derive_lane(segs, "right")
    merged["meta"]["seams_repaired"] = max(0, len(parts) - 1)
    Path(out_path).write_text(json.dumps(merged, indent=2))
    return merged


_YOLO = HERE.parent / "yolo_hands"
_HAND_MODEL = _YOLO / "yolo_bundle" / "hand_yolo_detector@20260314.pt"


# ===================== LEAN flow (v49-style, circle-grounded) ==================
# Skip 1A/1C: no facts seed. Direction from video+bursts, then a v49-style
# segment+label pass directly on the CIRCLED video (L/R from the circles), iterated
# against the gate. Isolated from the facts-first path so that one is unaffected.
def _segs_from_bounds(bounds, dur, prov):
    cuts = sorted(set(round(x, 2) for x in bounds if 0.2 < x < dur - 0.2))
    pts = [0.0] + cuts + [round(dur, 2)]
    kept = [pts[0]]
    for t in pts[1:-1]:
        if t - kept[-1] >= MIN_SEG:
            kept.append(t)
    kept.append(pts[-1])
    if len(kept) >= 3 and kept[-1] - kept[-2] < MIN_SEG:
        kept.pop(-2)
    out = [Segment(start=a, end=b, boundary_provenance=prov, confidence=0.6)
           for a, b in _spans(kept)]
    return out or [Segment(start=0.0, end=dur)]


def _lean_segment(s, gv, system, wd):
    r = gv.watch(_p("segment_lean.txt", A=0.0, B=round(s.duration, 1),
                    GOAL=s.goal or "(unknown)",
                    BURSTS="\n".join(s.bursts_reduced) or "(none)"),
                 system, SC.V49_SEGMENT, a=0.0, b=s.duration, fps=FPS_SEGMENT, max_tokens=4000)
    cuts = sorted(float(x) for x in r.get("boundaries", [])
                  if x is not None and 0.3 < float(x) < s.duration - 0.3)
    s.segments = _segs_from_bounds(cuts, s.duration, "lean_segment")
    _log(wd, f"LEAN segment -> {len(s.segments)} segments ({len(cuts)} cuts)")


def _lean_label(s, gv, system, wd, feedback=""):
    def _one(i):
        seg = s.segments[i]
        oa, ob = max(0.0, seg.start - LABEL_CTX), min(s.duration, seg.end + LABEL_CTX)
        prev = s.segments[i - 1] if i > 0 else None
        nxt = s.segments[i + 1] if i + 1 < len(s.segments) else None
        prompt = _p("label_lean.txt", OA=round(oa, 1), OB=round(ob, 1),
                    A=round(seg.start, 1), B=round(seg.end, 1),
                    GOAL=s.goal or "(unknown)", DIRECTION=s.direction or "(unknown)",
                    INVENTORY=s.objects_line() or "(name objects plainly as you see them)",
                    BURSTS="\n".join(s.bursts_in(seg.start, seg.end)) or "(none in window)",
                    PREV_LEFT=(prev.left if prev else "(start)"),
                    PREV_RIGHT=(prev.right if prev else "(start)"),
                    NEXT_LEFT=(nxt.left if nxt else "(end)"),
                    NEXT_RIGHT=(nxt.right if nxt else "(end)"))
        if feedback:
            prompt += f"\n\nReviewer feedback to address this pass: {feedback}"
        out = gv.watch(prompt, system, SC.VIDEO_LABEL_NATIVE, a=oa, b=ob,
                       fps=FPS_LABEL, max_tokens=2000)
        return i, out
    for r in _parallel(_one, list(range(len(s.segments))), wd=wd, tag="LEAN label"):
        if not r:
            continue
        i, out = r
        seg = s.segments[i]
        seg.left = out.get("left", "N/A") or "N/A"
        seg.right = out.get("right", "N/A") or "N/A"
        seg.draft = {"labeled": True, "label_think": out.get("think", ""),
                     "origin": "lean_label", "label": {"left": seg.left, "right": seg.right}}
    _log(wd, f"LEAN label -> {len(s.segments)} segments labeled")


def _seg_reconcile(s, gv, system, wd, min_len: float = 1.6) -> bool:
    """STAGE S+ (split) — per-segment temporal-structure audit (Gemini native, ported from
    v49). For each span longer than min_len a focused pass over [a,b] decides ONE action or
    several and returns the onset of each change (esp. a put-down that completes a cycle);
    the span is split there. Runs on UNLABELED cuts (before labeling), refining only the cut
    list. Complements the delete-only merge-critic — together they fix granularity BOTH ways
    (split under-segmented, merge over-segmented). Returns True if the cuts changed."""
    if not s.segments:
        return False

    def _one(idx_seg):
        i, seg = idx_seg
        if seg.end - seg.start < min_len:
            return i, []
        nfr = int(round((seg.end - seg.start) * FPS_SEGMENT))
        prompt = _p("seg_reconcile.txt", N=nfr, A=round(seg.start, 1), B=round(seg.end, 1),
                    GOAL=s.goal or "(unknown)",
                    LEFT=(seg.left if seg.left and seg.left != "N/A" else "(unlabeled)"),
                    RIGHT=(seg.right if seg.right and seg.right != "N/A" else "(unlabeled)"))
        r = gv.watch(prompt, system, SC.SEG_RECONCILE, a=seg.start, b=seg.end,
                     fps=FPS_SEGMENT, max_tokens=1500)      # raises -> _parallel retries
        cuts = sorted(float(x) for x in r.get("boundaries", [])
                      if x is not None and seg.start + 0.3 < float(x) < seg.end - 0.3)
        return i, cuts

    extra = []
    for r in _parallel(_one, list(enumerate(s.segments)), wd=wd, tag="seg_reconcile"):
        if r is None:
            continue
        _i, cuts = r
        extra.extend(cuts)
    if not extra:
        _log(wd, "seg_reconcile: no splits")
        return False
    interior = [seg.end for seg in s.segments[:-1]] + extra
    bounds = sorted({round(c, 2) for c in interior if 0.3 < c < s.duration - 0.3})
    n0 = len(s.segments)
    s.segments = _segs_from_bounds(bounds, s.duration, "seg_reconcile")
    _log(wd, f"seg_reconcile: +{len(extra)} split point(s) -> {n0} -> {len(s.segments)} segments")
    return True


# --------------------------------------------------------------------------- #
#  STAGE R — atomic-label contract (deterministic, NO model) + helpers ported  #
#  from v49: fuzzy same-action merge (token overlap, not a wordlist).          #
# --------------------------------------------------------------------------- #
def _norm_lbl(x) -> str:
    return " ".join(str(x or "N/A").lower().split())


def _label_sim(a, b) -> float:
    """Generic word-overlap (Jaccard) — token overlap, NOT a semantic wordlist."""
    wa, wb = set(_norm_lbl(a).split()), set(_norm_lbl(b).split())
    if not wa or not wb:
        return 1.0 if wa == wb else 0.0
    return len(wa & wb) / len(wa | wb)


def _labels_mergeable(p_l, p_r, s_l, s_r, thresh: float = 0.6) -> bool:
    """Two adjacent segments merge when BOTH hands read as the SAME action — exact OR
    near-identical wording (token overlap >= thresh). An idle hand (N/A) must match N/A
    exactly: idle vs active is never 'similar'."""
    def one(a, b):
        na, nb = _norm_lbl(a), _norm_lbl(b)
        if na == nb:
            return True
        if na == "n/a" or nb == "n/a":
            return False
        return _label_sim(a, b) >= thresh
    return one(p_l, s_l) and one(p_r, s_r)


_CHAIN_RE = re.compile(r"\b(and|then|while|before|after)\b", re.I)


def _atomic_contract(s, wd):
    """STAGE R (deterministic, no model): (1) fuzzy-merge consecutive segments whose BOTH
    hands read the SAME action ('thread the nut'+'tighten the nut' -> one), collapsing the
    micro pick/place cycles of one continuous handling; (2) FLAG any chained-narrative
    label ('pick up and thread') for the frame verifier to atomise — times untouched, the
    label rewrite is V's job (it sees the pixels)."""
    if not s.segments:
        return
    merged = [s.segments[0]]
    for seg in s.segments[1:]:
        p = merged[-1]
        if _labels_mergeable(p.left, p.right, seg.left, seg.right):
            p.end = seg.end
            p.draft = {**(p.draft or {}), "merged_run": True}
        else:
            merged.append(seg)
    n0 = len(s.segments)
    s.segments = merged
    if n0 != len(merged):
        _log(wd, f"R atomic-contract: fuzzy-merge {n0} -> {len(merged)} segments")
    nflag = 0
    for i, seg in enumerate(s.segments):
        for hand in ("left", "right"):
            lab = getattr(seg, hand)
            if lab and lab.upper() != "N/A" and _CHAIN_RE.search(lab):
                s.flags.append(Flag(i, "chained_narrative?", "atomic_contract",
                                    f"{hand} label not atomic: {lab!r} — atomise to one action"))
                nflag += 1
    if nflag:
        _log(wd, f"R atomic-contract: flagged {nflag} chained-narrative label(s)")


def _failing_stage(s, dir_before: str) -> str:
    """Which stage the verifier (gate) sends the rerun back to. The gate auto-applies its
    own merges/relabels within its call, and its rerun_feedback is a punch-list for the
    LABELER — so a rerun is a relabel (cuts kept) by default. The one case that warrants a
    full re-segment is a DIRECTION FLIP: the whole interpretation changed, so re-derive the
    cuts (S) then relabel. Returns 'segment' or 'label'."""
    return "segment" if s.direction != dir_before else "label"


# --------------------------------------------------------------------------- #
#  STAGE W — global consistency audit on a 3rd model family (GPT-5.5), full    #
#  whole-clip frames with labels overlaid, NO other context.                   #
# --------------------------------------------------------------------------- #
def _global_audit_gpt(s, system: str, wd: Path):
    """FINAL whole-clip pass on GPT-5.5 (3rd family): sees the clip with the assigned L/R
    labels overlaid and NOTHING else, returns a corrected timeline — catches what windowed
    opus verify shares-blindspot on (swapped hand roles, reversed destination, N/A on a
    releasing hand). Same degenerate-over-merge guard as fresh-eye; GPT is advisory-final,
    never re-graded."""
    if not s.segments:
        return
    frames = render_labeled(s.clocked, s.segments, FPS_FRESH, str(wd),
                            max_side=720, cap_frames=80)
    if not frames:
        return
    try:
        r = gpt_call(_p("fresh_eye.txt", TIMELINE=s.timeline_text()),
                     frames, system, SC.FRESH_EYE, model=GPT_MODEL, max_tokens=12000)
    except RuntimeError:
        _log(wd, "W global-audit (gpt): call failed, kept timeline")
        return
    clean = []
    for seg in r.get("segments", []):
        if not isinstance(seg, dict):
            continue
        a, b = seg.get("start_sec"), seg.get("end_sec")
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if b - a < 0.05 or a < -0.1 or b > s.duration + 0.5:
            continue
        clean.append((max(0.0, a), min(s.duration, b),
                      str(seg.get("left") or "N/A"), str(seg.get("right") or "N/A")))
    clean.sort()
    n0 = len(s.segments)
    if len(clean) < 1 or (len(clean) == 1 and s.duration > 6) \
            or (n0 >= 5 and len(clean) < 0.6 * n0):
        s.fresh_eye_note = (f"W global-audit rejected rewrite ({n0} -> {len(clean)}; "
                            f">40% drop or degenerate) — kept timeline. "
                            f"notes: {str(r.get('notes',''))[:120]}")
        _log(wd, f"W global-audit (gpt): {s.fresh_eye_note}")
        return
    rebuilt = []
    for (a, b, lft, rgt) in clean:
        sg = Segment(start=round(a, 2), end=round(b, 2), left=lft, right=rgt)
        sg.boundary_provenance = "global_audit_gpt"
        rebuilt.append(sg)
    s.segments = rebuilt
    s.fresh_eye_note = (f"W global-audit (gpt-5.5): {n0} -> {len(rebuilt)} segs. "
                        f"notes: {str(r.get('notes',''))[:160]}")
    _log(wd, f"W global-audit (gpt): {n0} -> {len(rebuilt)} segments | {str(r.get('notes',''))[:90]}")


def _annotate(video, out_path, workdir=None, max_attempts=3):
    """The pipeline for one clip — circle-grounded, no modes. The segment->verify block runs
    in a keep-best rerun loop (<=max_attempts); the verifier's feedback decides re-entry
    (relabel by default; full re-segment on a direction flip). Stages:
      S  segmentation        (pro 10fps)        -> cuts only
      S+ seg-reconcile       (pro, per long seg)-> split swallowed multi-action cycles
      A  annotation          (pro 10fps, parallel/seg) -> per-hand labels on fixed spans
      R  atomic-label contract (deterministic)  -> fuzzy same-action merge + chained flag
      S2 merge-critic        (opus, frames)     -> delete-only merge of over-split fragments
      V  verifier            (opus, frames)     -> verdict + corrections + rerun feedback
    then once on the best attempt:
      W  global audit        (GPT-5.5, frames)  -> 3rd-family whole-clip corrected timeline."""
    import copy as _copy
    t0 = time.time()
    wd = Path(workdir or f"logs/{Path(video).stem}")
    wd.mkdir(parents=True, exist_ok=True)
    system = (PROMPTS / "caption.system.txt").read_text()
    s = ClipState(video=video)
    s.duration = probe_duration(video)
    _log(wd, f"=== annotate LEAN+ {Path(video).name} ({s.duration:.1f}s) — circle-grounded, no 1A/1C ===")
    s.clocked = str(wd / "clocked.mp4")
    burn_clock(video, s.clocked)
    gv = GeminiVideo(GEMINI_NATIVE)
    gv.upload(s.clocked)
    _log(wd, "clock burned + video ACTIVE (LEAN+)")

    phase2_bursts(s, system, wd); _snap(s, "2 bursts", wd)          # for direction only
    phase3_direction(s, gv, system, wd); _snap(s, "3 direction", wd)
    facts_snaps = list(s.stage_snapshots)
    _log(wd, f"LEAN+ direction={s.direction} | goal={s.goal}")

    feedback, stage, best = "", "segment", None
    for attempt in range(1, max_attempts + 1):
        _log(wd, f"=== LEAN+ ATTEMPT {attempt}/{max_attempts} (re-enter @ {stage})"
                + (f" fb {feedback[:80]!r}" if feedback else "") + " ===")
        s.flags, s.ran, s.stage_snapshots, s.transitions = [], set(), list(facts_snaps), []
        if attempt == 1 or stage == "segment":
            _lean_segment(s, gv, system, wd); _snap(s, "S segment", wd)            # STAGE S
            _seg_reconcile(s, gv, system, wd); _snap(s, "S+ seg-reconcile", wd)    # STAGE S+ (split)
        _lean_label(s, gv, system, wd, feedback=feedback if attempt > 1 else "")   # STAGE A
        _snap(s, "A label", wd)
        _atomic_contract(s, wd); _snap(s, "R atomic-contract", wd)                 # STAGE R
        delete_only_critic(s, system, wd); _snap(s, "S2 merge-critic", wd)         # STAGE S2
        completeness_check(s, gv, system, wd); _snap(s, "PP pick<->place match", wd)  # STAGE PP
        dir_before = s.direction
        gate = phase5_gate(s, gv, system, wd); _snap(s, "V verifier", wd)          # STAGE V
        verdict = gate.get("quality_verdict", "good")
        score = (0 if verdict == "good" else 1, -len(s.segments))
        if best is None or score < best[0]:
            best = (score, _copy.deepcopy(s.segments), list(s.stage_snapshots),
                    s.gate_findings, s.purpose_verdict, attempt)
        if verdict == "good" or attempt == max_attempts:
            _log(wd, f"=== LEAN+ quality={verdict}; keep attempt {best[5]} ===")
            break
        stage = _failing_stage(s, dir_before)
        feedback = gate.get("rerun_feedback") or "Re-check segment cuts and per-hand (circle) labels."
    s.segments, s.stage_snapshots, s.gate_findings, s.purpose_verdict = best[1], best[2], best[3], best[4]

    _global_audit_gpt(s, system, wd); _snap(s, "W global-audit (gpt-5.5, final)", wd)  # STAGE W
    episode = QA.export_episode(s, out_path)
    _log(wd, f"=== DONE {time.time()-t0:.0f}s | {len(s.segments)} segments | {USAGE.summary()} ===")
    return episode


def _hand_overlay(video: str, workdir: str | None) -> str:
    """Pre-step: burn GREEN=LEFT / BLUE=RIGHT hand circles (slight spotlight) into the
    raw video at SOURCE fps via the yolo_hands handpose model, so the clocked working
    video every stage reads carries authoritative L/R hand grounding. Returns the
    overlaid path; on ANY failure falls back to the raw video so the pipeline still runs."""
    wd = Path(workdir or f"logs/{Path(video).stem}")
    wd.mkdir(parents=True, exist_ok=True)
    indir, outdir = wd / "_ov_in", wd / "_ov_out"
    indir.mkdir(exist_ok=True); outdir.mkdir(exist_ok=True)
    link = indir / Path(video).name
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(video).resolve())          # isolate one video for --videos-dir
        cmd = [sys.executable, str(_YOLO / "detect_hands.py"),
               "--handpose", "--model", str(_HAND_MODEL),
               "--shape", "circle", "--circle-expand", "60", "--dim", "0.6",
               "--detect-mult", "1", "--hold-sec", "0", "--radius-cap", "0.15",
               "--native-fps", "--fps", "10", "--workers", "1",
               "--videos-dir", str(indir), "--out-dir", str(outdir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        produced = outdir / f"{Path(video).stem}_hands.mp4"
        if r.returncode == 0 and produced.exists():
            _log(wd, f"hand overlay (green=L / blue=R circles, slight spotlight) -> {produced.name}")
            return str(produced)
        _log(wd, f"WARNING hand overlay failed rc={r.returncode}; using raw video :: {(r.stderr or '')[-300:]}")
    except Exception as e:                                # noqa: BLE001
        _log(wd, f"WARNING hand overlay error: {e}; using raw video")
    return video


def annotate(video: str, out_path: str, workdir: str | None = None,
             max_attempts: int = 3) -> dict:
    """Public entry — ONE pipeline, no modes/flags:
      raw video -> YOLO L/R-circle overlay -> clock burn -> bursts + direction
      -> [ S segment -> S+ seg-reconcile -> A label -> R atomic-contract -> S2 merge-critic
           -> V verify ] looped <=max_attempts (keep best; verifier feedback picks re-entry)
      -> W global audit (GPT-5.5) -> export_episode.
    Long clips (> CHUNK_SEC) are split into even <= CHUNK_SEC parts, each run through the
    SAME pipeline in parallel, then merged and every seam re-flowed across the boundary."""
    video = _hand_overlay(video, workdir)                # burn L/R circles -> all stages see them
    dur = probe_duration(video)
    if dur <= CHUNK_SEC:                                 # <=5min: one pass; longer -> split
        return _annotate(video, out_path, workdir, max_attempts)

    wd = Path(workdir or f"logs/{Path(video).stem}")
    wd.mkdir(parents=True, exist_ok=True)
    n = max(1, math.ceil(dur / CHUNK_SEC))
    clen = dur / n
    _log(wd, f"=== annotate {Path(video).name} ({dur:.1f}s) -> CHUNKED into {n} parts "
            f"of {clen:.1f}s each ===")
    specs = []
    for i in range(n):
        a = i * clen
        part_len = min(clen, dur - a)
        part_video = str(wd / f"part{i + 1}.mp4")
        _extract_clip(video, a, part_len, part_video)
        specs.append((i, a, part_video, part_len))
        _log(wd, f"-- part {i + 1}/{n}: [{a:.1f}-{a + part_len:.1f}s] -> {Path(part_video).name}")

    def _do_part(spec):
        i, a, pv, _plen = spec
        ep = _annotate(pv, str(wd / f"part{i + 1}.json"),
                       str(wd / f"part{i + 1}"), max_attempts)
        return (a, ep)

    _log(wd, f"-- annotating {n} parts IN PARALLEL")
    results = _parallel(_do_part, specs, levels=(n, max(1, n // 2), 1), wd=wd, tag="chunk parts")
    parts = sorted([r for r in results if r], key=lambda p: p[0])
    if not parts:
        raise RuntimeError("all chunk parts failed")
    if len(parts) < n:
        _log(wd, f"WARNING: {n - len(parts)} of {n} parts failed -> merging the {len(parts)} that succeeded")
    merged = _merge_chunk_episodes(video, out_path, dur, parts)
    merged = _repair_seams(video, out_path, parts, merged, wd, max_attempts)
    _log(wd, f"=== DONE (chunked) {len(merged['segments'])} segments across {n} parts "
            f"({n - 1} seams re-flowed) -> {out_path} ===")
    return merged


