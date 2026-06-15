#!/usr/bin/env python3
"""pipeline.py — the facts-first orchestrator.

One ClipState flows through five phases; every stage reads a COMPACTED view and
writes back. Spine = the 10fps object+hand-contact track (the ground truth).

  P1  facts        contact_track (Gemini 10fps: objects + fg/bg + per-hand contact)
                   -> transition scan (Claude opus, sliding 30fps windows)
  P2  evidence     burst sweep (Gemini 30fps) -> burst_reduce (deterministic)
  P3  direction    direction_decide (Gemini 10fps whole) — also derives the goal
  P4  segment+label v49_segment (Gemini 10fps; sole cut authority) -> per-seg label
                    (Gemini native 10fps) -> merge -> seg_reconcile -> verifiers [<=2 passes]
  P5  gate+export  gate (Claude opus-4-8 2fps whole) -> deterministic QA -> export
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import analysis as AN
import bursts as B
import qa as QA
import schemas as SC
from clipstate import (ClipState, Flag, Segment, build_track,
                       derive_track_from_labels, track_possession_changes)
from media import burn_clock, probe_duration, render_labeled, render_strip
from models import (CLAUDE_GATE, GEMINI_NATIVE, USAGE,
                    GeminiFrames, GeminiVideo, claude_call)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

# frame budgets (facts-first; refine-heavy then label-heavy)
FPS_CONTACT = 10.0
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
        rows = " | ".join(f"#{i+1} {x['left']}/{x['right']}" for i, x in enumerate(segs))
        _log(wd, f"TRACE [{stage}] {len(segs)} segs: {rows}")


# =========================================================================== #
#  PHASE 1 — the fact layer                                                    #
# =========================================================================== #
def phase1_contact(s: ClipState, gv: GeminiVideo, system: str, wd: Path):
    objs: dict[str, dict] = {}
    frames: list[dict] = []
    wins = plan_windows(s.duration, CONTACT_WIN)
    _log(wd, f"P1a contact_track: {len(wins)} window(s) @ {FPS_CONTACT}fps")

    def _one(win):
        a, b = win
        r = gv.watch(_p("contact_track.txt"), system, SC.CONTACT_TRACK,
                     a=a, b=b, fps=FPS_CONTACT, max_tokens=24000)
        return r

    with cf.ThreadPoolExecutor(max_workers=min(4, len(wins))) as ex:
        results = list(ex.map(_one, wins))
    for r in results:
        for o in r.get("objects", []):
            nm = str(o.get("name", "")).strip()
            if nm and nm.lower() not in {k.lower() for k in objs}:
                objs[nm] = o
        for fr in r.get("frames", []):
            if fr.get("t") is not None:
                frames.append(fr)
    frames.sort(key=lambda f: f["t"])
    s.objects = list(objs.values())
    s.contact_frames = frames
    s.track = build_track(frames)
    nL, nR = len(s.track["left"]), len(s.track["right"])
    _log(wd, f"P1a -> {len(s.objects)} objects, {len(frames)} contact frames, "
            f"track L={nL} R={nR} intervals")


TRANS_WIN = 3.5           # sliding window length (s)
TRANS_STRIDE = 2.0        # window stride (s)


def phase1_transitions(s: ClipState, system: str, wd: Path, hint: str = ""):
    """Detect place/pickup/handoff by SLIDING a short window across the clip and asking
    Claude opus, per window, ONLY 'is anything set down / picked up / handed off here?'.
    Focused attention is where the model actually perceives these brief events (the
    whole-clip pass and the contact track both miss them). Events dedup within 0.6s."""
    wins = []
    t = 0.0
    while t < s.duration - 0.5:
        wins.append((round(t, 2), round(min(t + TRANS_WIN, s.duration), 2)))
        t += TRANS_STRIDE
    _log(wd, f"P1c transition scan: {len(wins)} windows x {TRANS_WIN}s @ {FPS_EDGE}fps (opus)")

    def _one(win):
        wa, wb = win
        frames, _ = render_strip(s.clocked, wa, wb, FPS_EDGE, s.track, str(wd),
                                 ctx=0.0, cap_frames=36, max_side=720)
        prompt = _p("transition_window.txt", WA=round(wa, 1), WB=round(wb, 1))
        if hint:
            prompt += f"\n\nReviewer hint about likely missed transitions: {hint}"
        r = claude_call(prompt, frames, system, SC.WINDOW_TRANSITIONS,
                        model=CLAUDE_GATE, max_tokens=1500)   # raises -> _parallel retries
        out = []
        for e in r.get("events", []):
            tt = e.get("t")
            if tt is not None and wa - 0.05 <= float(tt) <= wb + 0.05:
                out.append(e)
        return out

    allev = []
    for evs in _parallel(_one, wins, wd=wd, tag="P1c transition scan"):
        allev.extend(evs or [])                            # None (failed all levels) -> skip
    # Keep the initial pick and final place (the user wants them). Only drop edge HAND-OFFS
    # (a hand-off at t=0 is the spurious "receive/give" artifact); keep place/pickup at edges.
    def _edge_ok(e):
        t = float(e["t"])
        if not (0.1 < t < s.duration - 0.1):
            return False
        if e.get("kind") == "handoff" and (t < 0.5 or t > s.duration - 0.5):
            return False
        return True
    allev = [e for e in allev if e.get("t") is not None and _edge_ok(e)]
    # dedup: collapse same-kind events within 1.2s into one (kills redundant pick/pick
    # clusters that produced duplicate "pick up" segments); keep distinct kinds apart.
    allev.sort(key=lambda e: float(e["t"]))
    merged = []
    for e in allev:
        if merged and abs(float(e["t"]) - float(merged[-1]["t"])) < 1.2 \
                and e.get("kind") == merged[-1].get("kind"):
            continue
        merged.append(e)
    s.transitions = merged
    kinds = {}
    for e in merged:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
    _log(wd, f"P1c -> {len(merged)} transition events {kinds} "
            f"at {[round(float(e['t']),1) for e in merged]}")


# =========================================================================== #
#  PHASE 2 — bursts                                                            #
# =========================================================================== #
def phase2_bursts(s: ClipState, system: str, wd: Path):
    reqs = []
    # seed 2-3 direction probes across the clip (where rotation shows). (No cycle stage:
    # it was vestigial/unstable and never drove anything — removed.)
    spans = plan_windows(s.duration, max(4.0, s.duration / 4))   # already (a,b) pairs
    qdir = ("Watching ONLY the visible rotation/motion of the part being worked, is "
            "this moment ASSEMBLY (tightening/screwing-on/joining/inserting) or "
            "DISASSEMBLY (loosening/unscrewing/removing/separating)? Verdict must be "
            "exactly 'assembly', 'disassembly', or 'cannot determine'.")
    for (a, b) in spans[:3]:
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
    if s.direction in ("other_or_ambiguous", "") and summ.get("rule") == "single_direction_majority" \
            and summ.get("majority") in ("assembly", "disassembly"):
        _log(wd, f"P3 direction was '{s.direction}' but bursts commit "
                f"{summ['majority']} (single-direction) -> override")
        s.direction = summ["majority"]
    _log(wd, f"P3 direction={s.direction} switch={s.phase_switch_sec} | goal: {s.goal}")


# =========================================================================== #
#  PHASE 4 — rough -> refine                                                   #
# =========================================================================== #
def _segs_from_bounds(bounds: list[float], dur: float, prov: str) -> list[Segment]:
    """Build contiguous segments from cut times. A span shorter than MIN_SEG is NOT
    dropped (that would leave a hole) — its cut is removed so the span fuses with the
    previous one. The timeline always covers [0, dur] with no gaps."""
    cuts = sorted(set(round(x, 2) for x in bounds if 0.2 < x < dur - 0.2))
    pts = [0.0] + cuts + [round(dur, 2)]
    kept = [pts[0]]
    for t in pts[1:-1]:
        if t - kept[-1] >= MIN_SEG:
            kept.append(t)
    kept.append(pts[-1])
    if len(kept) >= 3 and kept[-1] - kept[-2] < MIN_SEG:
        kept.pop(-2)                                 # fuse a too-short final span back
    out = [Segment(start=a, end=b, boundary_provenance=prov, confidence=0.6)
           for a, b in _spans(kept)]
    return out or [Segment(start=0.0, end=dur)]


def _transitions_in(s: ClipState, a: float, b: float) -> list[dict]:
    return [e for e in s.transitions
            if e.get("t") is not None and a - 0.3 <= float(e["t"]) <= b + 0.3]


def _tag_transitions(s: ClipState):
    """Tag each segment with the detected transition(s) inside it, so the labeler hints
    place/pickup (with N/A on the idle hand) and the mergers protect that boundary. Must
    be re-run after seg_reconcile splits, since new sub-spans start untagged."""
    for seg in s.segments:
        evs = _transitions_in(s, seg.start, seg.end)
        if evs:
            seg.draft = {**seg.draft, "transitions": evs}


def v49_segment(s: ClipState, gv: GeminiVideo, system: str, wd: Path) -> list[float]:
    """SEGMENTATION (v49, ported): one native whole-clip pass with v49's battle-tested
    atomic-action cut rules (one complete sub-goal per segment, anti-swallowing guards,
    conservative 2-8s). This is the SOLE cut authority: it returns every boundary. The
    Phase-1c place/pickup/handoff transitions are handed in only as SOFT HINTS the
    segmenter confirms against its own 10fps read — they are never pre-injected as cuts."""
    # Detected place/pickup/handoff times are passed as SOFT HINTS only — the segmenter
    # confirms each against its own 10fps read and decides. We do NOT pre-inject them as
    # hard cuts: ALL cutting happens here, in segmentation.
    thint = ("; ".join(f"{round(float(e['t']),1)}s {e.get('kind','?')} "
                       f"({e.get('hand','?')})" for e in s.transitions)
             or "(none detected)")
    r = gv.watch(_p("v49_segment.txt", A=0.0, B=round(s.duration, 1),
                    GOAL=s.goal or "(unknown)", INVENTORY=s.objects_line(),
                    TRACK=s.track_lines(),
                    TRANSITION_HINTS=thint,
                    BURSTS="\n".join(s.bursts_reduced) or "(none)"),
                 system, SC.V49_SEGMENT, a=0.0, b=s.duration, fps=FPS_SEGMENT,
                 max_tokens=4000)
    cuts = sorted(float(x) for x in r.get("boundaries", [])
                  if x is not None and 0.3 < float(x) < s.duration - 0.3)
    _log(wd, f"P4 v49 segmentation: {len(cuts)} cuts (segmenter-only, no preliminary union)")
    return cuts


def _norm_lbl(x: str) -> str:
    return " ".join(str(x or "N/A").lower().split())


def _label_sim(a: str, b: str) -> float:
    """Generic word-overlap (Jaccard) between two labels — NOT a semantic wordlist, just
    token overlap. ~1.0 = identical wording; high = same action worded slightly
    differently ('thread the nut' vs 'tighten the nut')."""
    wa, wb = set(_norm_lbl(a).split()), set(_norm_lbl(b).split())
    if not wa or not wb:
        return 1.0 if wa == wb else 0.0
    return len(wa & wb) / len(wa | wb)


def _labels_mergeable(p_l, p_r, s_l, s_r, thresh: float = 0.6) -> bool:
    """Two adjacent segments are mergeable when BOTH hands read as the SAME action —
    exact OR highly similar wording (token overlap >= thresh). An idle hand (N/A) must
    match N/A exactly: idle vs active is never 'similar'."""
    def one(a, b):
        na, nb = _norm_lbl(a), _norm_lbl(b)
        if na == nb:
            return True
        if na == "n/a" or nb == "n/a":
            return False
        return _label_sim(a, b) >= thresh
    return one(p_l, s_l) and one(p_r, s_r)


def merge_identical_labels(s: ClipState, wd: Path):
    """Deterministically merge consecutive segments whose (left,right) labels read as the
    SAME action — exact OR near-identical wording (generic token overlap, no wordlist), so
    'thread the nut' and 'tighten the nut' collapse. A boundary that coincides with a
    DETECTED place/pickup/handoff is PROTECTED: never merged away (the transition makes it
    a real action change); an idle hand (N/A) must match N/A exactly."""
    if not s.segments:
        return
    protect = [float(e["t"]) for e in s.transitions if e.get("t") is not None]
    merged = [s.segments[0]]
    for seg in s.segments[1:]:
        prev = merged[-1]
        guarded = any(abs(seg.start - t) <= 0.5 for t in protect)
        if (not guarded
                and _labels_mergeable(prev.left, prev.right, seg.left, seg.right)):
            prev.end = seg.end                       # extend the run
            prev.draft = {**prev.draft, "merged_run": True}
        else:
            merged.append(seg)
    n0 = len(s.segments)
    s.segments = merged
    if n0 != len(merged):
        _log(wd, f"P4 merge-by-label: {n0} -> {len(merged)} segments")


def _neighbors(segs, i):
    prev = segs[i - 1] if i > 0 else None
    nxt = segs[i + 1] if i + 1 < len(segs) else None
    return prev, nxt


def label_segments(s: ClipState, gv: GeminiVideo, system: str, wd: Path,
                   only_unlabeled=True):
    """Label each segment SEPARATELY with Gemini NATIVE at its exact place: the call
    references the uploaded video with start/end offsets (segment +-1s context) at
    10fps, so Gemini decodes that window itself. Each call is isolated — it sees only
    its window, the window-scoped track/contact-facts/bursts, and the two neighbor
    labels as text."""
    todo = [i for i, seg in enumerate(s.segments)
            if (not only_unlabeled) or not seg.draft.get("labeled")]
    if not todo:
        return
    _log(wd, f"P4b label {len(todo)} segment(s) NATIVE @ {FPS_LABEL}fps (gemini-pro)")

    def _one(i):
        seg = s.segments[i]
        oa, ob = max(0.0, seg.start - LABEL_CTX), min(s.duration, seg.end + LABEL_CTX)
        prev, nxt = _neighbors(s.segments, i)
        facts = s.contact_facts_in(seg.start, seg.end)
        track = s.track_lines(seg.start, seg.end)
        bursts = "\n".join(s.bursts_in(seg.start, seg.end)) or "(none in window)"
        prompt = _p("video_label.txt",
                    OA=round(oa, 1), OB=round(ob, 1), A=round(seg.start, 1),
                    B=round(seg.end, 1), GOAL=s.goal, DIRECTION=s.direction,
                    INVENTORY=s.objects_line(), TRACK=track, CONTACT_FACTS=facts,
                    BURSTS=bursts,
                    PREV_LEFT=(prev.left if prev else "(start)"),
                    PREV_RIGHT=(prev.right if prev else "(start)"),
                    NEXT_LEFT=(nxt.left if nxt else "(end)"),
                    NEXT_RIGHT=(nxt.right if nxt else "(end)"))
        trans = seg.draft.get("transitions") if isinstance(seg.draft, dict) else None
        hint = ""
        if trans:
            hint = "; ".join(f"{e['kind']} of {e.get('object','the object')} by the "
                             f"{e['hand']} hand at t={float(e['t']):.1f}s" for e in trans)
            prompt += ("\n\nDETECTED TRANSITION(S) in this segment, from a dense per-frame "
                       f"scan (strong evidence): {hint}. If the frames show it, your label "
                       "MUST reflect it — for a PLACE: the placing hand = 'place the "
                       "<object>' (name the destination if visible) and the OTHER hand = "
                       "N/A; for a PICKUP: the picking hand = 'pick up the <object>' and "
                       "the other = N/A; for a HANDOFF: name give/receive. Do NOT label "
                       "ongoing twisting/threading for a hand that is releasing or "
                       "acquiring in this window.")
        out = gv.watch(prompt, system, SC.VIDEO_LABEL_NATIVE, a=oa, b=ob,
                       fps=FPS_LABEL, max_tokens=2000)
        return i, out, {"window": f"{round(oa,1)}-{round(ob,1)}s @ {FPS_LABEL}fps",
                        "facts": facts, "track": track, "bursts": bursts,
                        "transition_hint": hint or "(none)"}

    for r in _parallel(_one, todo, wd=wd, tag="P4b label"):
        if r is None:                                  # all worker-levels failed for it
            continue
        i, out, dbg = r
        seg = s.segments[i]
        seg.left = out.get("left", "N/A") or "N/A"
        seg.right = out.get("right", "N/A") or "N/A"
        seg.draft = {"labeled": True, "label_think": out.get("think", ""),
                     "label": {"left": seg.left, "right": seg.right},
                     "origin": seg.boundary_provenance,
                     "transitions": (seg.draft.get("transitions")
                                     if isinstance(seg.draft, dict) else None),
                     "label_in": dbg}
        _log(wd, f"  P4b seg#{i+1} [{seg.start:.1f}-{seg.end:.1f}] -> "
                f"L:{seg.left!r} R:{seg.right!r} | hint:{dbg['transition_hint']}")


def seg_reconcile_pass(s: ClipState, gv: GeminiVideo, system: str, wd: Path,
                       min_len: float = 1.6) -> bool:
    """v49 seg_reconcile (ported): per-segment temporal-structure audit. For each segment
    longer than min_len, a focused native pass over [a,b] decides whether it is ONE action
    or several, returning the onset of each action change (especially the put-down that
    completes a cycle). Segments are split at those onsets; new sub-spans go back to the
    labeler. Replaces our edge verifier. Returns True if the segment set changed."""
    def _one(idx_seg):
        i, seg = idx_seg
        if seg.end - seg.start < min_len:
            return i, []
        nfr = int(round((seg.end - seg.start) * FPS_SEGMENT))
        prompt = _p("seg_reconcile.txt", N=nfr, A=round(seg.start, 1), B=round(seg.end, 1),
                    GOAL=s.goal or "(unknown)", LEFT=seg.left, RIGHT=seg.right)
        r = gv.watch(prompt, system, SC.SEG_RECONCILE, a=seg.start, b=seg.end,
                     fps=FPS_SEGMENT, max_tokens=1500)      # raises -> _parallel retries
        cuts = sorted(float(x) for x in r.get("boundaries", [])
                      if x is not None and seg.start + 0.3 < float(x) < seg.end - 0.3)
        return i, cuts

    results = {}
    for r in _parallel(_one, list(enumerate(s.segments)), wd=wd, tag="P4c seg_reconcile"):
        if r is None:
            continue
        i, cuts = r
        if cuts:
            results[i] = cuts
    if not results:
        _log(wd, "P4c seg_reconcile: no splits")
        return False
    # rebuild the segment list, inserting the sub-cuts (new sub-spans unlabeled)
    new_segs = []
    for i, seg in enumerate(s.segments):
        if i not in results:
            new_segs.append(seg)
            continue
        pts = [seg.start] + results[i] + [seg.end]
        for a, b in _spans(pts):
            if b - a < 0.05:
                continue
            if abs(a - seg.start) < 1e-6:             # first sub-span keeps the label
                new_segs.append(Segment(start=round(a, 2), end=round(b, 2),
                                        left=seg.left, right=seg.right,
                                        boundary_provenance="seg_reconcile",
                                        confidence=0.7, draft=seg.draft))
            else:                                     # new sub-spans need labeling
                new_segs.append(Segment(start=round(a, 2), end=round(b, 2),
                                        boundary_provenance="seg_reconcile", confidence=0.7))
    n0 = len(s.segments)
    s.segments = new_segs
    _log(wd, f"P4c seg_reconcile: split {len(results)} stretch(es) -> {n0} -> {len(s.segments)} segments")
    return True


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


def neighbor_review(s: ClipState, gframes_pro: GeminiFrames, wd: Path):
    """v49 neighbor-context reviewer (text only, no video): reads the label sequence and
    flags a segment whose label looks wrong vs its neighbours — adjacent duplicates,
    object/colour mismatch, role flips, monotone runs, stuck holds, missed phases. Cheap;
    its flags route to the gate for frame-level re-check."""
    r = gframes_pro.call(_p("neighbor_review.txt", GOAL=s.goal or "(unknown)",
                            DIRECTION=s.direction or "(unknown)",
                            TIMELINE=s.timeline_text()),
                         [], schema=SC.NEIGHBOR_REVIEW, max_tokens=2000, reasoning="low")
    for fl in r.get("flags", []):
        seg = int(fl.get("seg", 0)) - 1
        if 0 <= seg < len(s.segments):
            s.flags.append(Flag(seg, str(fl.get("issue", "neighbor")), "neighbor_review",
                                str(fl.get("reason", ""))[:120]))
    _log(wd, f"P4 neighbor_review: +{len(r.get('flags', []))} flags "
            f"{[f.get('issue') for f in r.get('flags', [])]}")


def template_match(s: ClipState, gframes_pro: GeminiFrames, wd: Path):
    prompt = _p("template_match.txt", DIRECTION=s.direction, GOAL=s.goal,
                TRACK=s.track_lines(), TIMELINE=s.timeline_text())
    r = gframes_pro.call(prompt, [], schema=SC.TEMPLATE_MATCH, max_tokens=2000,
                         reasoning="low")
    for fl in r.get("flags", []):
        seg = int(fl.get("seg", 0)) - 1
        if 0 <= seg < len(s.segments):
            s.flags.append(Flag(seg, str(fl.get("type", "deviation")),
                                "template_match", str(fl.get("detail", ""))[:120]))
    _log(wd, f"P4d template_match: +{len(r.get('flags', []))} flags; notes: "
            f"{str(r.get('notes',''))[:120]}")


def phase4(s: ClipState, gv: GeminiVideo, gframes_pro: GeminiFrames,
           system: str, wd: Path, max_passes=2, feedback: str = ""):
    # SEGMENTATION — v49 single-pass cut rules (our proprietary structure keeps the rest).
    cuts = v49_segment(s, gv, system, wd)
    s.segments = _segs_from_bounds(cuts, s.duration, "v49_segment")
    _tag_transitions(s)                               # tag for labeler-hint + merge-protect
    _log(wd, f"P4 segmentation -> {len(s.segments)} segments")
    # ANNOTATION — our per-segment focused labeler (one segment at a time, native 10fps).
    label_segments(s, gv, system, wd, only_unlabeled=True)
    merge_identical_labels(s, wd)                     # collapse identical runs
    s.track = derive_track_from_labels(s.segments)    # track mirrors the labels
    for k in range(max_passes):
        changed = seg_reconcile_pass(s, gv, system, wd)  # per-stretch split audit (v49)
        _tag_transitions(s)                           # re-tag new sub-spans so the labeler hints place/pickup
        label_segments(s, gv, system, wd, only_unlabeled=True)   # label any new spans
        merge_identical_labels(s, wd)
        s.track = derive_track_from_labels(s.segments)
        if not changed:
            break
        _log(wd, f"P4 refine pass {k+1} changed the segment set -> another pass")
    template_match(s, gframes_pro, wd)
    neighbor_review(s, gframes_pro, wd)               # v49 label-sequence consistency
    rep, afl = AN.analyze(s)                          # deterministic code analysis (advisory)
    s.analysis_report = rep
    s.flags.extend(afl)
    _log(wd, f"P4 code-analysis: +{len(afl)} advisory flags "
            f"{[f.type for f in afl]}")


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
                       ANALYSIS=s.analysis_report or ""),
                    frames, system, SC.GATE, model=CLAUDE_GATE)
    s.gate_findings = str(r.get("findings", ""))     # full findings (no truncation)
    s.purpose_verdict = r.get("purpose_verdict", "")
    if r.get("purpose_check"):
        s.goal = r["purpose_check"] if s.purpose_verdict == "corrected" else s.goal
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
    for sr in r.get("split_request", []):
        idx = int(sr) - 1
        if 0 <= idx < len(s.segments):
            s.flags.append(Flag(idx, "gate_split_request", "gate",
                                "gate asked for a split; routed to review"))
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
def fresh_eye(s: ClipState, system: str, wd: Path):
    """FINAL pass: show opus the whole clip with the assigned L/R labels overlaid and
    NOTHING else (no flags, no track, no analysis, no bursts) — fresh eyes re-read the
    pixels and return a corrected timeline so the labels match the video and the
    sequence makes sense. Last word before export; degenerate rewrites are rejected."""
    if not s.segments:
        return
    frames = render_labeled(s.clocked, s.segments, FPS_FRESH, str(wd),
                            max_side=720, cap_frames=80)
    if not frames:
        return
    try:
        r = claude_call(_p("fresh_eye.txt", TIMELINE=s.timeline_text()),
                        frames, system, SC.FRESH_EYE, model=CLAUDE_GATE, max_tokens=4000)
    except RuntimeError:
        _log(wd, "P6 fresh-eye: call failed, kept timeline")
        return
    clean = []
    for seg in r.get("segments", []):
        a, b = seg.get("start_sec"), seg.get("end_sec")
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if b - a < 0.05 or a < -0.1 or b > s.duration + 0.5:
            continue
        clean.append((max(0.0, a), min(s.duration, b),
                      str(seg.get("left") or "N/A"), str(seg.get("right") or "N/A")))
    clean.sort()
    if len(clean) < 1 or (len(clean) == 1 and s.duration > 6):
        s.fresh_eye_note = (f"rejected degenerate rewrite ({len(clean)} segs) — kept "
                            f"timeline. notes: {str(r.get('notes',''))[:120]}")
        _log(wd, f"P6 fresh-eye: {s.fresh_eye_note}")
        return
    rebuilt = []
    for (a, b, lft, rgt) in clean:
        sg = Segment(start=round(a, 2), end=round(b, 2), left=lft, right=rgt)
        sg.boundary_provenance = "fresh_eye"
        rebuilt.append(sg)
    n0 = len(s.segments)
    s.segments = rebuilt
    s.track = derive_track_from_labels(s.segments)
    s.fresh_eye_note = (f"reads_correctly={r.get('reads_correctly')}; {n0} -> "
                        f"{len(rebuilt)} segs. {str(r.get('notes',''))[:160]}")
    _log(wd, f"P6 fresh-eye: {s.fresh_eye_note}")


# =========================================================================== #
#  TOP-LEVEL                                                                   #
# =========================================================================== #
def annotate(video: str, out_path: str, workdir: str | None = None,
             max_passes: int = 2, max_attempts: int = 3,
             apply_overrides: bool = False) -> dict:
    t0 = time.time()
    wd = Path(workdir or f"logs/{Path(video).stem}")
    wd.mkdir(parents=True, exist_ok=True)
    system = (PROMPTS / "caption.system.txt").read_text()

    s = ClipState(video=video)
    s.duration = probe_duration(video)
    _log(wd, f"=== annotate {Path(video).name} ({s.duration:.1f}s) ===")

    s.clocked = str(wd / "clocked.mp4")
    burn_clock(video, s.clocked)
    _log(wd, "clock burned (audio off) -> upload to Gemini File API")

    gv = GeminiVideo(GEMINI_NATIVE)              # gemini-3.1-pro: facts + native label
    gv.upload(s.clocked)
    gframes_pro = GeminiFrames(GEMINI_NATIVE)    # pro frames: template_match (text only)
    _log(wd, "video ACTIVE")                     # bursts/edge/gate = Claude opus (no flash)

    # ---- FACTS (computed ONCE, cached across attempts) ----
    phase1_contact(s, gv, system, wd)
    phase1_transitions(s, system, wd)
    phase2_bursts(s, system, wd)
    phase3_direction(s, gv, system, wd)

    # ---- BOUNDED QUALITY LOOP over Phase 4-5 (max_attempts total) ----
    # Keep the BEST attempt, not the last: a strict gate that never says "good" must not
    # leave us with a worse final run. Score: gate-good beats not-good, then fewest flags,
    # then more segments (more granular). Lower score tuple = better.
    import copy as _copy
    feedback, best = "", None
    for attempt in range(1, max_attempts + 1):
        _log(wd, f"=== ATTEMPT {attempt}/{max_attempts}"
                + (f" (feedback: {feedback[:90]!r})" if feedback else "") + " ===")
        if attempt > 1 and feedback:
            phase1_transitions(s, system, wd, hint=feedback)   # focused re-scan
        s.segments, s.flags, s.ran = [], [], set()    # fresh per attempt (critics run once per timeline)
        phase4(s, gv, gframes_pro, system, wd, max_passes=max_passes, feedback=feedback)
        gate = phase5_gate(s, gv, system, wd)
        verdict = gate.get("quality_verdict", "good")
        score = (0 if verdict == "good" else 1, len(s.flags), -len(s.segments))
        if best is None or score < best[0]:
            best = (score, _copy.deepcopy(s.segments), list(s.flags),
                    _copy.deepcopy(s.track), attempt)
        if verdict == "good" or attempt == max_attempts:
            _log(wd, f"=== quality={verdict} after attempt {attempt}; "
                    f"best=attempt {best[4]} (score {best[0]}) -> accept ===")
            break
        feedback = (gate.get("rerun_feedback") or
                    "Re-check for missed place/pickup/handoff boundaries.")
    # restore the best attempt (clear the per-attempt guard so the final critics run on it)
    s.segments, s.flags, s.track, s.ran = best[1], best[2], best[3], set()
    s.stage_snapshots = []                            # trace the EXPORTED lineage only
    _snap(s, f"P4+P5 accepted (attempt {best[4]}: labeler + gate)", wd)


    # ---- FINAL polish + audit, ONCE on the accepted timeline (before export) ----
    delete_only_critic(s, system, wd)                 # merge any over-split runs
    s.track = derive_track_from_labels(s.segments)
    _snap(s, "P4d delete-only critic", wd)
    added = completeness_check(s, gv, system, wd)     # folds missing pick/place into s.transitions
    if added:                                         # turn those findings into real segments
        _log(wd, f"=== completeness added {added} missing transition(s) -> final round (broad->refine) ===")
        s.ran.discard("merge_critic")                 # let the critic clean the new timeline
        phase4(s, gv, gframes_pro, system, wd, max_passes=1)   # re-segment+label with augmented transitions
        delete_only_critic(s, system, wd)
        s.track = derive_track_from_labels(s.segments)
        _snap(s, "P4e completeness re-segment", wd)

    # ---- PHASE 6: fresh-eye final review (context-free), the last word ----
    fresh_eye(s, system, wd)
    _snap(s, "P6 fresh-eye (final, exported)", wd)

    episode = QA.export_episode(s, out_path, apply_overrides=apply_overrides)
    _log(wd, f"=== DONE {time.time()-t0:.0f}s | {len(s.segments)} segments, "
            f"{len(episode['_qa']['violations'])} qa flags | usage: {USAGE.summary()} ===")
    _log(wd, f"episode -> {out_path}")
    return episode
