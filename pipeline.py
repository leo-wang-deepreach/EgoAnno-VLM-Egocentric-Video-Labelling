#!/usr/bin/env python3
"""pipeline.py — the facts-first orchestrator.

ONE timeline, progressively refined: each stage reads it and rewrites it in place.
Facts are computed once; the dense 0.1s contact facts ARE the starting timeline.

  FACTS (once):
    1A contact_track  (Gemini 10fps) objects + fg/bg + per-hand contact  -> SEED the timeline
    1C transitions    (Claude opus, sliding 30fps windows) place/pickup/throw/handoff
    2  bursts         (Gemini 30fps) rotation/role/colour -> deterministic reduce
    3  direction      (Gemini 10fps whole) direction + derived goal
  REFINE the one timeline (retry loop, keep best by gate verdict; then fresh-eye once):
    4  label+collapse (Gemini 10fps) dense facts -> per-hand action sentences (bottom-up)
    5  verifier       (opus, text) refine_timeline -> model-judged corrected timeline
    6  code-analysis  (analysis.py) READ-ONLY advisory signals (no timeline change)
    7  gate + QA      (opus, frames) edits timeline + quality verdict -> deterministic QA
    8  fresh-eye      (opus) context-free review of clip w/ labels overlaid -> final edit
  export -> episode.json (+ _stages trace, _qa) -> dashboard

Retired (v49 top-down path) lives under unused/.
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


# --------------------------------------------------------------------------- #
#  THE TIMELINE — one structure, progressively refined                        #
# --------------------------------------------------------------------------- #
def _contact_summary(s: ClipState) -> str:
    """The RELIABLE 1A possession, collapsed to per-hand object intervals (the authority
    for who-holds-what / who-is-idle), as compact text for the timeline stages."""
    tr = build_track(s.contact_frames)
    out = []
    for hand, H in (("left", "L"), ("right", "R")):
        ivs = tr.get(hand, [])
        parts = [f"{iv.get('interacting_with', '?')} [{iv['start_sec']:.1f}-{iv['end_sec']:.1f}]"
                 for iv in ivs]
        out.append(f"{H}: " + ("; ".join(parts) if parts else "empty/absent"))
    return "\n".join(out)


def _transitions_text(s: ClipState) -> str:
    ev = [e for e in s.transitions if e.get("t") is not None]
    if not ev:
        return "(none detected)"
    return "\n".join(f"{float(e['t']):.1f}s {e.get('hand', '?')} {e.get('kind', '?')} "
                     f"of {e.get('object', '?')}" for e in ev)


def _apply_timeline(s: ClipState, segs_raw, prov: str, wd: Path, label: str) -> bool:
    """Validate a model-returned timeline and make it THE timeline (in place). Rejects an
    empty/degenerate result (keeps the previous timeline). Returns True if applied."""
    clean = []
    for seg in segs_raw or []:
        a, b = seg.get("start_sec"), seg.get("end_sec")
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if b - a < 0.05 or a < -0.1 or b > s.duration + 0.5:
            continue
        clean.append((max(0.0, a), min(s.duration, b),
                      str(seg.get("left") or "N/A"), str(seg.get("right") or "N/A")))
    clean.sort()
    if not clean or (len(clean) == 1 and s.duration > 6):
        _log(wd, f"{label}: degenerate timeline ({len(clean)} segs) -> kept previous")
        return False
    rebuilt = []
    for a, b, lft, rgt in clean:
        sg = Segment(start=round(a, 2), end=round(b, 2), left=lft, right=rgt)
        sg.boundary_provenance = prov
        rebuilt.append(sg)
    s.segments = rebuilt
    s.track = derive_track_from_labels(s.segments)
    return True


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


def label_and_collapse(s: ClipState, gv: GeminiVideo, system: str, wd: Path):
    """STEP 4: watch the clip at 10fps + the reliable dense facts, and lay out the per-hand
    action timeline — cut at EVERY action change, collapse only truly-steady runs. Detected
    transitions are REQUIRED cuts (a transition is never swallowed)."""
    cut_times = sorted({round(float(e["t"]), 2) for e in s.transitions
                        if e.get("t") is not None and 0.3 < float(e["t"]) < s.duration - 0.3})
    prompt = _p("label_collapse.txt", B=round(s.duration, 1),
                DIRECTION=s.direction or "(unknown)", GOAL=s.goal or "(unknown)",
                INVENTORY=s.objects_line(), CONTACT=_contact_summary(s),
                TRANSITIONS=_transitions_text(s),
                REQUIRED_CUTS=(", ".join(f"{t:.1f}s" for t in cut_times) or "(none)"),
                BURSTS="\n".join(s.bursts_reduced) or "(none)")
    r = gv.watch(prompt, system, SC.TIMELINE, a=0.0, b=s.duration,
                 fps=FPS_LABEL, max_tokens=8000)
    _apply_timeline(s, r.get("segments"), "collapse", wd, "P4 label+collapse")
    _enforce_cuts(s, cut_times, wd)                       # GUARANTEE the transition cuts
    _log(wd, f"P4 label+collapse -> {len(s.segments)} segments (transitions enforced)")


def refine_timeline(s: ClipState, system: str, wd: Path):
    """STEP 5 verifier (model-judged, text-only): read the timeline + the reliable
    possession facts, RETURN the corrected timeline (object consistency, hand-role,
    contact-grounded N/A, redundant/missing). No hardcoded rules."""
    if not s.segments:
        return
    prompt = _p("refine_timeline.txt", DIRECTION=s.direction or "(unknown)",
                GOAL=s.goal or "(unknown)", CONTACT=_contact_summary(s),
                TIMELINE=s.timeline_text())
    try:
        r = claude_call(prompt, [], system, SC.TIMELINE_LC, model=CLAUDE_GATE,
                        max_tokens=8000)
    except RuntimeError:
        _log(wd, "P5 verifier: call failed -> kept timeline")
        return
    note = str(r.get("think", ""))[:120]
    if _apply_timeline(s, r.get("segments"), "verifier", wd, "P5 verifier"):
        _log(wd, f"P5 verifier -> {len(s.segments)} segments | {note}")


def _seed_timeline(s: ClipState):
    """STEP 1 seed: THE timeline starts as the dense 0.1s facts — one segment per
    contact-state, each hand's slot = the OBJECT it holds (or N/A if empty). The labeler
    later collapses these into action sentences."""
    tr = build_track(s.contact_frames)
    edges = {0.0, round(s.duration, 2)}
    for h in ("left", "right"):
        for iv in tr.get(h, []):
            edges.add(round(iv["start_sec"], 2)); edges.add(round(iv["end_sec"], 2))
    bounds = sorted(e for e in edges if 0.0 <= e <= s.duration)

    def _obj_at(hand, t):
        for iv in tr.get(hand, []):
            if iv["start_sec"] - 0.01 <= t < iv["end_sec"] + 0.01:
                o = str(iv.get("interacting_with", "") or "").strip()
                return o if o and o.lower() not in ("empty", "none", "n/a", "out of frame") else "N/A"
        return "N/A"

    segs = []
    for a, b in _spans(bounds):
        if b - a < 0.05:
            continue
        mid = (a + b) / 2
        sg = Segment(start=round(a, 2), end=round(b, 2),
                     left=_obj_at("left", mid), right=_obj_at("right", mid))
        sg.boundary_provenance = "1a_seed"
        segs.append(sg)
    s.segments = segs


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
    s.contact_intervals = {h: [dict(iv) for iv in s.track[h]] for h in ("left", "right")}
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

    # ============================================================================ #
    #  ONE TIMELINE, refined step by step. Facts computed once; the dense 0.1s      #
    #  facts ARE the starting timeline; each stage rewrites it in place.            #
    # ============================================================================ #
    # INVARIANT: every stage emits the full timeline (via _snap), even if unchanged — so
    # each stage is a guaranteed timeline output, one row in the trace.
    phase1_contact(s, gv, system, wd)                 # STEP 1A: objects + per-0.1s contact
    _seed_timeline(s)                                 # the dense facts ARE the timeline
    _snap(s, "1A seed (0.1s contact facts)", wd)
    phase1_transitions(s, system, wd)                 # STEP 1C: transitions annotate it
    _snap(s, "1C transitions (annotate)", wd)
    phase2_bursts(s, system, wd)                      # STEP 2: rotation bursts
    _snap(s, "2 bursts (annotate)", wd)
    phase3_direction(s, gv, system, wd)               # STEP 3: direction + goal
    _snap(s, "3 direction (annotate)", wd)
    facts_snaps = list(s.stage_snapshots)             # preserved across the per-attempt reset
    _log(wd, f"facts ready: dir={s.direction}, {len(s.transitions)} transitions")

    # ---- RETRY LOOP over the refinement chain (keep BEST by gate verdict) ----
    import copy as _copy
    feedback, best = "", None
    for attempt in range(1, max_attempts + 1):
        _log(wd, f"=== ATTEMPT {attempt}/{max_attempts}"
                + (f" (feedback: {feedback[:90]!r})" if feedback else "") + " ===")
        if attempt > 1 and feedback:
            phase1_transitions(s, system, wd, hint=feedback)   # focused re-scan
        s.flags, s.ran, s.stage_snapshots = [], set(), list(facts_snaps)
        label_and_collapse(s, gv, system, wd)         # STEP 4: collapse facts -> sentences
        _snap(s, "4 label + collapse", wd)
        refine_timeline(s, system, wd)                # STEP 5: verifier (model edit)
        _snap(s, "5 verifier (consistency edit)", wd)
        rep, afl = AN.analyze(s)                      # STEP 6: code analysis — READ ONLY
        s.analysis_report = rep
        s.flags.extend(afl)
        _snap(s, "6 code-analysis (read-only, re-emit)", wd)
        _log(wd, f"6 code-analysis: +{len(afl)} advisory flags (no timeline change)")
        gate = phase5_gate(s, gv, system, wd)         # STEP 7: gate + QA (model edit)
        _snap(s, "7 gate + QA (model edit)", wd)
        verdict = gate.get("quality_verdict", "good")
        score = (0 if verdict == "good" else 1, len(s.flags), -len(s.segments))
        if best is None or score < best[0]:
            best = (score, _copy.deepcopy(s.segments), list(s.flags),
                    _copy.deepcopy(s.track), attempt, list(s.stage_snapshots),
                    s.analysis_report, s.gate_findings, s.purpose_verdict)
        if verdict == "good" or attempt == max_attempts:
            _log(wd, f"=== quality={verdict} after attempt {attempt}; "
                    f"best=attempt {best[4]} (score {best[0]}) -> accept ===")
            break
        feedback = (gate.get("rerun_feedback") or
                    "Re-check for missed place/pickup/throw boundaries.")
    # restore the best attempt
    (s.segments, s.flags, s.track, _, s.stage_snapshots,
     s.analysis_report, s.gate_findings, s.purpose_verdict) = best[1:]

    # ---- STEP 8: fresh-eye final review (context-free), once on the best ----
    fresh_eye(s, system, wd)
    _snap(s, "8 fresh-eye (final, exported)", wd)

    episode = QA.export_episode(s, out_path, apply_overrides=apply_overrides)
    _log(wd, f"=== DONE {time.time()-t0:.0f}s | {len(s.segments)} segments, "
            f"{len(episode['_qa']['violations'])} qa flags | usage: {USAGE.summary()} ===")
    _log(wd, f"episode -> {out_path}")
    return episode
