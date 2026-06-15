# Retired in v30 — the old v49 TOP-DOWN segment+label path (phase4 + its helpers:
# _segs_from_bounds, _transitions_in, _tag_transitions, _norm_lbl, _label_sim,
# _labels_mergeable, merge_identical_labels, _neighbors, label_segments,
# seg_reconcile_pass) and the flag-only verifiers (template_match, neighbor_review).
# Replaced by the bottom-up label_and_collapse + model-judged refine_timeline.
# Archived for reference; NOT imported/runnable as-is. Kept per request, not deleted.

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
