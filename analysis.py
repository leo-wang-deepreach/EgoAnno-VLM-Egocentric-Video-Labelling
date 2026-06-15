#!/usr/bin/env python3
"""analysis.py — deterministic, code-only analysis of the pipeline's own data.

NO models, NO CV, NO open-source libraries, NO semantic wordlists. Just plain Python
(Counter/statistics/string ops) over data the pipeline already produced: the 1A object
catalogue + per-frame contact track, the detected transitions, and the label timeline.

Everything here is ADVISORY. It is fed to the VLM (gate / fresh-eye) clearly marked as
CODE-GENERATED so the model treats each line as a hint to RE-CHECK against the video,
never as ground truth — edge cases (a clip starting/ending mid-cycle, near-identical
wording that is genuinely two actions) make raw counts wrong sometimes, on purpose.

`analyze(s)` -> (report_text, flags). report_text goes into prompts; flags route to the
gate. Flags are deliberately low-confidence: type names end in "?" intent.
"""
from __future__ import annotations

import re
import statistics as st
from collections import Counter

from clipstate import ClipState, Flag

_STOP = {"the", "a", "an", "of", "to", "into", "onto", "from", "on", "off", "and",
         "with", "at", "in", "its", "it", "for", "n/a"}


def _words(label: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9-]+", str(label or "").lower())
            if w not in _STOP]


def _verb_head(label: str) -> str:
    toks = re.findall(r"[a-z-]+", str(label or "").lower())
    return toks[0] if toks else ""


def _catalogue_words(s: ClipState) -> set[str]:
    out: set[str] = set()
    for o in s.objects:
        out.update(_words(o.get("name", "")))
        out.update(_words(o.get("colour", "")))
    return out


def _contact_objects_in(s: ClipState, hand: str, a: float, b: float) -> set[str]:
    """Set of object-words the RAW 1A contact track shows `hand` touching in [a,b]."""
    key = f"{hand}_touching"
    words: set[str] = set()
    held = False
    for fr in s.contact_frames:
        t = fr.get("t")
        if t is None or not (a - 0.05 <= t <= b + 0.05):
            continue
        v = str(fr.get(key, "") or "").lower()
        if v and v not in ("empty", "out of frame", "none", "n/a"):
            words.update(_words(v))
            held = True
    return words if held else set()


def _contact_empty_frac(s: ClipState, hand: str, a: float, b: float) -> float:
    key = f"{hand}_touching"
    n = empty = 0
    for fr in s.contact_frames:
        t = fr.get("t")
        if t is None or not (a - 0.05 <= t <= b + 0.05):
            continue
        n += 1
        v = str(fr.get(key, "") or "").lower()
        if v in ("", "empty", "out of frame", "none", "n/a"):
            empty += 1
    return (empty / n) if n else 1.0


# --------------------------------------------------------------------------- #
def analyze(s: ClipState) -> tuple[str, list[Flag]]:
    segs = s.segments
    flags: list[Flag] = []
    lines: list[str] = []
    if not segs:
        return "", []

    # 1. word / object-name frequency across all labels --------------------------
    wc: Counter = Counter()
    for seg in segs:
        wc.update(_words(seg.left))
        wc.update(_words(seg.right))
    top = ", ".join(f"{w}×{c}" for w, c in wc.most_common(8))
    lines.append(f"label word frequency (most used): {top}")

    # 2. label words vs the 1A catalogue ----------------------------------------
    # No verb wordlist: we just list which label words are absent from the catalogue
    # and let the VLM judge which are verbs (ignore) vs object nouns (a rename/
    # hallucination worth re-checking).
    cat = _catalogue_words(s)
    if cat:
        oov = sorted({w for w in wc if w not in cat and len(w) > 2 and not w.isdigit()})
        lines.append("catalogue (1A) objects: " + (s.object_names() or "(none)"))
        if oov:
            lines.append("label words NOT in the 1A catalogue (may include verbs — "
                         "judge which are objects; an object here is a likely rename/"
                         f"hallucination): {', '.join(oov)}")

    # 3. pick/place balance from DETECTED transitions (not verbs) ----------------
    if s.transitions:
        kind_obj: Counter = Counter()
        for e in s.transitions:
            kind_obj[(e.get("kind", "?"),
                      " ".join(_words(e.get("object", "")))[:24] or "?")] += 1
        picks = sum(v for (k, _), v in kind_obj.items() if k == "pickup")
        places = sum(v for (k, _), v in kind_obj.items() if k == "place")
        lines.append(f"detected transitions: {picks} pickup, {places} place "
                     f"(a clip starting/ending mid-cycle can legitimately leave one "
                     f"side unmatched — verify, do not assume a gap)")

    # 4. label <-> raw-contact cross-check (independent of the derived track) -----
    for i, seg in enumerate(segs):
        for hand in ("left", "right"):
            lab = getattr(seg, hand)
            nlab = str(lab or "n/a").strip().lower()
            touched = _contact_objects_in(s, hand, seg.start, seg.end)
            empty_frac = _contact_empty_frac(s, hand, seg.start, seg.end)
            if nlab not in ("n/a", ""):
                lab_objs = set(_words(lab)) & (cat or set(_words(lab)))
                # active label but the hand looks empty most of the span
                if empty_frac >= 0.8:
                    flags.append(Flag(i, "contact_says_idle?", "analysis",
                                      f"{hand} label '{lab}' but 1A shows it empty "
                                      f"{empty_frac*100:.0f}% of the span"))
                # names an object the hand's contact never shows
                elif touched and lab_objs and not (lab_objs & touched):
                    flags.append(Flag(i, "object_unsupported?", "analysis",
                                      f"{hand} label names {sorted(lab_objs)} but 1A "
                                      f"contact shows {sorted(touched)}"))
            else:  # N/A label but contact shows a held object
                if empty_frac <= 0.3 and touched:
                    flags.append(Flag(i, "idle_but_holding?", "analysis",
                                      f"{hand} label N/A but 1A shows it holding "
                                      f"{sorted(touched)}"))

    # 5. monotone run-length (deterministic; the 'word repeated N times' idea) ----
    def _pair(seg):
        return (str(seg.left or "n/a").lower(), str(seg.right or "n/a").lower())
    run_start, run_len = 0, 1
    longest = (0, 1)
    for i in range(1, len(segs)):
        if _pair(segs[i]) == _pair(segs[i - 1]):
            run_len += 1
        else:
            if run_len > longest[1]:
                longest = (run_start, run_len)
            run_start, run_len = i, 1
    if run_len > longest[1]:
        longest = (run_start, run_len)
    if longest[1] >= 3:
        flags.append(Flag(longest[0], "monotone_run?", "analysis",
                          f"{longest[1]} consecutive segments share identical labels "
                          f"from #{longest[0]+1} — a pick/place may have been swallowed"))

    # 6. segment-duration outliers (swallow detector) ----------------------------
    durs = [seg.end - seg.start for seg in segs]
    if len(durs) >= 3:
        med = st.median(durs)
        lines.append(f"segment durations: median {med:.1f}s, "
                     f"range {min(durs):.1f}-{max(durs):.1f}s over {len(durs)} segs")
        for i, d in enumerate(durs):
            if med > 0 and d >= max(8.0, 6 * med):
                flags.append(Flag(i, "duration_outlier?", "analysis",
                                  f"#{i+1} is {d:.1f}s vs median {med:.1f}s — likely "
                                  f"a swallow (several actions in one segment)"))

    # 7. transition <-> boundary alignment ---------------------------------------
    bounds = sorted({round(seg.start, 2) for seg in segs[1:]}
                    | {round(seg.end, 2) for seg in segs[:-1]})
    missed = []
    for e in s.transitions:
        t = e.get("t")
        if t is None:
            continue
        if not any(abs(float(t) - bt) <= 0.6 for bt in bounds):
            missed.append((round(float(t), 1), e.get("kind", "?"), e.get("hand", "?")))
    if missed:
        # attach to the segment that currently spans each missed transition
        for (mt, kind, hand) in missed:
            owner = next((j for j, sg in enumerate(segs)
                          if sg.start - 0.05 <= mt <= sg.end + 0.05), 0)
            flags.append(Flag(owner, "missed_boundary?", "analysis",
                              f"detected {kind} ({hand}) at {mt}s has no segment "
                              f"boundary near it — #{owner+1} may need a cut there"))

    # 8. contact-track staticness (the L=1/R=1 'holds all clip' failure) ---------
    for hand in ("left", "right"):
        objs = _contact_objects_in(s, hand, 0.0, s.duration)
        ef = _contact_empty_frac(s, hand, 0.0, s.duration)
        if len(objs) <= 1 and ef <= 0.02 and s.duration > 8:
            lines.append(f"NOTE: 1A shows the {hand} hand holding {sorted(objs) or '?'} "
                         f"for ~the entire clip with no release — the contact track may "
                         f"be unreliable for this hand; trust the frames.")

    header = ("=== CODE-GENERATED ANALYSIS (advisory only) ===\n"
              "Deterministic counts computed by code — NOT a model opinion, NOT ground "
              "truth. They can be wrong at edges (a clip starting/ending mid-cycle "
              "legitimately leaves a pick or place unmatched; near-identical wording is "
              "not always the same action). Treat each line as a HINT to RE-CHECK "
              "against the video; never obey it blindly.\n")
    report = header + "\n".join(f"- {ln}" for ln in lines)
    return report, flags
