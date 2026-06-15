#!/usr/bin/env python3
"""bursts.py — burst sweep (Gemini 30fps micro-calls) + DETERMINISTIC burst_reduce.

A burst answers ONE precise question (direction / role / colour) from ~15
consecutive frames spanning 0.5s, where sub-second motion (rotation direction,
a brief release, a hand-off) is actually visible. burst_reduce then turns the raw
transcripts into a small, high-signal verdict block so downstream stages spend
their attention on SIGNAL, not noise.

The rule that fixes the failing trace (encoded here, not left to prose):
  mixed_or_alternating is justified ONLY when runs >= 2 AND each run has >= 2
  decisive verdicts. A lone dissenter is NOISE -> majority direction, dissenter
  routed to recheck_times for one targeted re-burst.
"""
from __future__ import annotations

import concurrent.futures as cf

import schemas
from media import extract_frames

# phrases that mean "no evidence for THIS field" -> drop the field (not the record)
_INDET = ("cannot determine", "can't determine", "no clear", "not possible",
          "not clear", "unclear", "static", "neither", "stationary relative",
          "very slight", "indeterminate", "ambiguous", "none", "n/a", "no rotation",
          "no visible", "too fast", "too blurry", "not visible")

_ASM = ("assembl", "tighten", "screw on", "screw in", "thread on", "thread onto",
        "clockwise", "insert", "join", "attach", "onto", "into place")
_DIS = ("disassembl", "loosen", "unscrew", "screw off", "unthread", "remove from",
        "counter-clockwise", "counterclockwise", "anti-clockwise", "extract",
        "detach", "pull out", "take apart", "separate")


def _is_indet(txt: str) -> bool:
    t = (txt or "").lower()
    return (not t) or any(p in t for p in _INDET)


# disassembly-specific forms that CONTAIN an assembly substring ("disassembl" ⊃
# "assembl"; "counter-clockwise" ⊃ "clockwise") — strip them before counting
# assembly hits so the negative direction is never misread as the positive one.
_DIS_MASK = ("disassembl", "counter-clockwise", "counterclockwise", "anti-clockwise",
             "anticlockwise", "screw off", "unscrew")


def _classify_direction(txt: str):
    """-> 'ASSEMBLY' | 'DISASSEMBLY' | None (indeterminate). Double-sided: both
    directions are matched symmetrically; ties / no-evidence -> None."""
    t = (txt or "").lower()
    if _is_indet(t):
        return None
    d = sum(p in t for p in _DIS)
    masked = t
    for m in _DIS_MASK:
        masked = masked.replace(m, " ")
    a = sum(p in masked for p in _ASM)
    if a > d:
        return "ASSEMBLY"
    if d > a:
        return "DISASSEMBLY"
    return None


def _classify_role(txt: str):
    """Actor forearm from a role burst's TERSE verdict ('left'|'right'|'both').
    The burst prompt demands the verdict be exactly one token; prose lives in the
    evidence field. A verdict naming BOTH hands is ambiguous -> None (no signal)."""
    t = (txt or "").lower()
    if "both" in t:
        return "both_active"
    if _is_indet(t) and "left" not in t and "right" not in t:
        return None
    left = "left" in t or t.strip() in ("l", "l-hand", "l hand")
    right = "right" in t or t.strip() in ("r", "r-hand", "r hand")
    if right and not left:
        return "actor=right"
    if left and not right:
        return "actor=left"
    return None


# ---- COLOUR: a closed neutral palette so 'mixed' stays terminal -------------- #
_COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "black",
            "white", "grey", "gray", "silver", "gold", "brown", "tan", "beige")


def _classify_colour(txt: str):
    t = (txt or "").lower()
    if "mixed" in t or "assorted" in t or "various" in t or "multiple colour" in t:
        return "mixed"
    if _is_indet(t):
        return None
    hits = [c for c in _COLOURS if c in t]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "mixed"
    return None


# --------------------------------------------------------------------------- #
#  burst sweep                                                                 #
# --------------------------------------------------------------------------- #
def run_bursts(requests: list[dict], clocked: str, workdir: str, system: str = "",
               inv_names: str = "", cap: int = 16, parallel: int = 4) -> list[dict]:
    """requests: [{t, kind, question}]. Each -> ~15 consecutive 30fps frames over
    0.5s, answered by a focused Claude opus micro-call (flash was too weak). A failed
    call is NOT evidence; it is dropped (downstream routes the unanswered t to review)."""
    from models import CLAUDE_GATE, claude_call
    reqs = requests[:cap]

    def _one(req):
        try:
            t = float(req.get("t", -1))
        except (TypeError, ValueError):
            return None
        kind = str(req.get("kind", "direction"))
        q = str(req.get("question", ""))[:450]
        if t < 0 or not q:
            return None
        imgs = extract_frames(clocked, t - 0.25, 0.5, 30, 960, workdir)
        imgs = [b for _, b in imgs]
        if not imgs:
            return None
        inv = (f" Known objects (use these exact names): {inv_names}." if inv_names else "")
        try:
            ans = claude_call(
                f"These {len(imgs)} frames are CONSECUTIVE, spanning 0.5s around "
                f"t={t:.1f}s (clock top-right).{inv} Question: {q}\n"
                f"Answer ONLY from the visible motion in these frames. If the frames "
                f"do not show it, your verdict must be 'cannot determine'.",
                imgs, system, schemas.BURST_ANSWER, model=CLAUDE_GATE, max_tokens=1500)
        except RuntimeError:
            return None
        verdict = str(ans.get("verdict", "")) if isinstance(ans, dict) else str(ans)
        evidence = str(ans.get("evidence", "")) if isinstance(ans, dict) else ""
        return {"t": round(t, 2), "kind": kind, "question": q,
                "verdict": verdict.strip()[:200], "evidence": evidence.strip()[:200]}

    out = []
    with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
        for r in ex.map(_one, reqs):
            if r:
                out.append(r)
    out.sort(key=lambda r: r["t"])
    return out


# --------------------------------------------------------------------------- #
#  burst_reduce — DETERMINISTIC                                                #
# --------------------------------------------------------------------------- #
def burst_reduce(raw: list[dict]) -> dict:
    """Per burst_reduce.md: drop indeterminate FIELDS (not whole records), emit a
    compact verdict block, and compute the direction summary with the runs>=2 rule.
    """
    dir_pts, lines = [], []
    for r in raw:
        t = r.get("t")
        kind = r.get("kind", "direction")
        verdict = r.get("verdict", "")
        if t is None:
            continue
        if kind == "direction":
            d = _classify_direction(verdict)
            if d:
                dir_pts.append((t, d))
                lines.append(f"t={t:.1f} {d}")
        elif kind == "role":
            role = _classify_role(verdict)
            if role:
                lines.append(f"t={t:.1f} {role}")
            # a role burst can ALSO carry colour (drop indeterminate FIELDS, not
            # records) -> scan verdict + evidence to keep the decisive field
            col = _classify_colour(f"{verdict} {r.get('evidence','')}")
            if col:
                lines.append(f"t={t:.1f} colour={col}")
        elif kind == "colour":
            col = _classify_colour(verdict)
            if col:
                lines.append(f"t={t:.1f} colour={col}")
        else:                                       # unknown kind: try all fields
            d = _classify_direction(verdict)
            if d:
                dir_pts.append((t, d))
                lines.append(f"t={t:.1f} {d}")

    dir_pts.sort(key=lambda x: x[0])
    A = sum(1 for _, d in dir_pts if d == "ASSEMBLY")
    D = sum(1 for _, d in dir_pts if d == "DISASSEMBLY")

    # contiguous same-verdict runs over time
    runs, switch_times = [], []
    for i, (t, d) in enumerate(dir_pts):
        if not runs or runs[-1][0] != d:
            if runs:
                switch_times.append(round(t, 2))
            runs.append([d, 1, [t]])
        else:
            runs[-1][1] += 1
            runs[-1][2].append(t)

    # isolated dissenters (size-1 runs that break a longer majority) -> recheck
    recheck = []
    for j, (d, size, ts) in enumerate(runs):
        if size == 1:
            nbr_bigger = ((j > 0 and runs[j - 1][1] >= 2)
                          or (j < len(runs) - 1 and runs[j + 1][1] >= 2))
            if nbr_bigger or len(runs) > 1:
                recheck.append(round(ts[0], 2))

    if A + D == 0:
        majority = "none"
        summary = {"assembly": 0, "disassembly": 0, "majority": "none",
                   "runs": 0, "switch_times": [], "rule": "no_decisive_rotation"}
    else:
        majority = ("assembly" if A > D else "disassembly" if D > A else "tie")
        # mixed only if >=2 real (size>=2) runs of OPPOSITE direction
        big_runs = [r for r in runs if r[1] >= 2]
        opp = len({r[0] for r in big_runs}) >= 2 and len(big_runs) >= 2
        summary = {
            "assembly": A, "disassembly": D, "majority": majority,
            "runs": len(runs), "switch_times": switch_times,
            "rule": ("mixed_eligible" if opp else "single_direction_majority"),
        }

    # de-dup lines, keep order
    seen, reduced = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            reduced.append(ln)

    return {"bursts_reduced": reduced,
            "direction_burst_summary": summary,
            "recheck_times": sorted(set(recheck))}
