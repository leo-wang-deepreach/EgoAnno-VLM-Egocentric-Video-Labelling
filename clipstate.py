#!/usr/bin/env python3
"""clipstate.py — the ONE explicit object every stage reads from and writes to.

Context flows through ClipState, but each stage receives a COMPACTED view, never
the raw history. The compaction helpers below are the attention-budget discipline:
they return exactly the high-signal slice a stage needs and nothing else.

The precedence ladder (stated once; every stage obeys it):
  human override > burst rotation/role/color verdict > possession track >
  destination accumulation > neighbor context > overview calibration > task wording
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    left: str = "N/A"
    right: str = "N/A"
    boundary_provenance: str = "segment"          # how seg.start was set
    confidence: float = 0.6
    draft: dict = field(default_factory=dict)     # per-stage trace of this seg

    def as_row(self, i: int) -> str:
        # Multi-line, explicitly L:/R: on their own lines — never a "left/right" slash,
        # so the model can never confuse which hand is which.
        return (f"#{i+1}  [{self.start:.1f}-{self.end:.1f}s]\n"
                f"    L: {self.left}\n"
                f"    R: {self.right}")


@dataclass
class Flag:
    seg: int                                       # 0-based index into segments
    type: str
    raised_by: str
    detail: str = ""

    def as_row(self, segs: list[Segment]) -> str:
        s = segs[self.seg] if 0 <= self.seg < len(segs) else None
        span = f"[{s.start:.1f}-{s.end:.1f}]" if s else ""
        lab = f"L: {s.left} | R: {s.right}" if s else ""
        return f"seg#{self.seg+1} {span} ({self.type}, by {self.raised_by}) {lab} :: {self.detail}".strip()


@dataclass
class ClipState:
    video: str = ""
    clocked: str = ""
    duration: float = 0.0

    # --- Phase 1: the fact layer ---
    objects: list[dict] = field(default_factory=list)     # {name, colour, function}
    contact_frames: list[dict] = field(default_factory=list)  # 10fps per-frame facts
    track: dict = field(default_factory=lambda: {"left": [], "right": []})  # intervals
    contact_intervals: dict = field(default_factory=lambda: {"left": [], "right": []})  # RAW 1A objects (never overwritten)
    phase_boundaries: list[float] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)   # {t,hand,object,kind}

    # --- Phase 2: bursts reduced to signal ---
    bursts_raw: list[dict] = field(default_factory=list)
    bursts_reduced: list[str] = field(default_factory=list)
    direction_burst_summary: dict = field(default_factory=dict)
    recheck_times: list[float] = field(default_factory=list)

    # --- Phase 3 ---
    goal: str = ""                                         # DERIVED here, not given
    direction: str = ""
    phase_switch_sec: float | None = None

    # --- Phase 4 ---
    segments: list[Segment] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    analysis_report: str = ""                              # code-generated advisory facts
    fresh_eye_note: str = ""                               # what the fresh-eye pass changed
    stage_snapshots: list = field(default_factory=list)    # timeline after each transform (trace)

    # --- run-once guards for on-demand critics (merge_critic / completeness) ---
    ran: set = field(default_factory=set)

    # --- Phase 5 / export ---
    qa: list[dict] = field(default_factory=list)
    gate_findings: str = ""
    purpose_verdict: str = ""

    # --------------------------------------------------------------------- #
    #  COMPACTION VIEWS — what each stage actually SEES                      #
    # --------------------------------------------------------------------- #
    def objects_line(self) -> str:
        if not self.objects:
            return "(none catalogued yet)"
        return "; ".join(
            f"{o.get('name','?')}"
            + (f" ({o.get('colour')})" if o.get("colour") else "")
            for o in self.objects)

    def object_names(self) -> str:
        return ", ".join(str(o.get("name", "")) for o in self.objects if o.get("name"))

    def track_lines(self, a: float | None = None, b: float | None = None) -> str:
        """Whole track (a,b=None) or window-clipped track as text."""
        lo = a if a is not None else 0.0
        hi = b if b is not None else self.duration
        out = []
        for hand in ("left", "right"):
            H = "L" if hand == "left" else "R"
            rows = []
            for iv in self.track.get(hand, []):
                s = max(lo, iv.get("start_sec", 0.0))
                e = min(hi, iv.get("end_sec", 0.0))
                if e <= s + 0.05:
                    continue
                obj = str(iv.get("interacting_with", "") or "empty")
                rows.append(f"{s:.1f}-{e:.1f}: {obj}")
            out.append(f"{H}: " + ("; ".join(rows) if rows else "empty/absent"))
        return "\n".join(out)

    def bursts_in(self, a: float, b: float) -> list[str]:
        """bursts_reduced lines whose time falls in [a,b] (window-scoped)."""
        out = []
        for line in self.bursts_reduced:
            t = _parse_t(line)
            if t is not None and a - 0.05 <= t <= b + 0.05:
                out.append(line)
        return out

    def contact_facts_in(self, a: float, b: float) -> str:
        """Per-frame contact facts inside [a,b], as compact text for the labeler.
        This is what lets GPT name the action KNOWING what each hand holds."""
        rows = []
        for fr in self.contact_frames:
            t = fr.get("t")
            if t is None or not (a - 0.05 <= t <= b + 0.05):
                continue
            lt = fr.get("left_touching") or "empty"
            rt = fr.get("right_touching") or "empty"
            fg = fr.get("foreground") or "?"
            rows.append(f"t={t:.1f} fg:{fg} L:{lt} R:{rt}")
        return "\n".join(rows) if rows else "(no contact facts in window)"

    def timeline_text(self) -> str:
        return "\n".join(s.as_row(i) for i, s in enumerate(self.segments))

    def flags_text(self) -> str:
        if not self.flags:
            return "(none)"
        return "\n".join(f.as_row(self.segments) for f in self.flags)


    def transition_cuts(self, merge_gap: float = 0.6, long_gap: float = 1.0) -> list[float]:
        """Primary segment boundaries — the STRUCTURAL moments, NOT every consumable grab:
          1. place / pickup / handoff events from transition_detect,
          2. workpiece identity changes (held-object token set changes, e.g. bolt -> bolt+nut unit),
          3. LONG empty gaps in a hand's track (>= long_gap s) = a real release/reacquire.
        long_gap is set ABOVE a typical consumable regrab (~0.5s) so repeated same-object
        grabs produce NO cut — a continuous threading phase stays one segment — while a real
        place->pickup (the workpiece rests ~1-2s) does cut."""
        def _toks(name: str) -> set[str]:
            return {w for w in str(name or "").lower().split() if len(w) > 3}
        cuts = [float(e["t"]) for e in self.transitions
                if e.get("t") is not None and e.get("kind") in ("place", "pickup", "handoff")
                and 0.2 < float(e["t"]) < self.duration - 0.2]
        for hand in ("left", "right"):
            ivs = self.track.get(hand, [])
            for a, b in zip(ivs, ivs[1:]):
                ae, bs = a.get("end_sec", 0), b.get("start_sec", 0)
                t = round((ae + bs) / 2, 2)
                if not (0.2 < t < self.duration - 0.2):
                    continue
                if (bs - ae) >= long_gap:                       # real release/reacquire
                    cuts.append(t)
                elif _toks(a.get("interacting_with")) != _toks(b.get("interacting_with")):
                    cuts.append(t)                              # identity change
        cuts.sort()
        merged = []
        for t in cuts:
            if merged and t - merged[-1] < merge_gap:
                continue
            merged.append(round(t, 2))
        return merged


def _parse_t(line: str):
    """Pull the leading t=NN.N from a bursts_reduced line."""
    line = line.strip()
    if line.startswith("t="):
        try:
            return float(line[2:].split()[0].rstrip(","))
        except (ValueError, IndexError):
            return None
    return None


# ------------------------------------------------------------------------- #
#  per-frame contact facts -> possession intervals (the deterministic spine) #
# ------------------------------------------------------------------------- #
_EMPTY = {"", "none", "empty", "out of frame", "nothing", "n/a", "null", "-"}


def _norm(x) -> str:
    return str(x or "").strip()


def build_track(contact_frames: list[dict], min_dur: float = 0.25,
                empty_split_frames: int = 3) -> dict:
    """Collapse 10fps per-frame {left_touching, right_touching} into possession
    intervals per hand. A grasp ends when the hand is empty for >= empty_split_frames
    consecutive frames (a real release/re-grab boundary, even if the next item looks
    the same) OR when the held object's name changes. Single-frame flicker is
    tolerated; runs shorter than min_dur are dropped. This sensitivity is what lets
    repeated cycles surface as separate intervals rather than one smeared hold."""
    track = {}
    frames = sorted([f for f in contact_frames if f.get("t") is not None],
                    key=lambda f: f["t"])
    for hand, key in (("left", "left_touching"), ("right", "right_touching")):
        ivs = []
        cur = None
        empty_run = 0
        for fr in frames:
            t = fr["t"]
            obj = _norm(fr.get(key))
            is_empty = obj.lower() in _EMPTY
            if is_empty:
                empty_run += 1
                # close the current grasp once the release has lasted long enough
                if cur and empty_run >= empty_split_frames:
                    ivs.append(cur)
                    cur = None
                continue
            empty_run = 0
            ck = obj.lower()
            same = cur and (cur["_key"] == ck or _same(cur["_key"], ck))
            if same:
                cur["end_sec"] = t
                cur["_last"] = t
                cur["_names"][obj] = cur["_names"].get(obj, 0) + 1
            else:
                if cur:
                    ivs.append(cur)
                cur = {"start_sec": t, "end_sec": t, "_last": t, "_key": ck,
                       "_names": {obj: 1}}
        if cur:
            ivs.append(cur)
        clean = []
        for iv in ivs:
            if iv["end_sec"] - iv["start_sec"] < min_dur:
                continue
            name = max(iv["_names"].items(), key=lambda kv: kv[1])[0]
            clean.append({"start_sec": round(iv["start_sec"], 2),
                          "end_sec": round(iv["end_sec"], 2),
                          "interacting_with": name})
        # close sub-frame holes left by frame-sampling: object A's last frame at t and B's
        # first at t+0.1 leave a ~0.1s gap that otherwise surfaces as a spurious "N/A" sliver.
        # Snap any small gap (<=0.3s, incl. min_dur-dropped blips) to its midpoint so the
        # per-hand timeline is contiguous. Larger gaps (a genuinely empty hand) are left.
        for i in range(1, len(clean)):
            gap = clean[i]["start_sec"] - clean[i - 1]["end_sec"]
            if 0 < gap <= 0.3:
                mid = round((clean[i - 1]["end_sec"] + clean[i]["start_sec"]) / 2, 2)
                clean[i - 1]["end_sec"] = mid
                clean[i]["start_sec"] = mid
        track[hand] = clean
    return track


def _same(a: str, b: str) -> bool:
    """Loose object-name match: share a significant token (handles 'silver bolt'
    vs 'bolt')."""
    ta = {w for w in a.split() if len(w) > 3}
    tb = {w for w in b.split() if len(w) > 3}
    return bool(ta & tb)


def derive_track_from_labels(segments) -> dict:
    """Build the possession track FROM the per-segment labels (which are reliable),
    instead of the flaky whole-clip contact call. Each hand's activity per segment is
    its label verbatim; consecutive segments with the same label merge into one
    interval; N/A is a gap. No wordlists — the track simply mirrors what we labeled, so
    it can never contradict the labels."""
    track = {"left": [], "right": []}
    for hand in ("left", "right"):
        cur = None
        for seg in segments:
            lab = getattr(seg, hand, None) if not isinstance(seg, dict) else seg.get(hand)
            a = seg.start if not isinstance(seg, dict) else seg["start_sec"]
            b = seg.end if not isinstance(seg, dict) else seg["end_sec"]
            if str(lab or "N/A").strip().upper() in ("N/A", "NA", ""):
                if cur:
                    track[hand].append(cur)
                    cur = None
                continue
            key = " ".join(str(lab).lower().split())
            if cur and cur["_k"] == key:
                cur["end_sec"] = round(b, 2)
            else:
                if cur:
                    track[hand].append(cur)
                cur = {"start_sec": round(a, 2), "end_sec": round(b, 2),
                       "interacting_with": str(lab), "_k": key}
        if cur:
            track[hand].append(cur)
        for iv in track[hand]:
            iv.pop("_k", None)
    return track


def track_possession_changes(track: dict, a: float, b: float) -> bool:
    """True if EITHER hand starts or ends an interval strictly inside (a,b) —
    i.e. a real possession change the edge verifier must not merge across."""
    for hand in ("left", "right"):
        for iv in track.get(hand, []):
            for edge in (iv.get("start_sec"), iv.get("end_sec")):
                if edge is not None and a + 0.15 < edge < b - 0.15:
                    return True
    return False
