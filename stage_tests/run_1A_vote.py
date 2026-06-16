#!/usr/bin/env python3
"""Stage-1A with SAMPLE-AND-RECONCILE — runs the contact track N times and merges the
runs into one consensus timeline by per-0.1s majority vote, to cancel the run-to-run
variance of a single VLM pass.

Voting key (normalisation is for GROUPING runs only — the displayed name keeps full detail):
  - empty/idle -> "N/A"
  - otherwise: lowercase, drop articles/conjunctions, drop generic fillers
    ("assembly/unit/combined/...") and generic descriptors ("silver/metal/small/..."),
    singularise -> a sorted set of core nouns. So "silver bolt" == "bolt" == "bolts",
    but "bolt and nut" ({bolt,nut}) stays DISTINCT from "bolt" ({bolt}).
  NOTE: these little lists are parsing/grouping aids, not semantic judgement.

Usage:  python stage_tests/run_1A_vote.py <video> [N=5] [workdir]
"""
import re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P
from clipstate import ClipState, build_track

EMPTY = {"", "n/a", "na", "empty", "none", "nothing", "out of frame"}
STOP = {"and", "the", "a", "an", "of", "with", "plus", "to", "into", "its", "on", "in", "or"}
FILLER = {"assembly", "assembled", "unit", "combined", "joined", "set", "piece", "pieces",
          "two", "both", "together", "group", "bunch", "handful", "whole", "thing", "object"}
DESC = {"silver", "metal", "metallic", "steel", "grey", "gray", "shiny", "small", "large",
        "clear", "single", "loose", "one", "another", "the", "tiny", "little"}


def vkey(name):
    n = (name or "").strip().lower()
    if n in EMPTY:
        return ("N/A",)
    core = []
    for t in re.findall(r"[a-z]+", n):
        if t in STOP or t in FILLER or t in DESC:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        core.append(t)
    core = sorted(set(core))
    return tuple(core) if core else ("N/A",)


def main():
    video = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    wd = Path(sys.argv[3] if len(sys.argv) > 3 else f"logs/1Avote_{Path(video).stem[:8]}")
    wd.mkdir(parents=True, exist_ok=True)
    system = (P.PROMPTS / "caption.system.txt").read_text()

    dur = P.probe_duration(video)
    clocked = str(wd / "clocked.mp4")
    P.burn_clock(video, clocked)
    gv = P.GeminiVideo(P.GEMINI_NATIVE)
    gv.upload(clocked)
    P._log(wd, f"=== 1A-VOTE {Path(video).name} ({dur:.1f}s) x{N} runs ===")

    runs = []
    for i in range(N):
        s = ClipState(video=video)
        s.duration = dur
        s.clocked = clocked
        P.phase1_contact(s, gv, system, wd)
        runs.append(s.contact_frames)
        P._log(wd, f"run {i+1}/{N}: L={len(s.track['left'])} R={len(s.track['right'])} intervals")

    # per-frame majority vote (frames share the same 0.1s time grid across runs)
    m = min(len(r) for r in runs)
    voted = []
    for k in range(m):
        row = {"t": runs[0][k]["t"], "foreground": "", "background": ""}
        for hand, key in (("left", "left_touching"), ("right", "right_touching")):
            names = [r[k].get(key, "") for r in runs]
            keys = [vkey(n) for n in names]
            win = Counter(keys).most_common(1)[0][0]
            if win == ("N/A",):
                row[key] = "empty"
            else:                                 # display the most detailed winning name
                cands = [names[j] for j in range(len(names)) if keys[j] == win]
                row[key] = max(Counter(cands).most_common(),
                               key=lambda kv: (kv[1], len(kv[0])))[0]
        voted.append(row)

    sc = ClipState(video=video)
    sc.duration = dur
    sc.contact_frames = voted
    tr = build_track(voted)
    P._seed_timeline(sc)
    P._snap(sc, f"1A CONSENSUS of {N} runs", wd)
    P._log(wd, f"=== consensus: L={len(tr['left'])} R={len(tr['right'])} intervals ===")


if __name__ == "__main__":
    main()
