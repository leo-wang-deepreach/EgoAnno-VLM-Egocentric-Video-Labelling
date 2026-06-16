#!/usr/bin/env python3
"""Run ONLY Stage 1A (contact track) on a video — fast isolated testing of the 1A prompt.

Replicates exactly the pre-1A setup that annotate() does (clock burn + upload), then runs
phase1_contact and prints the 1A seed timeline. Nothing downstream runs.

Usage:  python stage_tests/run_1A.py <video> [workdir]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root (factsfirst/)
import pipeline as P
from clipstate import ClipState


def main():
    video = sys.argv[1]
    stem = Path(video).stem[:8]
    wd = Path(sys.argv[2] if len(sys.argv) > 2 else f"logs/1Atest_{stem}")
    wd.mkdir(parents=True, exist_ok=True)
    system = (P.PROMPTS / "caption.system.txt").read_text()

    s = ClipState(video=video)
    s.duration = P.probe_duration(video)
    P._log(wd, f"=== 1A-ONLY {Path(video).name} ({s.duration:.1f}s) ===")

    s.clocked = str(wd / "clocked.mp4")
    P.burn_clock(video, s.clocked)
    P._log(wd, "clock burned -> upload to Gemini File API")
    gv = P.GeminiVideo(P.GEMINI_NATIVE)
    gv.upload(s.clocked)
    P._log(wd, "video ACTIVE -> running ONLY phase1_contact (Stage 1A)")

    P.phase1_contact(s, gv, system, wd)          # the stage under test
    P._seed_timeline(s)                          # the dense facts -> timeline rows
    P._snap(s, "1A seed (0.1s contact facts)", wd)

    # also dump the raw per-hand object intervals (object names, no actions)
    P._log(wd, "RAW per-hand contact intervals:\n" + json.dumps(s.contact_intervals, indent=2))
    P._log(wd, f"=== 1A done: {len(s.objects)} objects, "
               f"L={len(s.track['left'])} R={len(s.track['right'])} intervals ===")


if __name__ == "__main__":
    main()
