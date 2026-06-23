#!/usr/bin/env python3
"""llm_manip.py — the MANIPULATION GATE (3.10 venv). Decides hold-vs-empty (N/A) with two BROAD yes/no
questions on a clean + overlay crop, voted k times for stability. Replaces the flaky single-pick-or-N/A +
geometry-override. "Manipulating" is intentionally broad (grasp/use/pinch/press/steady), and a SEPARATE
transparent-object question catches clear cups/jars that a plain "holding?" misses.

Run: ../.venv/bin/python perception/llm_manip.py <clean.jpg,overlay.jpg> <LEFT|RIGHT> [k]
Prints one JSON line: {"manip": bool, "q1": int_yes, "q2": int_yes, "k": k}
"""
from __future__ import annotations
import base64
import io
import json
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import CLAUDE_GATE, claude_call  # noqa: E402

SYS = (
    "You see TWO crops of the {hand} hand region in an egocentric (head-mounted) manipulation video: the FIRST "
    "is a CLEAN crop (no marks), the SECOND is the same crop with candidate outlines + a RED grasp dot. Decide "
    "whether the {hand} hand is MANIPULATING an object at THIS moment.\n"
    "MANIPULATING (broad) = holding, gripping, pinching, using as a tool, picking up, pressing, threading, "
    "assembling, pouring, turning, or STEADYING an object. It does NOT require a firm closed 'hold' — a light "
    "grasp or working-on still counts.\n"
    "NOT manipulating = an empty / resting / gesturing hand, or a hand merely HOVERING or REACHING over the "
    "table / a pile / a rack with no object actually in its grasp.\n"
    "Answer TWO yes/no questions:\n"
    "  q1_transparent: is the hand manipulating a TRANSPARENT / clear see-through object (a clear vessel, "
    "bottle, vial, or container)? Transparent objects are EASY TO MISS — look carefully for clear rims, edges, "
    "refractions, or contents visible inside a clear vessel.\n"
    "  q2_object: is the hand manipulating ANY object (transparent or not)?\n"
    "Be INCLUSIVE: if the hand is pinching / working on / steadying something small or partly occluded, answer "
    "yes. Judge ONLY this single moment."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"},
    "q1_transparent": {"type": "string", "enum": ["yes", "no"]},
    "q2_object": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["reason", "q1_transparent", "q2_object"]}


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    imgs = sys.argv[1].split(","); hand = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    frames = [_b64(p) for p in imgs]
    sys_txt = SYS.format(hand=hand)
    user = (f"The {hand} hand. Is it manipulating an object right now? Answer q1_transparent and q2_object.")

    def _vote(_):                                            # one independent gate vote (HTTP -> threads OK)
        try:
            r = claude_call(user, frames, sys_txt, SCHEMA, model=CLAUDE_GATE, max_tokens=300)
            return (1 if r.get("q1_transparent", "no").lower() == "yes" else 0,
                    1 if r.get("q2_object", "no").lower() == "yes" else 0)
        except Exception:
            return (0, 0)
    with ThreadPoolExecutor(max_workers=k) as ex:           # k votes CONCURRENTLY (was sequential)
        votes = list(ex.map(_vote, range(k)))
    q1 = sum(v[0] for v in votes); q2 = sum(v[1] for v in votes)
    manip = (q1 * 2 > k) or (q2 * 2 > k)                      # majority on EITHER question
    print(json.dumps({"manip": bool(manip), "q1": q1, "q2": q2, "k": k}))


if __name__ == "__main__":
    main()
