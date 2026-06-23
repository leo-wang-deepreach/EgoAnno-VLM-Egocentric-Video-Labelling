#!/usr/bin/env python3
"""llm_transparent.py — TRANSPARENT-OBJECT GAP CHECK (3.10 venv; Claude). Transparent/clear objects
(clear cups, jars) are the objects SAM most often MISSES, so a hand holding one ends up N/A. This
explicitly asks: given the regions already segmented (outlined), is the hand holding a CLEAR/
TRANSPARENT object that is NOT among them? If yes, name it from the inventory so we can recover it.

Run: ../.venv/bin/python perception/llm_transparent.py <img> <LEFT|RIGHT> <names>
Prints one JSON line: {"name": str}   ('N/A' if no missed transparent object)
"""
from __future__ import annotations
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import CLAUDE_GATE, claude_call  # noqa: E402

SYS = (
    "A region is HIGHLIGHTED in GREEN in an egocentric frame — edge analysis guessed it might be an "
    "object the {hand} hand is holding (the RED DOT is the grasp point). Your job is to decide whether "
    "the {hand} hand is ACTUALLY HOLDING an object right now, and if so NAME it.\n"
    "HOLDING means the {hand} hand is in CONTACT with and SUPPORTING or GRIPPING the object: fingers "
    "wrapped/pinched on it, or the object resting cradled in/against the hand and moving with it. The "
    "hand shape itself does NOT decide this — a hand can cradle a rounded object with open fingers "
    "(= holding) or pinch the empty air (= not holding). Look for the object actually IN the hand at "
    "the red dot.\n"
    "Answer 'N/A' (do NOT name anything) if ANY of these is true:\n"
    "  - the {hand} hand is EMPTY — open or gesturing in the air with nothing in contact;\n"
    "  - the hand is REACHING TOWARD an object it has not yet grasped (about to pick it up, not holding);\n"
    "  - the highlighted region is the bare hand/arm, the table, a tray, a storage rack, or an object "
    "sitting on a surface (not lifted/supported by THIS hand);\n"
    "  - the highlighted thing is held by the OTHER hand, not the {hand} hand;\n"
    "  - it does not match any name in the list.\n"
    "Only if the {hand} hand is genuinely holding one of these, NAME it using EXACTLY one canonical "
    "name (copy verbatim):\n  {names}\n"
    "A rough or partial outline of a genuinely-held object is fine — name it. When unsure whether it is "
    "held vs merely reached-for, answer 'N/A'."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"}, "name": {"type": "string"}}, "required": ["reason", "name"]}


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    imgs = sys.argv[1].split(",")                            # frame [, reference sheet]
    hand = sys.argv[2]
    names = [s.strip() for s in sys.argv[3].split(",") if s.strip()]
    r = claude_call(f"Name the GREEN-highlighted object the {hand} hand is holding (or N/A).",
                    [_b64(p) for p in imgs], SYS.format(hand=hand, names="\n  ".join(f"- {n}" for n in names)),
                    SCHEMA, model=CLAUDE_GATE, max_tokens=300)
    name = (r.get("name") or "").strip()
    low = {n.lower(): n for n in names}
    if name.lower() in low:
        name = low[name.lower()]
    elif not any(n.lower() in name.lower() for n in names):
        name = "N/A"
    print(json.dumps({"name": name}))


if __name__ == "__main__":
    main()
