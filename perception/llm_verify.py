#!/usr/bin/env python3
"""llm_verify.py — STEP 6b of the v16 grounder (3.10 venv; Claude). Confirms the picked MASK actually
covers the named object the hand holds. Shows a frame with ONE region highlighted + the hand's grasp
point; answers YES only if that region is really a <name> the <hand> hand is touching. Right-name /
wrong-mask -> NO -> the grounder emits N/A. Called as a subprocess by ground_v16.py.

Run: ../.venv/bin/python perception/llm_verify.py <img> <LEFT|RIGHT> <name>
Prints one JSON line: {"ok": bool}
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
    "You sanity-check ONE object grounding in an egocentric video frame. A region is HIGHLIGHTED "
    "(green) and the {hand} hand's grasp point is the RED DOT. Be LENIENT: masks are often rough or "
    "partial — that is FINE. Answer ok=true if the highlighted region is plausibly ON a '{name}' "
    "near the {hand} hand (even if the outline is imprecise or includes a bit of the hand/surroundings). "
    "Answer ok=false ONLY when it is clearly WRONG: the region is almost entirely the bare hand/arm "
    "with no object, OR it is empty table/background, OR it is unmistakably a DIFFERENT kind of object "
    "than '{name}'. When unsure, answer ok=true."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"}, "ok": {"type": "boolean"}}, "required": ["reason", "ok"]}


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    img, hand, name = sys.argv[1], sys.argv[2], sys.argv[3]
    r = claude_call(f"Is the highlighted region a '{name}' the {hand} hand is holding?",
                    [_b64(img)], SYS.format(hand=hand, name=name), SCHEMA,
                    model=CLAUDE_GATE, max_tokens=300)
    print(json.dumps({"ok": bool(r.get("ok", False))}))


if __name__ == "__main__":
    main()
