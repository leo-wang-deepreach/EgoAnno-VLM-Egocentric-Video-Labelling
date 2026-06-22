#!/usr/bin/env python3
"""llm_pick.py — STEP 4 of the v16 grounder (3.10 venv; has Claude via models.py). Shows Claude a
zoomed hand region with NUMBERED candidate-object outlines (from the SAM3 segment-everything pass)
and asks which ONE the named hand is actively MANIPULATING. Returns {choice, name}; choice = the
candidate index, or -1 for N/A (empty hand / worn item / bare hand / camera-rig). The recording
rig is explicitly NOT a task object. Called as a subprocess by ground_v16.py.

Naming is CONSTRAINED to the clip's canonical inventory (manipulable objects only) so the same
object gets the same name every frame AND non-inventory items (the recording camera, worn items)
fall through to N/A automatically.

Run: ../.venv/bin/python perception/llm_pick.py <img1[,img2]> <LEFT|RIGHT> <n_candidates> <names> [feedback]
  <names> = comma-joined canonical inventory names (may be empty)
Prints one JSON line: {"choice": int, "name": str}
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
    "You analyze ONE frame of an egocentric (head-mounted) video of a person doing a tabletop "
    "manipulation task. Colored outlines with NUMBERS mark candidate objects detected near the "
    "{hand} hand; the RED DOT is that hand's grasp point. Decide which ONE numbered object the "
    "{hand} hand is actively MANIPULATING — holding, using as a tool, moving, pouring, or working on.\n"
    "Name it using one of these canonical object names (copy verbatim):\n"
    "  {names}\n"
    "A second image (the REFERENCE SHEET) shows a CLEAR cropped view of each named object. Use it to "
    "identify the held object — match what is in the hand to the correct reference, so the SAME "
    "physical object always gets the SAME name even when it is transparent or occluded in this frame.\n"
    "CONTAINER RULE — read carefully: a container may hold loose MATERIAL/SUBSTANCE (loose granular "
    "material, powder, or liquid). ONLY in that case add the suffix '<container> with <material> inside'. "
    "A separate TOOL, DEVICE, or OBJECT (a separate tool, device, lid, or another container) is NEVER "
    "'contents' — do NOT write '<container> with <that object> inside'. If the hand holds a plain "
    "container with nothing loose in it, just name the container. Label ONLY the single object THIS "
    "hand is holding; an object held in the OTHER hand, or hovering / pouring above this one, is a "
    "DIFFERENT object and is NOT inside it. Never label loose material on its own while it sits in a "
    "held container.\n"
    "A handheld device or camera the hand is gripping IS a manipulated object — label it from the "
    "list, do not answer N/A for a held device.\n"
    "Judge ONLY this single moment — is the hand FIRMLY HOLDING an object right now (object lifted "
    "out / enclosed in the grasp)? Answer N/A (choice = -1, name 'N/A') if: the hand is empty / "
    "resting / gesturing; the hand is REACHING toward or merely TOUCHING an object that is still "
    "sitting in a rack / tray / on the table (not yet lifted out and held) — reaching is unstable, "
    "treat it as N/A; the point is on the BARE hand / arm / a WORN item (watch, wrist strap); OR "
    "the held object does NOT match any name in the list. Do NOT invent a name not in the list. "
    "Choose ONLY from the numbered candidates shown."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"},
    "choice": {"type": "integer"},
    "name": {"type": "string"}},
    "required": ["reason", "choice", "name"]}


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    imgs = sys.argv[1].split(","); hand = sys.argv[2]; n = int(sys.argv[3])
    names = [s.strip() for s in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if s.strip()]
    feedback = sys.argv[5] if len(sys.argv) > 5 else ""
    frames = [_b64(p) for p in imgs]
    namelist = "\n  ".join(f"- {nm}" for nm in names) if names else "(none provided)"
    sys_txt = SYS.format(hand=hand, names=namelist)
    user = (f"The {hand} hand. Candidate objects are numbered 0..{n - 1}. Which one is this hand "
            f"manipulating? Give its number and a canonical name FROM THE LIST, or -1 / 'N/A'.")
    if feedback:
        user += f"\n\nNOTE: {feedback}"
    r = claude_call(user, frames, sys_txt, SCHEMA, model=CLAUDE_GATE, max_tokens=400)
    choice = int(r.get("choice", -1))
    name = (r.get("name") or "").strip()
    # enforce the canonical vocabulary: keep composites ("<container> with <contents> inside"),
    # snap a bare partial to its inventory name, else N/A
    if names and name and name.upper() not in ("N/A", "NA"):
        low = {nm.lower(): nm for nm in names}
        if name.lower() in low:
            name = low[name.lower()]
        elif any(nm.lower() in name.lower() for nm in names):
            pass                                              # composite contains a canonical container -> keep
        else:
            near = [nm for nm in names if name.lower() in nm.lower()]
            name = near[0] if near else "N/A"
    if choice < 0 or choice >= n or name.upper() in ("N/A", "NA", ""):
        choice, name = -1, "N/A"
    print(json.dumps({"choice": choice, "name": name}))


if __name__ == "__main__":
    main()
