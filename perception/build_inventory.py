#!/usr/bin/env python3
"""build_inventory.py — per-clip object inventory (3.10 venv). A VLM lists the distinct manipulable
objects on the work surface → the CANONICAL names + the text prompts SAM3 will segment. Leak-safe:
generic table-object names, eval/dev only. Output: out/v2_grounded/_inventory_<tag>.json

Run: ../.venv/bin/python perception/build_inventory.py <tag> <video>
"""
from __future__ import annotations
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import CLAUDE_GATE, claude_call  # noqa: E402

SYS = ("You inventory the distinct PHYSICAL OBJECTS a person manipulates on a tabletop in an "
       "egocentric video. List each distinct object that a hand picks up / holds / uses ONCE with "
       "a short canonical name = colour + material + form (<=4 words). INCLUDE any handheld device "
       "or camera the person picks up and manipulates. EXCLUDE only: hands, arms, the table "
       "surface, the person's body/clothing, and WORN items (watch, wrist strap). Be "
       "THOROUGH about coverage: also list small tools, parts, fasteners, caps and lids, and "
       "anything held even if visible only briefly or partly occluded. For EACH object give a "
       "ROLE: 'manipulable' = picked up / held / used as a tool / moved / worked on; 'fixture' = a "
       "fixed surface / container / holder the task happens ON or IN and is NOT itself picked up "
       "(a fixed work surface, holder, bin, mat, or jig). When unsure, choose 'manipulable'. "
       "Output strict JSON.")
SCHEMA = {"type": "object", "properties": {"objects": {"type": "array", "items": {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "role": {"type": "string", "enum": ["manipulable", "fixture"]}},
    "required": ["name", "role"]}}}, "required": ["objects"]}


def main():
    tag, video = sys.argv[1], sys.argv[2]
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else None    # optional window [t0,t1]s (default: whole clip)
    t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    lo = 0 if t0 is None else max(0, int(t0 * fps))
    hi = (n - 1) if t1 is None else min(n - 1, int(t1 * fps))
    frames = []
    for i in np.linspace(lo, hi, 10).astype(int):             # more frames -> catch briefly-seen objects
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max(im.size) > 1024:
            im.thumbnail((1024, 1024))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    cap.release()
    r = claude_call("List the distinct manipulable objects across these frames.", frames, SYS,
                    SCHEMA, model=CLAUDE_GATE, max_tokens=700)
    objs = [{"name": o["name"].strip(), "role": o.get("role", "manipulable")}
            for o in r.get("objects", []) if o.get("name", "").strip()]
    # VERIFICATION PASS: show the model its own list + the frames and drop anything not actually
    # visible / merge duplicates -> kills hallucinated objects (no hand-editing, generalizes per clip).
    if objs:
        vlist = "\n".join(f"- {o['name']} ({o['role']})" for o in objs)
        vsys = ("You are auditing a proposed object inventory against the actual frames. For EACH "
                "proposed object, KEEP it only if you can actually SEE it in at least one frame. "
                "REMOVE objects that are not visibly present (hallucinations) and MERGE near-"
                "duplicates that are the same physical object into one canonical name. Keep the same "
                "role (manipulable/fixture). Output strict JSON of the surviving objects only.")
        vr = claude_call("Proposed inventory to audit against the frames:\n" + vlist, frames, vsys,
                         SCHEMA, model=CLAUDE_GATE, max_tokens=700)
        verified = [{"name": o["name"].strip(), "role": o.get("role", "manipulable")}
                    for o in vr.get("objects", []) if o.get("name", "").strip()]
        if verified:
            print(f"{tag} verify: {len(objs)} -> {len(verified)} (dropped: "
                  + ", ".join(sorted({o['name'] for o in objs} - {v['name'] for v in verified})) + ")")
            objs = verified
    out = Path("out/v2_grounded") / f"_inventory_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "video": video, "objects": objs}, indent=2))
    nm = sum(1 for o in objs if o["role"] == "manipulable")
    print(f"{tag} inventory: {nm} manipulable + {len(objs) - nm} fixture | "
          + ", ".join(f"{o['name']}[{o['role'][0]}]" for o in objs))


if __name__ == "__main__":
    main()
