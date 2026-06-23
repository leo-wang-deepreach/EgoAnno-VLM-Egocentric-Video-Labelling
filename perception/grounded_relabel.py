#!/usr/bin/env python3
"""grounded_relabel.py — S5 integration step 3 (3.10 venv).

For each spotlighted crop, ask Claude (CLAUDE_GATE — the SAME family the baseline labeler uses, so
this is a clean A/B that isolates the GROUNDING effect) to NAME the green-outlined object. Build
out/v2_grounded/<tag>.json with the baseline v2 segment boundaries but left/right replaced by the
grounded object names — ready for object_eval (which judges object identity).

Run: ../.venv/bin/python perception/grounded_relabel.py <tag> [workers]
"""
from __future__ import annotations
import base64
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import CLAUDE_GATE, claude_call                  # noqa: E402

ROOT = Path("/home/ubuntu/local/factsfirst")
SYS = ("You name the single object OUTLINED IN GREEN in an egocentric two-handed manipulation "
       "frame (everything outside it is dimmed). Give its specific visible identity — form + "
       "colour, <=4 words, NO verb (e.g. '<color> <material> <container>', '<color> <tool>'). If the "
       "outlined region is clearly NOT a manipulable task object (just hand/skin, the table, a "
       "tray, or background), answer exactly 'N/A'.")
SCHEMA = {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}


def name_crop(path):
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")                     # PNG crop -> JPEG (claude_call sends image/jpeg)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    b = base64.b64encode(buf.getvalue()).decode()
    r = claude_call("Name the green-outlined object.", [b], SYS, SCHEMA,
                    model=CLAUDE_GATE, max_tokens=200)
    return (r.get("object") or "N/A").strip()


def main():
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gdir = ROOT / "out" / "v2_grounded"
    crops = json.loads((gdir / f"crops_{tag}" / f"_crops_{tag}.json").read_text())["crops"]
    pred = json.loads((ROOT / "out" / "v2" / f"{tag}.json").read_text())

    todo = [(c["seg"], c["hand"], c["crop"]) for c in crops if c.get("grounded") and c.get("crop")]
    obj = {}
    with cf.ThreadPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as ex:
        futs = {ex.submit(name_crop, p): (s, h) for (s, h, p) in todo}
        for f in cf.as_completed(futs):
            try:
                obj[futs[f]] = f.result()
            except Exception as e:                           # noqa: BLE001
                print(f"  name failed {futs[f]}: {e}")

    # HYBRID: use the grounded object where we have one; otherwise FALL BACK to the baseline
    # label for that hand (so grounding can only improve a slot, never blank one the baseline got).
    segs = []
    nfb = 0
    for si, s in enumerate(pred["segments"]):
        gl, gr = obj.get((si, "L")), obj.get((si, "R"))
        L = gl if gl else (s.get("left") or "N/A")
        R = gr if gr else (s.get("right") or "N/A")
        nfb += (gl is None) + (gr is None)
        segs.append({"start_sec": s["start_sec"], "end_sec": s["end_sec"], "left": L, "right": R})
    print(f"  (fell back to baseline on {nfb} hand-slots without a confident grounded object)")
    out = {"clip": tag, "direction": pred.get("direction"), "objects": [], "segments": segs}
    (gdir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"{tag}: named {len(obj)}/{len(todo)} grounded slots -> {gdir}/{tag}.json")


if __name__ == "__main__":
    main()
