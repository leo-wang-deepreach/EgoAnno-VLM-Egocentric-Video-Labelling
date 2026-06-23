#!/usr/bin/env python3
"""pipeline2.py — measured-spine -> semantic-head pipeline (v2, experimental).

ISOLATED from the production pipeline.py. The thesis: the VLM never MEASURES
(boundaries, hand, grip, possession are measured off the handpose model); it only
NAMES fixed measured spans. One VLM direction call + one BATCHED VLM labeling call,
versus the dozens of native-video calls + 3-attempt loop in pipeline.py.

  raw video
   -> burn clock (working video the VLM frames come from)
   -> PERCEPTION (handpose) : typed segments (action|transition) + per-frame hand facts
   -> FACT-PACK (code)      : per-segment {acting hand, grip release/grasp, motion}
   -> DIRECTION (1 VLM call): assembly|disassembly|ambiguous (high abstention)
   -> LABEL (1 BATCHED VLM) : per-hand atomic label on every fixed span, fed the facts
   -> CROSS-CHECK (code)    : measured acting-hand vs labelled active-hand -> flags
   -> export episode.json

Run: ../.venv/bin/python perception/pipeline2.py <video> --out out/v2/<clip>.json [--eval 76a67a82]
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.append(str(PKG))                      # factsfirst/ for models, media, qa, clipstate

import media                                   # noqa: E402
import qa as QA                                # noqa: E402
from clipstate import ClipState, Flag, Segment, derive_track_from_labels  # noqa: E402
from models import CLAUDE_GATE, USAGE, claude_call  # noqa: E402
import perception as PCP                        # noqa: E402  (sibling perception.py)
import rotation as ROT                           # noqa: E402  (sibling rotation.py — Spec B)

# Lean, v2-specific, EXAMPLE-FREE system prompt. The old caption.system.txt commands the
# model to obey a "possession track", "role bursts" and "NAMED CHECKS" that v2 does not
# produce — reusing it made the model invent structure. This references only what v2 gives
# (cropped frames + measured facts) and names no specific objects (no leakage).
SYSTEM = (
    "You label fine-grained egocentric (head-mounted, top-down) two-handed manipulation "
    "video for robot learning.\n"
    "- FRAMES ARE GROUND TRUTH: report only what is visible. The images are CROPPED to the "
    "region around the HANDS — name the object actually IN/BETWEEN THE HANDS, never the "
    "background container, pile, bag, tray, or table the work happens over.\n"
    "- L/R IDENTITY: if a hand is tagged with a GREEN 'L' or BLUE 'R' marker at its wrist, "
    "that tag is authoritative — green=LEFT, blue=RIGHT, never swap them even when hands "
    "cross. Only if there is no marker, fall back to: forearms enter from the bottom, "
    "left-edge = LEFT, right-edge = RIGHT.\n"
    "- A hand out of frame, empty, or merely resting/hovering is 'N/A'. Use 'hold'/'steady' "
    "only for a hand keeping an object still while the other does the work.\n"
    "- Two hands on one item usually have DIFFERENT roles (one steadies, one works) — do not "
    "give both the same active verb unless both clearly act independently.\n"
    "- Name each object by its specific visible identity (form + colour) and keep the SAME "
    "name while the same physical object is held; a fastened pair takes a combined-unit name "
    "only once it is joined.\n"
    "- LABEL FORMAT: 2-8 words, exactly ONE verb, '<verb> the <object> [modifier]'. No "
    "'and/then/while', no pronouns, never the words 'left hand'/'right hand' in a label. Use "
    "exactly 'N/A' when idle.\n"
    "OUTPUT: raw JSON matching the requested schema; start '{', end '}', no markdown.")

# --- inline schemas (Gemini-style; claude_call uses them as the forced-tool input_schema) ---
DIR_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string",
                      "enum": ["assembly", "disassembly", "transfer_or_portioning",
                               "mixed_or_alternating", "other_or_ambiguous"]},
        "reason": {"type": "string"}},
    "required": ["direction", "reason"]}

OBJ_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "description"]}}},
    "required": ["objects"]}


# --------------------------------------------------------------------------- #
#  per-segment measured fact-pack (pure code, no model)                       #
# --------------------------------------------------------------------------- #
def _span_idx(t, a, b):
    return int(np.searchsorted(t, a)), int(np.searchsorted(t, b))


def _edge(arr, i0, i1, head=True, k=3):
    """median of the first/last k non-nan samples in a span (grip start/end)."""
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return np.nan
    return float(np.median(seg[:k] if head else seg[-k:]))


def seg_facts(sig, speeds, seg, thr):
    """MEASURED descriptors for one span — motion + grip change per hand. No verbs:
    these are physical facts handed to the labeler, not action words.
      * motion is scaled to the clip's adaptive transition threshold `thr`
      * grip release/grasp is computed ONLY on short transition spans (a start-vs-end
        grip delta over a long action span is meaningless and was misleading the labeler)
    """
    t = sig["t"]; a, b = seg["start"], seg["end"]
    i0, i1 = _span_idx(t, a, b)
    is_trans = seg["type"] == "transition"
    facts = {"type": seg["type"]}
    for side, name in (("L", "left"), ("R", "right")):
        sp = speeds[side][i0:i1 + 1]
        mean_sp = float(np.nanmean(sp)) if np.any(~np.isnan(sp)) else 0.0
        mot = ("moved a lot" if mean_sp > thr else
               "moved a little" if mean_sp > 0.4 * thr else "stayed still")
        grip = "grip steady (held)"
        if is_trans and (b - a) <= 2.5:                  # grip delta only where meaningful
            g0 = _edge(sig["grip"][side], i0, i1, head=True)
            g1 = _edge(sig["grip"][side], i0, i1, head=False)
            if np.isnan(g0) or np.isnan(g1):
                grip = "grip unknown"
            elif g1 - g0 > 0.10:
                grip = "fingers OPENED (likely released/placed)"
            elif g0 - g1 > 0.10:
                grip = "fingers CLOSED (grasped/picked up)"
        gr = sig["grip"][side][i0:i1 + 1]
        gr = gr[~np.isnan(gr)]
        gripr = float(np.mean(gr)) if len(gr) else float("nan")
        pres = float(np.mean(sig["pres"][side][i0:i1 + 1])) if i1 >= i0 else 0.0
        facts[name] = {"motion": mot, "grip": grip, "mean_speed": round(mean_sp, 2),
                       "gripr": gripr, "pres": pres}
    lm, rm = facts["left"]["mean_speed"], facts["right"]["mean_speed"]
    facts["acting"] = "left" if lm > rm * 1.4 else "right" if rm > lm * 1.4 else "both"
    return facts


# --------------------------------------------------------------------------- #
#  SPEC E — idle / N-A detection (anti-hallucination): a hand that is barely    #
#  present, OR static with an OPEN (non-gripping) hand and not turning, is idle. #
#  A segment with both hands idle is N/A both — labelled deterministically, NOT  #
#  sent to the VLM (which otherwise invents an action on every span).           #
# --------------------------------------------------------------------------- #
def _hand_idle(fh, is_turner, open_thr=0.40):
    if fh["pres"] < 0.5:                                  # hand mostly absent
        return True
    if is_turner:                                         # measured rotation -> active
        return False
    gripr = fh["gripr"]
    if fh["motion"] == "stayed still" and not (gripr != gripr) and gripr >= open_thr:
        return True                                       # static + open hand = resting, not holding
    return False


def idle_segments(facts, seg_rots):
    idle = set()
    for i, f in enumerate(facts):
        rot = seg_rots[i] if seg_rots else {}
        lturn = rot.get("fastening") and rot.get("turner") == "left"
        rturn = rot.get("fastening") and rot.get("turner") == "right"
        if _hand_idle(f["left"], lturn) and _hand_idle(f["right"], rturn):
            idle.add(i)
    return idle


def facts_line(f):
    return (f"type={f['type']}; acting≈{f['acting']} | "
            f"LEFT: {f['left']['motion']}, {f['left']['grip']} | "
            f"RIGHT: {f['right']['motion']}, {f['right']['grip']}")


# --------------------------------------------------------------------------- #
#  SPEC A — action-class binding (pure code): measured facts dictate the       #
#  REQUIRED action class per hand; the labeler may only fill object + wording. #
# --------------------------------------------------------------------------- #
def hand_class(fh, is_acting, seg_type):
    """Return (CLASS, instruction) for one hand from its MEASURED facts.
    CLASS ∈ {PICK, PLACE, MOVE, FINE, NA}. The measured signal is RELIABLE only for clear
    translation (pick/place/carry) — it CANNOT tell a steadying hand (small wobble) from a
    hand turning in place during fastening, since neither translates. So those go to FINE
    (the VLM decides hold-vs-turn from frames); the measured holder/turner split waits for
    Spec B (rotation). This avoids forcing 'turn the <fastener>' onto a hand that is just holding."""
    grip = fh["grip"]
    opened = "OPENED" in grip
    closed = "CLOSED" in grip
    gripping = "held" in grip or opened or closed
    big_move = fh["motion"] == "moved a lot"             # clear translation, reliable
    if seg_type == "transition" and closed and is_acting:
        return ("PICK", "GRASPED an object (fingers closed) — name a PICK-UP and where it "
                        "comes FROM: '<verb> the <object> from <source>'. Never 'hold'/'N/A'.")
    if seg_type == "transition" and opened and is_acting:
        return ("PLACE", "RELEASED an object (fingers opened) — name a PLACE/HAND-OFF and "
                         "where it GOES: '<verb> the <object> into/onto <destination>'. "
                         "Never 'hold'/'N/A'.")
    if big_move:
        return ("MOVE", "clearly MOVES across the scene — name the translation (carry/bring/"
                        "pull/move the <object> to <where>). 'hold'/'steady' is FORBIDDEN here.")
    if gripping:
        return ("FINE", "is roughly in place: EITHER just steadies an object ('hold/steady "
                        "the <object>') OR does fine in-place work on it (turn/screw/unscrew/"
                        "press). Decide from the frames; do NOT invent translation.")
    return ("NA", "is static and not gripping — almost certainly 'N/A'.")


def seg_constraints(f, rot=None):
    """Per-hand class block for the labeler prompt + the machine classes for the post-check.
    If Spec-B rotation says this span is FASTENING, MEASURED rotation overrides: the turner
    hand becomes FASTEN with the measured screw/unscrew verb, the other hand HOLDS — this
    fixes both the holder/turner wobble and the screw-vs-unscrew flip."""
    lc, li = hand_class(f["left"], f["acting"] in ("left", "both"), f["type"])
    rc, ri = hand_class(f["right"], f["acting"] in ("right", "both"), f["type"])
    block = (f"REQUIRED per measured facts — obey strictly:\n"
             f"  LEFT  [{lc}]: the left hand {li}\n"
             f"  RIGHT [{rc}]: the right hand {ri}")
    # ROTATION as a DIRECTION HINT only (do NOT force which hand turns — flow-based turner
    # detection is too noisy and forcing it hurt hand-role). Whichever hand the VLM sees
    # turning must carry the measured verb.
    if rot and rot.get("fastening") and rot.get("direction"):
        block += (f"\n  ROTATION (measured): a '{rot['direction']}' turning action occurs this "
                  f"span. Whichever hand actually turns the fastener MUST use the verb "
                  f"'{rot['direction']}' (not the opposite, not generic twist/press). Name the "
                  f"FASTENER/part being turned (the <fastener>) as the object "
                  f"— NOT the tool used to turn it (a <tool> is the means, not the "
                  f"object). The other hand holds/steadies. If no hand is clearly turning "
                  f"(e.g. prying/pulling/levering, not rotating), IGNORE this hint and name "
                  f"what you see.")
    return {"left": lc, "right": rc, "block": block}


_HOLD_RE = re.compile(r"\b(hold|holds|holding|steady|steadies|steadying|keep\w*)\b", re.I)
_UNSCREW_RE = re.compile(r"\b(unscrew\w*|loosen\w*|undo|undoes|unfasten\w*)\b", re.I)
_SCREW_RE = re.compile(r"((?<!un)\bscrew\w*|tighten\w*|fasten\w*|thread\w*)", re.I)


def _dir_ok(lab, d):
    """Does the label carry the MEASURED rotation direction (and not the opposite)?"""
    l = lab or ""
    if d == "unscrew":
        return bool(_UNSCREW_RE.search(l)) and not bool(_SCREW_RE.search(l))
    if d == "screw":
        return bool(_SCREW_RE.search(l)) and not bool(_UNSCREW_RE.search(l))
    return True


def contract_violations(seglabels, facts, seg_rots=None):
    """Deterministic post-check enforcing MEASURED facts (not semantic judgement): the
    'no hold while clearly translating' ban, 'measured pick/place must not be N/A', and
    'a FASTENING turner must carry the MEASURED screw/unscrew verb'. Returns per-seg
    {hand: reason} for a targeted relabel."""
    bad = {}
    for i, (lft, rgt) in enumerate(seglabels):
        rot = seg_rots[i] if seg_rots else None
        cons = seg_constraints(facts[i], rot)
        for hand, lab, cls in (("left", lft, cons["left"]), ("right", rgt, cons["right"])):
            isna = (lab or "N/A").strip().upper() == "N/A"
            if cls == "MOVE" and not isna and _HOLD_RE.search(lab or ""):
                bad.setdefault(i, {})[hand] = ("measured clear TRANSLATION but labelled "
                                               "'hold/steady' — name the actual motion")
            if cls in ("PICK", "PLACE") and isna:
                bad.setdefault(i, {})[hand] = (f"measured a {cls} (grip event) but labelled "
                                               "N/A — name the action + endpoint")
            # measured rotation: flag a hand that used a rotational verb in the WRONG direction
            # (don't touch holding/generic labels — only contradictions of the measured sense).
            if rot and rot.get("fastening") and rot.get("direction") and not isna:
                d = rot["direction"]
                has_rot = _UNSCREW_RE.search(lab or "") or _SCREW_RE.search(lab or "")
                if has_rot and not _dir_ok(lab, d):
                    bad.setdefault(i, {})[hand] = (f"MEASURED rotation = '{d}' but this hand used "
                                                   f"the opposite direction — use '{d}'")
    return bad


# --------------------------------------------------------------------------- #
#  COMBING — merge consecutive segments that are ONE continuous action/process  #
#  (de-over-segmentation). Runs AFTER labelling.                                #
# --------------------------------------------------------------------------- #
def _norm(x):
    return " ".join(str(x or "N/A").lower().split())


def _sim(a, b):
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    if not wa or not wb:
        return 1.0 if wa == wb else 0.0
    return len(wa & wb) / len(wa | wb)


def _same_action(pl, pr, sl, sr, th=0.55):
    def one(a, b):
        na, nb = _norm(a), _norm(b)
        if na == nb:
            return True
        if na == "n/a" or nb == "n/a":
            return False
        return _sim(a, b) >= th
    return one(pl, sl) and one(pr, sr)


def _nonempty(seg):
    return sum(1 for x in (seg.left, seg.right) if x and x.upper() != "N/A")


def comb_segments(segments, seg_rots):
    """Merge consecutive segments that are ONE continuous process — a sustained fastening run
    (same MEASURED direction) OR near-identical per-hand labels — so an action is never cut
    mid-process. Brief pick/place transitions carry distinct labels and aren't fastening, so
    they survive as real boundaries (never merged across)."""
    if not segments:
        return segments
    rots = seg_rots or [{}] * len(segments)
    out, rk = [segments[0]], [rots[0]]
    for s, r in zip(segments[1:], rots[1:]):
        p, pr = out[-1], rk[-1]
        cont_fasten = bool(r.get("fastening") and pr.get("fastening")
                           and r.get("direction") == pr.get("direction"))
        if cont_fasten or _same_action(p.left, p.right, s.left, s.right):
            p.end = s.end
            if _nonempty(s) > _nonempty(p):           # keep the more informative label
                p.left, p.right = s.left, s.right
            p.boundary_provenance = "combed"
        else:
            out.append(s); rk.append(r)
    return out


def absorb_tiny(segments, min_dur=0.5):
    """Merge any segment shorter than min_dur into its LONGER neighbour, extending that
    neighbour to cover its span (timeline stays gap-free). Removes the sub-0.5s micro-
    segments that are not separate actions."""
    segs = list(segments)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s.end - s.start < min_dur:
                prev = segs[i - 1] if i > 0 else None
                nxt = segs[i + 1] if i + 1 < len(segs) else None
                if prev is not None and (nxt is None or
                                         (prev.end - prev.start) >= (nxt.end - nxt.start)):
                    prev.end = s.end                  # extend previous over the tiny one
                else:
                    nxt.start = s.start               # or pull next back to cover it
                del segs[i]
                changed = True
                break
    return segs


# --------------------------------------------------------------------------- #
#  frame helpers                                                              #
# --------------------------------------------------------------------------- #
def grasp_box(sig, a, b, expand=0.5, floor=0.3):
    """Normalized crop box around the HANDS over span [a,b] — the union of both hand boxes,
    expanded so the manipulated object (usually just above/between the fingers) is included.
    The VLM then sees what is IN the hands, not the background container. None if no hands."""
    t = sig["t"]; i0, i1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
    xs0, ys0, xs1, ys1 = [], [], [], []
    for side in ("L", "R"):
        bx = sig["box"][side][i0:i1 + 1]
        for row in bx:
            if not np.any(np.isnan(row)):
                xs0.append(row[0]); ys0.append(row[1]); xs1.append(row[2]); ys1.append(row[3])
    if not xs0:
        return None
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w, h = x1 - x0, y1 - y0
    x0 -= expand * w; x1 += expand * w; y0 -= expand * h; y1 += expand * h
    # enforce a minimum crop size so the object keeps visual context
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if x1 - x0 < floor:
        x0, x1 = cx - floor / 2, cx + floor / 2
    if y1 - y0 < floor:
        y0, y1 = cy - floor / 2, cy + floor / 2
    return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]


def _crop_b64(b64jpeg, box):
    """Crop a base64 JPEG to a normalized [x0,y0,x1,y1] box; re-encode JPEG -> base64."""
    if box is None:
        return b64jpeg
    import base64
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(b64jpeg))).convert("RGB")
    W, H = im.size
    c = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    # the crop is a small region — 768px long side keeps the part SHARP but caps payload
    if max(c.size) > 768:
        c.thumbnail((768, 768), Image.LANCZOS)
    buf = io.BytesIO(); c.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _lr_crop(b64jpeg, t, sig, grasp):
    """Crop to the grasp region AND burn the MEASURED L/R identity onto the frame: a green
    'L' at the left hand's wrist, a blue 'R' at the right hand's — taken from the handpose
    detection at time `t`. The VLM then READS which hand is which (authoritative YOLO
    identity) instead of guessing from screen position — this is the L/R-swap fix. Markers
    sit at the wrist (box bottom, where the forearm enters) so they don't cover the held part."""
    import base64
    import io
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(io.BytesIO(base64.b64decode(b64jpeg))).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    idx = int(np.argmin(np.abs(sig["t"] - t)))
    r = max(12, int(0.018 * W))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(r * 1.5))
    except Exception:                                    # noqa: BLE001
        fnt = ImageFont.load_default()
    for side, col in (("L", (0, 210, 0)), ("R", (0, 140, 255))):
        bx = sig["box"][side][idx]
        if np.any(np.isnan(bx)):
            continue
        cx = (bx[0] + bx[2]) / 2 * W
        cy = bx[3] * H - r                                # near the wrist (box bottom)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(3, int(0.004 * W)))
        d.text((cx, cy), side, fill=col, anchor="mm", font=fnt)
    if grasp is not None:
        im = im.crop((int(grasp[0] * W), int(grasp[1] * H),
                      int(grasp[2] * W), int(grasp[3] * H)))
    if max(im.size) > 768:
        im.thumbnail((768, 768), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _pick(all_frames, a, b, per=3):
    """Choose up to `per` frames inside [a,b] (start / mid / end)."""
    inside = [(t, f) for (t, f) in all_frames if a - 0.05 <= t <= b + 0.05]
    if not inside:
        # nearest single frame
        return [min(all_frames, key=lambda tf: abs(tf[0] - (a + b) / 2))]
    if len(inside) <= per:
        return inside
    idxs = np.linspace(0, len(inside) - 1, per).round().astype(int)
    return [inside[i] for i in idxs]


# --------------------------------------------------------------------------- #
#  stages                                                                     #
# --------------------------------------------------------------------------- #
def decide_direction(all_frames, wd):
    """ONE VLM call: assembly vs disassembly ONLY (NO goal — a coarse goal guess primes and
    poisons the labels). High abstention: a fastener on a thread looks the same either way.
    Used downstream only as a SOFT verb hint when not ambiguous."""
    idxs = np.linspace(0, len(all_frames) - 1, min(16, len(all_frames))).round().astype(int)
    frames = [all_frames[i][1] for i in idxs]
    prompt = (
        "These frames span a two-handed egocentric tabletop manipulation video "
        "(timestamp burned top-left). Decide ONLY the OVERALL direction of change.\n"
        "ASSEMBLY = parts joined / fastener screwed ON / engagement INCREASING.\n"
        "DISASSEMBLY = parts separated / fastener unscrewed OFF / engagement DECREASING.\n"
        "A fastener turning on a thread looks almost identical either way — answer "
        "'assembly' or 'disassembly' ONLY if the direction of change is CLEAR (compare early "
        "vs late frames); otherwise 'other_or_ambiguous'.")
    r = claude_call(prompt, frames, SYSTEM, DIR_SCHEMA, model=CLAUDE_GATE, max_tokens=1500)
    return r.get("direction", "other_or_ambiguous"), r.get("reason", "")


def name_objects(all_frames, segs, sig, wd, tries=2):
    """CONSERVATIVE manipulated-object inventory (anti-hallucination): sample the longest
    ACTION spans, CROP to the hands, and list ONLY objects the hands actually grip/work on.
    A whole-scene 'list everything' over-lists and pulls in background + the camera rig (which
    then leak into labels as 'wrist camera'/'yellow tape'), so we deliberately stay tight."""
    actions = sorted([s for s in segs if s["type"] == "action"],
                     key=lambda s: s["end"] - s["start"], reverse=True)[:4]
    frames = []
    for s in actions:
        gb = grasp_box(sig, s["start"], s["end"])
        frames += [_crop_b64(f, gb) for (_t, f) in _pick(all_frames, s["start"], s["end"], per=3)]
    if not frames:
        idxs = np.linspace(0, len(all_frames) - 1, 12).round().astype(int)
        frames = [all_frames[i][1] for i in idxs]
    prompt = (
        "These frames are CROPPED to the hands of a two-handed manipulation video. List ONLY "
        "the distinct physical objects the HANDS actually GRIP / work on / manipulate — NOT "
        "the table, fixtures, background, or any camera / recording equipment (a head- or "
        "wrist-mounted camera is NOT a task object). Each: a SHORT canonical name (<=3 words, "
        "form + colour) + a one-phrase description. List ONLY objects you CLEARLY see being "
        "handled; never guess or invent. Distinguish similar parts (a bolt vs a nut vs an "
        "assembled unit) explicitly.")
    for _ in range(tries):
        try:
            r = claude_call(prompt, frames[:40], SYSTEM, OBJ_SCHEMA, model=CLAUDE_GATE,
                            max_tokens=2000)
            objs = [o for o in r.get("objects", []) if o.get("name")]
            if objs:
                return objs
        except Exception:                                # noqa: BLE001
            pass
    return []


def derive_goal(seglabels):
    """Goal is SYNTHESISED from the per-segment labels (after the fact), never guessed up
    front. The dominant distinct actions, most-common first."""
    from collections import Counter
    acts = [l.strip() for (a, b) in seglabels for l in (a, b)
            if l and l.strip().upper() != "N/A"]
    if not acts:
        return ""
    top = [a for a, _ in Counter(a.lower() for a in acts).most_common(4)]
    # map normalized back to a representative original casing
    rep = {}
    for a in acts:
        rep.setdefault(a.lower(), a)
    return "; ".join(rep[t] for t in top if t in rep)


_LABEL1 = {
    "type": "object",
    "properties": {"think": {"type": "string"},
                   "left": {"type": "string"}, "right": {"type": "string"}},
    "required": ["left", "right"]}


def label_segments(all_frames, segs, facts, direction, objects, sig, seg_rots, idle, wd, workers=6):
    """PER-SEGMENT, parallel: each VLM call sees ONLY its own span's frames (no cross-span
    confusion) + its measured facts + the fixed object VOCABULARY. The span is fixed; the
    model only names the dominant sustained action per hand from what it SEES — NO task-goal
    prime (that poisons labels). Direction is a soft verb hint, only when not ambiguous."""
    n = len(segs)
    dir_hint = ("" if direction in ("other_or_ambiguous", "mixed_or_alternating", "")
                else f" The overall task is {direction}; match verbs to it (e.g. screw "
                      f"ON / OFF) when relevant.")
    obj_line = ""
    if objects:
        obj_line = ("\nVISIBLE-OBJECT INVENTORY (the COMPLETE list of objects in this clip): "
                    + "; ".join(f"{o['name']}" for o in objects) + ". You MUST name objects "
                    "using ONLY names from this inventory — NEVER introduce an object that is "
                    "not in it. If what a hand holds is not clearly one of these, use the "
                    "closest inventory name or 'N/A'; do not invent.")

    def one(i, feedback=""):
        s = segs[i]
        per = min(8, max(3, int(round(s["end"] - s["start"]))))
        picks = _pick(all_frames, s["start"], s["end"], per=per)
        gb = grasp_box(sig, s["start"], s["end"])        # crop to the hands -> what's MANIPULATED
        frames = [_lr_crop(f, _t, sig, gb) for (_t, f) in picks]  # + burn measured L/R identity
        prompt = (
            "Two-handed egocentric manipulation (timestamp burned top-left). These frames are "
            "CROPPED to the region the hands work in. Name ONLY the object a hand is actually "
            "HOLDING or WORKING ON in THESE frames — the thing in or under the fingers. Do NOT "
            "name other objects elsewhere in the scene, and do NOT pick an object the hand is "
            "not actually on this span. If you cannot identify what a hand holds in these "
            "frames, use 'N/A' rather than guess. Describe ONLY what is VISIBLE."
            + dir_hint + obj_line +
            "\n\nHAND IDENTITY (authoritative): each hand is tagged at the wrist — GREEN 'L' = "
            "LEFT hand, BLUE 'R' = RIGHT hand. Put the LEFT hand's action in 'left' and the "
            "RIGHT hand's in 'right' BY THESE TAGS — never by screen side, never swap on cross.\n"
            f"\nLabel ONE fixed span [{s['start']:.1f}-{s['end']:.1f}s]: the SUSTAINED, dominant "
            "atomic action per hand (2-8 words, one verb), not a momentary edge action.\n\n"
            + seg_constraints(facts[i], seg_rots[i])["block"] + "\n\nRULES:\n"
            "- 'hold'/'steady' is allowed ONLY for a hand that is static and merely supporting; "
            "a hand that MOVES/TURNS must name that motion.\n"
            "- a PICK names where the object comes FROM; a PLACE names where it GOES (endpoint), "
            "when visible — robot training needs the spatial target.\n"
            "- if a hand-held TOOL is being used to act on something, name the TOOL as the "
            "manipulated object and the part it acts on as the endpoint (no fixed example).\n"
            "- name each object by specific visible identity (form + colour); same physical "
            "object keeps the SAME name across segments.\n"
            "- ANTI-HALLUCINATION (critical): it is BETTER to output 'N/A' than to invent. "
            "Output 'N/A' for a hand UNLESS you CLEARLY see it performing a purposeful TASK "
            "step (pick from a source, place to a destination, fasten/unfasten, insert/remove, "
            "press, pry). Merely repositioning/adjusting/resting/handling the workpiece, or "
            "hands idle at the very start or end of the clip, is 'N/A'. If NEITHER hand is "
            "doing a task step this span, output 'N/A' for BOTH. 'hold/steady' is valid ONLY "
            "when the OTHER hand is doing a task action it supports.\n"
            + (f"\nFIX (your previous label broke the contract): {feedback}" if feedback else ""))
        r = claude_call(prompt, frames[:60], SYSTEM, _LABEL1,
                        model=CLAUDE_GATE, max_tokens=1500)
        return i, (r.get("left") or "N/A", r.get("right") or "N/A")

    out = [("N/A", "N/A")] * n
    todo = [i for i in range(n) if i not in idle]        # idle segs stay N/A (Spec E)
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as ex:
        for fut in cf.as_completed([ex.submit(one, i) for i in todo]):
            try:
                i, lab = fut.result()
                out[i] = lab
            except Exception as e:                       # noqa: BLE001
                print(f"  [label] seg failed: {e}")
    # CONTRACT POST-CHECK -> one targeted relabel pass for violators (skip idle segs)
    bad = {i: v for i, v in contract_violations(out, facts, seg_rots).items() if i not in idle}
    if bad:
        print(f"  [contract] {len(bad)} segment(s) violated the contract -> relabel")
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(bad))) as ex:
            futs = {ex.submit(one, i, "; ".join(f"{h}: {r}" for h, r in v.items())): i
                    for i, v in bad.items()}
            for fut in cf.as_completed(futs):
                try:
                    i, lab = fut.result(); out[i] = lab
                except Exception:                        # noqa: BLE001
                    pass
    return out


def cross_check(segs, facts, seglabels):
    """MEASURED vs LABELLED contradiction (no wordlist): if perception says a hand stayed
    still all span but the label gives it an active (non-N/A) verb, or vice-versa, flag it."""
    flags = []
    for i, (lft, rgt) in enumerate(seglabels):
        f = facts[i]
        for side, lab in (("left", lft), ("right", rgt)):
            still = f[side]["motion"] == "stayed still" and "steady" in f[side]["grip"]
            active = lab and lab.strip().upper() != "N/A"
            # only flag the strong case: clearly active label on a hand measured fully static
            # during a TRANSITION span (steady-holding during an action span is legitimate).
            if f["type"] == "transition" and still and active and f["acting"] != side:
                flags.append(Flag(i, "label_vs_motion?", "cross_check",
                                  f"{side} labelled active ({lab!r}) but measured static "
                                  f"in a transition span"))
    return flags


# --------------------------------------------------------------------------- #
#  top level                                                                  #
# --------------------------------------------------------------------------- #
def run(video, out_path, workdir=None, fps=30.0):
    t0 = time.time()
    wd = Path(workdir or f"logs/v2_{Path(video).stem}")
    wd.mkdir(parents=True, exist_ok=True)

    clocked = str(wd / "clocked.mp4")
    media.burn_clock(video, clocked)
    dur = media.probe_duration(clocked)

    # 1) PERCEPTION on the raw video (clean keypoints) -> measured spine
    sig = PCP.extract(video, fps_target=fps)
    energy, speeds = PCP.motion_energy(sig)
    res = PCP.detect_segments(sig, energy)
    segs = res["segments"]
    n0 = len(segs)
    # SEGMENTATION FIX: split mega-segments of in-place fastening using rotation energy
    segs = ROT.refine_long_segments(video, sig, segs)
    print(f"[perception] {n0} motion segments -> {len(segs)} after rotation-split "
          f"in {time.time()-t0:.0f}s")

    # 2) FACT-PACK (code) — motion scaled to the clip's adaptive transition threshold
    facts = [seg_facts(sig, speeds, s, res["threshold"]) for s in segs]

    # 3) frames (extract clocked ONCE at 4fps; reused by direction + labeling). HIGH res
    # (1600) so that when cropped to the grasp region the small manipulated part (a nut on a
    # bolt) is still SHARP enough to recognise — the crop, not the whole frame, is what the
    # labeler sees, so resolution is spent where it matters.
    all_frames = media.extract_frames(clocked, 0.0, dur, 4.0, 1600, str(wd))

    # 4) OBJECT VOCABULARY (1 VLM call, hand-cropped) — fixed names for every later label
    objects = name_objects(all_frames, segs, sig, wd)
    print(f"[objects] {', '.join(o['name'] for o in objects) or '(none)'}")

    # 5) DIRECTION (1 VLM call; soft verb hint only, NO goal prime)
    direction, reason = decide_direction(all_frames, wd)
    print(f"[direction] {direction} | {reason[:80]}")

    # 5b) SPEC B — MEASURED rotation per segment (10fps optical-flow curl): turner + screw/
    #     unscrew, calibrated by the global direction. The VLM never decides direction now.
    seg_rots = ROT.segment_rotation(video, sig, segs)
    ROT.calibrate(seg_rots, direction)
    nf = sum(1 for r in seg_rots if r["fastening"])
    fd = {}
    for r in seg_rots:
        if r["fastening"]:
            fd[r["direction"]] = fd.get(r["direction"], 0) + 1
    print(f"[rotation] {nf}/{len(segs)} fastening segs; {fd}")

    # 5c) SPEC E — idle/N-A detection (anti-hallucination): idle segs are N/A, not labelled
    idle = idle_segments(facts, seg_rots)
    print(f"[idle] {len(idle)}/{len(segs)} segments idle -> N/A (no VLM call)")

    # 6) LABEL (per-segment, parallel, hand-cropped frames) — name what is MANIPULATED
    seglabels = label_segments(all_frames, segs, facts, direction, objects, sig, seg_rots, idle, wd)

    # 7) GOAL synthesised from the labels (after the fact)
    goal = derive_goal(seglabels)

    # 8) CROSS-CHECK (code)
    flags = cross_check(segs, facts, seglabels)

    # 7) assemble + export
    state = ClipState(video=video)
    state.clocked = clocked
    state.duration = dur
    state.direction = direction
    state.goal = goal
    state.segments = [Segment(start=s["start"], end=s["end"],
                              left=lab[0], right=lab[1],
                              boundary_provenance="measured_" + s["type"],
                              confidence=0.7)
                      for s, lab in zip(segs, seglabels)]
    n_pre = len(state.segments)
    state.segments = comb_segments(state.segments, seg_rots)   # COMBING: de-over-segment
    state.segments = absorb_tiny(state.segments)               # drop sub-0.5s micro-segments
    print(f"[comb] {n_pre} -> {len(state.segments)} segments (combed + tiny absorbed)")
    state.objects = [{"name": o["name"], "colour": "", "function": o.get("description", "")}
                     for o in objects]
    state.track = derive_track_from_labels(state.segments)
    state.transitions = [{"t": round((s["start"] + s["end"]) / 2, 2),
                          "hand": facts[i]["acting"], "kind": "transition", "object": ""}
                         for i, s in enumerate(segs) if s["type"] == "transition"]
    state.flags = flags
    episode = QA.export_episode(state, out_path)

    print(f"\n=== v2 DONE {time.time()-t0:.0f}s | {len(state.segments)} segs | "
          f"{len(flags)} cross-check flags | {USAGE.summary()} ===")
    for i, sg in enumerate(state.segments):
        tag = "»T«" if "transition" in sg.boundary_provenance else "   "
        print(f"  #{i+1:<2}[{sg.start:6.2f}-{sg.end:6.2f}] {tag} L: {sg.left:<28} R: {sg.right}")

    return episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out/v2/episode.json")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    run(a.video, a.out, a.workdir or None, a.fps)


if __name__ == "__main__":
    main()
