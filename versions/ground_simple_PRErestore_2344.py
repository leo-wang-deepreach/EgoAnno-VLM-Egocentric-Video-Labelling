#!/usr/bin/env python3
"""ground_simple.py — LEAN object grounder (sam3py). ONE segmentation source, ONE decision rule.

  STEP 1  SAM3 concept text-seg for each verified object -> all instance masks, named + scored.
          (This is SAM3's native "segment everything"; unlike a point-grid/AMG it also finds TRANSPARENT
           objects, and the masks come pre-named.)
  STEP 2  RULE 1 near-hand filter: keep a mask only if it is AT the pinch, INTERSECTS the hand box, and is
          NOT the wrist/arm (and a sane size).
  STEP 3  LLM names: show Claude the numbered near-hand candidates + the reference sheet; it picks which one
          the hand manipulates and names it (disambiguating visually-similar look-alike objects), or N/A.
          SAFETY NET: if Claude says N/A but a STRONG (>= TRUST_THR) RULE-1 candidate exists, ground it
          anyway (Claude was vetoing correct transparent detections that SAM3 had clearly found).
  STEP 4  The chosen text-seg mask IS the clean mask.

Dropped vs the old pipeline: custom point-grid AMG, depth/FG gate, container-preference, retry loop,
edge-recovery, advisory verify. Easier to debug, more robust.

Run (sam3py): ground_simple.py <outdir> <inventory.json> <prompts.json>
  env: REFS_DIR (refsheet + _refs verified vocab), TRUST_THR (0.6), PINCH (0.045), DBG.
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

HERE = os.path.dirname(os.path.abspath(__file__))
PY310 = "/home/ubuntu/local/.venv/bin/python"
TRUST_THR = float(os.environ.get("TRUST_THR", "0.6"))
PINCH = float(os.environ.get("PINCH", "0.045"))
REFS_DIR = os.environ.get("REFS_DIR", "")
DBG = os.environ.get("DBG", "")
COLORS = [(80, 255, 80), (80, 160, 255), (255, 255, 60), (255, 80, 255), (80, 255, 255), (255, 160, 80)]
_DBGF = None


def _dbg(m):
    if _DBGF:
        _DBGF.write(m + "\n"); _DBGF.flush()


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def call_llm(imgs, hand, n, names, feedback=""):
    """STEP 3: Claude picks the manipulated candidate # + names it from the vocab (or -1/N/A)."""
    try:
        out = subprocess.run([PY310, os.path.join(HERE, "llm_pick.py"), ",".join(imgs), hand, str(n),
                              ",".join(names), feedback], capture_output=True, text=True, timeout=150)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        return int(r.get("choice", -1)), (r.get("name") or "N/A").strip()
    except Exception as e:
        _dbg(f"  call_llm error: {e}")
        return -1, "N/A"


def main():
    global _DBGF
    outdir, inv_file, prompts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    if DBG:
        _DBGF = open(os.path.join(outdir, "_dbgtrace.log"), "w")
    inv = json.load(open(inv_file)); tag = inv["tag"]
    names = [o["name"] if isinstance(o, dict) else o for o in inv["objects"]
             if not (isinstance(o, dict) and o.get("role") == "fixture")]
    all_names = list(names)            # FULL inventory — used for NAMING (so objects SAM couldn't verify,
    #                                    e.g. thin pen parts, can still be named on a point-prompt mask)
    refsheet = ""
    if REFS_DIR:
        rs = os.path.abspath(os.path.join(REFS_DIR, f"_refsheet_{tag}.png"))
        refsheet = rs if os.path.exists(rs) else ""
        rm = os.path.join(REFS_DIR, f"_refs_{tag}.json")
        if os.path.exists(rm):
            verified = set(json.load(open(rm)).keys())
            kept = [n for n in names if n in verified]
            if kept:
                print(f"{tag}: vocab restricted to SAM-verified: {kept}"); names = kept
    pj = json.load(open(prompts_file)); video = pj["video"]; W, H = pj["W"], pj["H"]
    model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    cap = cv2.VideoCapture(video); tmpd = tempfile.mkdtemp()
    by_t = {}
    for p in pj["prompts"]:
        by_t.setdefault(round(p["t"], 3), []).append(p)

    index = []
    for t, plist in sorted(by_t.items()):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), ac:
            state = proc.set_image(Image.fromarray(rgb))
            segs = []                                            # STEP 1: (name, mask, score) per verified object
            for nm in names:
                out = proc.set_text_prompt(state=state, prompt=nm.split(" with ")[0].strip())
                tm = _np(out["masks"]); ts = _np(out["scores"]).ravel()
                if tm.ndim == 4:
                    tm = tm[:, 0]
                for k in range(len(tm) if tm.ndim == 3 else 0):
                    mk = tm[k] > 0.5
                    if mk.ndim > 2:
                        mk = mk.squeeze()
                    if mk.any() and mk.shape == (H, W):
                        segs.append((nm, mk, float(ts[k])))

        for p in plist:
            gx, gy = _clamp(p["x_px"], 0, W - 1), _clamp(p["y_px"], 0, H - 1)
            x0, y0, x1, y1 = p["box_px"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            wpx, wpy = _clamp(2 * cx - gx, x0, x1), _clamp(2 * cy - gy, y0, y1)
            hand = "LEFT" if p["hand"] == "L" else "RIGHT"
            pw = max(20, int(PINCH * W)); yl, yh = max(0, gy - pw), min(H, gy + pw + 1)
            xl, xh = max(0, gx - pw), min(W, gx + pw + 1)

            # STEP 2: RULE-1 near-hand candidates (dedup near-identical masks, keep higher score).
            # Track whether each contains the EXACT pinch pixel — the held object is at the fingertips,
            # while nearby clutter (e.g. a pile of identical bolts) only clips the pinch window.
            cands = []
            for nm, mk, sc in sorted(segs, key=lambda s: -s[2]):
                if mk[wpy, wpx] or not mk[yl:yh, xl:xh].any() or mk[y0:y1, x0:x1].sum() == 0:
                    continue
                if not (0.0005 < float(mk.mean()) < 0.40):
                    continue
                if all(_iou(mk, c[1]) < 0.8 for c in cands):
                    cands.append((nm, mk, sc, bool(mk[gy, gx])))
            # POINT-PROMPT at the pinch: segment WHATEVER the hand is holding, by clicking the grasp point
            # (positive) + the wrist (negative). Catches small/thin/unnamed objects that text-seg-by-name
            # misses (e.g. pen parts). The mask is unnamed here — the LLM names it from the full inventory.
            with torch.inference_mode(), ac:
                pm, ps, _ = model.predict_inst(state, point_coords=np.array([[gx, gy], [wpx, wpy]]),
                                               point_labels=np.array([1, 0]), multimask_output=True)
            pm = _np(pm); ps = _np(ps).ravel()
            for i in range(len(ps) if pm.ndim == 3 else 0):
                m = pm[i] > 0.5
                if m.ndim > 2:
                    m = m.squeeze()
                if not m.any() or m.shape != (H, W) or m[wpy, wpx] or not m[gy, gx]:
                    continue
                if m[y0:y1, x0:x1].sum() == 0 or not (0.0002 < float(m.mean()) < 0.40):
                    continue
                if all(_iou(m, c[1]) < 0.8 for c in cands):
                    cands.append(("", m, float(ps[i]), True))   # unnamed held-object mask -> LLM names it
            # CONTAINER-PREFERENCE: drop a candidate whose mask sits mostly INSIDE a larger one — loose
            # contents (beads) live inside the held container (cup), so keep the container, not the contents.
            if len(cands) > 1:
                cands.sort(key=lambda c: -float(c[1].sum()))     # largest area first
                kept = []
                for c in cands:
                    a = float(c[1].sum())
                    if not any(float(np.logical_and(c[1], k[1]).sum()) > 0.7 * a for k in kept):
                        kept.append(c)
                cands = kept
            # PREFER candidates the pinch is INSIDE (the actually-held object) over near-but-offset clutter.
            # Only narrow when some candidate contains the pinch — else keep the window set (a transparent
            # object's see-through interior may not be in its mask, so we must not require containment).
            pinchin = [c for c in cands if c[3]]
            if pinchin:
                cands = pinchin
            cands = [(n, m, s) for n, m, s, _ in cands]
            _dbg(f"[DBG] {tag} t={t:.1f} {hand}: candidates={[(n, round(s,2)) for n,_,s in cands]} "
                 f"(pinch-inside={len(pinchin)})")

            # STEP 3: LLM names; SAFETY NET overrides a false N/A when a strong RULE-1 candidate exists
            name, conf, chosen, src = "N/A", 0.0, None, ""
            if cands:
                # PINCH-CENTERED crop: zoom into the grasp so a small held object (pen part, bolt) is
                # isolated for Claude, not lost in a cluttered hand-box view of the whole pile.
                rad = int(0.12 * W)
                ex0, ey0 = max(0, gx - rad), max(0, gy - rad)
                ex1, ey1 = min(W, gx + rad), min(H, gy + rad)
                zoom = rgb[ey0:ey1, ex0:ex1].copy()
                for j, (nm, mk, sc) in enumerate(cands):
                    col = COLORS[j % len(COLORS)]
                    cont, _ = cv2.findContours(mk[ey0:ey1, ex0:ex1].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(zoom, cont, -1, col, 2)
                    yx = np.argwhere(mk[ey0:ey1, ex0:ex1])
                    if len(yx):
                        yc, xc = map(int, yx.mean(0))
                        cv2.putText(zoom, str(j), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)
                cv2.circle(zoom, (gx - ex0, gy - ey0), 7, (255, 40, 40), -1)
                zoom = cv2.resize(zoom, (zoom.shape[1] * 2, zoom.shape[0] * 2))
                zpath = os.path.join(tmpd, f"z_{tag}_{t:.1f}_{p['hand']}.jpg")
                cv2.imwrite(zpath, cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
                imgs = [zpath] + ([refsheet] if refsheet else [])
                choice, nm = call_llm(imgs, hand, len(cands), all_names)
                _dbg(f"[DBG] {tag} t={t:.1f} {hand}: LLM choice={choice} name={nm!r}")
                # RE-ASK once if Claude rejected a candidate that is RIGHT at the grasp (a point-prompt mask
                # contains the pinch). Claude over-rejects small in-place manipulation as "not lifted out";
                # this focused prompt counts it as holding, while still allowing N/A for a truly open/empty hand.
                if choice < 0 and any(c[0] == "" for c in cands):
                    # TIGHT crop on the candidate(s) — isolate + scale up the held part so Claude can name a
                    # tiny object (a pen refill) that's lost in the wider pile view.
                    union = np.logical_or.reduce([c[1] for c in cands])
                    ys2, xs2 = np.nonzero(union)
                    imgs2 = imgs
                    if len(xs2):
                        mg = int(0.05 * W)
                        tx0, ty0 = max(0, int(xs2.min()) - mg), max(0, int(ys2.min()) - mg)
                        tx1, ty1 = min(W, int(xs2.max()) + mg), min(H, int(ys2.max()) + mg)
                        tz = rgb[ty0:ty1, tx0:tx1].copy()
                        for j, (nm2, mk2, sc2) in enumerate(cands):
                            col = COLORS[j % len(COLORS)]
                            cont, _ = cv2.findContours(mk2[ty0:ty1, tx0:tx1].astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(tz, cont, -1, col, 2)
                            yx = np.argwhere(mk2[ty0:ty1, tx0:tx1])
                            if len(yx):
                                yc, xc = map(int, yx.mean(0))
                                cv2.putText(tz, str(j), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                        if 0 <= gy - ty0 < tz.shape[0] and 0 <= gx - tx0 < tz.shape[1]:
                            cv2.circle(tz, (gx - tx0, gy - ty0), 5, (255, 40, 40), -1)
                        scl = max(2, int(560 / max(1, tz.shape[1])))
                        tz = cv2.resize(tz, (tz.shape[1] * scl, tz.shape[0] * scl), interpolation=cv2.INTER_LINEAR)
                        tzp = os.path.join(tmpd, f"zt_{tag}_{t:.1f}_{p['hand']}.jpg")
                        cv2.imwrite(tzp, cv2.cvtColor(tz, cv2.COLOR_RGB2BGR))
                        imgs2 = [tzp] + ([refsheet] if refsheet else [])
                    fb = ("The highlighted object is RIGHT AT the grasp point (red dot), shown zoomed in. If the "
                          "hand is touching / holding / manipulating it — INCLUDING small in-place work like "
                          "assembling, threading, or picking a part from a pile — pick it and NAME it from the "
                          "list. Answer N/A ONLY if the hand is clearly OPEN or EMPTY with nothing in it.")
                    choice, nm = call_llm(imgs2, hand, len(cands), all_names, fb)
                    _dbg(f"[DBG] {tag} t={t:.1f} {hand}: RE-ASK(tight) choice={choice} name={nm!r}")
                # The LLM is the sole decider: it picks the manipulated candidate (+ name from the FULL
                # inventory) or N/A. A point-prompt candidate has no built-in name, so it relies on the LLM
                # naming it; if the LLM gives no valid name, it stays N/A.
                if 0 <= choice < len(cands):
                    cnm = nm if nm and nm.upper() not in ("N/A", "NA") else cands[choice][0]
                    if cnm:
                        name, chosen, conf, src = cnm, cands[choice][1], cands[choice][2], "llm"

            vis = rgb.copy(); locked = chosen is not None
            if locked:
                f = vis.astype(np.float32); f[chosen] = 0.5 * f[chosen] + 0.5 * np.array(COLORS[0], np.float32)
                vis = f.clip(0, 255).astype(np.uint8)
                cont, _ = cv2.findContours(chosen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, COLORS[0], 2)
                cap_txt = f"{tag} {hand}: {name}  (conf {conf:.2f}, {src})"
            else:
                cap_txt = f"{tag} {hand}: N/A"
            cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 160, 255), 1); cv2.circle(vis, (gx, gy), 7, (255, 40, 40), -1)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(vis, cap_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            fn = f"{tag}_t{p['t']:05.1f}_{p['hand']}.png"
            cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            index.append({"file": fn, "tag": tag, "hand": hand, "t": p["t"], "obj": bool(locked),
                          "name": name if locked else "N/A", "conf": round(conf, 3),
                          "low_conf": bool(locked and conf < TRUST_THR), "src": src})
            print(cap_txt, flush=True)
    cap.release()
    json.dump(index, open(os.path.join(outdir, f"_index_{tag}.json"), "w"), indent=2)
    n = sum(1 for r in index if r["obj"])
    print(f"\n{tag}: {len(index)} hand-frames, {n} grounded, {len(index) - n} N/A")


if __name__ == "__main__":
    main()
