#!/usr/bin/env python3
"""media.py — the single frame-rendering contract for the facts-first pipeline.

Every stage that reads pixels goes through here, so the labeler and the gate never
disagree because they saw different frames. Three things are attached to a strip:
  (a) CLOCK       — burned once into the working video (top-right, ms precision).
  (b) EDGE TABS   — "L|" bottom-left, "|R" bottom-right of every frame, so the
                    FOREARM-L/R rule is visual, not a thing the model must remember.
  (c) TRACK CAP   — a scoped one-line possession caption for the call's window,
                    returned as text to paste into the prompt.

GPU is used for the one heavy encode (clock burn -> NVENC); per-call frame
extraction is cheap CPU ffmpeg. Audio is stripped at clock-burn time so EVERY
downstream native-Gemini call is audio-off by construction.
"""
from __future__ import annotations

import base64
import glob
import os
import subprocess
import tempfile
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ----------------------------------------------------------------------------- #
#  ffprobe / clock burn                                                         #
# ----------------------------------------------------------------------------- #
def probe_duration(video: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def burn_clock(video: str, out_path: str) -> str:
    """Burn a top-right MICROSECOND-precision seconds clock and STRIP AUDIO (-an).

    The clock reads absolute time in SECONDS at microsecond precision (e.g.
    "4.233333"), not HH:MM:SS — every stage talks in seconds, and at 10fps/30fps
    each frame must show a distinct, reasonable time. Audio off is a facts-first
    requirement for every Gemini call; doing it here once means the uploaded video
    carries no audio at all. GPU (h264_nvenc) first, libx264 fallback.
    """
    vf = (f"drawtext=fontfile={FONT_PATH}:text='%{{pts\\:flt}}s':fontsize=44:"
          f"fontcolor=yellow:box=1:boxcolor=black@0.7:boxborderw=8:x=w-tw-16:y=16")
    err = b""
    for codec in ("h264_nvenc", "libx264"):
        hw = ["-hwaccel", "cuda"] if codec == "h264_nvenc" else []
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *hw, "-i", video, "-map", "0:v:0",
             "-vf", vf, "-an", "-c:v", codec, "-pix_fmt", "yuv420p", out_path],
            capture_output=True)
        if r.returncode == 0:
            return out_path
        err = r.stderr
    raise SystemExit(f"clock burn failed: {err[:300]}")


# ----------------------------------------------------------------------------- #
#  frame extraction                                                             #
# ----------------------------------------------------------------------------- #
def extract_frames(video: str, a: float, dur: float, fps: float, max_side: int,
                   workdir: str) -> list[tuple[float, str]]:
    """Extract frames over [a, a+dur] at `fps` -> [(t_sec, b64_jpeg), ...].

    The clock is already burned into `video`, so t_sec here is for our own
    bookkeeping (edge tabs / dimming); the model reads time off the pixels.
    """
    sub = tempfile.mkdtemp(dir=workdir)
    pat = os.path.join(sub, "f_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, a):.3f}", "-t", f"{dur:.3f}",
         "-i", video, "-vf", f"fps={fps},scale=w='min(iw\\,{max_side})':h=-2",
         "-q:v", "4", pat], capture_output=True)
    out = []
    files = sorted(glob.glob(os.path.join(sub, "f_*.jpg")))
    for i, f in enumerate(files):
        t = max(0.0, a) + (i + 0.5) / fps          # frame center time
        try:
            data = open(f, "rb").read()
        except OSError:
            continue
        # SKIP empty / truncated JPEGs (e.g. ffmpeg cut short by a full disk) — a single bad
        # frame must never reach PIL and crash the whole run. Valid JPEG starts with FFD8.
        if len(data) < 128 or data[:2] != b"\xff\xd8":
            continue
        out.append((round(t, 3), base64.b64encode(data).decode()))
    return out


# ----------------------------------------------------------------------------- #
#  edge tabs + context dimming (PIL)                                            #
# ----------------------------------------------------------------------------- #
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _tab(draw: ImageDraw.ImageDraw, xy, text, fnt, anchor):
    x, y = xy
    w = draw.textlength(text, font=fnt)
    h = fnt.size
    pad = 6
    if anchor == "left":
        box = (x, y - h - pad, x + w + 2 * pad, y)
        tx = x + pad
    else:  # right
        box = (x - w - 2 * pad, y - h - pad, x, y)
        tx = x - w - pad
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((tx, y - h - pad + 1), text, fill=(0, 255, 120), font=fnt)


def add_edge_tabs(b64_jpeg: str, dim: bool = False) -> str:
    """Draw 'L|' bottom-left and '|R' bottom-right. If dim=True, darken the frame
    and badge it 'ctx' (a context/overlap frame the model must not label)."""
    im = Image.open(BytesIO(base64.b64decode(b64_jpeg))).convert("RGB")
    if dim:
        im = Image.eval(im, lambda p: int(p * 0.45))
    W, H = im.size
    d = ImageDraw.Draw(im)
    fnt = _font(max(20, H // 28))
    _tab(d, (8, H - 8), "L|", fnt, "left")
    _tab(d, (W - 8, H - 8), "|R", fnt, "right")
    if dim:
        cf = _font(max(18, H // 32))
        cw = d.textlength("ctx", font=cf)
        d.rectangle((W // 2 - cw // 2 - 6, 8, W // 2 + cw // 2 + 6, 8 + cf.size + 6),
                    fill=(0, 0, 0))
        d.text((W // 2 - cw // 2, 10), "ctx", fill=(255, 180, 0), font=cf)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- #
#  scoped track caption                                                         #
# ----------------------------------------------------------------------------- #
def track_caption(track: dict, a: float, b: float) -> str:
    """One-line possession caption scoped to [a, b]: what each forearm holds.

    'you cannot act on what you don't hold' is put IN the model's view, not hoped
    for. Intervals are clipped to the window; empty/absent spans are omitted.
    """
    parts = []
    for hand in ("left", "right"):
        H = "L" if hand == "left" else "R"
        segs = []
        for iv in track.get(hand, []):
            s = max(a, iv.get("start_sec", iv.get("start", a)))
            e = min(b, iv.get("end_sec", iv.get("end", b)))
            if e <= s + 0.05:
                continue
            obj = str(iv.get("interacting_with", "") or "").strip()
            if obj.lower() in ("", "none", "empty", "out of frame", "nothing", "n/a"):
                obj = "empty"
            segs.append(f"{obj} [{s:.1f}..{e:.1f}]")
        parts.append(f"{H}: " + ("; ".join(segs) if segs else "empty/absent"))
    return "track  " + "  |  ".join(parts)


# ----------------------------------------------------------------------------- #
#  the render_strip contract                                                    #
# ----------------------------------------------------------------------------- #
def render_strip(clocked: str, a: float, b: float, fps: float, track: dict,
                 workdir: str, max_side: int = 960, ctx: float = 0.0,
                 cap_frames: int = 100) -> tuple[list[str], str]:
    """Render the strip EVERY stage uses.

    Extract [a-ctx, b+ctx] at `fps`, burn edge tabs into every frame, dim the
    frames that fall in the context margins (outside the core [a,b]), and return
    (frames_b64, scoped_track_caption). `ctx` adds neighbor-overlap frames so a
    boundary action is never seen half-cut.
    """
    oa, ob = max(0.0, a - ctx), b + ctx
    raw = extract_frames(clocked, oa, ob - oa, fps, max_side, workdir)
    if len(raw) > cap_frames:                       # decimate uniformly, keep span
        step = len(raw) / cap_frames
        raw = [raw[int(i * step)] for i in range(cap_frames)]
    frames = []
    for t, b64 in raw:
        is_ctx = t < a - 0.05 or t > b + 0.05
        frames.append(add_edge_tabs(b64, dim=is_ctx and ctx > 0))
    return frames, track_caption(track, a, b)


def _overlay_labels(b64_jpeg: str, left: str, right: str) -> str:
    """Burn the CURRENTLY-ASSIGNED L/R labels onto a frame for the fresh-eye review:
    'L: <left>' top-left, 'R: <right>' top-right, plus the L|/|R edge tabs.
    Returns None for an undecodable frame so the caller can skip it (never crash)."""
    try:
        im = Image.open(BytesIO(base64.b64decode(b64_jpeg))).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    d = ImageDraw.Draw(im)
    fnt = _font(max(16, H // 34))
    pad = 5
    for text, anchor in ((f"L: {left}", "left"), (f"R: {right}", "right")):
        tw = d.textlength(text, font=fnt)
        if anchor == "left":
            box = (0, 0, tw + 2 * pad, fnt.size + 2 * pad); tx = pad
        else:
            box = (W - tw - 2 * pad, 0, W, fnt.size + 2 * pad); tx = W - tw - pad
        d.rectangle(box, fill=(0, 0, 0))
        col = (120, 200, 255) if anchor == "left" else (255, 160, 160)
        d.text((tx, pad), text, fill=col, font=fnt)
    _tab(d, (8, H - 8), "L|", _font(max(20, H // 28)), "left")
    _tab(d, (W - 8, H - 8), "|R", _font(max(20, H // 28)), "right")
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def render_labeled(clocked: str, segments, fps: float, workdir: str,
                   max_side: int = 720, cap_frames: int = 80) -> list[str]:
    """Whole-clip frames with each segment's assigned L/R label overlaid — the input to
    the fresh-eye review. The model sees the video AS LABELLED and corrects mismatches.
    `segments` is a list of objects with .start/.end/.left/.right."""
    if not segments:
        return []
    dur = max(s.end for s in segments)
    raw = extract_frames(clocked, 0.0, dur, fps, max_side, workdir)
    if len(raw) > cap_frames:
        step = len(raw) / cap_frames
        raw = [raw[int(i * step)] for i in range(cap_frames)]
    out = []
    for t, b64 in raw:
        seg = next((s for s in segments if s.start - 0.05 <= t <= s.end + 0.05),
                   segments[-1])
        lab = _overlay_labels(b64, seg.left or "N/A", seg.right or "N/A")
        if lab is not None:                         # skip any frame that failed to decode
            out.append(lab)
    return out
