#!/usr/bin/env python3
"""er16_overlay.py — burn the per-hand action timeline onto a video (GPU-fast, high quality).

Strategy: render each segment's panel ONCE as a transparent PNG with PIL (nice DejaVu
TTF, L=green / R=blue), then let ffmpeg do all the heavy lifting in one pass —
NVDEC decode + C-level `overlay=enable='between(t,a,b)'` + NVENC encode. No per-frame
Python loop. Segments are contiguous & non-overlapping, so exactly one panel is live
at any time. The clocked video already carries the yellow timestamp top-right, so the
goal pill goes top-left.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from media import FONT_PATH
except Exception:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

L_RGB = (61, 220, 132)        # green  (left hand)
R_RGB = (74, 163, 255)        # blue   (right hand)
PANEL_RGB = (16, 18, 22)


def _font(sz: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def _fit(draw, text: str, font, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def _seg_png(path: str, W: int, H: int, goal: str, idx: int, n: int,
             left: str, right: str) -> None:
    """Render one segment's overlay panel as a full-frame transparent PNG."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = max(1.0, W / 640.0)
    f_lab, f_chip, f_goal = _font(int(19 * s)), _font(int(15 * s)), _font(int(14 * s))
    pad, row_h = int(12 * s), int(30 * s)
    panel_h = row_h * 2 + pad * 3
    y0 = H - panel_h
    d.rectangle([0, y0, W, H], fill=(*PANEL_RGB, 175))
    d.rectangle([0, y0, W, y0 + max(1, int(2 * s))], fill=(255, 255, 255, 45))
    chip_w = int(26 * s)
    for i, (lab, col) in enumerate(((left, L_RGB), (right, R_RGB))):
        ry = y0 + pad + i * (row_h + pad)
        tag = "L" if i == 0 else "R"
        d.rounded_rectangle([pad, ry, pad + chip_w, ry + row_h], radius=int(6 * s),
                            fill=(*col, 255))
        cw = d.textlength(tag, font=f_chip)
        d.text((pad + (chip_w - cw) / 2, ry + row_h / 2 - int(9 * s)), tag,
               font=f_chip, fill=(10, 12, 16, 255))
        is_na = str(lab or "N/A").strip().upper().startswith("N/A")
        txt = _fit(d, str(lab or "N/A"), f_lab, W - (pad * 3 + chip_w))
        d.text((pad * 2 + chip_w, ry + row_h / 2 - int(11 * s)), txt,
               font=f_lab, fill=(235, 238, 242, 150 if is_na else 255))
    if goal:
        g = _fit(d, f"{idx + 1}/{n}  ·  {goal}", f_goal, int(W * 0.72))
        gw = d.textlength(g, font=f_goal)
        d.rounded_rectangle([pad, pad, pad * 2 + gw, pad + int(24 * s)],
                            radius=int(6 * s), fill=(*PANEL_RGB, 170))
        d.text((pad * 1.5, pad + int(3 * s)), g, font=f_goal, fill=(235, 238, 242, 255))
    img.save(path)


def _dims(video: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def overlay_video(video: str, segments: list[dict], goal: str, out_path: str) -> str:
    W, H = _dims(video)
    n = len(segments)
    if not n:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-c", "copy",
                        out_path], check=True)
        return out_path

    tmp = Path(tempfile.mkdtemp(prefix="er16ov_"))
    pngs = []
    for i, sg in enumerate(segments):
        p = str(tmp / f"seg_{i:04d}.png")
        _seg_png(p, W, H, goal, i, n, sg.get("left", "N/A"), sg.get("right", "N/A"))
        pngs.append(p)

    # one chained overlay per segment, gated to its time window (contiguous -> one live)
    parts, prev = [], "0:v"
    for i, sg in enumerate(segments):
        a, b = float(sg["start_sec"]), float(sg["end_sec"])
        cur = f"v{i}"
        parts.append(f"[{prev}][{i + 1}:v]overlay=0:0:"
                     f"enable='between(t,{a:.3f},{b:.3f})'[{cur}]")
        prev = cur
    fc = ";".join(parts)

    def _cmd(codec, hw):
        c = ["ffmpeg", "-v", "error", "-y", *hw, "-i", video]
        for p in pngs:
            c += ["-i", p]
        c += ["-filter_complex", fc, "-map", f"[{prev}]", "-an",
              "-c:v", codec, "-pix_fmt", "yuv420p"]
        if codec == "h264_nvenc":            # quality-targeted VBR, web-sane size for 640x360
            c += ["-preset", "p5", "-rc", "vbr", "-cq", "30", "-b:v", "2M", "-maxrate", "3M"]
        c += ["-movflags", "+faststart", out_path]   # moov atom up front -> instant seek/stream
        return c

    # GPU first (NVDEC decode + NVENC encode), then CPU fallback — each is ONE fast pass
    for codec, hw in (("h264_nvenc", ["-hwaccel", "cuda"]), ("libx264", [])):
        r = subprocess.run(_cmd(codec, hw), capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[overlay] {codec}: {n} panels -> {out_path}")
            return out_path
        if codec == "h264_nvenc":
            print(f"[overlay] nvenc failed ({r.stderr.strip()[-160:]}), trying libx264")
    raise SystemExit(f"overlay: ffmpeg failed: {r.stderr.strip()[-300:]}")


if __name__ == "__main__":
    import json
    ep = json.loads(Path(sys.argv[1]).read_text())
    out = sys.argv[2] if len(sys.argv) > 2 else "overlaid.mp4"
    src = ep.get("_clocked_plain") or ep.get("_clocked")
    overlay_video(src, ep["segments"], ep.get("goal", ""), out)
