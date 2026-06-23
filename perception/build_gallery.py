#!/usr/bin/env python3
"""build_gallery.py — merge ground_review shard indexes (_index_*.json) into one scrollable
index.html gallery. Run after the parallel SAM workers finish.

Run: ../.venv/bin/python perception/build_gallery.py <review_dir>
"""
import glob
import json
import os
import sys

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "_index_*.json"))):
    rows += json.load(open(f))
rows.sort(key=lambda r: (r["tag"], r["t"], r["hand"]))

cards = []
for r in rows:
    flag = "" if r["obj"] else " · NONE"
    cards.append(
        f'<figure><img src="{r["file"]}?v=ts" loading="lazy">'
        f'<figcaption>{r["tag"]} · {r["hand"]} → <b>{r.get("name", "?")}</b>{flag} · t={r["t"]:.1f}s'
        f'</figcaption></figure>')
nobj = sum(1 for r in rows if r["obj"])
html = (
    "<!doctype html><meta charset=utf-8><title>SAM grounding review (" + str(len(rows)) + ")</title>"
    "<style>body{background:#111;color:#eee;font-family:system-ui;margin:0;padding:14px}"
    "h1{font-size:15px;font-weight:600}.grid{display:grid;"
    "grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}"
    "figure{margin:0;background:#1b1b1b;border-radius:8px;overflow:hidden}"
    "img{width:100%;display:block}figcaption{font-size:12px;padding:6px 8px;color:#bbb}</style>"
    f"<h1>SAM grounding review — {len(rows)} hand-frames ({nobj} object-locked). "
    "Green = masked region (what the hand is judged to manipulate) · red dot = measured fingertip · "
    "blue ✕ = negative hand points.</h1><div class=grid>" + "".join(cards) + "</div>")
open(os.path.join(d, "index.html"), "w").write(html)
print(f"gallery: {len(rows)} frames ({nobj} object-locked) -> {d}/index.html")
