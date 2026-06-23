#!/usr/bin/env python3
"""build_artifact.py — render a grounding review dir into ONE self-contained HTML gallery (all frames
embedded as downscaled JPEG data URIs) for publishing as a claude.ai Artifact (no server / no ports).
Re-run each iteration on the new review dir; publish to the same Artifact URL.

Run: ../.venv/bin/python perception/build_artifact.py <review_dir> <out.html> [title]
"""
import base64
import glob
import io
import json
import sys

import cv2
from PIL import Image


def main():
    revs = sys.argv[1].split(",")          # one or more review dirs (comma-separated)
    out_html = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else revs[0].rstrip("/").split("/")[-1]
    rows = []
    for rev in revs:
        for f in glob.glob(f"{rev}/_index_*.json"):
            for r in json.load(open(f)):
                r["_dir"] = rev; rows.append(r)
    rows.sort(key=lambda r: (r.get("tag", ""), r["t"], r["hand"]))
    g = [r for r in rows if r["obj"]]
    low = [r for r in g if r.get("low_conf")]
    na = [r for r in rows if not r["obj"]]

    cards = []
    for r in rows:
        img = cv2.imread(f"{r['_dir']}/{r['file']}")
        if img is None:
            continue
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im.thumbnail((620, 620))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=80)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        if not r["obj"]:
            cls, label = "na", "N/A"
        elif r.get("low_conf"):
            cls, label = "low", f"{r['name']} · conf {r.get('conf')} (LOWCONF)"
        else:
            cls, label = "ok", f"{r['name']} · conf {r.get('conf')}"
        cap = f"{r.get('tag', '')} · t={r['t']:.1f}s · {r['hand']}"
        cards.append(f'<figure class="{cls}"><img loading="lazy" src="{uri}">'
                     f'<figcaption><b>{cap}</b><br>{label}</figcaption></figure>')

    html = f"""<title>{title} — grounding review</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#15171c; color:#e8eaed;
         font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:#1d2026; padding:14px 20px;
            border-bottom:1px solid #2c313a; z-index:5; }}
  h1 {{ margin:0 0 6px; font-size:18px; }}
  .stats span {{ display:inline-block; margin-right:16px; font-size:13px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px;
          vertical-align:middle; }}
  .grid {{ display:grid; gap:12px; padding:16px;
           grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); }}
  figure {{ margin:0; background:#1d2026; border-radius:8px; overflow:hidden;
            border-left:4px solid #555;
            content-visibility:auto; contain-intrinsic-size:auto 280px; }}  /* skip off-screen render */
  figure.ok {{ border-left-color:#21d07a; }}
  figure.low {{ border-left-color:#ff9d2e; }}
  figure.na {{ border-left-color:#5b6270; opacity:.85; }}
  img {{ width:100%; display:block; cursor:zoom-in;
         aspect-ratio:16/9; object-fit:cover; background:#0d0f13; }}  /* reserve space -> no layout jump */
  figcaption {{ padding:8px 10px; font-size:13px; }}
  .ok figcaption b {{ color:#21d07a; }} .low figcaption b {{ color:#ff9d2e; }}
  .na figcaption b {{ color:#9aa3b2; }}
  #ov {{ position:fixed; inset:0; background:rgba(0,0,0,.92); display:none; cursor:zoom-out;
         align-items:center; justify-content:center; z-index:20; }}
  #ov img {{ max-width:97%; max-height:97%; width:auto; }}
</style>
<header>
  <h1>{title} — per-hand object grounding</h1>
  <div class="stats">
    <span><i class="dot" style="background:#21d07a"></i>grounded {len(g) - len(low)}</span>
    <span><i class="dot" style="background:#ff9d2e"></i>low-conf {len(low)}</span>
    <span><i class="dot" style="background:#5b6270"></i>N/A {len(na)}</span>
    <span>· {len(rows)} hand-frames</span>
  </div>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
<div id="ov" onclick="this.style.display='none'"><img id="ovimg"></div>
<script>
  document.querySelectorAll('.grid img').forEach(im => im.onclick = e => {{
    e.stopPropagation(); const o=document.getElementById('ov');
    document.getElementById('ovimg').src = im.src; o.style.display='flex';
  }});
</script>
"""
    open(out_html, "w").write(html)
    print(f"wrote {out_html} ({len(cards)} frames, {len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
