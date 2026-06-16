#!/usr/bin/env python3
"""Stage-1A extractor — pulls the `TRACE [1A seed ...]` block verbatim from each clip's
console log into one document. PURE EXTRACTION: no scoring, no verdicts, no interpretation —
the human reads the raw 1A output and evaluates it.

Usage:  python stage_tests/eval_1A.py [out_dir]      # default out/v38
Writes: docs/stage_1A_contact_track.md
"""
import glob, os, re, sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/v38"
DOC = "docs/stage_1A_contact_track.md"
TS = re.compile(r"^\[\d\d:\d\d:\d\d\]")          # a timestamped log line


def seed_block(text):
    """Lines of the `TRACE [1A seed ...]` block (header + rows), verbatim."""
    out, grabbing = [], False
    for ln in text.splitlines():
        if "TRACE [1A seed" in ln:
            grabbing, out = True, [ln]
            continue
        if grabbing:
            if TS.match(ln):                      # next log line -> block ended
                break
            out.append(ln)
    return out


def dur_of(text):
    m = re.search(r"annotate .*?\(([\d.]+)s\)", text)
    return f"{m.group(1)}s" if m else "?"


def main():
    files = sorted(f for f in glob.glob(f"{OUT_DIR}/*.console.log") if "_batch" not in f)
    items = []
    for f in files:
        clip = os.path.basename(f).replace(".console.log", "")
        text = open(f, errors="replace").read()
        items.append((clip, dur_of(text), seed_block(text)))

    out = [f"# Stage 1A — contact track — raw output ({OUT_DIR})\n",
           "_Verbatim `TRACE [1A seed]` block from each clip's console log. No analysis added._\n",
           "## Index\n", "| clip | dur | rows |", "|------|----:|----:|"]
    for clip, dur, block in items:
        out.append(f"| {clip} | {dur} | {max(0, len(block) - 1)} |")
    out.append("")
    for clip, dur, block in items:
        out.append(f"## {clip} — {dur}\n")
        out.append("```\n" + ("\n".join(block) if block else "(no 1A block found)") + "\n```\n")

    os.makedirs("docs", exist_ok=True)
    open(DOC, "w").write("\n".join(out))
    print(f"wrote {DOC}  ({len(items)} clips)")


if __name__ == "__main__":
    main()
