#!/usr/bin/env python3
"""PreToolUse / Edit|Write guard for the egoanno (factsfirst) project. Block-and-ask:
exit 2 BLOCKS the write; exit 0 allows it.

Blocks writing leaky content INTO a model-facing prompt file (factsfirst/prompts/*.txt):
clip UUIDs, ground-truth paths, or reviewer-calibration material. Prompts must use typed
placeholders (<part>, <counterpart>, <container>, <color>, <stable location>) only.

FAIL-OPEN: on any internal error the guard allows the write. It blocks ONLY on a confirmed hit.
Scoped to factsfirst/prompts/ — no-ops for every other file.
"""
import json
import re
import sys

PROMPTS_DIR = "/home/ubuntu/local/factsfirst/prompts/"
BAD = [
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"), "clip UUID"),
    (re.compile(r"out/(gt|leo_edited|overrides)\b"), "ground-truth path"),
    (re.compile(r"review_calibration", re.I), "calibration file reference"),
    (re.compile(r"REVIEWER CALIBRATION", re.I), "calibration block"),
    (re.compile(r"\{calib", re.I), "calibration injection variable"),
]


def main():
    try:
        ti = (json.load(sys.stdin).get("tool_input") or {})
    except Exception:
        sys.exit(0)  # fail-open
    fp = ti.get("file_path", "") or ""
    if not (fp.startswith(PROMPTS_DIR) and fp.endswith(".txt")):
        sys.exit(0)
    content = ti.get("content")
    if content is None:
        content = ti.get("new_string", "")
    content = content or ""
    try:
        hits = [why for rx, why in BAD if rx.search(content)]
    except Exception:
        sys.exit(0)  # fail-open
    if hits:
        sys.stderr.write(
            "BLOCKED by egoanno prompt guard — leaky content in a prompt file:\n  "
            + fp + "\n  contains: " + ", ".join(hits)
            + "\nPrompts must use typed placeholders, never GT / clip-IDs / calibration.\n")
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
