#!/usr/bin/env python3
"""PreToolUse / Bash guard for the egoanno (factsfirst) project. Block-and-ask:
exit 2 (with a reason on stderr) BLOCKS the Bash call; exit 0 allows it.

Two concerns, one process (cheap):
  1. GIT LEAK GUARD — block `git commit`/`git push` that would put a secret or
     held-out ground truth into a repo, and block `git add -f/--force` (which
     bypasses .gitignore). The repo is PUBLIC, so a leak is permanent.
  2. PRE-RUN LEAK GATE — before launching a pipeline OR grounder run, verify every
     model-facing prompt (prompts/*.txt AND the inline prompts in perception/llm_*.py)
     is leak-clean via judge_leak_check.py; block the run if any prompt leaks.

FAIL-OPEN: on any internal error the guard allows the call (a guard bug must never
brick all git/runs). It blocks ONLY on a confirmed hit. .gitignore is the backstop.
"""
import glob
import json
import os
import re
import subprocess
import sys

LOCAL = "/home/ubuntu/local"
FACTS = LOCAL + "/factsfirst"
CHECKER = LOCAL + "/judge_leak_check.py"
PY = "/usr/bin/python3"

# git repos to scan when a commit/push is attempted (only those with staged content matter)
REPOS = [FACTS, LOCAL + "/sam3",
         "/home/ubuntu/Internship/egoanno", "/home/ubuntu/Internship/egoanno/egoanno-pipeline"]

# secret / GT / held-out material on a staged PATH
PATH_BAD = re.compile(
    r"(^|/)\.(gemini|openai|anthropic|claude)_key$|_key$|\.key$|(^|/)\.env|\.s3creds|\.lark"
    r"|out/gt/|out/leo_edited/|out/overrides/|review_calibration"
    r"|_gt\.json$|\.gt\.json$|OUTPUT_AND_GT|FEEDBACK_BUNDLE|ground_truth", re.I)
# secret material in staged CONTENT (brackets in these literals keep them from self-matching)
CONTENT_BAD = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "OpenAI-style secret key"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "Google API key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
]
LAUNCH = re.compile(r"\b(run\.sh|run\.py|pipeline2?\.py|leo_batch\.py|ground_simple\.py|ground_v16\.py)\b")

# REMOVAL GUARD — block-and-ask before anything that DELETES/DISCARDS, so an impulsive `rm -rf`
# can't nuke source/snapshots/logs/GT (the v30/v31 thrash). Destructive verbs:
# _CMD = a COMMAND-position prefix (start, or after a shell separator) + optional sudo/time, so the
# verb is the command being RUN — not a substring of an argument (`grep rm`, `echo "...rm..."`).
_CMD = r"(?:^|[\n;|&(]\s*)(?:sudo\s+|time\s+)?"
DESTRUCTIVE = re.compile(
    _CMD + r"(?:rm|rmdir|shred)\b"
    r"|" + _CMD + r"truncate\s+-s\s*0\b"
    r"|" + _CMD + r"git\s+(?:rm|clean)\b"
    r"|" + _CMD + r"git\s+reset\s+--hard\b"
    r"|" + _CMD + r"git\s+(?:checkout|restore)\b[^|;&\n]*\s--(?:\s|$)"
    r"|" + _CMD + r"find\b[^|;&\n]*\s-delete\b"
    r"|" + _CMD + r"find\b[^|;&\n]*-exec\s+rm\b"
    r"|" + _CMD + r"xargs\s+(?:-\S+\s+)*rm\b", re.I)
# A destructive op is allowed ONLY when it is plainly confined to a scratch zone (regenerable temp).
SAFE_ZONE = re.compile(r"/tmp/|/scratchpad/")
PROTECTED = re.compile(r"/home/ubuntu/local|factsfirst|versions/|\.git\b|leo_edited|Videos-Leo|out/", re.I)


def block(msg):
    sys.stderr.write("BLOCKED by egoanno bash guard:\n" + msg + "\n")
    sys.exit(2)


def remove_guard(cmd):
    if not DESTRUCTIVE.search(cmd):
        return
    if SAFE_ZONE.search(cmd) and not PROTECTED.search(cmd):
        return                                    # purely /tmp or scratchpad cleanup — allowed
    block("about to REMOVE / DISCARD something (rm · git rm · git clean · reset --hard · restore · "
          "find -delete · shred · truncate).\n"
          "  This guard is BLOCK-AND-ASK: stop and get the user's explicit OK first.\n"
          "  Protected = anything outside /tmp or the scratchpad (source, versions/ snapshots, .git,\n"
          "  out/ outputs, GT, Videos-Leo). If the user approves, THEY run it (type `! <cmd>`) — do\n"
          "  NOT bypass this guard yourself.")


def git_guard(cmd):
    if re.search(r"\bgit\b.*\badd\b.*(-f\b|--force\b)", cmd):
        block("`git add -f/--force` bypasses .gitignore — staging is blocked. "
              "Stage files explicitly (no -f). The repo is PUBLIC.")
    if not re.search(r"\bgit\b.*\b(commit|push)\b", cmd):
        return
    hits = []
    for repo in REPOS:
        if not os.path.isdir(os.path.join(repo, ".git")):
            continue
        try:
            names = subprocess.run(["git", "-C", repo, "diff", "--cached", "--name-only"],
                                   capture_output=True, text=True, timeout=10).stdout
        except Exception:
            continue
        for n in names.splitlines():
            if n and PATH_BAD.search(n):
                hits.append(f"  [{os.path.basename(repo)}] staged path hits secret/GT rule: {n}")
        try:
            diff = subprocess.run(["git", "-C", repo, "diff", "--cached"],
                                  capture_output=True, text=True, timeout=15).stdout
        except Exception:
            diff = ""
        for rx, why in CONTENT_BAD:
            if rx.search(diff):
                hits.append(f"  [{os.path.basename(repo)}] staged content contains {why}")
    if hits:
        block("commit/push would leak secret or held-out GT into a PUBLIC repo:\n"
              + "\n".join(hits) + "\nUnstage the offending file(s) before committing.")


def run_gate(cmd):
    if not LAUNCH.search(cmd) or not os.path.exists(CHECKER):
        return
    # Model-facing prompts live BOTH in prompts/*.txt (frozen pipeline) AND inline in the
    # active grounder's LLM helpers (perception/llm_*.py). Scan ALL of them before any run,
    # so the grounder launch path (sam3py ground_simple.py ...) can't bypass the leak gate.
    targets = (sorted(glob.glob(os.path.join(FACTS, "prompts", "*.txt")))
               + sorted(glob.glob(os.path.join(FACTS, "perception", "llm_*.py"))))
    leaks = []
    for p in targets:
        try:
            r = subprocess.run([PY, CHECKER, p], capture_output=True, text=True, timeout=15)
        except Exception:
            continue
        if r.returncode != 0:
            tail = (r.stdout or "").strip().splitlines()
            leaks.append(f"  {os.path.basename(p)}: {tail[-1] if tail else 'leak'}")
    if leaks:
        block("pipeline run blocked — prompts are NOT leak-clean (no GT/clip-IDs/dataset "
              "objects allowed in a model-facing prompt):\n" + "\n".join(leaks))


def main():
    try:
        data = json.load(sys.stdin)
        cmd = (data.get("tool_input") or {}).get("command", "") or ""
    except Exception:
        sys.exit(0)  # fail-open
    try:
        remove_guard(cmd)
        git_guard(cmd)
        run_gate(cmd)
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)  # fail-open on any unexpected error
    sys.exit(0)


if __name__ == "__main__":
    main()
