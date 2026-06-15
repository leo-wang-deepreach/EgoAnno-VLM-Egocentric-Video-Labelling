# CLAUDE.md — facts-first egocentric annotation pipeline

A VLM pipeline that turns a short egocentric two-handed manipulation video into a
per-hand action timeline for robot learning. SDK-free (raw HTTP). Models:
`gemini-3.1-pro-preview` (native video: facts, segmentation, labeling) and
`claude-opus-4-8` (verifying / critics / gate / fresh-eye). No gemini-flash, no GPT.

## Working style (follow on EVERY prompt)
- **Plan before acting.** For every request, think through the approach first; for
  non-trivial work lay out the plan before editing.
- **Ask, don't guess.** When something is ambiguous or a choice is genuinely the user's,
  ASK a clarifying question rather than guessing — EXCEPT when the user has said to run
  autonomously, or the user is currently away / not present. In those cases, pick the
  best option, state the assumption, and proceed.

## Version control (MANDATORY)
- Repo: `https://github.com/leo-wang-deepreach/EgoAnno-VLM-Egocentric-Video-Labelling`
  (this `factsfirst/` dir is its own git repo; auth as `leo-wang-deepreach`).
- **ALWAYS commit + push the current version to GitHub BEFORE launching a new version's
  run.** The prior version must be safely saved first — never start a new run with
  uncommitted pipeline changes. Each version: commit, tag `vNN`, push `main` + the tag.
- **NEVER commit** secrets (`.gemini_key` / `.openai_key` / `.anthropic_key`), run logs,
  videos, generated images, or held-out ground truth. `.gitignore` enforces this; still
  scan staged content before pushing (the repo is PUBLIC).

## Pipeline (phases)
- P1a `contact_track` (Gemini 10fps): objects catalogue + per-frame foreground/background
  + which object each hand touches. Facts only, no actions; tracks object decomposition.
- P1c transition scan (opus, sliding 30fps windows): place/pickup/handoff events.
- P2 rotation bursts → deterministic reduce. P3 `direction_decide` (+ derived goal).
- P4 segmentation (`v49_segment`, 10fps) is the SOLE cut authority — transitions are only
  soft hints, never pre-injected cuts. Per-segment native labeling (parallel) → merge
  (same-action, generic token overlap, no wordlist) → seg_reconcile → verifiers.
- P5 gate (opus, sees frames + advisory code-analysis). P6 fresh-eye: opus reviews the
  clip with labels overlaid and NO other context, returns a corrected timeline.
- `analysis.py`: deterministic code-only signals (no CV/models/wordlists), ADVISORY,
  fed to the gate clearly marked code-generated.

## Hard constraints
- No held-out ground truth / calibration / dataset examples in any model-facing prompt.
- No hard-coded semantic wordlists (determinism for parsing, not for judgement).
- Double-sided: every directional instruction names assembly AND disassembly.

## Run / publish
- `./run.sh <video> --out out/<dir>/<clip>.json --workdir logs/<clip>_<ver> --attempts 3`
  (uses the parent egoanno `.venv`). Per-segment work is parallel (`max_workers=4`).
- Publish to the dashboard (version-aware): `python to_viewer.py <episode.json>` — version
  is inferred from the output dir (out/v20 → v20). Served at `:8800/viewer/factsfirst.html`.
