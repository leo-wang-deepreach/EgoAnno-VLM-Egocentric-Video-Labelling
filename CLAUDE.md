# CLAUDE.md — egocentric annotation pipeline

A VLM pipeline that turns an egocentric two-handed manipulation video into a per-hand
action timeline for robot learning. SDK-free (raw HTTP). ONE clean pipeline, no modes/flags.
Three model families: `gemini-3.1-pro-preview` (native video: segmentation, seg-reconcile,
labeling, pick<->place completeness, direction), `claude-opus-4-8` (merge-critic + frame
verifier/gate), and `gpt-5.5` (3rd-family whole-clip global audit). No gemini-flash.

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

## Pipeline (one clean pass, no flags)
- P0 hand overlay (yolo handpose): burn GREEN=L / BLUE=R circles into the video so every
  stage reads authoritative hand identity. P0b clock burn (timestamp + audio strip) → the
  single working video every stage reads.
- P1 bursts (opus, frames) rotation/role/colour → reduce. P2 direction (Gemini 10fps whole)
  → direction + derived goal.
- LOOP (≤ max_attempts, keep best by (verdict, −#segs); the verifier's feedback picks the
  re-entry stage — relabel by default, full re-segment on a direction flip):
  - **S** segment (Gemini 10fps) cut times only · **S+** seg-reconcile splits swallowed
    multi-action cycles · **A** label (Gemini 10fps, parallel/seg) per-hand on fixed spans ·
    **R** atomic-label contract (deterministic) fuzzy same-action merge + chained flag ·
    **S2** merge-critic (opus frames) delete-only merge · **PP** pick↔place completeness
    (Gemini 10fps) every pick has a matching place/handoff & vice-versa · **V** verifier
    (opus frames) verdict + corrections + rerun feedback.
- **W** global audit (gpt-5.5, frames): 3rd-family whole-clip context-free corrected timeline.
- Long clips (> CHUNK_SEC = 300s / 5min) are split into even ≤5min parts, each run through the
  SAME pipeline in parallel, then merged and every seam re-flowed.

## Hard constraints
- No held-out ground truth / calibration / dataset examples in any model-facing prompt.
- No hard-coded semantic wordlists (determinism for parsing, not for judgement).
- Double-sided: every directional instruction names assembly AND disassembly.

## Run / publish
- `./run.sh <video> --out out/<dir>/<clip>.json [--workdir logs/<clip>_<ver>]` — NO behavioral
  flags (single pipeline). Uses the parent egoanno `.venv`. Per-segment work is parallel.
- Publish to the dashboard (version-aware): `python to_viewer.py <episode.json>` — version is
  inferred from the output dir (out/v44 → v44). Served at `:8800/factsfirst/factsfirst.html`.
