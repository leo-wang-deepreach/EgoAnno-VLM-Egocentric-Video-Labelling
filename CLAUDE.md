# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A VLM pipeline that turns an egocentric two-handed manipulation video into a per-hand atomic
action timeline (`{start_sec, end_sec, left, right}` per segment) for robot learning. SDK-free
(raw HTTP to each model). Three model families: `gemini-3.1-pro-preview` (native video),
`claude-opus-4-8` (frame verifier/gate, bursts, merge-critic), `gpt-5.5` (3rd-family global
audit). No gemini-flash.

## Two pipelines (this is the thing to understand first)
**Direction (locked 2026-06-18): the measured spine (#2) is the path we invest in; the VLM-heavy
flow (#1) is frozen prior production, kept runnable for comparison.** Both paths share the support
modules but differ fundamentally in how they find structure:

1. **`pipeline.py` — FROZEN prior production, VLM-heavy.** Entry: `run.sh` → `run.py` → `annotate()`.
   Kept runnable for comparison; new work does NOT go here.
   The VLM both *measures* (cut times, hand roles) and *names*. One clean flow, no behavioral
   flags. Pipeline:
   - **P0** hand overlay (yolo handpose): burn GREEN=L / BLUE=R circles so every stage reads
     authoritative hand identity. **P0b** clock burn (timestamp + audio strip) → the single
     working video every stage reads.
   - **P1** bursts (opus, frames) rotation/role/colour → reduce. **P2** direction (Gemini 10fps).
   - **LOOP** (≤ max_attempts, keep best by `(verdict, −#segs)`; the verifier's feedback picks
     the re-entry stage — relabel by default, full re-segment on a direction flip): **S** segment
     (Gemini) cut times · **S+** seg-reconcile splits swallowed multi-action cycles · **A** label
     (Gemini, per-seg) · **R** atomic-contract (deterministic) fuzzy same-action merge · **S2**
     merge-critic (opus, frames) delete-only · **PP** pick↔place completeness (Gemini) · **V**
     verifier (opus, frames) verdict + corrections + rerun feedback.
   - **W** global audit (gpt-5.5, frames): 3rd-family whole-clip corrected timeline.
   - Long clips (> `CHUNK_SEC` = 300s) split into ≤5min parts, each run through the SAME
     pipeline in parallel, then merged and every seam re-flowed.

2. **`perception/pipeline2.py` — THE CHOSEN PATH (measured spine).** This is where new work
   lands, including the SAM object-grounding that targets the wrong-object problem.
   Thesis: **the VLM never measures — it only NAMES fixed measured spans.** Boundaries, acting
   hand, grip events, and rotation are all measured off the handpose model in pure code; the
   only VLM calls are one direction call + per-segment labeling, and labeling uses **only**
   `claude-opus-4-8` (no Gemini/GPT). ~20× faster and deterministic (same input → same
   segmentation). Flow: clock burn → `perception.extract` (per-frame handpose signals) →
   `motion_energy` → `detect_segments` (action vs transition spans) → `rotation.refine_long_segments`
   → `seg_facts` (fact-pack) → `name_objects` (1 cropped Claude call → fixed object vocabulary) →
   `decide_direction` (1 Claude call) → `segment_rotation`+`calibrate` (Spec B) → `idle_segments`
   (Spec E) → `label_segments` (per-seg parallel Claude, **cropped to the grasp box + L/R wrist
   tags burned in**) → deterministic contract post-check → `derive_goal` → `cross_check` →
   `comb_segments`/`absorb_tiny` → export.
   - The grounding trick lives in `grasp_box` (crop to what's *in the hands*, not the background)
     and the "Specs": **Spec A** = measured facts dictate a required action class per hand
     (PICK/PLACE/MOVE/FINE/NA), the VLM only fills object + wording; **Spec B** = optical-flow
     curl measures the turner hand + screw/unscrew direction (the VLM never decides direction);
     **Spec E** = idle hands are deterministically `N/A`, never sent to the VLM.

Both paths export the same `episode.json` via `qa.export_episode` and share `models.py`
(HTTP clients + token ledger), `media.py` (clock burn, frame extraction, overlays),
`clipstate.py` (`ClipState`/`Segment`/`Flag`, track derivation), `qa.py`, `schemas.py`.

## Commands
Run **from `factsfirst/`**. Everything uses the workspace venv (`../.venv`, see Environment).

- **Annotate one clip (production):**
  `./run.sh <video> --out out/<ver>/<clip>.json [--workdir logs/<clip>_<ver>]` — no behavioral
  flags. Per-segment work is parallel.
- **Annotate one clip (measured spine):**
  `../.venv/bin/python perception/pipeline2.py <video> --out out/v2/<clip>.json [--workdir DIR] [--fps 30]`
- **Perception only (no VLM, no cost):**
  `../.venv/bin/python perception/perception.py <video> [--fps 30]` — prints the measured
  segments and writes `perception/facts.json`. Use this to inspect the spine in isolation.
- **Batch (Videos-Leo, resumable, deadline-bounded — run inside tmux):**
  `.venv/bin/python leo_batch.py` (reads `../Videos-Leo/_index.csv`, writes `Videos-Leo-PROGRESS.md`).
- **Publish to the dashboard (version-aware):**
  `python to_viewer.py out/<ver>/<clip>.json [--ver vNN]` — version inferred from the output
  dir. Served at `:8800/factsfirst/factsfirst.html`.

### "Tests" (there is no pytest suite — these are the verification scripts)
A change must move the relevant number on these GT-aware, **zero-API-cost** scripts:
- **Segmentation determinism + boundary recall:**
  `../.venv/bin/python perception/eval.py [clip.mp4 ...]` — runs perception twice (boundaries
  must be byte-identical) and scores recall/precision where held-out boundaries exist.
- **Per-stage scorecard vs human GT:**
  `../.venv/bin/python perception/scorecard.py [H466 H459 ...]` — boundary recall/precision/F1
  vs `out/leo_edited/` GT (reads GT as text only; opens no video).
- **Build the v2-vs-GT file for a codex grader:**
  `../.venv/bin/python perception/make_grade_input.py <tag>` (the VLM label stages are graded by
  a fresh codex agent, not in-process).

## Environment (non-obvious)
- **Workspace layout:** `/home/ubuntu/local/` holds `factsfirst/` next to its deps (`.venv`,
  `yolo_hands`, `sam3`, the API keys, `Videos-Leo`). Relocated here 2026-06-18 from
  `Internship/egoanno/` to separate it from the legacy pipeline; the relative-path layout is
  preserved, so `../.venv`, `HERE.parent` (keys), and `HERE.parent.parent/yolo_hands` all resolve.
- **venv:** `.venv` is a symlink to `../.venv` (the workspace venv at `/home/ubuntu/local/.venv`
  with PIL, numpy, cv2, ultralytics). Never create a local venv — always invoke `.venv/bin/python`.
- **API keys:** `read_key` looks in **this dir first, then the parent workspace dir** for
  `.gemini_key`, `.openai_key`, `.anthropic_key`. They live in `/home/ubuntu/local`.
- **Handpose model:** `../yolo_hands/yolo_bundle/hand_yolo_detector@20260314.pt`, loaded via the
  bundle's `hand_yolo_infer.detect_hands` + ultralytics (perception.py puts the bundle on
  `sys.path`). Classes: `0`=left hand, `1`=right hand, with 21 keypoints (the grip ratio and
  fingertip "grasp" centroid are derived from tips 4/8/12/16/20).
- An L4 GPU + ffmpeg NVENC/NVDEC are available; prefer GPU for local decode/encode and the
  handpose detector rather than leaving it idle.

## Working style (follow on EVERY prompt)
- **Plan before acting.** For non-trivial work, lay out the approach before editing.
- **Ask, don't guess.** When something is ambiguous or a choice is genuinely the user's, ASK —
  EXCEPT when told to run autonomously or the user is away; then pick the best option, state the
  assumption, and proceed.
- **No editing the pipeline while a run is in flight.** Commit/snapshot the current version first.

## Version control (MANDATORY)
- Repo: `https://github.com/leo-wang-deepreach/EgoAnno-VLM-Egocentric-Video-Labelling`
  (this `factsfirst/` dir is its own git repo; auth as `leo-wang-deepreach`).
- **ALWAYS commit + push the current version BEFORE launching a new version's run.** Each
  version: commit, tag `vNN`, push `main` + the tag. Never start a run with uncommitted pipeline
  changes.
- **NEVER commit** secrets (`.gemini_key` / `.openai_key` / `.anthropic_key`), run logs, videos,
  generated images, or held-out ground truth. `.gitignore` enforces this; still scan staged
  content before pushing (the repo is PUBLIC).

## Definition of done (production labeler)
The goal is a production labeler that autonomously annotates a video corpus into atomic per-hand
action timelines for robot training. A clip/version is "done" only when: (1) the held-out
**GPT-5.5** review says it's really good, (2) **claude-fable-5** independently agrees, and (3) the
**leak audit is clean**. North-star metric: **object-identity accuracy** (is the named manipulated
object correct?). Honesty first — GT disagreements are logged as failures, never papered over.

## Hard constraints
- **GT is evaluation-only.** Held-out ground truth, calibration, and dataset examples live ONLY
  in the eval modules (`perception/eval.py`, `perception/scorecard.py`, `perception/make_grade_input.py`)
  — `out/leo_edited/`, `Videos-Leo/`. They must NEVER appear in any model-facing prompt or be read
  by `perception.py` / `pipeline.py` / `pipeline2.py`. Pipeline modules are GT-blind by design.
- **Leak rule (operational, repo is PUBLIC — a leak is permanent).** Prompt examples use typed
  placeholders only (`<part>`, `<counterpart>`, `<container>`, `<color>`, `<stable location>`),
  never real object names / clip-IDs / directions. Two-step model-assisted fixes: a model that saw
  a clip or its GT may only NAME the error *class*; a separate context-free step turns that class
  into a generic rule with no clip-specifics. A GPT/Fable review counts as held-out only if the
  reviewer received no text mentioning the reviewed clip. Full text: `../LEAK_RULE.md`.
- **No hard-coded semantic wordlists** for judgement (determinism for parsing only, not for
  deciding what an action is).
- **Double-sided:** every directional instruction names assembly AND disassembly.
