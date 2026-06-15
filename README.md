# EgoAnno — facts-first egocentric video labelling

Upload a short egocentric (head-mounted, top-down) video of **two-handed tabletop
manipulation** → local + VLM processing → an accurate **per-hand action timeline** for
robot learning: a list of segments, each `{start_sec, end_sec, left_label, right_label}`
where a label is a 2–8-word atomic action (`thread the silver nut onto the bolt`) or `N/A`.

It is **facts-first**: derive everything from observed facts (what each hand touches, when
objects change hands) rather than guessing the task up front. The pipeline is SDK-free —
every model call is raw HTTP (Gemini File API for native video; Claude via forced
tool-call for structured output).

## Pipeline

```
INPUT → burn µs-clock (audio off) → upload once to Gemini File API
PHASE 1 (facts, cached):
  1a contact_track    Gemini native 10fps → objects[] + per-frame {left_touching,right_touching}
                                          → build_track(): collapse to possession intervals
  1c transition scan  Claude opus, sliding 3.5s windows @30fps → place/pickup/handoff events
PHASE 2  bursts (opus 30fps×0.5s) → burst_reduce (deterministic; "mixed" needs runs≥2)
PHASE 3  direction_decide (Gemini native 10fps) → direction + derived goal (+ burst stabilizer)
QUALITY LOOP (≤3 attempts, keep BEST by (gate-good, fewest-flags, more-segments)):
  PHASE 4
    SEGMENT  v49_segment (Gemini native 10fps cut-rules) ∪ transition cuts
    LABEL    per-segment native labeler (Gemini, one segment at a time, 10fps, +contact facts)
    MERGE    merge_identical_labels (exact, transition-protected)
    SPLIT    seg_reconcile (Gemini native 10fps, per-stretch "one action or several?", ≤2 passes)
    VERIFY   template_match (cycle completeness) + neighbor_review (label-sequence consistency)
  PHASE 5
    gate     Claude opus 2fps whole-clip → quality verdict + corrections + seam-merges
FINALIZE (once, on best attempt):
    delete_only_critic (opus, merge over-split, transition-protected)
    completeness (Gemini native 10fps; missing pick/place → fold into transitions → re-run P4)
    QA + export → episode.json (+ _trace, _qa)
```

The spine is `ClipState` (`clipstate.py`): every stage reads a **compacted** view and writes
back. Window-scoped context for per-segment label calls; whole-clip only for direction and
the gate. The exported possession `track` is derived from the final labels.

## Models

| role | model |
|---|---|
| facts · segmentation · labeling (native video) | `gemini-3.1-pro-preview` |
| verifying · critics · gate (sees frames) | `claude-opus-4-8` |

fps is a real `video_metadata.fps` upload parameter (not prompt text); native fps caps at 24,
so 30fps transition/burst work uses local frame extraction.

## Run

```bash
./run.sh path/to/video.mp4 --out out/ep.json        # wraps ../.venv/bin/python
# or: python run.py path/to/video.mp4 --out out/ep.json
```

SDK-free: needs PIL + stdlib only. Provide API keys as `.gemini_key` and `.anthropic_key`
(and `.openai_key` if used) in this directory or the parent — these files are git-ignored.

## Constraints

- **No held-out reference, calibration, or dataset examples** in any model-facing prompt.
- **No hard-coded semantic wordlists** — determinism for parsing, not for judgement.
- **Double-sided**: every directional instruction names assembly AND disassembly.

## Files

`pipeline.py` orchestrator · `clipstate.py` state + compaction · `models.py` raw-HTTP
clients · `media.py` ffmpeg/PIL render · `bursts.py` sweep + reduce · `qa.py` QA + export ·
`schemas.py` structured-output schemas · `to_viewer.py` dashboard publish · `prompts/`.
