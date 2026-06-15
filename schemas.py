#!/usr/bin/env python3
"""schemas.py — structured-output schemas.

Two conventions live here on purpose:
  * NATIVE Gemini (generateContent responseSchema) wants UPPERCASE types
    ("OBJECT"/"ARRAY"/"STRING"/"NUMBER"/"BOOLEAN") and supports `nullable`.
  * Frame models (GPT strict json_schema, Claude output_config, Gemini OpenAI-compat
    bursts) want lowercase JSON Schema ("object"/"array"/"string"/...).
Each schema is named for the stage and prompt it backs.
"""

# =========================================================================== #
#  NATIVE GEMINI (uppercase)                                                   #
# =========================================================================== #
_NEED_BURST = {"type": "ARRAY", "items": {
    "type": "OBJECT",
    "properties": {"t": {"type": "NUMBER"}, "question": {"type": "STRING"}},
    "required": ["t", "question"]}}

# Phase 1a — object + hand-contact track (10fps, THE GROUND TRUTH)
CONTACT_TRACK = {
    "type": "OBJECT",
    "properties": {
        "objects": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"},
                           "colour": {"type": "STRING"},
                           "function": {"type": "STRING"}},
            "required": ["name"]}},
        "frames": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "t": {"type": "NUMBER"},
                "foreground": {"type": "STRING"},
                "background": {"type": "STRING"},
                "left_touching": {"type": "STRING"},
                "right_touching": {"type": "STRING"}},
            "required": ["t", "foreground", "background",
                         "left_touching", "right_touching"]}},
    },
    "required": ["objects", "frames"],
}

# Phase 4 — seg_reconcile (v49): per-stretch "one action or several?" -> split cuts
SEG_RECONCILE = {
    "type": "OBJECT",
    "properties": {"boundaries": {"type": "ARRAY", "items": {"type": "NUMBER"}}},
    "required": ["boundaries"],
}

# Phase 4 — v49 segmentation (single native pass: cuts only, NO labels)
V49_SEGMENT = {
    "type": "OBJECT",
    "properties": {
        "boundaries": {"type": "ARRAY", "items": {"type": "NUMBER"}},
        "need_burst": _NEED_BURST,
    },
    "required": ["boundaries"],
}


# Phase 4e — completeness audit (NATIVE Gemini 10fps): pick/place chain gaps
COMPLETENESS = {
    "type": "OBJECT",
    "properties": {
        "gaps": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "t": {"type": "NUMBER"},
                "hand": {"type": "STRING", "enum": ["left", "right", "both"]},
                "kind": {"type": "STRING",
                         "enum": ["missing_pick", "missing_place", "missing_handoff"]},
                "object": {"type": "STRING"},
                "in_seg": {"type": "INTEGER"},
                "evidence": {"type": "STRING"}},
            "required": ["t", "hand", "kind", "object", "evidence"]}},
        "notes": {"type": "STRING"},
    },
    "required": ["gaps"],
}

# Phase 4b — per-segment label, NATIVE Gemini variant (uppercase)
VIDEO_LABEL_NATIVE = {
    "type": "OBJECT",
    "properties": {
        "think": {"type": "STRING"},
        "left": {"type": "STRING"},
        "right": {"type": "STRING"},
    },
    "required": ["think", "left", "right"],
}

# Phase 3 — direction (also DERIVES the goal)
DIRECTION_DECIDE = {
    "type": "OBJECT",
    "properties": {
        "direction": {"type": "STRING",
                      "enum": ["assembly", "disassembly", "transfer_or_portioning",
                               "mixed_or_alternating", "other_or_ambiguous"]},
        "goal": {"type": "STRING"},
        "basis": {"type": "STRING"},
        "phase_switch_sec": {"type": "NUMBER", "nullable": True},
        "agrees_with_stated_task": {"type": "BOOLEAN", "nullable": True},
        "recheck_times": {"type": "ARRAY", "items": {"type": "NUMBER"}},
    },
    "required": ["direction", "goal", "basis"],
}

# =========================================================================== #
#  FRAME MODELS (lowercase)                                                    #
# =========================================================================== #

# Phase 1c — sliding-window place/pickup/handoff detector (opus, focused frames)
WINDOW_TRANSITIONS = {
    "type": "object",
    "properties": {
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "t": {"type": "number"},
                "hand": {"type": "string", "enum": ["left", "right", "both"]},
                "kind": {"type": "string", "enum": ["place", "pickup", "handoff"]},
                "object": {"type": "string"},
                "evidence": {"type": "string"}},
            "required": ["t", "hand", "kind", "object", "evidence"],
            "additionalProperties": False}},
    },
    "required": ["events"],
    "additionalProperties": False,
}

# Phase 4d — delete-only merge critic (Claude, frames): merge over-segmented runs
MERGE_CRITIC = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "delete_t": {"type": "number"},
                "keep": {"type": "string", "enum": ["before", "after"]},
                "reason": {"type": "string"}},
            "required": ["delete_t", "keep", "reason"],
            "additionalProperties": False}},
        "notes": {"type": "string"},
    },
    "required": ["merges", "notes"],
    "additionalProperties": False,
}

# Phase 2 — one burst question -> one verdict
BURST_ANSWER = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}

# Phase 4 — neighbor-context review (text only): flag label-sequence inconsistencies
NEIGHBOR_REVIEW = {
    "type": "object",
    "properties": {
        "flags": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "seg": {"type": "integer"},
                "issue": {"type": "string"},
                "reason": {"type": "string"}},
            "required": ["seg", "issue", "reason"],
            "additionalProperties": False}},
    },
    "required": ["flags"],
    "additionalProperties": False,
}

# Phase 4d — template-match (GPT text): flag, do not fix
TEMPLATE_MATCH = {
    "type": "object",
    "properties": {
        "flags": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "seg": {"type": "integer"},
                "type": {"type": "string"},
                "detail": {"type": "string"}},
            "required": ["seg", "type", "detail"],
            "additionalProperties": False}},
        "notes": {"type": "string"},
    },
    "required": ["flags", "notes"],
    "additionalProperties": False,
}

# Phase 5a — the gate (Claude opus-4-8, sees frames)
GATE = {
    "type": "object",
    "properties": {
        "purpose_check": {"type": "string"},
        "purpose_verdict": {"type": "string", "enum": ["confirmed", "corrected"]},
        "findings": {"type": "string"},
        "merge_at_sec": {"type": "array", "items": {"type": "number"}},
        "split_request": {"type": "array", "items": {"type": "integer"}},
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "seg": {"type": "integer"},
                "hand": {"type": "string", "enum": ["left", "right"]},
                "label": {"type": "string"}},
            "required": ["seg", "hand", "label"],
            "additionalProperties": False}},
        "quality_verdict": {"type": "string", "enum": ["good", "needs_rerun"]},
        "rerun_feedback": {"type": "string"},
        "request_merge_critic": {"type": "boolean"},
        "request_completeness": {"type": "boolean"},
    },
    "required": ["purpose_check", "purpose_verdict", "findings",
                 "merge_at_sec", "split_request", "corrections",
                 "quality_verdict", "rerun_feedback",
                 "request_merge_critic", "request_completeness"],
    "additionalProperties": False,
}
