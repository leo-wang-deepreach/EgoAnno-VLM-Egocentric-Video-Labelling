#!/usr/bin/env bash
# run_lean.sh — run the LEAN grounding pipeline (v39) end-to-end on ONE video.
# Chains: emit_grasp_prompts (handpose) -> build_inventory (Claude) -> build_references (SAM3) -> ground_simple.
# Usage: run_lean.sh <video> [tag]
set -uo pipefail
cd /home/ubuntu/local/factsfirst
VENV=/home/ubuntu/local/.venv/bin/python
SAM3PY=/home/ubuntu/local/sam3/sam3py
VIDEO="$1"
base=$(basename "$VIDEO"); base="${base%.*}"
if [[ "${2:-}" != "" ]]; then TAG="$2"
elif [[ "$base" =~ H[0-9]+ ]]; then TAG="${BASH_REMATCH[0]}"
else TAG=$(echo "$base" | cut -d- -f1 | cut -c1-12); fi

BATCH=out/v2_grounded/batch/$TAG
mkdir -p "$BATCH/refs" "$BATCH/review"
GRASP="$BATCH/_grasp_$TAG.json"
INV=out/v2_grounded/_inventory_$TAG.json
LOG="$BATCH/run.log"
echo "==== $TAG  ($VIDEO) ====" | tee "$LOG"

step(){ echo "-- $1 --" | tee -a "$LOG"; }

step "1/4 grasp prompts"
$VENV perception/emit_grasp_prompts.py "$VIDEO" "$GRASP" >>"$LOG" 2>&1 || { echo "$TAG FAIL grasp"; exit 11; }
ng=$($VENV -c "import json;print(len(json.load(open('$GRASP'))['prompts']))" 2>/dev/null || echo 0)
echo "   grasp prompts: $ng" | tee -a "$LOG"
[ "$ng" -eq 0 ] && { echo "$TAG: no grasps — skip"; exit 0; }

step "2/4 inventory (Claude)"
$VENV perception/build_inventory.py "$TAG" "$VIDEO" >>"$LOG" 2>&1 || { echo "$TAG FAIL inventory"; exit 12; }

step "3/4 references (SAM3)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $SAM3PY perception/build_references.py "$TAG" "$INV" "$VIDEO" "$BATCH/refs" >>"$LOG" 2>&1 || { echo "$TAG FAIL refs"; exit 13; }

step "4/4 ground (lean v39)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REFS_DIR="$BATCH/refs" DBG=1 $SAM3PY perception/ground_simple.py "$BATCH/review" "$INV" "$GRASP" >>"$LOG" 2>&1 || { echo "$TAG FAIL ground"; exit 14; }

res=$(grep -E "hand-frames," "$LOG" | tail -1)
echo "$TAG DONE: $res" | tee -a "$LOG"
