#!/usr/bin/env bash
# run_grounding_pipeline.sh — ONE clean end-to-end object-grounding pipeline for an egocentric video.
#
#   <video>
#     ├─ [A] handpose grasp prompts          emit_grasp_prompts.py   (.venv / py3.10)   -> _prompts_<tag>.json
#     ├─ [B] geometric grasp-filter inventory build_inventory_grasp.py (sam3py, Gemini) -> _inventory_<tag>.json
#     └─ [C] per-hand grounding              ground_simple.py        (sam3py, Gemini)   -> grounded masks + names
#
# Everything VLM = gemini-3.1-pro-preview (no Opus). Usage:
#   ./run_grounding_pipeline.sh <video> <tag> [outdir]
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="$1"; TAG="$2"; OUT="${3:-out/grounded/$TAG}"
PROMPTS="out/v2_grounded/_prompts_${TAG}.json"
INV="out/v2_grounded/_inventory_${TAG}.json"
VENV=/home/ubuntu/local/.venv/bin/python
SAM=/home/ubuntu/local/sam3/sam3py
mkdir -p "$OUT" out/v2_grounded
export LLM_PROVIDER=gemini GEMINI_MODEL=gemini-3.1-pro-preview
export TOKEN_LEDGER="$OUT/_ledger.txt"

echo "==================================================================="
echo " GROUNDING PIPELINE  |  $TAG  |  $VIDEO"
echo " VLM = gemini-3.1-pro-preview (no Opus)  |  out -> $OUT"
echo "==================================================================="

echo ">>> [A] grasp prompts (handpose, measured spine; .venv)"
$VENV perception/emit_grasp_prompts.py "$VIDEO" "$PROMPTS"

echo ">>> [B] geometric grasp-filter inventory (Gemini; sam3py)"
GRASP_EXPAND=0.2 N_FRAMES=12 $SAM perception/build_inventory_grasp.py "$TAG" "$VIDEO"

echo ">>> [C] per-hand grounding (ground_simple, Gemini; sam3py)"
$SAM perception/ground_simple.py "$OUT" "$INV" "$PROMPTS"

echo "==================================================================="
echo " DONE  ->  $OUT"
echo "   prompts:   $PROMPTS"
echo "   inventory: $INV"
echo "   ledger:    $TOKEN_LEDGER"
echo "==================================================================="
