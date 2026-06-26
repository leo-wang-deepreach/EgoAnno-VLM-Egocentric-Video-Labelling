#!/usr/bin/env bash
# run_batch_grounding.sh — batch the grounding pipeline over dataset clips:
#   frames -> proxy mp4 -> per-frame handpose prompts -> geometric grasp-filter inventory -> ground_simple.
# Manifest lines: "<meta|visor> <clip>".  All VLM = gemini-3.1-pro-preview.
# Usage: ./run_batch_grounding.sh <manifest> [outbase]
cd "$(dirname "$0")"
VENV=/home/ubuntu/local/.venv/bin/python
SAM=/home/ubuntu/local/sam3/sam3py
export LLM_PROVIDER=gemini GEMINI_MODEL=gemini-3.1-pro-preview
MANIFEST="$1"; BASE="${2:-out/batch_grounded}"
mkdir -p "$BASE" /tmp/clipvids
while read -r DS CLIP; do
  [ -z "${DS:-}" ] && continue
  TAG="${DS}_${CLIP}"; VID="/tmp/clipvids/${TAG}.mp4"; OUT="$BASE/$TAG"; mkdir -p "$OUT"
  export TOKEN_LEDGER="$OUT/_ledger.txt"
  echo "================= $TAG ================="
  $VENV perception/make_clip_video.py "$DS" "$CLIP" "$VID"   || { echo "VID FAIL $TAG"; continue; }
  MAX_PROMPTS=8 SAMPLE_FPS=2 $VENV perception/emit_prompts_simple.py "$VID" "$OUT/_prompts.json" "$TAG" 2>&1 | grep -avE "FutureWarning|warn" | tail -1 || { echo "PROMPTS FAIL $TAG"; continue; }
  GRASP_EXPAND=0.2 N_FRAMES=10 OUTDIR="$OUT" $SAM perception/build_inventory_grasp.py "$TAG" "$VID" 2>&1 | grep -aE "grasp inventory|invC|FAIL" | tail -1 || { echo "INV FAIL $TAG"; continue; }
  $SAM perception/ground_simple.py "$OUT" "$OUT/_inventory_${TAG}.json" "$OUT/_prompts.json" 2>&1 | grep -aE "${TAG} (LEFT|RIGHT):|Error|Traceback" | tail -40
  echo "<<< DONE $TAG"
done < "$MANIFEST"
echo "===== BATCH COMPLETE ====="
