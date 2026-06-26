#!/usr/bin/env bash
# run_lean_batch.sh — run the lean grounding pipeline (run_lean.sh) over a list of videos
# (one path per line) with N parallel workers. Usage: run_lean_batch.sh <list.txt> [workers]
cd /home/ubuntu/local/factsfirst
LIST="$1"; WORKERS="${2:-2}"
xargs -P "$WORKERS" -d '\n' -I {} bash run_lean.sh "{}" < "$LIST"
echo "LEAN BATCH COMPLETE"
