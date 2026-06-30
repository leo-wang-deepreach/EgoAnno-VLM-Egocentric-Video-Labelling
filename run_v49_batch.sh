#!/bin/bash
# Re-run ground_simple (v49) on every batch_grounded clip's existing inventory + prompts.
# Only the grounder changed, so prompts/inventory are reused; outputs -> out/batch_grounded_v49/<clip>.
set -u
cd /home/ubuntu/local/factsfirst
SAM=/home/ubuntu/local/sam3/sam3py
SRC=out/batch_grounded
OUT=out/batch_grounded_v49
mkdir -p "$OUT"
LEDGER="$OUT/_run_ledger.txt"
: > "$LEDGER"
for d in "$SRC"/*/; do
  c=$(basename "$d")
  inv="$d/_inventory_$c.json"; pr="$d/_prompts.json"
  [ -f "$inv" ] && [ -f "$pr" ] || { echo "SKIP $c (no inv/prompts)" | tee -a "$LEDGER"; continue; }
  echo "=== $c ===" | tee -a "$LEDGER"
  DBG=1 RENDER=1 LLM_PROVIDER=gemini GEMINI_MODEL=gemini-3.1-pro-preview \
    $SAM perception/ground_simple.py "$OUT/$c" "$inv" "$pr" > "$OUT/$c.log" 2>&1
  echo "  exit=$? : $(grep -cE 'RIGHT:|LEFT:' "$OUT/$c.log" 2>/dev/null) hand-results, $(grep -oE '[0-9]+ grounded' "$OUT/$c.log" | tail -1)" | tee -a "$LEDGER"
done
echo "ALL DONE" | tee -a "$LEDGER"
