#!/usr/bin/env bash
# Run MemAgent inference over all four unified datasets sequentially.
#
# Memory model: BytedTsinghua-SIA/RL-MemoryAgent-14B  (recurrent memory updates)
# QA model:     Qwen/Qwen3.6-35B-A3B                  (final boxed answer)
#
# Endpoints are taken from env vars; override before invoking if needed:
#   URL, API_KEY        -> memory model endpoint   (default: http://localhost:8000/v1)
#   QA_URL, QA_API_KEY  -> QA model endpoint       (default: http://localhost:8001/v1)
#
# Other knobs:
#   BABI_FACTS_PER_CHUNK (default 15; horizonbench/github/wiki always use 1 entry/chunk)
#   CONCURRENCY          (default 4)
#   OUTPUT_DIR           (default ./results)
#   EXTRA_ARGS           (anything else forwarded to run_memagent_unified.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_ENTRY="$REPO_ROOT/src/memagent/run_memagent_unified.py"
cd "$REPO_ROOT"

MEM_MODEL="${MEM_MODEL:-BytedTsinghua-SIA/RL-MemoryAgent-14B}"
QA_MODEL="${QA_MODEL:-Qwen/Qwen3.6-35B-A3B}"

export URL="${URL:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-123-abc}"
export QA_URL="${QA_URL:-http://localhost:8001/v1}"
export QA_API_KEY="${QA_API_KEY:-123-abc}"

BABI_FACTS_PER_CHUNK="${BABI_FACTS_PER_CHUNK:-15}"
CONCURRENCY="${CONCURRENCY:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"

echo "memory model : $MEM_MODEL @ $URL"
echo "QA     model : $QA_MODEL @ $QA_URL"
echo "output dir   : $OUTPUT_DIR"
echo "concurrency  : $CONCURRENCY"
echo "babi facts/chunk: $BABI_FACTS_PER_CHUNK"
echo

for DATASET in babi github horizonbench wiki; do
    OUT="$OUTPUT_DIR/${DATASET}.jsonl"
    LOG="$OUTPUT_DIR/${DATASET}.log"
    echo "============================================================"
    echo "[$(date +%F\ %T)] dataset=$DATASET  -> $OUT"
    echo "============================================================"
    python "$PY_ENTRY" \
        --model "$MEM_MODEL" \
        --qa-model "$QA_MODEL" \
        --dataset "$DATASET" \
        --babi-facts-per-chunk "$BABI_FACTS_PER_CHUNK" \
        --concurrency "$CONCURRENCY" \
        --output "$OUT" \
        $EXTRA_ARGS 2>&1 | tee "$LOG"
    echo
done

echo "[$(date +%F\ %T)] all datasets complete."
