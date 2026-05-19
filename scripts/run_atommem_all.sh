#!/usr/bin/env bash
# Run AtomMem inference over all four unified datasets sequentially.
#
# Memory model: Yupeng123/AtomMem-8B        (atomic-CRUD memory agent, served by vLLM)
# QA     model: Qwen/Qwen3.6-35B-A3B        (final boxed answer, served by a separate vLLM)
# Embedding:    Qwen/Qwen3-Embedding-4B     (long-memory retrieval, served by vLLM)
#
# Prereqs (launch before running this script):
#   1. vLLM serving the AtomMem memory-builder:
#        vllm serve <AtomMem-8B path> --served-model-name AtomMem --port 8000
#   2. vLLM serving the QA / answering model (shared with memagent / memalpha):
#        vllm serve Qwen/Qwen3.6-35B-A3B --port 8001 ...
#   3. vLLM serving the embedding model on its own GPU:
#        CUDA_VISIBLE_DEVICES=2 vllm serve <embedding model path> \
#            --served-model-name qwen3-embedding --task embed --port 9007
#
# Endpoints (env-overridable):
#   URL,    API_KEY           -> memory-builder endpoint  (default: http://localhost:8000/v1)
#   QA_URL, QA_API_KEY        -> QA / final-answer endpoint (default: http://localhost:8001/v1)
#   EMBED_URL, EMBED_API_KEY  -> embedding endpoint       (default: http://localhost:9007/v1)
#
# Per-run knobs:
#   ATOMMEM_MODEL         served_model_name of the memory-builder vLLM (default: AtomMem)
#   QA_MODEL              served_model_name of the QA vLLM             (default: Qwen/Qwen3.6-35B-A3B)
#   EMBED_MODEL           served_model_name of the embedding vLLM      (default: qwen3-embedding)
#   MAX_CHUNKS            cap on memory-building turns per item   (default: 40 == upstream eval)
#   MAX_QUERY_UPDATES     final-stage <update_query> retries      (default: 3 == upstream)
#   MEMORY_CHAR_BUDGET    char cap on retrieved long-memory block (default: 16000)
#   BABI_FACTS_PER_CHUNK  facts packed per babi chunk             (default: 15;
#                                                                   horizonbench/github/wiki
#                                                                   always use 1 entry/chunk)
#   CONCURRENCY           items processed in parallel             (default: 4)
#   TEMPERATURE           sampling temperature                    (default: 0.7)
#   TOP_P                 nucleus sampling                        (default: 0.95)
#   MAX_NEW               max_tokens per chat completion          (default: 2048)
#   OUTPUT_DIR            log + JSONL destination                 (default: ./results/atommem)
#   DATASETS              space-separated list to run             (default: "babi github horizonbench wiki")
#   DATA_ROOT             local directory with <dataset>.json     (default: unset = pull from
#                                                                  dinobby/LongMINT on HF)
#   LIMIT                 first-N items (debug)                   (default: unset = all)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_ENTRY="$REPO_ROOT/src/atommem/run_atommem_unified.py"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

ATOMMEM_MODEL="${ATOMMEM_MODEL:-AtomMem}"
QA_MODEL="${QA_MODEL:-Qwen/Qwen3.6-35B-A3B}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding}"

export URL="${URL:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-123-abc}"
export QA_URL="${QA_URL:-http://localhost:8001/v1}"
export QA_API_KEY="${QA_API_KEY:-123-abc}"
export EMBED_URL="${EMBED_URL:-http://localhost:9007/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-sk-123}"

MAX_CHUNKS="${MAX_CHUNKS:-40}"
MAX_QUERY_UPDATES="${MAX_QUERY_UPDATES:-3}"
MEMORY_CHAR_BUDGET="${MEMORY_CHAR_BUDGET:-16000}"
BABI_FACTS_PER_CHUNK="${BABI_FACTS_PER_CHUNK:-15}"
CONCURRENCY="${CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW="${MAX_NEW:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/atommem}"
DATASETS="${DATASETS:-babi github horizonbench wiki}"
DATA_ROOT="${DATA_ROOT:-}"
LIMIT="${LIMIT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

DATA_ROOT_ARG=""
if [ -n "$DATA_ROOT" ]; then
    DATA_ROOT_ARG="--data-root $DATA_ROOT"
fi

echo "memory model  : $ATOMMEM_MODEL @ $URL"
echo "QA     model  : $QA_MODEL @ $QA_URL"
echo "embed   model : $EMBED_MODEL @ $EMBED_URL"
echo "output dir    : $OUTPUT_DIR"
echo "datasets      : $DATASETS"
echo "data source   : ${DATA_ROOT:-hf:dinobby/LongMINT}"
echo "concurrency   : $CONCURRENCY   max_chunks: $MAX_CHUNKS   max_query_updates: $MAX_QUERY_UPDATES"
echo "sampling      : temperature=$TEMPERATURE  top_p=$TOP_P  max_new=$MAX_NEW"
echo "babi facts/chunk: $BABI_FACTS_PER_CHUNK"
[ -n "$LIMIT" ] && echo "LIMIT         : $LIMIT (debug)"
echo

for DATASET in $DATASETS; do
    OUT="$OUTPUT_DIR/${DATASET}.jsonl"
    LOG="$OUTPUT_DIR/${DATASET}.log"
    echo "============================================================"
    echo "[$(date +%F\ %T)] dataset=$DATASET  -> $OUT"
    echo "============================================================"
    "$PYTHON" "$PY_ENTRY" \
        --model "$ATOMMEM_MODEL" \
        --qa-model "$QA_MODEL" \
        --embed-model "$EMBED_MODEL" \
        --dataset "$DATASET" \
        --max-chunks "$MAX_CHUNKS" \
        --max-query-updates "$MAX_QUERY_UPDATES" \
        --memory-char-budget "$MEMORY_CHAR_BUDGET" \
        --babi-facts-per-chunk "$BABI_FACTS_PER_CHUNK" \
        --concurrency "$CONCURRENCY" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --max-new "$MAX_NEW" \
        --output "$OUT" \
        $DATA_ROOT_ARG \
        $LIMIT_ARG \
        $EXTRA_ARGS 2>&1 | tee "$LOG"
    echo
done

echo "[$(date +%F\ %T)] all datasets complete."
