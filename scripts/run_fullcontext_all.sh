#!/usr/bin/env bash
# Full-context baseline 
#
# Model:  Qwen/Qwen3.6-35B-A3B (or any OpenAI-compatible endpoint)
#
# Prereq:
#   vLLM serving the QA model:
#     vllm serve Qwen/Qwen3.6-35B-A3B --port 8001 --tensor-parallel-size 2 ...
#
# Endpoints (env-overridable):
#   URL, API_KEY          QA model endpoint   (default: http://localhost:8001/v1)
#
# Per-run knobs:
#   QA_MODEL              served model name                 (default: Qwen/Qwen3.6-35B-A3B)
#   CONCURRENCY           items processed in parallel       (default: 4)
#   TEMPERATURE           sampling temperature              (default: 0.7)
#   TOP_P                 nucleus sampling                  (default: 0.95)
#   MAX_NEW               max_tokens per completion         (default: 2048)
#   VLLM_MAX_LEN          vLLM --max-model-len (token cap)  (default: 65536 = 2**16)
#   SAFETY_MARGIN         token buffer for chat template    (default: 256)
#   DISABLE_THINKING      Qwen3 thinking off (1/0)          (default: 0)
#   OUTPUT_DIR            log + JSONL destination           (default: ./results/fullcontext)
#   DATASETS              space-separated list to run       (default: "babi github horizonbench wiki")
#   DATA_ROOT             local dir with <dataset>.json     (default: unset = pull from
#                                                            dinobby/LongMINT on HF)
#   LIMIT                 first-N items (debug)             (default: unset = all)
#   PYTHON                python interpreter                (default: python on PATH)
#   EXTRA_ARGS            anything else forwarded to the runner
#
# Usage:
#   ./run_fullcontext_all.sh                       # all four datasets
#   DATASETS=wiki ./run_fullcontext_all.sh         # one dataset
#   DATASETS="babi github" LIMIT=5 ./run_fullcontext_all.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_ENTRY="$REPO_ROOT/src/fullcontext/run_fullcontext_unified.py"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

QA_MODEL="${QA_MODEL:-Qwen/Qwen3.6-35B-A3B}"

export URL="${URL:-http://localhost:8001/v1}"
export API_KEY="${API_KEY:-123-abc}"

CONCURRENCY="${CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW="${MAX_NEW:-2048}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-65536}"
SAFETY_MARGIN="${SAFETY_MARGIN:-256}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/fullcontext}"
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

THINKING_ARG=""
if [ "$DISABLE_THINKING" = "1" ]; then
    THINKING_ARG="--disable-thinking"
fi

echo "QA      model : $QA_MODEL @ $URL"
echo "output dir    : $OUTPUT_DIR"
echo "datasets      : $DATASETS"
echo "data source   : ${DATA_ROOT:-hf:dinobby/LongMINT}"
echo "concurrency   : $CONCURRENCY"
echo "sampling      : temperature=$TEMPERATURE  top_p=$TOP_P  max_new=$MAX_NEW"
echo "vllm_max_len  : $VLLM_MAX_LEN   safety_margin: $SAFETY_MARGIN"
[ -n "$LIMIT" ] && echo "LIMIT         : $LIMIT (debug)"
echo

for DATASET in $DATASETS; do
    OUT="$OUTPUT_DIR/${DATASET}.jsonl"
    LOG="$OUTPUT_DIR/${DATASET}.log"
    echo "============================================================"
    echo "[$(date +%F\ %T)] dataset=$DATASET  -> $OUT"
    echo "============================================================"
    "$PYTHON" "$PY_ENTRY" \
        --model "$QA_MODEL" \
        --dataset "$DATASET" \
        --concurrency "$CONCURRENCY" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --max-new "$MAX_NEW" \
        --vllm-max-len "$VLLM_MAX_LEN" \
        --safety-margin "$SAFETY_MARGIN" \
        --output "$OUT" \
        $DATA_ROOT_ARG \
        $LIMIT_ARG \
        $THINKING_ARG \
        $EXTRA_ARGS 2>&1 | tee "$LOG"
    echo
done

echo "[$(date +%F\ %T)] all datasets complete."
