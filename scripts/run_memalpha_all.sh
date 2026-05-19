#!/usr/bin/env bash
# Run Mem-alpha inference over all four unified datasets sequentially.
#
# Memory builder : YuWangX/Memalpha-4B          (local vLLM inside main.py)
# Answering model: Qwen/Qwen3.6-35B-A3B         (served by a separate vLLM and
#                                                hit via memory_server.py)
#
# Prereqs (launch before running this script):
#   1. vLLM serving the answering model at $QA_URL:
#        vllm serve Qwen/Qwen3.6-35B-A3B --port 8001 --tensor-parallel-size 2 ...
#   2. memory_server.py talking to that vLLM and exposing /batch_process at $MEM_SERVER_URL:
#        python src/mem_alpha/memory_server.py \
#            --port 5000 \
#            --server_url "$QA_URL" \
#            --model_name "$QA_MODEL"
#
# Endpoints (env-overridable):
#   QA_URL          vLLM serving the answering model      (default: http://localhost:8001/v1)
#   MEM_SERVER_URL  memory_server.py base URL             (default: http://127.0.0.1:5000)
#
# Per-run knobs:
#   AGENT_CONFIG    yaml passed to run_memalpha_unified.py (default: src/mem_alpha/config/memalpha_unified.yaml)
#   BATCH_SIZE      main.py --batch_size                  (default: 32)
#   SAMPLE_SIZE     main.py --sample_size                 (default: unset = all)
#   OUTPUT_DIR      log destination                       (default: ./results/memalpha)
#   EXTRA_ARGS      anything else forwarded to run_memalpha_unified.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_ENTRY="$REPO_ROOT/src/mem_alpha/run_memalpha_unified.py"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

QA_MODEL="${QA_MODEL:-Qwen/Qwen3.6-35B-A3B}"
QA_URL="${QA_URL:-http://localhost:8001/v1}"
MEM_SERVER_URL="${MEM_SERVER_URL:-http://127.0.0.1:5000}"

AGENT_CONFIG="${AGENT_CONFIG:-$REPO_ROOT/src/mem_alpha/config/memalpha_unified.yaml}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/memalpha}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"

SAMPLE_FLAG=""
[ -n "$SAMPLE_SIZE" ] && SAMPLE_FLAG="--sample_size $SAMPLE_SIZE"

echo "memory builder : YuWangX/Memalpha-4B  (loaded by main.py via vLLM)"
echo "answer  model  : $QA_MODEL @ $QA_URL"
echo "memory_server  : $MEM_SERVER_URL"
echo "agent config   : $AGENT_CONFIG"
echo "batch size     : $BATCH_SIZE"
echo "sample size    : ${SAMPLE_SIZE:-all}"
echo "log dir        : $OUTPUT_DIR"
echo "python         : $PYTHON"
echo

# Per-source dataset names defined in src/mem_alpha/conversation_creator.py.
# babi / github / horizonbench / wiki — same source order as run_memagent_all.sh.
for SOURCE in babi github horizonbench wiki; do
    DATASET="unified_${SOURCE}"
    LOG="$OUTPUT_DIR/${SOURCE}.log"
    echo "============================================================"
    echo "[$(date +%F\ %T)] dataset=$DATASET  -> $LOG"
    echo "============================================================"
    "$PYTHON" "$PY_ENTRY" \
        --agent_config "$AGENT_CONFIG" \
        --dataset "$DATASET" \
        --batch_size "$BATCH_SIZE" \
        $SAMPLE_FLAG \
        $EXTRA_ARGS 2>&1 | tee "$LOG"
    echo
done

echo "[$(date +%F\ %T)] all datasets complete."
