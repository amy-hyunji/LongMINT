#!/usr/bin/env bash
# Run HippoRAG over all four MINTEval datasets sequentially.
# Assumes a vLLM (or OpenAI-compatible) server is already serving the LLM at $LLM_BASE_URL.
#
# Data source: by default each dataset is pulled from the `dinobby/MINTEval`
#
# Usage:
#   ./scripts/run_hipporag_all.sh                                # all 4, GPU 0
#   GPUS=0,1,2,3 ./scripts/run_hipporag_all.sh                   # shard each dataset across 4 GPUs
#   DATASETS="babi wiki" ./scripts/run_hipporag_all.sh           # subset
#   DATASETS=horizonbench MAX_DOC_CHARS=8000 ./scripts/run_hipporag_all.sh  # single + override
#   LIMIT=2 ./scripts/run_hipporag_all.sh                        # smoke test (2 instances each)
#   DATA_PATH=./data/babi.json DATASETS=babi ./scripts/run_hipporag_all.sh  # local JSON override

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
# torch from PyPI may need the bundled libnvJitLink to satisfy libcusparse.
if [ "$PYTHON" != "python" ]; then
    _PY_VER=$("$PYTHON" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo python3.12)
    NVJITLINK_DIR="$(dirname "$(dirname "$PYTHON")")/lib/$_PY_VER/site-packages/nvidia/nvjitlink/lib"
    [ -d "$NVJITLINK_DIR" ] && export LD_LIBRARY_PATH="$NVJITLINK_DIR:${LD_LIBRARY_PATH:-}"
fi

DATASETS="${DATASETS:-babi github horizonbench wiki}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

LLM="${LLM:-Qwen/Qwen3.6-35B-A3B}"
EMBED="${EMBED:-Qwen/Qwen3-Embedding-4B}"
EMBED_DTYPE="${EMBED_DTYPE:-bfloat16}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
GPUS="${GPUS:-0}"
LOG_ROOT="${LOG_ROOT:-logs/hipporag}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-0}"
QA_TOP_K="${QA_TOP_K:-5}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
QA_CONCURRENCY="${QA_CONCURRENCY:-32}"
LIMIT="${LIMIT:-}"

LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

cd "$REPO_ROOT"

if [ -n "$DATA_PATH" ] && [ ! -f "$DATA_PATH" ]; then
    echo "DATA_PATH was set but file is missing: $DATA_PATH" >&2
    exit 1
fi
DATA_PATH_FLAG=""
[ -n "$DATA_PATH" ] && DATA_PATH_FLAG="--data_path $DATA_PATH"
DATA_DESC="${DATA_PATH:-hf:dinobby/MINTEval}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

echo "HippoRAG · datasets: $DATASETS · GPUs: ${GPU_ARR[*]}"
echo "LLM: $LLM @ $LLM_BASE_URL  EMBED: $EMBED  (dtype=$EMBED_DTYPE)"
echo "DATA: $DATA_DESC  MAX_DOC_CHARS: $MAX_DOC_CHARS  QA_TOP_K: $QA_TOP_K"
echo "QA_CONCURRENCY: $QA_CONCURRENCY  DISABLE_THINKING: $DISABLE_THINKING"
echo "Output: $OUTPUT_DIR/  Logs: $LOG_ROOT/<dataset>/"
echo

for DATASET in $DATASETS; do
    echo "=================================================================="
    echo "[$(date '+%F %T')] dataset=$DATASET"
    echo "=================================================================="
    LOG_DIR="$LOG_ROOT/$DATASET"
    mkdir -p "$LOG_DIR"

    pids=()
    labels=()
    for ((i=0; i<N_GPUS; i++)); do
        gpu="${GPU_ARR[$i]}"
        log="$LOG_DIR/hipporag.$DATASET.k${QA_TOP_K}.mc${MAX_DOC_CHARS}.gpu${gpu}.log"
        label="dataset=$DATASET gpu=$gpu"
        if (( N_GPUS > 1 )); then
            label="$label shard=$i/$N_GPUS"
            SHARD_FLAGS="--shard_idx $i --num_shards $N_GPUS"
        else
            SHARD_FLAGS=""
        fi
        echo "[$(date '+%F %T')] start: $label -> $log"
        (
            CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON" src/hipporag/tests_hipporag.py $LIMIT_FLAG $SHARD_FLAGS \
                $DATA_PATH_FLAG \
                --dataset "$DATASET" \
                --output_dir "$OUTPUT_DIR" \
                --llm_model_name "$LLM" \
                --llm_base_url "$LLM_BASE_URL" \
                --embedding_model_name "$EMBED" \
                --embedding_model_dtype "$EMBED_DTYPE" \
                --max_doc_chars "$MAX_DOC_CHARS" \
                --qa_top_k "$QA_TOP_K" \
                --qa_concurrency "$QA_CONCURRENCY" \
                --llm_disable_thinking "$DISABLE_THINKING" \
                > "$log" 2>&1
        ) &
        pids+=($!)
        labels+=("$label")
    done

    for ((m=0; m<${#pids[@]}; m++)); do
        if ! wait "${pids[$m]}"; then
            echo "[$(date '+%F %T')] FAILED: ${labels[$m]} (see log)"
        else
            echo "[$(date '+%F %T')] done:   ${labels[$m]}"
        fi
    done
    echo
done

echo "[$(date '+%F %T')] All HippoRAG datasets done."
