#!/usr/bin/env bash
# Dense-retrieval baseline RAG over all four LongMINT datasets sequentially.
# Same prompt as run_hipporag_all.sh — the only axis that changes is retrieval.
#
# Data source: by default each dataset is pulled from the `dinobby/LongMINT`
#
# Usage:
#   ./scripts/run_baserag_all.sh                                            # all 4, top_k=5, GPU 0
#   GPUS=0,1,2,3 ./scripts/run_baserag_all.sh                               # fan top_k×gpu across GPUs
#   DATASETS="babi wiki" ./scripts/run_baserag_all.sh                       # subset
#   TOP_KS="1 3 5 10" GPUS=0,1,2,3 ./scripts/run_baserag_all.sh             # sweep top_k
#   LIMIT=2 ./scripts/run_baserag_all.sh                                    # smoke test
#   DATA_PATH=./data/babi.json DATASETS=babi ./scripts/run_baserag_all.sh   # local JSON override

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"

DATASETS="${DATASETS:-babi github horizonbench wiki}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

TOP_KS="${TOP_KS:-5}"
LLM="${LLM:-Qwen/Qwen3.6-35B-A3B}"
EMBED="${EMBED:-Qwen/Qwen3-Embedding-4B}"
EMBED_DTYPE="${EMBED_DTYPE:-bfloat16}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-32}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-1}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-0}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
GPUS="${GPUS:-0}"
LOG_ROOT="${LOG_ROOT:-logs/baserag}"
LIMIT="${LIMIT:-}"

EMBEDDER_PROVIDER="${EMBEDDER_PROVIDER:-hf}"
EMBEDDER_BASE_URL="${EMBEDDER_BASE_URL:-}"
EMBEDDER_API_KEY_FILE="${EMBEDDER_API_KEY_FILE:-}"
EMBEDDER_CONCURRENCY="${EMBEDDER_CONCURRENCY:-8}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-50}"
EMBEDDER_OUTPUT_DIM="${EMBEDDER_OUTPUT_DIM:-0}"
EMBEDDER_CACHE_DB="${EMBEDDER_CACHE_DB:-}"

LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

cd "$REPO_ROOT"

if [ -n "$DATA_PATH" ] && [ ! -f "$DATA_PATH" ]; then
    echo "DATA_PATH was set but file is missing: $DATA_PATH" >&2
    exit 1
fi
DATA_PATH_FLAG=""
[ -n "$DATA_PATH" ] && DATA_PATH_FLAG="--data_path $DATA_PATH"
DATA_DESC="${DATA_PATH:-hf:dinobby/LongMINT}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

# top_k × GPU combo plan (round-robin pinning).
COMBOS=()
for k in $TOP_KS; do COMBOS+=("$k"); done
N=${#COMBOS[@]}

echo "BaseRAG · datasets: $DATASETS · GPUs: ${GPU_ARR[*]}  TOP_KS: $TOP_KS"
echo "LLM: $LLM @ $LLM_BASE_URL  EMBED: $EMBED ($EMBEDDER_PROVIDER, dtype=$EMBED_DTYPE)"
echo "DATA: $DATA_DESC  MAX_DOC_CHARS: $MAX_DOC_CHARS"
echo "Output: $OUTPUT_DIR/  Logs: $LOG_ROOT/<dataset>/"
echo

for DATASET in $DATASETS; do
    echo "=================================================================="
    echo "[$(date '+%F %T')] dataset=$DATASET"
    echo "=================================================================="
    LOG_DIR="$LOG_ROOT/$DATASET"
    mkdir -p "$LOG_DIR"

    for ((batch_start=0; batch_start<N; batch_start+=N_GPUS)); do
        pids=()
        labels=()
        for ((j=0; j<N_GPUS; j++)); do
            idx=$((batch_start + j))
            if (( idx >= N )); then break; fi
            gpu="${GPU_ARR[$j]}"
            k="${COMBOS[$idx]}"
            log="$LOG_DIR/baserag.$DATASET.k${k}.mc${MAX_DOC_CHARS}.gpu${gpu}.log"
            label="dataset=$DATASET gpu=$gpu top_k=$k"
            echo "[$(date '+%F %T')] start: $label -> $log"
            (
                CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON" src/hipporag/tests_baserag.py $LIMIT_FLAG \
                    $DATA_PATH_FLAG \
                    --dataset "$DATASET" \
                    --output_dir "$OUTPUT_DIR" \
                    --top_k "$k" \
                    --max_doc_chars "$MAX_DOC_CHARS" \
                    --llm_model_name "$LLM" \
                    --llm_base_url "$LLM_BASE_URL" \
                    --llm_concurrency "$LLM_CONCURRENCY" \
                    --embedding_model_name "$EMBED" \
                    --embedder_provider "$EMBEDDER_PROVIDER" \
                    --embedding_model_dtype "$EMBED_DTYPE" \
                    --encode_batch_size "$ENCODE_BATCH_SIZE" \
                    --disable_thinking "$DISABLE_THINKING" \
                    ${EMBEDDER_BASE_URL:+--embedder_base_url "$EMBEDDER_BASE_URL"} \
                    ${EMBEDDER_API_KEY_FILE:+--embedder_api_key_file "$EMBEDDER_API_KEY_FILE"} \
                    --embedder_concurrency "$EMBEDDER_CONCURRENCY" \
                    --embedder_batch_size "$EMBEDDER_BATCH_SIZE" \
                    --embedder_output_dim "$EMBEDDER_OUTPUT_DIM" \
                    ${EMBEDDER_CACHE_DB:+--embedder_cache_db "$EMBEDDER_CACHE_DB"} \
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
    done
    echo
done

echo "[$(date '+%F %T')] All BaseRAG datasets done."
