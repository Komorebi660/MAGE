#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash run.sh <dataset|all> [base_url] [model]"
    echo "Datasets: webshop websearch math phys travelplan"
    echo "Default base_url: http://localhost:8004/v1"
    echo "Default model: Qwen/Qwen3.6-27B"
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage
    exit 1
fi
case "$1" in
    -h|--help) usage; exit 0 ;;
    all) DATASETS=(webshop websearch math phys travelplan) ;;
    webshop|websearch|math|phys|travelplan) DATASETS=("$1") ;;
    *) usage; exit 1 ;;
esac

CONDA_BASE=$(conda info --base)
ENV_DIR="$CONDA_BASE/envs/mage"
PY="$ENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Python executable not found: $PY" >&2
    exit 1
fi

BASE_URL="${2:-http://localhost:8004/v1}"
LLM_MODEL="${3:-Qwen/Qwen3.6-27B}"
VLLM_EMBEDDING_URL="http://localhost:8001/v1/embeddings"
WORKERS=8
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_TAG="${LLM_MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"

for DATASET in "${DATASETS[@]}"; do
    SESSION="mage_${DATASET}_${MODEL_TAG}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session already exists: $SESSION" >&2
        exit 1
    fi
done

STAMP=$(date +%Y%m%d-%H%M%S)

for DATASET in "${DATASETS[@]}"; do
    SESSION="mage_${DATASET}_${MODEL_TAG}"
    LOG_DIR="$WORKDIR/logs/$DATASET/$STAMP"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/run_${DATASET}.log"
    case "$DATASET" in
        webshop) NUM=150; MAX_RETRY_NUM=2; MAX_STEPS=60 ;;
        websearch) NUM=221; MAX_RETRY_NUM=0; MAX_STEPS=30 ;;
        math) NUM=40; MAX_RETRY_NUM=2; MAX_STEPS=30 ;;
        phys) NUM=20; MAX_RETRY_NUM=1; MAX_STEPS=30 ;;
        travelplan) NUM=270; MAX_RETRY_NUM=0; MAX_STEPS=40 ;;
    esac
    tmux new-session -d -s "$SESSION" \
        "set -o pipefail; cd \"$WORKDIR\" && \
        JAVA_HOME=\"$ENV_DIR\" \
        VLLM_EMBEDDING_URL=\"$VLLM_EMBEDDING_URL\" \
        \"$PY\" -u evaluate.py \
        --dataset $DATASET \
        --llm \"$LLM_MODEL\" \
        --base_url \"$BASE_URL\" \
        --num $NUM --workers $WORKERS --max_retry_num $MAX_RETRY_NUM \
        --max_steps $MAX_STEPS \
        --log_dir \"$LOG_DIR\" 2>&1 | tee \"$LOG_FILE\"; \
        status=\$?; echo \"Exit code: \$status\"; exit \$status"
    echo "$DATASET: tmux attach -t $SESSION"
    echo "Log: $LOG_FILE"
done