#!/bin/bash
# Serve one LLM via vLLM.
#
# Pass a model tag to switch models. Tags match download_models.sh:
#   bash agentcore/llm/serve_llm.sh qwen
#   bash agentcore/llm/serve_llm.sh gemma

set -e

CONDA_BASE=$(conda info --base 2>/dev/null)
VLLM_BIN="${CONDA_BASE}/envs/mage/bin/vllm"
if [[ ! -x "$VLLM_BIN" ]]; then
    echo "ERROR: vllm binary not found at $VLLM_BIN" >&2
    exit 1
fi

PORT=8004
export HF_HOME=~/.huggingface_cache
export HF_HUB_OFFLINE=1

MODEL_TAG="${1:-qwen}"

case "$MODEL_TAG" in
    qwen)
        MODEL_NAME="Qwen/Qwen3.6-27B"
        REASONING_PARSER="qwen3"
        ;;
    qwen_moe | qwen-moe)
        MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
        REASONING_PARSER="qwen3"
        ;;
    gemma)
        MODEL_NAME="google/gemma-4-31B-it"
        REASONING_PARSER="gemma4"
        ;;
    *)
        echo "ERROR: unsupported model tag: $MODEL_TAG" >&2
        echo "Supported tags: qwen, qwen_moe, gemma." >&2
        exit 1
        ;;
esac

MODEL_SLUG=$(printf '%s' "${MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')

LOG_DIR=/tmp/vllm_logs
mkdir -p "$LOG_DIR"

SESSION="vllm_${MODEL_SLUG}_${PORT}"
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/${SESSION}_${TS}.log"

echo "=== Starting ${MODEL_NAME} on port ${PORT} ==="
echo "=== Log: ${LOG_FILE} ==="

tmux new-session -d -s "$SESSION" \
  "CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 \
   $VLLM_BIN serve ${MODEL_NAME} \
     --port ${PORT} \
     --tensor-parallel-size 4 \
     --data-parallel-size 1 \
     --max-model-len 131072 \
     --gpu-memory-utilization 0.94 \
     --reasoning-parser ${REASONING_PARSER} \
     --dtype bfloat16 \
     --language-model-only \
     --enable-prefix-caching \
   2>&1 | tee ${LOG_FILE}; echo === DONE ===; read"

echo
echo "=== tmux session started ==="
tmux ls | grep "$SESSION"
echo "Logs:    tail -f ${LOG_FILE}"
echo "Attach:  tmux attach -t ${SESSION}    (Ctrl+B D to detach)"
echo "Endpoint: http://localhost:${PORT}/v1   (model: ${MODEL_NAME})"