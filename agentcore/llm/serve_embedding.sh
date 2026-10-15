#!/bin/bash
# Serve Qwen/Qwen3-Embedding-8B via vLLM as an OpenAI-compatible
# embedding endpoint on port 8001.
#
# This embedding service supports the websearch benchmark (dense retrieval).
#
# Requires a dedicated conda env with vLLM>=0.20.0 installed.
#
# Usage:
#   bash agentcore/llm/serve_embedding.sh

set -e

export HF_HOME=~/.huggingface_cache
export HF_HUB_OFFLINE=1

# Adjust to the conda env that has vLLM installed for the embedding service.
CONDA_BASE=$(conda info --base 2>/dev/null)
VLLM_BIN=${CONDA_BASE}/envs/mage/bin/vllm
LOG_DIR=/tmp/vllm_logs
mkdir -p "$LOG_DIR"

MAX_MODEL_LEN=40960
GPU_MEM_UTIL=0.92
PORT=8001
GPU_IDS=0

SESSION=vllm_qwen3_embed_${PORT}

echo "=== Starting Qwen3-Embedding-8B on port ${PORT} (GPUs ${GPU_IDS}) in tmux session: ${SESSION} ==="
tmux new-session -d -s "${SESSION}" \
  "CUDA_VISIBLE_DEVICES=${GPU_IDS} HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 \
   $VLLM_BIN serve Qwen/Qwen3-Embedding-8B \
     --port ${PORT} \
     --gpu-memory-utilization ${GPU_MEM_UTIL} \
     --max-model-len ${MAX_MODEL_LEN} \
   2>&1 | tee $LOG_DIR/${SESSION}.log; echo === DONE ===; read"

echo
echo "Logs:    tail -f $LOG_DIR/${SESSION}.log"
echo "Attach:  tmux attach -t ${SESSION}    (Ctrl+B D to detach)"
echo "Endpoint: http://localhost:${PORT}/v1   (model: Qwen/Qwen3-Embedding-8B)"
