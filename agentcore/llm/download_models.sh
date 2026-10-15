#!/bin/bash
# Download all required models for MAGE.
#
# Models:
#   1. Qwen/Qwen3.6-27B           — dense main LLM (serve_qwen3.6_27B.sh)
#   2. Qwen/Qwen3.6-35B-A3B       — MoE main LLM (serve_qwen3.6_35B.sh)
#   3. google/gemma-4-31B-it      — dense main LLM (serve_gemma4_31B.sh)
#   4. Qwen/Qwen3-Embedding-8B    — embedding model (serve_qwen3_embedding_8B.sh)
#
# All models will be cached under HF_HOME.
#
# Prerequisites:
#   conda env with huggingface_hub installed
#   huggingface-cli login             # if gated models require auth
#
# Usage:
#   bash agentcore/llm/download_models.sh            # download all
#   bash agentcore/llm/download_models.sh qwen       # download Qwen3.6 dense only
#   bash agentcore/llm/download_models.sh qwen_moe   # download Qwen3.6 MoE only
#   bash agentcore/llm/download_models.sh gemma      # download Gemma4 dense only
#   bash agentcore/llm/download_models.sh embed      # download embedding only

set -e

export HF_HOME=~/.huggingface_cache

CONDA_BASE=$(conda info --base 2>/dev/null)
ENV_DIR="${CONDA_BASE}/envs/mage"

if [[ ! -d "$ENV_DIR" ]]; then
    echo "ERROR: mage conda env not found at $ENV_DIR" >&2
    echo "Create it before running this script; this script only downloads models." >&2
    exit 1
fi

# Use the hf CLI from the mage conda env explicitly so this script
# works regardless of the currently activated env
HF_BIN="${HF_BIN:-${ENV_DIR}/bin/hf}"

if [[ ! -x "$HF_BIN" ]]; then
    echo "ERROR: hf binary not found at $HF_BIN" >&2
    echo "Set HF_BIN env var to the correct path, or install huggingface_hub in mage." >&2
    exit 1
fi

FILTER="${1:-all}"

download_model() {
    local model="$1"
    echo ""
    echo "============================================"
    echo "  Downloading: ${model}"
    echo "  Cache dir:   ${HF_HOME}"
    echo "  CLI:         ${HF_BIN}"
    echo "============================================"
    "$HF_BIN" download "$model"
    echo "  ✓ ${model} done."
}

if [[ "$FILTER" == "all" || "$FILTER" == "qwen" ]]; then
    download_model "Qwen/Qwen3.6-27B"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "qwen_moe" || "$FILTER" == "qwen-moe" ]]; then
    download_model "Qwen/Qwen3.6-35B-A3B"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "gemma" ]]; then
    download_model "google/gemma-4-31B-it"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "embed" ]]; then
    download_model "Qwen/Qwen3-Embedding-8B"
fi

echo ""
echo "============================================"
echo "  All requested models downloaded."
echo "  HF_HOME=${HF_HOME}"
echo "============================================"
