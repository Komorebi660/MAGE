#!/usr/bin/env bash
# Download (and decrypt) the BrowseComp-Plus dataset, the pre-built BM25 index,
# and the Qwen3-Embedding-8B FAISS index.
#
# Usage:
#   cd WebSearchEnv
#   bash download_data_and_index.sh
#
# Prerequisites:
#   pip install datasets huggingface_hub
#   huggingface-cli login   # if needed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/data"
INDEX_DIR="$SCRIPT_DIR/indexes"

mkdir -p "$DATA_DIR"
mkdir -p "$INDEX_DIR"

# ──────────────────────────────────────────────
# 1. Download & decrypt the dataset (original questions and answers)
# ──────────────────────────────────────────────
DECRYPTED_FILE="$DATA_DIR/browsecomp_plus_decrypted.jsonl"
QUERIES_TSV="$SCRIPT_DIR/topics-qrels/queries.tsv"

if [ -f "$DECRYPTED_FILE" ]; then
    echo "[INFO] Decrypted dataset already exists: $DECRYPTED_FILE"
else
    echo "[INFO] Downloading and decrypting dataset..."
    python scripts_build_index/decrypt_dataset.py \
        --output "$DECRYPTED_FILE" \
        --generate-tsv "$QUERIES_TSV"
    echo "[INFO] Dataset saved to $DECRYPTED_FILE"
fi

# ──────────────────────────────────────────────
# 2. Download MemoryArena data.jsonl
# ──────────────────────────────────────────────
MEMORY_ARENA_FILE="$DATA_DIR/data.jsonl"

if [ -f "$MEMORY_ARENA_FILE" ]; then
    echo "[INFO] MemoryArena data already exists: $MEMORY_ARENA_FILE"
else
    echo "[INFO] Downloading MemoryArena data.jsonl..."
    wget -O "$MEMORY_ARENA_FILE" \
        https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/progressive_search/data.jsonl
    echo "[INFO] MemoryArena data saved to $MEMORY_ARENA_FILE"
fi

# ──────────────────────────────────────────────
# 3. Download the pre-built BM25 Lucene index (includes the corpus and the index)
# ──────────────────────────────────────────────
BM25_INDEX_DIR="$INDEX_DIR/bm25"

if [ -d "$BM25_INDEX_DIR" ] && [ "$(ls -A "$BM25_INDEX_DIR" 2>/dev/null)" ]; then
    echo "[INFO] BM25 index already exists: $BM25_INDEX_DIR"
else
    echo "[INFO] Downloading pre-built BM25 index from HuggingFace..."
    hf download \
        Tevatron/browsecomp-plus-indexes \
        --repo-type dataset \
        --include "bm25/*" \
        --local-dir "$INDEX_DIR"
    echo "[INFO] BM25 index saved to $BM25_INDEX_DIR"
fi

# ──────────────────────────────────────────────
# 4. Download the pre-built Qwen3-Embedding-8B FAISS index
# ──────────────────────────────────────────────
FAISS_INDEX_DIR="$INDEX_DIR/qwen3-embedding-8b"

if [ -d "$FAISS_INDEX_DIR" ] && [ "$(ls -A "$FAISS_INDEX_DIR" 2>/dev/null)" ]; then
    echo "[INFO] Qwen3-Embedding-8B index already exists: $FAISS_INDEX_DIR"
else
    echo "[INFO] Downloading pre-built Qwen3-Embedding-8B index from HuggingFace..."
    hf download \
        Tevatron/browsecomp-plus-indexes \
        --repo-type dataset \
        --include "qwen3-embedding-8b/*" \
        --local-dir "$INDEX_DIR"
    echo "[INFO] Qwen3-Embedding-8B index saved to $FAISS_INDEX_DIR"
fi

echo ""
echo "========================================"
echo "  All downloads complete!"
echo "  Dataset      : $DECRYPTED_FILE"
echo "  MemArena     : $MEMORY_ARENA_FILE"
echo "  BM25 idx     : $BM25_INDEX_DIR"
echo "  FAISS idx    : $FAISS_INDEX_DIR"
echo "========================================"
