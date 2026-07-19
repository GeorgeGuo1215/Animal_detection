#!/usr/bin/env bash
# === LivestockMind Agent 启动脚本 (Linux) ===
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[Config] Loaded .env"
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export OPENAI_MODEL="${OPENAI_MODEL:-deepseek-chat}"
export REASONER_MODEL="${REASONER_MODEL:-deepseek-reasoner}"
export AGENT_ENABLE_CORS="${AGENT_ENABLE_CORS:-1}"
export AGENT_WARMUP_DEVICE="${AGENT_WARMUP_DEVICE:-cuda}"
export AGENT_WARMUP_RAG="${AGENT_WARMUP_RAG:-1}"

# Livestock-specific RAG index (cattle/pigs/sheep/horses seed corpus).
# To share PetMind index instead, set: RAG_INDEX_DIR=../agentAndRag/RAG/data/rag_index_e5
_LIVESTOCK_RAG_ROOT="${SCRIPT_DIR}/RAG/data"
export RAG_INDEX_DIR="${RAG_INDEX_DIR:-${_LIVESTOCK_RAG_ROOT}/rag_index_e5}"
export RAG_RAW_DIR="${RAG_RAW_DIR:-${_LIVESTOCK_RAG_ROOT}/raw}"
export RAG_RELEVANCE_THRESHOLD="${RAG_RELEVANCE_THRESHOLD:-0.10}"

echo "=== LivestockMind Agent ==="
echo "[Config] OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "[Config] OPENAI_API_KEY=${OPENAI_API_KEY:+${OPENAI_API_KEY:0:10}...}"
echo "[Config] OPENAI_MODEL=$OPENAI_MODEL"
echo "[Config] REASONER_MODEL=$REASONER_MODEL"
echo "[Config] AGENT_WARMUP_DEVICE=$AGENT_WARMUP_DEVICE"
echo "[Config] RAG_INDEX_DIR=$RAG_INDEX_DIR"
echo "[Config] RAG_RELEVANCE_THRESHOLD=$RAG_RELEVANCE_THRESHOLD"
echo "[Config] TAVILY_API_KEY=${TAVILY_API_KEY:+${TAVILY_API_KEY:0:10}...}"
echo ""

CONDA_PYTHON="/home/sam/anaconda3/envs/AnimalDetection/bin/python"
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "[Error] Conda env python not found: $CONDA_PYTHON"
    exit 1
fi

"$CONDA_PYTHON" -m uvicorn agent_api.app.main:app \
    --host "${LIVESTOCK_AGENT_HOST:-0.0.0.0}" \
    --port "${LIVESTOCK_AGENT_PORT:-8000}"
