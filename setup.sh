#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# DFlash Quick Setup
# Generates .env.dflash with a random API key and prompts for
# required configuration.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ENV_FILE=".env.dflash"

echo "=== DFlash vLLM Setup ==="
echo ""

# Generate API key
API_KEY=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
echo "Generated API key: ${API_KEY:0:8}...${API_KEY: -8}"

# Prompt for HF token
read -rp "HuggingFace token (leave blank to skip): " HF_TOKEN
HF_TOKEN="${HF_TOKEN:-}"

# Prompt for model path
DEFAULT_MODEL_PATH="/home/$(whoami)/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4"
read -rp "Model path [${DEFAULT_MODEL_PATH}]: " MODEL_PATH
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"

# Prompt for spec tokens
read -rp "Speculative tokens (15=fast single-stream, 5=high concurrency) [15]: " SPEC_TOKENS
SPEC_TOKENS="${SPEC_TOKENS:-15}"

# Write env file
cat > "$ENV_FILE" << EOF
# DFlash vLLM Configuration (generated $(date +%Y-%m-%d))

# Authentication
HF_TOKEN=${HF_TOKEN}
VLLM_API_KEY=${API_KEY}

# Model
MODEL_HOST_PATH=${MODEL_PATH}
DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash
DFLASH_NUM_SPEC_TOKENS=${SPEC_TOKENS}

# Resources (DGX Spark GB10)
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=8
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_BATCHED_TOKENS=8192
EOF

echo ""
echo "Wrote ${ENV_FILE}"
echo ""
echo "To start:"
echo "  docker compose --env-file ${ENV_FILE} -f docker-compose.dflash.yml up -d"
echo ""
echo "API endpoint: http://$(hostname -I | awk '{print $1}'):8000/v1"
echo "API key: ${API_KEY}"
