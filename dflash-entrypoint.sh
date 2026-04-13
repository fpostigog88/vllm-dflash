#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# DFlash vLLM Entrypoint
#
# Automatically downloads the DFlash drafter model from
# HuggingFace and configures speculative decoding.
#
# Environment variables:
#   MODEL_PATH           - Path or HF repo ID for the target model (required)
#   DFLASH_DRAFTER       - HF repo ID or local path for the DFlash drafter
#                          (e.g. "z-lab/Qwen3.5-27B-DFlash")
#                          Set to empty or "off" to disable spec decode.
#   DFLASH_NUM_SPEC_TOKENS - Number of speculative tokens (default: 15)
#   SERVED_MODEL_NAME    - Model name exposed via API (default: derived from MODEL_PATH)
#   MAX_MODEL_LEN        - Maximum sequence length (default: 4096)
#   MAX_NUM_SEQS         - Maximum concurrent sequences (default: 8)
#   GPU_MEMORY_UTILIZATION - GPU memory fraction (default: 0.80)
#   MAX_NUM_BATCHED_TOKENS - Max batched tokens (default: 8192)
#   KV_CACHE_DTYPE       - KV cache dtype (default: auto when DFlash enabled, fp8_e4m3 otherwise)
#   ATTENTION_BACKEND    - Attention backend (default: flash_attn)
#   QUANTIZATION         - Quantization method (default: auto-detected from model files)
#                          "modelopt" for NVFP4, "none" for BF16/FP16
#   VLLM_API_KEY         - Optional API key for authentication
#   EXTRA_ARGS           - Additional vllm serve arguments (space-separated)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DRAFTER_CACHE="/models/drafter-cache"

# ── Resolve target model ──────────────────────────────────────
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"

# If MODEL_PATH looks like a HF repo (contains /), download it
if [[ "$MODEL_PATH" == */* && ! -d "$MODEL_PATH" ]]; then
    echo "[dflash] Downloading target model: $MODEL_PATH"
    python3 -c "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$MODEL_PATH" /models/target
    MODEL_PATH="/models/target"
fi

# ── Resolve DFlash drafter ────────────────────────────────────
DFLASH_DRAFTER="${DFLASH_DRAFTER:-}"
DFLASH_NUM_SPEC_TOKENS="${DFLASH_NUM_SPEC_TOKENS:-15}"
DFLASH_ENABLED=false
DRAFTER_PATH=""

if [[ -n "$DFLASH_DRAFTER" && "$DFLASH_DRAFTER" != "off" ]]; then
    DFLASH_ENABLED=true

    if [[ -d "$DFLASH_DRAFTER" ]]; then
        # Local path
        DRAFTER_PATH="$DFLASH_DRAFTER"
        echo "[dflash] Using local drafter: $DRAFTER_PATH"
    elif [[ "$DFLASH_DRAFTER" == */* ]]; then
        # HuggingFace repo ID — download to cache
        REPO_NAME=$(echo "$DFLASH_DRAFTER" | tr '/' '_')
        DRAFTER_PATH="$DRAFTER_CACHE/$REPO_NAME"

        if [[ -f "$DRAFTER_PATH/config.json" ]]; then
            echo "[dflash] Drafter already cached: $DRAFTER_PATH"
        else
            echo "[dflash] Downloading drafter: $DFLASH_DRAFTER -> $DRAFTER_PATH"
            mkdir -p "$DRAFTER_PATH"
            python3 -c "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$DFLASH_DRAFTER" "$DRAFTER_PATH"
        fi
    else
        echo "[dflash] ERROR: DFLASH_DRAFTER='$DFLASH_DRAFTER' is not a valid path or HF repo ID"
        exit 1
    fi
fi

# ── Build vllm serve command ──────────────────────────────────
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"

# KV cache dtype: auto for DFlash (non-causal needs BF16), fp8 otherwise
if [[ "$DFLASH_ENABLED" == "true" ]]; then
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
else
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
fi

# Auto-detect quantization: check for NVFP4 hf_quant_config in model config
QUANTIZATION="${QUANTIZATION:-auto}"
if [[ "$QUANTIZATION" == "auto" ]]; then
    if [[ -d "$MODEL_PATH" ]] && python3 -c "
import json, sys
cfg = json.load(open('$MODEL_PATH/config.json'))
qc = cfg.get('quantization_config', cfg.get('hf_quantizer', {}))
sys.exit(0 if qc.get('quant_method','') == 'modelopt' or 'nvfp4' in json.dumps(cfg).lower() else 1)
" 2>/dev/null; then
        QUANTIZATION="modelopt"
        echo "[dflash] Auto-detected NVFP4 model -> --quantization modelopt"
    else
        QUANTIZATION="none"
        echo "[dflash] Auto-detected BF16/FP16 model -> no quantization flag"
    fi
fi

CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port 8000
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --dtype auto
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --attention-backend "$ATTENTION_BACKEND"
    --enable-chunked-prefill
    --enable-prefix-caching
    --limit-mm-per-prompt '{"image": 4, "video": 2}'
)

# Add quantization flag only for NVFP4 models
if [[ "$QUANTIZATION" != "none" ]]; then
    CMD+=(--quantization "$QUANTIZATION")
fi

# Add DFlash speculative config
if [[ "$DFLASH_ENABLED" == "true" ]]; then
    SPEC_CONFIG="{\"method\": \"dflash\", \"model\": \"$DRAFTER_PATH\", \"num_speculative_tokens\": $DFLASH_NUM_SPEC_TOKENS}"
    CMD+=(--speculative-config "$SPEC_CONFIG")
    echo "[dflash] Speculative decoding: ON (${DFLASH_NUM_SPEC_TOKENS} tokens, drafter: $DRAFTER_PATH)"
else
    echo "[dflash] Speculative decoding: OFF"
fi

# Add API key if set
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    CMD+=(--api-key "$VLLM_API_KEY")
fi

# Append any extra args
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -ra EXTRA <<< "$EXTRA_ARGS"
    CMD+=("${EXTRA[@]}")
fi

echo "[dflash] Starting: ${CMD[*]}"
exec "${CMD[@]}"
