# syntax=docker/dockerfile:1.6
#
# DFlash vLLM — Dedicated container for DFlash speculative decoding
# on NVIDIA DGX Spark (GB10 Blackwell, SM121)
#
# Extends the AEON-7 vLLM Spark image with:
#   - Smart entrypoint that auto-downloads DFlash drafter from HuggingFace
#   - Pre-configured for NVFP4 quantized models
#   - All GB10/Blackwell optimizations (FlashInfer CUTLASS, torch.compile, etc.)
#
# Usage:
#   docker buildx build -f Dockerfile.dflash -t ghcr.io/aeon-7/vllm-dflash:latest .
#
# Run with DFlash speculative decoding:
#   docker run --runtime nvidia --network host --ipc host \
#     -e MODEL_PATH=/models/target \
#     -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
#     -e DFLASH_NUM_SPEC_TOKENS=15 \
#     -v /path/to/model:/models/target \
#     ghcr.io/aeon-7/vllm-dflash:latest
#
# Run WITHOUT speculative decoding:
#   docker run --runtime nvidia --network host --ipc host \
#     -e MODEL_PATH=/models/target \
#     -v /path/to/model:/models/target \
#     ghcr.io/aeon-7/vllm-dflash:latest
# =========================================================================

FROM ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest

LABEL org.opencontainers.image.title="vLLM DFlash"
LABEL org.opencontainers.image.description="vLLM with DFlash block-diffusion speculative decoding for NVIDIA DGX Spark"
LABEL org.opencontainers.image.source="https://github.com/AEON-7/vllm-dflash"

# Drafter model cache directory (persist via volume for faster restarts)
RUN mkdir -p /models/drafter-cache

# Entrypoint that handles drafter download and vllm configuration
COPY dflash-entrypoint.sh /usr/local/bin/dflash-entrypoint.sh
RUN chmod +x /usr/local/bin/dflash-entrypoint.sh

# Default environment
ENV MODEL_PATH="/models/target"
ENV DFLASH_DRAFTER=""
ENV DFLASH_NUM_SPEC_TOKENS="15"
ENV SERVED_MODEL_NAME=""
ENV MAX_MODEL_LEN="65536"
ENV MAX_NUM_SEQS="4"
ENV GPU_MEMORY_UTILIZATION="0.85"
ENV MAX_NUM_BATCHED_TOKENS="65536"
ENV KV_CACHE_DTYPE=""
ENV ATTENTION_BACKEND="flash_attn"
ENV TORCH_MATMUL_PRECISION="high"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/dflash-entrypoint.sh"]
