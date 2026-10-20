#!/usr/bin/env bash
# Qwen3.8-27B-FP8 on one GPU (loopback :8001), same vLLM flags as OpenJev (scripts/serve_openjev.sh). Offline.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=models/qwen3.8-27b-fp8
${VLLM_BIN:-vllm} serve "$M" --host 127.0.0.1 --served-model-name Qwen/Qwen3.8-27B-FP8 --port 8001 --enable-prefix-caching \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code --max-num-seqs ${MAX_SEQS:-16} --max-logprobs 64 \
  --gdn-prefill-backend triton --quantization fp8 > runs/qwen38_local_vllm.log 2>&1 &
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 5; done
echo "vLLM up"
