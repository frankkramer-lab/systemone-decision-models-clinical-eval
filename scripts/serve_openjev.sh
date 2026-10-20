#!/usr/bin/env bash
# OpenJev-FP8 on one GPU: vLLM (loopback :8000) + the unmodified helper (loopback :3000), flags as in
# models/openjev-helper/serve/SERVE.md and the FP8 card (--max-num-seqs 16). Offline: nothing leaves the machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=models/openjev-fp8
${VLLM_BIN:-vllm} serve "$M" --host 127.0.0.1 --served-model-name qwen --port 8000 --enable-prefix-caching \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1}' --trust-remote-code \
  --max-num-seqs 16 --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8 > runs/openjev_vllm.log 2>&1 &
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo "vLLM up"
VLLM=http://localhost:8000/v1 TOKENIZER="$M" READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
  READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
  ${VLLM_PYTHON:-python} models/openjev-helper/helper/shim.py --host 127.0.0.1 --port 3000 > runs/openjev_shim.log 2>&1 &
echo "helper started"
