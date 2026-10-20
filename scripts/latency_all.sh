#!/usr/bin/env bash
# Latency (warm, batch 1) of every local arm on one GPU, one at a time: Jeff/Qwen3.5 in-process, then OpenJev
# (vLLM + helper) and the plain Qwen3.8-27B-FP8 served with the same vLLM flags. Jev is timed over the internet.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=${PYTHON:-python}
for b in jeff-0.8b qwen3.5-0.8b jeff-2b qwen3.5-2b; do $L -m jt.latency --backend $b || echo FAILED $b; done
stop() { for p in $(ps -eo pid,args | grep -E '[s]him.py --host 127.0.0.1|[v]llm serve models/' | awk '{print $1}'); do kill $p; done; sleep 30; }
scripts/serve_openjev.sh && until curl -sf http://127.0.0.1:3000/ >/dev/null 2>&1 || [ $? -eq 22 ]; do sleep 5; done
sleep 20; $L -m jt.latency --backend openjev-fp8 || echo FAILED openjev; stop
M=models/qwen3.8-27b-fp8
${VLLM_BIN:-vllm} serve "$M" --host 127.0.0.1 --served-model-name Qwen/Qwen3.8-27B-FP8 --port 8001 --enable-prefix-caching \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code --max-num-seqs 16 --max-logprobs 64 \
  --gdn-prefill-backend triton --quantization fp8 > runs/qwen38_local_vllm.log 2>&1 &
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 5; done
JT_BASE_URL=http://127.0.0.1:8001/v1 $L -m jt.latency --backend chat:Qwen/Qwen3.8-27B-FP8 || echo FAILED qwen38; stop
$L -m jt.latency --backend jev --tasks medabs hoc || echo FAILED jev
echo LATENCY DONE
