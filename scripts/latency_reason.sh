#!/usr/bin/env bash
# Latency of Qwen3.8-27B with reasoning (amendment 2): local vLLM on one GPU, batch 1, the tasks Jev also ran.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/serve_qwen38.sh
JT_BASE_URL=http://127.0.0.1:8001/v1 ${PYTHON:-python} -m jt.latency --backend reason:Qwen/Qwen3.8-27B-FP8 --tasks medabs hoc || echo FAILED
for p in $(ps -eo pid,args | grep -E '[v]llm serve models/' | awk '{print $1}'); do kill $p; done
echo LATENCY DONE
