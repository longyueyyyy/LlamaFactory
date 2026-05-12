#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-saves/egocross/qwen3vl4b/kto_lora_from_grpo_answer_reward_perm4_lr3e6_beta010_ftx002_ep2_memsafe_ctx24576_px131k_merged}"
SUPPORT_DIR="${SUPPORT_DIR:-/share/home/group9/data/egocross}"
GPU="${GPU:-4}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-egocross_kto}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/egocross_kto_support_eval_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-egocross_outputs/support_eval/kto_lora_from_grpo_answer_reward_perm4_direct_tail_dense_max12_vid006_4f_${RUN_ID}}"
SERVER_LOG="${LOG_ROOT}/vllm_gpu${GPU}_port${PORT}.log"

export PATH="/share/home/group9/miniconda3/envs/lsg/bin:${PATH}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_DIR}"

say() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/driver.log"
}

wait_vllm() {
  local url="http://127.0.0.1:${PORT}/v1/models"
  local deadline="$((SECONDS + 900))"
  while (( SECONDS < deadline )); do
    if python - "${url}" <<'PY' >/dev/null 2>&1
import sys
from urllib import request
request.urlopen(sys.argv[1], timeout=2).read()
PY
    then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      say "vLLM exited before ready. See ${SERVER_LOG}"
      return 1
    fi
    sleep 5
  done
  say "Timed out waiting for vLLM. See ${SERVER_LOG}"
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap stop_server EXIT

if [[ ! -d "${MODEL_PATH}" ]]; then
  say "ERROR: missing model path: ${MODEL_PATH}"
  exit 2
fi

say "Starting vLLM for fixed support eval: model=${MODEL_PATH}, GPU=${GPU}, PORT=${PORT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --enforce-eager \
  --mm-processor-cache-gb 0 \
  --max-num-seqs 1 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

wait_vllm
say "vLLM ready"

python scripts/egocross_support_eval.py \
  --support-dir "${SUPPORT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --prompt-mode direct \
  --max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --temperature 0 \
  --max-tokens 8 \
  > "${LOG_ROOT}/support_eval.log" 2>&1

say "Fixed support eval done: ${OUTPUT_DIR}/support_metrics.json"
python - "${OUTPUT_DIR}/support_metrics.json" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(m["overall"], indent=2, ensure_ascii=False, sort_keys=True))
PY
