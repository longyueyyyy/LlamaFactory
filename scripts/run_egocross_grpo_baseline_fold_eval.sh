#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212}"
SUPPORT_DIR="${SUPPORT_DIR:-/share/home/group9/data/egocross}"
GPU="${GPU:-4}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-egocross}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EGOCROSS_EVAL_LOG_ROOT:-logs/egocross_grpo_baseline_fold_eval_${RUN_ID}}"
OUTPUT_ROOT="${EGOCROSS_EVAL_OUTPUT_ROOT:-egocross_outputs/support_eval/grpo_baseline_folds_${RUN_ID}}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-0}"

export VLLM_USE_V1="${VLLM_USE_V1:-0}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
SUMMARY_TSV="${LOG_ROOT}/summary.tsv"
SERVER_LOG="${LOG_ROOT}/vllm_gpu${GPU}_port${PORT}.log"
printf "model\tfold\tstatus\tacc\tanswer_only_format_rate\tcoverage\tparse_fail\terror\terror_fallback\tavg_used_frames\truntime_seconds\toutput_dir\tmetrics_json\n" > "${SUMMARY_TSV}"

read -r -a FOLD_LIST <<< "${FOLDS:-0 1 2 3}"

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

say "Starting GRPO baseline vLLM on GPU=${GPU}, PORT=${PORT}, model=${MODEL_PATH}"
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

for fold in "${FOLD_LIST[@]}"; do
  eval_json="${SUPPORT_DIR}/pref_answer_only_all_equal_folds/eval_answer_only_all_equal_fold${fold}.json"
  output_dir="${OUTPUT_ROOT}/fold${fold}_direct_tail_dense_max12_vid006_4f"
  eval_log="${LOG_ROOT}/eval_fold${fold}.log"

  if [[ -f "${output_dir}/support_metrics.json" && "${SKIP_EXISTING_EVAL}" == "1" ]]; then
    say "SKIP fold${fold}: existing ${output_dir}/support_metrics.json"
  else
    say "Eval GRPO baseline fold${fold}"
    python scripts/egocross_support_eval.py \
      --support-dir "${SUPPORT_DIR}" \
      --eval-json "${eval_json}" \
      --output-dir "${output_dir}" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${SERVED_MODEL_NAME}" \
      --prompt-mode direct \
      --max-frames 12 \
      --frame-sampling tail_dense \
      --frame-route VID006=4 \
      --temperature 0 \
      --max-tokens 8 \
      > "${eval_log}" 2>&1
  fi

  metrics_json="${output_dir}/support_metrics.json"
  python - "${metrics_json}" "${SUMMARY_TSV}" "GRPO" "${fold}" "${output_dir}" <<'PY'
import json, sys
metrics_path, summary_path, model, fold, output_dir = sys.argv[1:6]
m = json.load(open(metrics_path, encoding="utf-8"))
o = m["overall"]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(
        "\t".join(
            [
                model,
                str(fold),
                "OK",
                str(o["acc"]),
                str(o["answer_only_format_rate"]),
                str(o["coverage"]),
                str(o["parse_fail"]),
                str(o["error"]),
                str(o["error_fallback"]),
                str(o["avg_used_frames"]),
                str(o["runtime_seconds"]),
                output_dir,
                metrics_path,
            ]
        )
        + "\n"
    )
PY
done

say "GRPO baseline fold eval done. summary=${SUMMARY_TSV}"
