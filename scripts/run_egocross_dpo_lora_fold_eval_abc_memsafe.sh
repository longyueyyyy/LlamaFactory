#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SUPPORT_DIR="${SUPPORT_DIR:-/share/home/group9/data/egocross}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EGOCROSS_EVAL_LOG_ROOT:-logs/egocross_dpo_lora_fold_eval_abc_memsafe_${RUN_ID}}"
OUTPUT_ROOT="${EGOCROSS_EVAL_OUTPUT_ROOT:-egocross_outputs/support_eval/dpo_lora_fold_eval_abc_memsafe_${RUN_ID}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-egocross}"
EXPORT_MISSING="${EXPORT_MISSING:-1}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-0}"

export VLLM_USE_V1="${VLLM_USE_V1:-0}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
SUMMARY_TSV="${LOG_ROOT}/summary.tsv"
printf "candidate\tfold\tgpu\tport\tstatus\tacc\tanswer_only_format_rate\tcoverage\tparse_fail\terror\terror_fallback\tavg_used_frames\truntime_seconds\tmodel_dir\toutput_dir\tmetrics_json\n" > "${SUMMARY_TSV}"

declare -A SUFFIX_BY_CANDIDATE=(
  [A]="lr1e5_beta003_ftx005_ep1"
  [B]="lr5e6_beta003_ftx005_ep1"
  [C]="lr1e5_beta005_ftx005_ep1"
)

read -r -a CANDIDATE_LIST <<< "${CANDIDATES:-A B C}"
read -r -a FOLD_LIST <<< "${FOLDS:-0 1 2 3}"
read -r -a GPU_LANE_LIST <<< "${GPU_LANES:-4:8001 5:8002 6:8003 7:8004}"

say() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/driver.log"
}

config_for() {
  local candidate="$1"
  local fold="$2"
  local suffix="${SUFFIX_BY_CANDIDATE[${candidate}]:-}"
  if [[ -z "${suffix}" ]]; then
    say "ERROR: unknown candidate '${candidate}'. Use A, B, C."
    return 2
  fi
  printf "configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_fold%s_%s.yaml" "${fold}" "${suffix}"
}

adapter_dir_for_config() {
  awk -F': ' '/^output_dir:/ {print $2; exit}' "$1"
}

build_tasks() {
  TASKS=()
  local candidate fold config adapter_dir model_dir
  for candidate in "${CANDIDATE_LIST[@]}"; do
    for fold in "${FOLD_LIST[@]}"; do
      config="$(config_for "${candidate}" "${fold}")"
      if [[ ! -f "${config}" ]]; then
        say "ERROR: missing config ${config}"
        exit 2
      fi
      adapter_dir="$(adapter_dir_for_config "${config}")"
      model_dir="${adapter_dir}_merged"
      TASKS+=("${candidate}:${fold}:${adapter_dir}:${model_dir}")
    done
  done
}

export_missing_models() {
  local task candidate fold adapter_dir model_dir log_file
  if [[ "${EXPORT_MISSING}" != "1" ]]; then
    return 0
  fi
  for task in "${TASKS[@]}"; do
    IFS=: read -r candidate fold adapter_dir model_dir <<< "${task}"
    if [[ -f "${model_dir}/config.json" ]]; then
      say "Merged model exists: ${candidate} fold${fold}"
      continue
    fi
    log_file="${LOG_ROOT}/export_${candidate}_fold${fold}.log"
    say "Export missing merged model: ${candidate} fold${fold}"
    bash scripts/export_egocross_dpo_lora_fold_memsafe.sh "${candidate}" "${fold}" > "${log_file}" 2>&1
  done
}

wait_vllm() {
  local port="$1"
  local pid="$2"
  local server_log="$3"
  local url="http://127.0.0.1:${port}/v1/models"
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
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      say "vLLM exited before ready. See ${server_log}"
      return 1
    fi
    sleep 5
  done
  say "Timed out waiting for vLLM. See ${server_log}"
  return 1
}

append_metrics() {
  local candidate="$1"
  local fold="$2"
  local gpu="$3"
  local port="$4"
  local model_dir="$5"
  local output_dir="$6"
  local metrics_json="${output_dir}/support_metrics.json"
  python - "${metrics_json}" "${SUMMARY_TSV}" "${candidate}" "${fold}" "${gpu}" "${port}" "${model_dir}" "${output_dir}" <<'PY'
import json, sys
metrics_path, summary_path, candidate, fold, gpu, port, model_dir, output_dir = sys.argv[1:9]
m = json.load(open(metrics_path, encoding="utf-8"))
o = m["overall"]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(
        "\t".join(
            [
                candidate,
                str(fold),
                str(gpu),
                str(port),
                "OK",
                str(o["acc"]),
                str(o["answer_only_format_rate"]),
                str(o["coverage"]),
                str(o["parse_fail"]),
                str(o["error"]),
                str(o["error_fallback"]),
                str(o["avg_used_frames"]),
                str(o["runtime_seconds"]),
                model_dir,
                output_dir,
                metrics_path,
            ]
        )
        + "\n"
    )
PY
}

run_one() {
  local task="$1"
  local lane="$2"
  local gpu="${lane%%:*}"
  local port="${lane##*:}"
  local candidate fold adapter_dir model_dir output_dir eval_json server_log eval_log pid status
  IFS=: read -r candidate fold adapter_dir model_dir <<< "${task}"

  if [[ ! -f "${model_dir}/config.json" ]]; then
    say "ERROR: missing merged model for ${candidate} fold${fold}: ${model_dir}"
    return 2
  fi

  output_dir="${OUTPUT_ROOT}/${candidate}_fold${fold}_direct_tail_dense_max12_vid006_4f"
  eval_json="${SUPPORT_DIR}/pref_answer_only_all_equal_folds/eval_answer_only_all_equal_fold${fold}.json"
  server_log="${LOG_ROOT}/vllm_${candidate}_fold${fold}_gpu${gpu}_port${port}.log"
  eval_log="${LOG_ROOT}/eval_${candidate}_fold${fold}.log"

  if [[ -f "${output_dir}/support_metrics.json" && "${SKIP_EXISTING_EVAL}" == "1" ]]; then
    say "SKIP ${candidate} fold${fold}: existing eval output"
    append_metrics "${candidate}" "${fold}" "${gpu}" "${port}" "${model_dir}" "${output_dir}"
    return 0
  fi

  say "START ${candidate} fold${fold} on GPU=${gpu}, PORT=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m vllm.entrypoints.openai.api_server \
    --model "${model_dir}" \
    --port "${port}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --enforce-eager \
    --mm-processor-cache-gb 0 \
    --max-num-seqs 1 \
    > "${server_log}" 2>&1 &
  pid="$!"

  set +e
  wait_vllm "${port}" "${pid}" "${server_log}"
  status="$?"
  set -e
  if [[ "${status}" != "0" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
    return "${status}"
  fi

  set +e
  python scripts/egocross_support_eval.py \
    --support-dir "${SUPPORT_DIR}" \
    --eval-json "${eval_json}" \
    --output-dir "${output_dir}" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --prompt-mode direct \
    --max-frames 12 \
    --frame-sampling tail_dense \
    --frame-route VID006=4 \
    --temperature 0 \
    --max-tokens 8 \
    > "${eval_log}" 2>&1
  status="$?"
  set -e

  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true

  if [[ "${status}" != "0" ]]; then
    say "FAIL ${candidate} fold${fold}; eval log=${eval_log}"
    printf "%s\t%s\t%s\t%s\tFAIL_%s\t\t\t\t\t\t\t\t\t%s\t%s\t%s\n" \
      "${candidate}" "${fold}" "${gpu}" "${port}" "${status}" "${model_dir}" "${output_dir}" "${output_dir}/support_metrics.json" >> "${SUMMARY_TSV}"
    return "${status}"
  fi

  append_metrics "${candidate}" "${fold}" "${gpu}" "${port}" "${model_dir}" "${output_dir}"
  say "DONE ${candidate} fold${fold}; metrics=${output_dir}/support_metrics.txt"
}

run_lane() {
  local lane_index="$1"
  local lane="$2"
  local lane_count="$3"
  local i
  for ((i = lane_index; i < ${#TASKS[@]}; i += lane_count)); do
    run_one "${TASKS[$i]}" "${lane}"
  done
}

cleanup() {
  jobs -pr | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT

build_tasks
say "tasks=${#TASKS[@]} candidates=${CANDIDATE_LIST[*]} folds=${FOLD_LIST[*]} gpu_lanes=${GPU_LANE_LIST[*]}"
export_missing_models

pids=()
for lane_index in "${!GPU_LANE_LIST[@]}"; do
  run_lane "${lane_index}" "${GPU_LANE_LIST[${lane_index}]}" "${#GPU_LANE_LIST[@]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  say "One or more eval lanes failed. Check ${SUMMARY_TSV} and logs under ${LOG_ROOT}."
  exit 1
fi

say "DPO fold eval done. summary=${SUMMARY_TSV}"
