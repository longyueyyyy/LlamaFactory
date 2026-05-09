#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LLAMAFACTORY_LOGPS_CHUNK_SIZE="${LLAMAFACTORY_LOGPS_CHUNK_SIZE:-128}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EGOCROSS_DPO_LOG_ROOT:-logs/egocross_dpo_lora_folds_abc_memsafe_${RUN_ID}}"
DRIVER_LOG="${LOG_ROOT}/driver.log"
SUMMARY_TSV="${LOG_ROOT}/summary.tsv"

mkdir -p "${LOG_ROOT}"
printf "candidate\tfold\tgpu_pair\tstatus\tstart_time\tend_time\tconfig\tlog_file\n" > "${SUMMARY_TSV}"

declare -A SUFFIX_BY_CANDIDATE=(
  [A]="lr1e5_beta003_ftx005_ep1"
  [B]="lr5e6_beta003_ftx005_ep1"
  [C]="lr1e5_beta005_ftx005_ep1"
)

read -r -a CANDIDATE_LIST <<< "${CANDIDATES:-A B C}"
read -r -a FOLD_LIST <<< "${FOLDS:-0 1 2 3}"
read -r -a GPU_LANE_LIST <<< "${GPU_LANES:-4,5}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

say() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${DRIVER_LOG}"
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

output_dir_for() {
  local config="$1"
  awk -F': ' '/^output_dir:/ {print $2; exit}' "${config}"
}

build_tasks() {
  TASKS=()
  local candidate fold config
  for candidate in "${CANDIDATE_LIST[@]}"; do
    for fold in "${FOLD_LIST[@]}"; do
      config="$(config_for "${candidate}" "${fold}")"
      TASKS+=("${candidate}:${fold}:${config}")
    done
  done
}

preflight() {
  local task candidate fold config output_dir
  say "log_root=${LOG_ROOT}"
  say "candidates=${CANDIDATE_LIST[*]} folds=${FOLD_LIST[*]} gpu_lanes=${GPU_LANE_LIST[*]}"
  say "LLAMAFACTORY_LOGPS_CHUNK_SIZE=${LLAMAFACTORY_LOGPS_CHUNK_SIZE}"
  say "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

  for task in "${TASKS[@]}"; do
    IFS=: read -r candidate fold config <<< "${task}"
    if [[ ! -f "${config}" ]]; then
      say "ERROR: missing config ${config}"
      exit 2
    fi
    output_dir="$(output_dir_for "${config}")"
    if [[ -z "${output_dir}" ]]; then
      say "ERROR: config has no output_dir: ${config}"
      exit 2
    fi
    if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" && "${SKIP_EXISTING}" != "1" ]]; then
      say "ERROR: output_dir already exists and is non-empty: ${output_dir}"
      say "Move/rename that directory, or rerun with SKIP_EXISTING=1 to skip finished/existing outputs."
      exit 2
    fi
  done
}

run_one() {
  local candidate="$1"
  local fold="$2"
  local config="$3"
  local gpu_pair="$4"
  local suffix="${SUFFIX_BY_CANDIDATE[${candidate}]}"
  local log_file="${LOG_ROOT}/${candidate}_fold${fold}_${suffix}.log"
  local output_dir start_time end_time status master_port

  output_dir="$(output_dir_for "${config}")"
  master_port="$((20000 + RANDOM % 40000))"
  if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" && "${SKIP_EXISTING}" == "1" ]]; then
    start_time="$(date '+%F %T')"
    end_time="${start_time}"
    say "SKIP ${candidate} fold${fold}: existing output_dir=${output_dir}"
    printf "%s\t%s\t%s\tSKIP_EXISTING\t%s\t%s\t%s\t%s\n" \
      "${candidate}" "${fold}" "${gpu_pair}" "${start_time}" "${end_time}" "${config}" "${log_file}" >> "${SUMMARY_TSV}"
    return 0
  fi

  start_time="$(date '+%F %T')"
  say "START ${candidate} fold${fold} on CUDA_VISIBLE_DEVICES=${gpu_pair}: ${config}"
  {
    printf "[%s] START candidate=%s fold=%s gpu_pair=%s\n" "${start_time}" "${candidate}" "${fold}" "${gpu_pair}"
    printf "config=%s\n" "${config}"
    printf "output_dir=%s\n" "${output_dir}"
    printf "MASTER_PORT=%s\n" "${master_port}"
    printf "LLAMAFACTORY_LOGPS_CHUNK_SIZE=%s\n" "${LLAMAFACTORY_LOGPS_CHUNK_SIZE}"
    printf "PYTORCH_CUDA_ALLOC_CONF=%s\n" "${PYTORCH_CUDA_ALLOC_CONF}"
  } | tee -a "${log_file}"

  set +e
  MASTER_PORT="${master_port}" CUDA_VISIBLE_DEVICES="${gpu_pair}" llamafactory-cli train "${config}" 2>&1 | tee -a "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e

  end_time="$(date '+%F %T')"
  if [[ "${status}" == "0" ]]; then
    say "DONE ${candidate} fold${fold}; log=${log_file}"
    printf "%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\n" \
      "${candidate}" "${fold}" "${gpu_pair}" "${start_time}" "${end_time}" "${config}" "${log_file}" >> "${SUMMARY_TSV}"
  else
    say "FAIL ${candidate} fold${fold} status=${status}; log=${log_file}"
    printf "%s\t%s\t%s\tFAIL_%s\t%s\t%s\t%s\t%s\n" \
      "${candidate}" "${fold}" "${gpu_pair}" "${status}" "${start_time}" "${end_time}" "${config}" "${log_file}" >> "${SUMMARY_TSV}"
  fi
  return "${status}"
}

run_lane() {
  local lane_index="$1"
  local gpu_pair="$2"
  local lane_count="$3"
  local i task candidate fold config

  for ((i = lane_index; i < ${#TASKS[@]}; i += lane_count)); do
    task="${TASKS[$i]}"
    IFS=: read -r candidate fold config <<< "${task}"
    run_one "${candidate}" "${fold}" "${config}" "${gpu_pair}"
  done
}

build_tasks
preflight

say "queued_tasks=${#TASKS[@]}"

if (( ${#GPU_LANE_LIST[@]} == 1 )); then
  run_lane 0 "${GPU_LANE_LIST[0]}" 1
else
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
    say "One or more lanes failed. Check ${SUMMARY_TSV} and per-task logs."
    exit 1
  fi
fi

say "All queued tasks completed. summary=${SUMMARY_TSV}"
