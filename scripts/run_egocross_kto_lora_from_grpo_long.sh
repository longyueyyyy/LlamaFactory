#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/share/home/group9/data/egocross}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/egocross_kto_lora_from_grpo_answer_reward_perm4_lr3e6_beta010_ftx002_ep2.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/egocross_kto_lora_from_grpo_answer_reward_perm4_lr3e6_beta010_ftx002_smoke.yaml}"
KTO_FILE="${KTO_FILE:-${DATA_DIR}/train_kto_answer_reward_perm4_wrong3_fmt2_seed2026.json}"
FOLD_DIR="${FOLD_DIR:-${DATA_DIR}/kto_answer_reward_perm4_wrong3_fmt2_folds}"
CHECK_REPORT="${CHECK_REPORT:-logs/egocross_kto_answer_reward_perm4_data_check.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/egocross_kto_lora_from_grpo_${RUN_ID}}"
TRAIN_LOG="${LOG_ROOT}/train_long.log"
SMOKE_LOG="${LOG_ROOT}/smoke.log"
PID_FILE="${LOG_ROOT}/train_long.pid"
TRAIN_GPUS="${TRAIN_GPUS:-4,5}"
SMOKE_GPUS="${SMOKE_GPUS:-${TRAIN_GPUS}}"

export PATH="/share/home/group9/miniconda3/envs/lsg/bin:${PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LLAMAFACTORY_LOGPS_CHUNK_SIZE="${LLAMAFACTORY_LOGPS_CHUNK_SIZE:-128}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

say() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/driver.log"
}

yaml_value() {
  local key="$1"
  local file="$2"
  awk -F': ' -v k="${key}" '$1 == k {print $2; exit}' "${file}"
}

require_clean_or_missing() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    say "ERROR: ${label} exists and is non-empty: ${path}"
    exit 2
  fi
}

mkdir -p "${LOG_ROOT}" "$(dirname "${CHECK_REPORT}")"

LONG_OUTPUT_DIR="$(yaml_value output_dir "${TRAIN_CONFIG}")"
SMOKE_OUTPUT_DIR="$(yaml_value output_dir "${SMOKE_CONFIG}")"
require_clean_or_missing "${LONG_OUTPUT_DIR}" "long output_dir"
require_clean_or_missing "${SMOKE_OUTPUT_DIR}" "smoke output_dir"

for protected in \
  "submission_template.json" \
  "saves/egocross/qwen3vl4b/full_sft_32k_200k" \
  "saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1" \
  "/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212" \
  "egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f"; do
  if [[ "${LONG_OUTPUT_DIR}" == "${protected}" || "${SMOKE_OUTPUT_DIR}" == "${protected}" ]]; then
    say "ERROR: refusing protected output path: ${protected}"
    exit 2
  fi
done

say "Preparing train-safe GRPO view"
python scripts/prepare_egocross_grpo_train_safe.py --overwrite-config

if [[ ! -f "${KTO_FILE}" || "${OVERWRITE_DATA:-0}" == "1" ]]; then
  say "Preparing KTO answer-only data: ${KTO_FILE}"
  python scripts/prepare_egocross_kto_answer_only.py \
    --data-dir "${DATA_DIR}" \
    --output "${KTO_FILE}" \
    --fold-output-dir "${FOLD_DIR}" \
    --permutations 4 \
    --format-negatives 2 \
    --seed 2026 \
    ${OVERWRITE_DATA:+--overwrite}
else
  say "KTO data already exists: ${KTO_FILE}"
fi

say "Checking KTO data"
python scripts/check_egocross_kto_data.py \
  --data-dir "${DATA_DIR}" \
  --kto-file "${KTO_FILE}" \
  --fold-dir "${FOLD_DIR}" \
  --output-json "${CHECK_REPORT}"

say "Running smoke train on CUDA_VISIBLE_DEVICES=${SMOKE_GPUS}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}" llamafactory-cli train "${SMOKE_CONFIG}" 2>&1 | tee "${SMOKE_LOG}"

say "Smoke passed; starting long KTO train in background on CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}"
(
  set -euo pipefail
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" llamafactory-cli train "${TRAIN_CONFIG}"
) > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID="$!"
printf "%s\n" "${TRAIN_PID}" > "${PID_FILE}"

say "Long training launched: pid=${TRAIN_PID}"
say "train_log=${TRAIN_LOG}"
say "output_dir=${LONG_OUTPUT_DIR}"
say "watch: tail -f ${TRAIN_LOG}"
