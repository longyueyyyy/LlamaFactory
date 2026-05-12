#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/egocross_export_kto_lora_from_grpo_answer_reward_perm4_lr3e6_beta010_ftx002_ep2.yaml}"
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:${PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

yaml_value() {
  local key="$1"
  awk -F': ' -v k="${key}" '$1 == k {print $2; exit}' "${CONFIG}"
}

ADAPTER_DIR="$(yaml_value adapter_name_or_path)"
EXPORT_DIR="$(yaml_value export_dir)"

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "ERROR: missing adapter dir: ${ADAPTER_DIR}" >&2
  exit 2
fi
if [[ -d "${EXPORT_DIR}" && -n "$(find "${EXPORT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" && "${OVERWRITE_EXPORT:-0}" != "1" ]]; then
  echo "ERROR: export dir exists and is non-empty: ${EXPORT_DIR}" >&2
  echo "Set OVERWRITE_EXPORT=1 after moving/removing the old export if you intentionally want to replace it." >&2
  exit 2
fi

llamafactory-cli export "${CONFIG}"

python - "${EXPORT_DIR}/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
if isinstance(config.get("text_config"), dict):
    config["text_config"]["rope_scaling"] = None
path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Patched vLLM inference config rope_scaling=null: {path}")
PY

echo "Exported merged KTO model: ${EXPORT_DIR}"
