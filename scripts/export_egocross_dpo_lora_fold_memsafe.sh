#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/export_egocross_dpo_lora_fold_memsafe.sh <A|B|C> <fold_id>" >&2
  echo "Example: bash scripts/export_egocross_dpo_lora_fold_memsafe.sh A 0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

candidate="$1"
fold="$2"

case "${candidate}" in
  A) suffix="lr1e5_beta003_ftx005_ep1" ;;
  B) suffix="lr5e6_beta003_ftx005_ep1" ;;
  C) suffix="lr1e5_beta005_ftx005_ep1" ;;
  *)
    echo "Unknown candidate '${candidate}'. Use A, B, or C." >&2
    exit 2
    ;;
esac

train_config="configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_fold${fold}_${suffix}.yaml"
if [[ ! -f "${train_config}" ]]; then
  echo "Missing train config: ${train_config}" >&2
  exit 2
fi

adapter_dir="$(awk -F': ' '/^output_dir:/ {print $2; exit}' "${train_config}")"
if [[ -z "${adapter_dir}" ]]; then
  echo "No output_dir found in ${train_config}" >&2
  exit 2
fi
if [[ ! -d "${adapter_dir}" ]]; then
  echo "Adapter directory does not exist: ${adapter_dir}" >&2
  exit 2
fi

export_dir="${adapter_dir}_merged"
mkdir -p logs/egocross_export_configs
export_config="logs/egocross_export_configs/export_${candidate}_fold${fold}_${suffix}_memsafe.yaml"

cat > "${export_config}" <<EOF
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters.

### model
model_name_or_path: saves/egocross/qwen3vl4b/grpo_v3_train_rope_default
adapter_name_or_path: ${adapter_dir}
template: qwen3_vl
trust_remote_code: true

### export
export_dir: ${export_dir}
export_size: 5
export_device: cpu
export_legacy_format: false
EOF

echo "Export config: ${export_config}"
echo "Adapter: ${adapter_dir}"
echo "Merged export dir: ${export_dir}"
llamafactory-cli export "${export_config}"
