#!/usr/bin/env bash
set -euo pipefail

# Export the latest TP=PP=1 single- or multi-tokenizer checkpoint as a Hugging
# Face model. The checkpoint type is auto-detected. Usage:
#   ./convert_latest_checkpoint_to_hf.sh [checkpoint_root] [output_dir]

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"

checkpoint_root="${1:-$script_dir/artifacts/checkpoints/dclm_stack_20m_single_large_10b_seq4096_seed1000}"
if [[ ! -f "$checkpoint_root/latest.txt" ]]; then
  echo "Could not find $checkpoint_root/latest.txt" >&2
  exit 2
fi

checkpoint_step="$(tr -d '[:space:]' < "$checkpoint_root/latest.txt")"
if [[ ! "$checkpoint_step" =~ ^[0-9]+$ ]] || [[ ! -d "$checkpoint_root/$checkpoint_step" ]]; then
  echo "Invalid latest checkpoint step in $checkpoint_root/latest.txt: $checkpoint_step" >&2
  exit 2
fi

checkpoint_path="$checkpoint_root/$checkpoint_step"
config_path="$checkpoint_path/config.yaml"
if [[ ! -f "$config_path" ]]; then
  echo "Missing checkpoint config: $config_path" >&2
  exit 2
fi
if grep -q '^multi_tokenizer:' "$config_path"; then
  checkpoint_kind="multi_tokenizer"
else
  checkpoint_kind="single_tokenizer"
fi
export_mode="${HF_EXPORT_MODE:-auto}"
case "$export_mode" in
  auto)
    output_kind="$checkpoint_kind"
    ;;
  multi)
    output_kind="multi_tokenizer"
    ;;
  single)
    output_kind="single_tokenizer"
    ;;
  endpoint)
    output_kind="endpoint_${HF_TOKENIZER_ID:-0}"
    ;;
  *)
    echo "HF_EXPORT_MODE must be one of: auto, single, multi, endpoint" >&2
    exit 2
    ;;
esac
output_dir="${2:-$script_dir/artifacts/hf/${checkpoint_root##*/}/step-$checkpoint_step/$output_kind}"

source "$repo_dir/mot_lms/bin/activate"
export PYTHONPATH="$script_dir/src:$script_dir${PYTHONPATH:+:$PYTHONPATH}"

overwrite_args=()
if [[ "${HF_OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Checkpoint: $checkpoint_path"
echo "Detected checkpoint type: $checkpoint_kind"
echo "Export mode: $export_mode"
echo "HF output: $output_dir"

python "$script_dir/tools/convert_multitokenizer_checkpoint_to_hf.py" \
  --checkpoint-path "$checkpoint_path" \
  --output-path "$output_dir" \
  --export-mode "$export_mode" \
  --tokenizer-id "${HF_TOKENIZER_ID:-0}" \
  "${overwrite_args[@]}"

echo "HF conversion complete: $output_dir"
