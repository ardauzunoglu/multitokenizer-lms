#!/usr/bin/env bash
set -euo pipefail

# Evaluate a single-tokenizer HF LM, a multi-tokenizer HF export, or both on
# the DCLM+Stack eval split.
#
# Usage:
#   ./evaluate_dclm_stack_nll.sh single /path/to/single-hf [output.json]
#   ./evaluate_dclm_stack_nll.sh multi  /path/to/multi-hf  [output.json]
#   ./evaluate_dclm_stack_nll.sh both /path/to/single-hf /path/to/multi-hf [output.json]
#
# The prior two-positional-argument form remains supported and means `both`.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
nanotron_dir="$repo_dir/nanotron"

mode="both"
single_model=""
multi_model=""
case "${1:-}" in
  single)
    [[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: $0 single /path/to/single-hf [output.json]" >&2; exit 2; }
    mode="single"
    single_model="$2"
    output_path="${3:-$nanotron_dir/artifacts/evaluations/dclm-stack-10b-p5p5-single-nll.json}"
    ;;
  multi)
    [[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: $0 multi /path/to/multi-hf [output.json]" >&2; exit 2; }
    mode="multi"
    multi_model="$2"
    output_path="${3:-$nanotron_dir/artifacts/evaluations/dclm-stack-10b-p5p5-multi-nll.json}"
    ;;
  both)
    [[ $# -ge 3 && $# -le 4 ]] || { echo "Usage: $0 both /path/to/single-hf /path/to/multi-hf [output.json]" >&2; exit 2; }
    single_model="$2"
    multi_model="$3"
    output_path="${4:-$nanotron_dir/artifacts/evaluations/dclm-stack-10b-p5p5-eval-nll.json}"
    ;;
  *)
    [[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: $0 [single|multi|both] ..." >&2; exit 2; }
    single_model="$1"
    multi_model="$2"
    output_path="${3:-$nanotron_dir/artifacts/evaluations/dclm-stack-10b-p5p5-eval-nll.json}"
    ;;
esac

single_tokenizer_args=()
if [[ -n "${SINGLE_TOKENIZER:-}" ]]; then
  single_tokenizer_args=(--single-tokenizer "$SINGLE_TOKENIZER")
fi

source "$repo_dir/mot_lms/bin/activate"
export PYTHONPATH="$nanotron_dir/src:$nanotron_dir${PYTHONPATH:+:$PYTHONPATH}"

model_args=()
if [[ -n "$single_model" ]]; then
  model_args+=(--single-model "$single_model" "${single_tokenizer_args[@]}")
fi
if [[ -n "$multi_model" ]]; then
  model_args+=(--multi-model "$multi_model")
fi

python "$script_dir/evaluate_hf_nll.py" \
  "${model_args[@]}" \
  --dataset "${DATASET_ID:-ardauzunoglu/dclm-stack-10b-p5p5}" \
  --split "${DATASET_SPLIT:-eval}" \
  --text-column "${TEXT_COLUMN:-text}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --block-size "${BLOCK_SIZE:-0}" \
  --max-documents "${MAX_DOCUMENTS:-0}" \
  --max-bytes "${MAX_BYTES:-0}" \
  --device "${DEVICE:-auto}" \
  --output "$output_path"
