#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
source "$repo_dir/mot_lms/bin/activate"

# Keep EVAL_FRACTION and PARTITION_SALT identical to the pretraining run.
# Set HF_REPO_ID to publish this as the eval split of the same dataset repository.
token_budget="${TOKEN_BUDGET:-10000000}"
output_dir="${OUTPUT_DIR:-$script_dir/outputs/dclm-stack-eval-${token_budget}}"
args=(
  --partition eval
  --token-budget "$token_budget"
  --output-dir "$output_dir"
  --eval-fraction "${EVAL_FRACTION:-0.01}"
  --partition-salt "${PARTITION_SALT:-dclm-stack-v1}"
  --tokenizer-batch-size "${TOKENIZER_BATCH_SIZE:-256}"
  --shard-token-budget "${SHARD_TOKEN_BUDGET:-100000000}"
  --overwrite
)

if [[ -n "${HF_REPO_ID:-}" ]]; then
  args+=(--repo-id "$HF_REPO_ID" --hub-config "${HF_CONFIG:-default}" --hub-split eval)
fi

python "$script_dir/sample_mixture.py" "${args[@]}"
