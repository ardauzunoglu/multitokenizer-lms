#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
source "$repo_dir/mot_lms/bin/activate"

# Override these at launch, e.g. TOKEN_BUDGET=10000000000 HF_REPO_ID=org/dclm-stack bash sample_pretraining.sh
token_budget="${TOKEN_BUDGET:-100000000}"
output_dir="${OUTPUT_DIR:-$script_dir/outputs/dclm-stack-train-${token_budget}}"
args=(
  --partition train
  --token-budget "$token_budget"
  --output-dir "$output_dir"
  --eval-fraction "${EVAL_FRACTION:-0.01}"
  --partition-salt "${PARTITION_SALT:-dclm-stack-v1}"
  --tokenizer-batch-size "${TOKENIZER_BATCH_SIZE:-256}"
  --shard-token-budget "${SHARD_TOKEN_BUDGET:-100000000}"
  --overwrite
)

if [[ -n "${HF_REPO_ID:-}" ]]; then
  args+=(--repo-id "$HF_REPO_ID" --hub-config "${HF_CONFIG:-default}" --hub-split train)
fi

python "$script_dir/sample_mixture.py" "${args[@]}"
