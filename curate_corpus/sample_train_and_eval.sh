#!/usr/bin/env bash
set -euo pipefail

# Samples a disjoint train/eval pair and pushes both splits to the repository
# below. Override token budgets or partition settings when launching.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

partition_salt="${PARTITION_SALT:-dclm-stack-v1}"
eval_fraction="${EVAL_FRACTION:-0.01}"
train_budget="${TRAIN_TOKEN_BUDGET:-10000000000}"
eval_budget="${EVAL_TOKEN_BUDGET:-10000000}"
hub_repo_id="ardauzunoglu/dclm-stack-10b-p5p5"

common_env=(
  "PARTITION_SALT=$partition_salt"
  "EVAL_FRACTION=$eval_fraction"
)

env "${common_env[@]}" \
  TOKEN_BUDGET="$train_budget" \
  OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$script_dir/outputs/dclm-stack-train-${train_budget}}" \
  HF_REPO_ID="$hub_repo_id" \
  HF_CONFIG="${HF_CONFIG:-default}" \
  bash "$script_dir/sample_pretraining.sh"

env "${common_env[@]}" \
  TOKEN_BUDGET="$eval_budget" \
  OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$script_dir/outputs/dclm-stack-eval-${eval_budget}}" \
  HF_REPO_ID="$hub_repo_id" \
  HF_CONFIG="${HF_CONFIG:-default}" \
  bash "$script_dir/sample_evaluation.sh"
