#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

source "$repo_dir/mot_lms/bin/activate"

# This config retains dp=2 from the reference run. Override only when the
# parallelism section of the YAML is changed to match a different allocation.
nproc_per_node="${NPROC_PER_NODE:-2}"
if [[ "$nproc_per_node" != "2" ]]; then
  echo "NPROC_PER_NODE must be 2 for the checked-in config (dp=2); edit the YAML to change parallelism." >&2
  exit 2
fi

mkdir -p "$script_dir/artifacts/checkpoints" "$script_dir/artifacts/benchmarks"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$script_dir"
torchrun --nproc_per_node="$nproc_per_node" run_train.py \
  --config-file "$script_dir/config_dclm_stack_20m_10b_seq4096_seed1000.yaml"
