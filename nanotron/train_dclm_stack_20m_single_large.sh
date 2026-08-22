#!/usr/bin/env bash
set -euo pipefail

# Train the single_large baseline described in docs/single_large_baseline.md.
# Optional overrides:
#   NPROC_PER_NODE=2  Number of local ranks (must match the config's dp=2)
#   MASTER_PORT=29501 Rendezvous port; use a distinct port per concurrent run
#   LOG_FILE=...      Destination for the combined stdout/stderr log

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
config_file="$script_dir/config_dclm_stack_20m_single_large_10b_seq4096_seed1000.yaml"
log_file="${LOG_FILE:-$script_dir/train_dclm_stack_20m_single_large.log}"
nproc_per_node="${NPROC_PER_NODE:-2}"
master_port="${MASTER_PORT:-29501}"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi
if [[ "$nproc_per_node" != "2" ]]; then
  echo "NPROC_PER_NODE must be 2 for this config (parallelism.dp=2). Edit the YAML before changing it." >&2
  exit 2
fi
if [[ ! -f "$config_file" ]]; then
  echo "Missing training config: $config_file" >&2
  exit 2
fi

source "$repo_dir/mot_lms/bin/activate"
mkdir -p "$script_dir/artifacts/checkpoints" "$script_dir/artifacts/benchmarks" "$(dirname -- "$log_file")"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Config: $config_file"
echo "Ranks:  $nproc_per_node"
echo "Port:   $master_port"
echo "Log:    $log_file"

cd "$script_dir"
torchrun --master_port="$master_port" --nproc_per_node="$nproc_per_node" run_train.py \
  --config-file "$config_file" \
  2>&1 | tee "$log_file"
