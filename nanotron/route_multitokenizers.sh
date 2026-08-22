#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
source "$repo_dir/mot_lms/bin/activate"

ROUTER_OVERWRITE=1 
ROUTER_WORKERS=16 
ROUTER_BATCH_SIZE=1024

args=(
  --input-dir "$repo_dir/curate_corpus/outputs/dclm-stack-train-10000000000"
  --registry "$script_dir/registry_v1.yaml"
  --output-dir "$script_dir/routed_train"
  --batch-size "${ROUTER_BATCH_SIZE:-1024}"
  --workers "${ROUTER_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
)
if [[ "${ROUTER_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

PYTHONPATH="$script_dir/src" python "$script_dir/tools/route_local_multi_tokenizer_data.py" "${args[@]}"
