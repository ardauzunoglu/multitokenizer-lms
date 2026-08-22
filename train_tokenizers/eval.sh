#!/usr/bin/env bash
set -euo pipefail

run_dir="/scratch/dkhasha1/auzunog1/multitokenizer-lms/runs/dclm-stack-50m50m"
eval_bytes="${1:-50000000}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$script_dir/eval.py" \
  --run-dir "$run_dir" \
  --eval-bytes "$eval_bytes" \
  --bootstrap 500 \
  --output-dir "$run_dir/compression_eval_both_dclm_and_thestack" \
  --overwrite
