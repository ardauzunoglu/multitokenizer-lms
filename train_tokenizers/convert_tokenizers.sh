#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source ../mot_lms/bin/activate

python "$script_dir/convert_tokenizers.py" \
  --input-dir "$script_dir/runs_v1/dclm-stack-50m50m/tokenizers" \
  --output-dir "$script_dir/runs_v1/dclm-stack-50m50m/hf_tokenizers"
