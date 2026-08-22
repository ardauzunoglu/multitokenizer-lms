#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
source "$repo_dir/mot_lms/bin/activate"

PYTHONPATH="$script_dir/src" python "$script_dir/tools/preprocess_multi_tokenizer_shards.py" \
  --input-dir "$script_dir/routed_train" \
  --registry "$script_dir/registry_v1.yaml" \
  --output-dir "$script_dir/tokenized_train" \
  --sequence-length 2048 \
  --batch-size "${PREPROCESS_BATCH_SIZE:-512}" \
  --tokens-per-output-shard "${TOKENS_PER_OUTPUT_SHARD:-100000000}" \
  --write-buffer-tokens "${WRITE_BUFFER_TOKENS:-1000000}" \
  --overwrite
