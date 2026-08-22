# DCLM + The Stack corpus curation

`sample_mixture.py` streams DCLM and The Stack, applies per-source token
quotas, and writes a JSONL dataset with `text`, `source`, `document_id`, and
`token_count`. It uses a content hash to put every document in exactly one
partition: `train` or `eval`. Use identical `--partition-salt` and
`--eval-fraction` values for both runs; changing either invalidates the
disjointness guarantee.

Run the pretraining mixture, then the held-out evaluation set:

```bash
TOKEN_BUDGET=10000000000 HF_REPO_ID=your-org/dclm-stack \
  bash sample_pretraining.sh

TOKEN_BUDGET=100000000 HF_REPO_ID=your-org/dclm-stack \
  bash sample_evaluation.sh
```

Or run both, with the shared partition settings wired together:

```bash
TRAIN_TOKEN_BUDGET=10000000000 EVAL_TOKEN_BUDGET=100000000 \
  HF_REPO_ID=your-org/dclm-stack bash sample_train_and_eval.sh
```

The first command pushes the Hub `train` split and the second pushes the `eval`
split. Authenticate beforehand with `hf auth login`. Without `HF_REPO_ID`, the
scripts only write the local JSONL and metadata files under `outputs/`.

Tokenization is batched (`TOKENIZER_BATCH_SIZE=256` by default), and output is
written as 100M-token JSONL shards (`SHARD_TOKEN_BUDGET=100000000` by default).
Increase the tokenizer batch size until host RAM becomes a constraint; use a
smaller shard budget when you want more completed upload-sized files.

The default mixture weights are 50/50 by token. To change the mixture or set a
different held-out fraction, call the Python command directly:

```bash
python sample_mixture.py --partition train --token-budget 100000000 \
  --source mlfoundations/dclm-baseline-1.0-parquet --source-weight 3 \
  --source bigcode/the-stack --source-weight 1 \
  --eval-fraction 0.01 --partition-salt dclm-stack-v1 \
  --output-dir outputs/train-100m
```

For a public repository, do not pass `--private`; add it to the direct command
or extend the shell scripts if a private dataset is desired.
