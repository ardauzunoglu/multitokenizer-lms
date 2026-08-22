# Training the `single_large` tokenizer baseline

This procedure trains a **single-tokenizer** Llama baseline on the same
DCLM+Stack routed corpus used by the multi-tokenizer run. It uses the
49,152-token `single_large` BPE located at:

```text
/scratch/dkhasha1/auzunog1/multitokenizer-lms/train_tokenizers/runs_v1/dclm-stack-50m50m/compression_eval_both_dclm_and_thestack/baselines/single_large
```

`single_large` has three times as many tokens as one expert and matches the
combined vocabulary capacity of the three 16,384-token disjoint endpoints.

Do not reuse `nanotron/tokenized_train`: it was encoded with the three expert
tokenizers, so its token IDs are not valid for `single_large`.

## 1. Enter the project and activate the environment

```bash
cd /scratch/dkhasha1/auzunog1/multitokenizer-lms
source mot_lms/bin/activate
```

## 2. Convert the tokenizer to a Hugging Face artifact

`single_large` contains the BPE model but only has `tokenizer.json`. The
conversion adds Hugging Face metadata without changing its vocabulary. Its
special-token IDs are `<unk>=0`, `<bos>=1`, `<eos>=2`, and `<pad>=3`.

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from train_tokenizers.convert_tokenizers import convert_one

convert_one(
    Path(
        "train_tokenizers/runs_v1/dclm-stack-50m50m/"
        "compression_eval_both_dclm_and_thestack/baselines/single_large"
    ),
    Path("nanotron/artifacts/tokenizers/single_large"),
    overwrite=True,
)
PY
```

Verify the conversion:

```bash
python - <<'PY'
from transformers import AutoTokenizer

path = "nanotron/artifacts/tokenizers/single_large"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
assert len(tokenizer) == 49152
assert (tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (0, 1, 2, 3)
print("Tokenizer verified:", len(tokenizer), tokenizer.special_tokens_map)
PY
```

## 3. Create a preprocessing registry

Create `nanotron/registry_single_large.yaml` with this content. The routed
corpus contains IDs 0, 1, and 2, so the preprocessing tool needs three entries.
All entries intentionally use the same `single_large` tokenizer; the trained
model below is a normal single-tokenizer model.

```yaml
multi_tokenizer:
  tokenizers:
    - id: 0
      name: source_00
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_large
      vocab_size: 49152
    - id: 1
      name: source_01
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_large
      vocab_size: 49152
    - id: 2
      name: source_02
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_large
      vocab_size: 49152
```

## 4. Retokenize the routed corpus

This reads the 101 files under `nanotron/routed_train`, appends EOS (`2`) to
each document, and writes `uint16` `.ds` files under
`nanotron/tokenized_single_large`. `uint16` is sufficient because 49,152 is
below 65,536.

The `--overwrite` option removes only the destination directory
`nanotron/tokenized_single_large`; omit it if that directory contains data you
intend to keep.

```bash
PYTHONPATH=nanotron/src python nanotron/tools/preprocess_multi_tokenizer_shards.py \
  --input-dir nanotron/routed_train \
  --registry nanotron/registry_single_large.yaml \
  --output-dir nanotron/tokenized_single_large \
  --sequence-length 4096 \
  --batch-size 512 \
  --tokens-per-output-shard 100000000 \
  --overwrite
```

Check that all partitions produced data:

```bash
find nanotron/tokenized_single_large -name '*.ds' -type f | wc -l
du -sh nanotron/tokenized_single_large
```

## 5. Create the single-tokenizer training config

Copy the current multi-tokenizer configuration:

```bash
cp nanotron/config_dclm_stack_20m_10b_seq4096_seed1000.yaml \
  nanotron/config_dclm_stack_20m_single_large_10b_seq4096_seed1000.yaml
```

Edit the copied configuration as follows.

1. Give the run unique outputs so it cannot overwrite the multi-tokenizer run:

   ```yaml
   general:
     run: dclm_stack_20m_single_large_10b_seq4096_seed1000
     benchmark_csv_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/benchmarks/dclm_stack_20m_single_large_10b_seq4096_seed1000.csv

   checkpoints:
     checkpoints_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/checkpoints/dclm_stack_20m_single_large_10b_seq4096_seed1000
   ```

2. Remove the complete top-level `multi_tokenizer:` block. Add this top-level
   `tokenizer:` block instead:

   ```yaml
   tokenizer:
     tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_large
     tokenizer_revision: null
     tokenizer_max_length: null
   ```

3. Replace the `data_stages[0].data.dataset` block with this standard
   single-tokenizer Nanoset dataset. The weights reproduce the current
   routed-corpus mixture.

   ```yaml
   dataset:
     dataset_folder:
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_large/000-source_00
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_large/001-source_01
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_large/002-source_02
     dataset_weights: [5464988301, 1237438227, 3710788176]
     tokenizer_name: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_large
     vocab_size: 49152
     token_size_in_bytes: 2
     return_positions: true
     skip_in_stream: false
     pad_samples_to_global_batch_size: false
     shuffle_files: true
   ```

4. Change the model vocabulary from 16,384 to 49,152. Keep the special-token
   IDs unchanged.

   ```yaml
   model:
     model_config:
       bos_token_id: 1
       eos_token_id: 2
       pad_token_id: 3
       vocab_size: 49152
   ```

5. Keep the token schedule unchanged: microbatch 32, accumulation 4, DP 2,
   sequence length 4096, and 9537 steps. This trains on 10,000,269,312 tokens.
   Keeping `optimizer.accumulate_grad_in_fp32: false` is also fine.

## 6. Validate the configuration

```bash
cd nanotron
PYTHONPATH=src python - <<'PY'
from nanotron.config import Config

config = Config.load_from_yaml("config_dclm_stack_20m_single_large_10b_seq4096_seed1000.yaml")
assert config.multi_tokenizer is None
assert config.model.model_config.vocab_size == 49152
assert config.tokenizer.tokenizer_name_or_path.endswith("artifacts/tokenizers/single_large")
print("Single-tokenizer configuration verified.")
PY
```

## 7. Start training

Run from `nanotron`, using two local processes to match `parallelism.dp: 2`.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=2 run_train.py \
  --config-file config_dclm_stack_20m_single_large_10b_seq4096_seed1000.yaml \
  2>&1 | tee train_dclm_stack_20m_single_large.log
```

The resolved configuration should report `vocab_size=49152`, a standard
`TokenizedBytes Dataloader`, and no `multi_tokenizer=` section.
