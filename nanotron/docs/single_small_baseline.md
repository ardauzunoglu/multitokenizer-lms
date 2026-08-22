# Training the `single_small` tokenizer baseline

This procedure trains a **single-tokenizer** Llama baseline on the same
DCLM+Stack routed corpus used by the multi-tokenizer run.  It uses the
16,384-token `single_small` BPE located at:

```text
/scratch/dkhasha1/auzunog1/multitokenizer-lms/train_tokenizers/runs_v1/dclm-stack-50m50m/compression_eval_both_dclm_and_thestack/baselines/single_small
```

Do not reuse `nanotron/tokenized_train`: it was encoded with the three expert
tokenizers, so its token IDs are not valid for `single_small`.

## 1. Enter the project and activate the environment

```bash
cd /scratch/dkhasha1/auzunog1/multitokenizer-lms
source mot_lms/bin/activate
```

## 2. Convert the tokenizer to a Hugging Face artifact

`single_small` contains the correct BPE model but only has `tokenizer.json`.
The conversion writes Hugging Face metadata while preserving the vocabulary and
the existing special-token IDs: `<unk>=0`, `<bos>=1`, `<eos>=2`, and `<pad>=3`.

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from train_tokenizers.convert_tokenizers import convert_one

convert_one(
    Path(
        "train_tokenizers/runs_v1/dclm-stack-50m50m/"
        "compression_eval_both_dclm_and_thestack/baselines/single_small"
    ),
    Path("nanotron/artifacts/tokenizers/single_small"),
    overwrite=True,
)
PY
```

Verify the conversion before continuing:

```bash
python - <<'PY'
from transformers import AutoTokenizer

path = "nanotron/artifacts/tokenizers/single_small"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
assert len(tokenizer) == 16384
assert (tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (0, 1, 2, 3)
print("Tokenizer verified:", len(tokenizer), tokenizer.special_tokens_map)
PY
```

## 3. Create a preprocessing registry

Create `nanotron/registry_single_small.yaml` with the following content.
There are three entries only because the existing routed corpus carries route
IDs 0, 1, and 2.  Each entry deliberately points to the same single tokenizer.
The resulting data is still a single-tokenizer baseline.

```yaml
multi_tokenizer:
  tokenizers:
    - id: 0
      name: source_00
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_small
      vocab_size: 16384
    - id: 1
      name: source_01
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_small
      vocab_size: 16384
    - id: 2
      name: source_02
      tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_small
      vocab_size: 16384
```

## 4. Retokenize the routed corpus

This reads the 101 files under `nanotron/routed_train` and writes new
`uint16` `.ds` files under `nanotron/tokenized_single_small`.  It appends the
tokenizer's EOS token (`2`) after every document.

The `--overwrite` option only removes the destination directory
`nanotron/tokenized_single_small`; omit it if that directory already contains
data you intend to keep.

```bash
PYTHONPATH=nanotron/src python nanotron/tools/preprocess_multi_tokenizer_shards.py \
  --input-dir nanotron/routed_train \
  --registry nanotron/registry_single_small.yaml \
  --output-dir nanotron/tokenized_single_small \
  --sequence-length 4096 \
  --batch-size 512 \
  --tokens-per-output-shard 100000000 \
  --overwrite
```

Check that all three routed partitions produced data:

```bash
find nanotron/tokenized_single_small -name '*.ds' -type f | wc -l
du -sh nanotron/tokenized_single_small
```

## 5. Create the single-tokenizer training config

Copy the current multi-tokenizer configuration:

```bash
cp nanotron/config_dclm_stack_20m_10b_seq4096_seed1000.yaml \
  nanotron/config_dclm_stack_20m_single_small_10b_seq4096_seed1000.yaml
```

Edit the copied file as follows.

1. Change the run name and output locations so that this baseline cannot
   overwrite multi-tokenizer logs, checkpoints, or benchmark rows.

   ```yaml
   general:
     run: dclm_stack_20m_single_small_10b_seq4096_seed1000
     benchmark_csv_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/benchmarks/dclm_stack_20m_single_small_10b_seq4096_seed1000.csv

   checkpoints:
     checkpoints_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/checkpoints/dclm_stack_20m_single_small_10b_seq4096_seed1000
   ```

2. Remove the complete top-level `multi_tokenizer:` block.  Add this top-level
   `tokenizer:` block instead:

   ```yaml
   tokenizer:
     tokenizer_name_or_path: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_small
     tokenizer_revision: null
     tokenizer_max_length: null
   ```

3. Replace the `data_stages[0].data.dataset` block with this ordinary
   single-tokenizer Nanoset dataset.  The three weights reproduce the current
   routed-corpus mixture.

   ```yaml
   dataset:
     dataset_folder:
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_small/000-source_00
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_small/001-source_01
       - /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/tokenized_single_small/002-source_02
     dataset_weights: [5464988301, 1237438227, 3710788176]
     tokenizer_name: /scratch/dkhasha1/auzunog1/multitokenizer-lms/nanotron/artifacts/tokenizers/single_small
     vocab_size: 16384
     token_size_in_bytes: 2
     return_positions: true
     skip_in_stream: false
     pad_samples_to_global_batch_size: false
     shuffle_files: true
   ```

4. Keep `model.model_config.vocab_size: 16384`.  Keep the existing token
   settings unchanged: microbatch 32, accumulation 4, DP 2, sequence length
   4096, and 9537 steps.  This is still a 10B-token run:

   ```text
   32 microbatch × 4 accumulation × 2 DP × 4096 sequence length × 9537 steps
   = 10,000,269,312 training tokens
   ```

5. Keeping `optimizer.accumulate_grad_in_fp32: false` is fine.  The
   single-tokenizer model has no dynamically-unused endpoint banks, so it does
   not require the multi-tokenizer DDP workaround.

## 6. Validate the configuration

```bash
cd nanotron
PYTHONPATH=src python - <<'PY'
from nanotron.config import Config

config = Config.load_from_yaml("config_dclm_stack_20m_single_small_10b_seq4096_seed1000.yaml")
assert config.multi_tokenizer is None
assert config.model.model_config.vocab_size == 16384
assert config.tokenizer.tokenizer_name_or_path.endswith("artifacts/tokenizers/single_small")
print("Single-tokenizer configuration verified.")
PY
```

## 7. Start training

Run this from `nanotron`.  It uses two local processes to match `parallelism.dp: 2`.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=2 run_train.py \
  --config-file config_dclm_stack_20m_single_small_10b_seq4096_seed1000.yaml \
  2>&1 | tee train_dclm_stack_20m_single_small.log
```

The first successful lines should report a standard `TokenizedBytes Dataloader`,
`vocab_size=16384`, and no `multi_tokenizer=` section in the resolved config.
