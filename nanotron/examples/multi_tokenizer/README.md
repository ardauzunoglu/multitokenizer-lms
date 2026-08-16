# Disjoint multi-tokenizer training

Route raw documents offline into JSONL records containing `text` and a stable
`tokenizer_id`, then preprocess them without mixing local ID spaces:

```bash
PYTHONPATH=src python tools/preprocess_multi_tokenizer_data.py \
  --input routed.jsonl --registry registry.yaml \
  --output-dir data/tokenized --sequence-length 2048
```

The registry and training-stage portion of `registry.yaml`/the Nanotron config
looks like this. Preprocessing computes fingerprints; copy the emitted values
into the final training config to pin them explicitly.

```yaml
multi_tokenizer:
  batching: step_homogeneous
  tie_word_embeddings: true
  mask_padded_vocab_logits: true
  schedule:
    strategy: weighted_deficit
    weights: [0.7, 0.3]
    weight_unit: tokens
    seed: 42
  tokenizers:
    - id: 0
      name: web
      tokenizer_name_or_path: ./tokenizers/web
      vocab_size: 16384
    - id: 1
      name: code
      tokenizer_name_or_path: ./tokenizers/code
      vocab_size: 16384

data_stages:
  - name: stable
    start_training_step: 1
    data:
      seed: 42
      num_loading_workers: 2
      dataset:
        type: multi_tokenizer_nanoset
        return_positions: false
        sources:
          - path: ./data/tokenized/000-web
            tokenizer_id: 0
            weight: 0.7
          - path: ./data/tokenized/001-code
            tokenizer_id: 1
            weight: 0.3
```

Each optimizer step, including all accumulation microbatches and data-parallel
replicas, selects one tokenizer. Checkpoint resume validates the ordered
fingerprinted registry and restores the weighted-deficit scheduler state.

Multi-tokenizer LightEval and non-Llama architectures are deliberately rejected
until they implement explicit endpoint selection.
