# Multi-tokenizer language models

Experiments and a Nanotron implementation for training a shared Transformer
backbone with completely disjoint tokenizer-specific embeddings and language
model heads.

## Contents

- `create_domains.py`: discovers/routs corpus domains and builds tokenizer assignments.
- `eval.py`: evaluates tokenizer compression and held-out statistics.
- `nanotron_implementation_plan.md`: design, compatibility contract, and test plan.
- `nanotron/`: the Nanotron training implementation, preprocessing tool, and tests.

Generated experiment outputs under `runs_v1/` are intentionally excluded from
version control.

See `nanotron/examples/multi_tokenizer/README.md` for preprocessing and training
configuration instructions.
