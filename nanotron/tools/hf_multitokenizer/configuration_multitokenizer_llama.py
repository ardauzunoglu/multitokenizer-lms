"""Hugging Face configuration for a Llama model with disjoint tokenizer banks."""

from transformers import LlamaConfig


class MultiTokenizerLlamaConfig(LlamaConfig):
    model_type = "multi_tokenizer_llama"

    def __init__(self, tokenizer_registry=None, **kwargs):
        self.tokenizer_registry = tokenizer_registry or []
        super().__init__(**kwargs)

