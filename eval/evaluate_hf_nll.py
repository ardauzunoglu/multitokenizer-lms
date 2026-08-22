#!/usr/bin/env python3
"""Evaluate single- and/or multi-tokenizer Hugging Face causal LMs on text.

The multi-tokenizer export produced by ``convert_latest_checkpoint_to_hf.sh``
does not have one canonical tokenizer.  This evaluator therefore reproduces
the training router: it assigns each document to the endpoint with the
shortest ``add_special_tokens=False`` encoding, breaking ties by endpoint ID.

NLL is measured independently for each model's tokenization.  ``nats_per_byte``
and ``bits_per_byte`` are the directly comparable metrics across the models.
Each document is evaluated as ``EOS + document + EOS``, matching the EOS
document separation used by this project's preprocessing scripts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_DATASET = "ardauzunoglu/dclm-stack-10b-p5p5"


@dataclass
class Metrics:
    documents: int = 0
    source_bytes: int = 0
    predicted_tokens: int = 0
    nll_sum: float = 0.0

    def add_loss(self, nll_sum: float, predicted_tokens: int) -> None:
        self.nll_sum += nll_sum
        self.predicted_tokens += predicted_tokens

    def as_dict(self) -> dict[str, float | int | None]:
        nats_per_token = self.nll_sum / self.predicted_tokens if self.predicted_tokens else None
        nats_per_byte = self.nll_sum / self.source_bytes if self.source_bytes else None
        return {
            "documents": self.documents,
            "source_bytes": self.source_bytes,
            "predicted_tokens": self.predicted_tokens,
            "nll_sum_nats": self.nll_sum,
            "nats_per_token": nats_per_token,
            "perplexity": math.exp(nats_per_token) if nats_per_token is not None and nats_per_token < 80 else None,
            "nats_per_byte": nats_per_byte,
            "bits_per_byte": nats_per_byte / math.log(2) if nats_per_byte is not None else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-model", type=Path, help="HF directory for a standard single-tokenizer LM")
    parser.add_argument("--single-tokenizer", type=Path, help="Tokenizer directory; defaults to --single-model")
    parser.add_argument("--multi-model", type=Path, help="All-endpoint HF export for the multi-tokenizer LM")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-documents", type=int, default=0, help="0 evaluates the entire split")
    parser.add_argument("--max-bytes", type=int, default=0, help="Stop before exceeding this UTF-8-byte budget; 0 disables it")
    parser.add_argument("--batch-size", type=int, default=1, help="Token chunks per forward pass (one endpoint per batch)")
    parser.add_argument("--block-size", type=int, default=0, help="Context length; 0 uses each model's max_position_embeddings")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON metrics file")
    args = parser.parse_args()
    if args.max_documents < 0 or args.max_bytes < 0 or args.batch_size <= 0 or args.block_size < 0:
        parser.error("--max-documents, --max-bytes, and --block-size must be non-negative; --batch-size must be positive")
    if args.single_model is None and args.multi_model is None:
        parser.error("Provide --single-model, --multi-model, or both")
    for model_path in (args.single_model, args.multi_model):
        if model_path is not None and not model_path.is_dir():
            parser.error(f"Model directory does not exist: {model_path}")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false")
    return device


def model_block_size(model, requested: int) -> int:
    limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if limit <= 1:
        limit = 4096
    if requested and requested > limit:
        raise ValueError(f"--block-size {requested} exceeds model context length {limit}")
    return requested or limit


def encode_document(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    token_ids = list(encoded["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        # Training concatenated ``document_tokens + [eos]``.  The leading EOS
        # supplies the same document-boundary context while allowing the NLL
        # for the first document token to be included.
        token_ids = [int(eos_id), *token_ids, int(eos_id)]
    elif tokenizer.bos_token_id is not None:
        token_ids.insert(0, int(tokenizer.bos_token_id))
    return token_ids


def chunk_tokens(token_ids: list[int], block_size: int) -> Iterable[list[int]]:
    """Yield overlapping chunks so every token after the first is scored once."""
    if len(token_ids) < 2:
        return
    start = 0
    while start < len(token_ids) - 1:
        end = min(start + block_size, len(token_ids))
        yield token_ids[start:end]
        if end == len(token_ids):
            break
        start = end - 1


class ChunkBatcher:
    def __init__(self, model, device: torch.device, batch_size: int, pad_token_id: int, tokenizer_id: int | None = None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer_id = tokenizer_id
        self.pending: list[list[int]] = []
        self.metrics = Metrics()

    def add(self, token_ids: list[int], block_size: int) -> None:
        for chunk in chunk_tokens(token_ids, block_size):
            self.pending.append(chunk)
            if len(self.pending) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        max_length = max(len(chunk) for chunk in self.pending)
        input_ids = torch.full((len(self.pending), max_length), self.pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(self.pending), max_length), dtype=torch.long, device=self.device)
        for row, chunk in enumerate(self.pending):
            input_ids[row, : len(chunk)] = torch.tensor(chunk, dtype=torch.long, device=self.device)
            attention_mask[row, : len(chunk)] = 1

        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if self.tokenizer_id is not None:
            model_kwargs["tokenizer_id"] = self.tokenizer_id
        with torch.inference_mode():
            logits = self.model(**model_kwargs).logits
            labels = input_ids[:, 1:].masked_fill(attention_mask[:, 1:] == 0, -100)
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        self.metrics.add_loss(float(loss.item()), int((labels != -100).sum().item()))
        self.pending.clear()


def load_model(path: Path, device: torch.device, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()
    return model


def load_multi_tokenizers(model_dir: Path) -> list[PreTrainedTokenizerBase]:
    metadata_path = model_dir / "conversion_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Multi-tokenizer export is missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    specs = metadata.get("tokenizers", [])
    if not specs:
        raise ValueError(f"{metadata_path} has no tokenizer endpoints; use an all-endpoint multi export")
    specs = sorted(specs, key=lambda spec: spec["id"])
    expected_ids = list(range(len(specs)))
    if [spec["id"] for spec in specs] != expected_ids:
        raise ValueError(f"Tokenizer endpoint IDs must be contiguous {expected_ids}")
    return [AutoTokenizer.from_pretrained(model_dir / spec["path"], trust_remote_code=False) for spec in specs]


def route_shortest(tokenizers: list[PreTrainedTokenizerBase], text: str) -> tuple[int, list[int]]:
    # Keep routing identical to route_local_multi_tokenizer_data.py: special
    # tokens do not participate in the length comparison.
    raw_encodings = [
        tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        for tokenizer in tokenizers
    ]
    endpoint_id = min(range(len(raw_encodings)), key=lambda index: len(raw_encodings[index]))
    return endpoint_id, encode_document(tokenizers[endpoint_id], text)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    single_batcher = None
    single_tokenizer = None
    single_block_size = None
    if args.single_model is not None:
        print(f"Loading single-tokenizer model from {args.single_model}", flush=True)
        single_model = load_model(args.single_model, device, trust_remote_code=True)
        single_tokenizer = AutoTokenizer.from_pretrained(args.single_tokenizer or args.single_model, trust_remote_code=False)
        single_pad_id = single_tokenizer.pad_token_id
        if single_pad_id is None:
            single_pad_id = single_tokenizer.eos_token_id if single_tokenizer.eos_token_id is not None else 0
        single_batcher = ChunkBatcher(single_model, device, args.batch_size, int(single_pad_id))
        single_block_size = model_block_size(single_model, args.block_size)

    multi_batchers: list[ChunkBatcher] | None = None
    multi_tokenizers = None
    multi_block_size = None
    if args.multi_model is not None:
        print(f"Loading multi-tokenizer model from {args.multi_model}", flush=True)
        multi_model = load_model(args.multi_model, device, trust_remote_code=True)
        multi_tokenizers = load_multi_tokenizers(args.multi_model)
        multi_batchers = []
        for endpoint_id, tokenizer in enumerate(multi_tokenizers):
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            multi_batchers.append(ChunkBatcher(multi_model, device, args.batch_size, int(pad_id), endpoint_id))
        multi_block_size = model_block_size(multi_model, args.block_size)

    block_description = ", ".join(
        f"{name} block={size}" for name, size in (("single", single_block_size), ("multi", multi_block_size)) if size is not None
    )
    print(f"Evaluating {args.dataset!r} split {args.split!r} on {device}; {block_description}", flush=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    endpoint_documents: Counter[int] = Counter()
    endpoint_bytes: Counter[int] = Counter()
    processed_documents = 0
    processed_bytes = 0
    skipped_documents = 0

    for record in dataset:
        text = record.get(args.text_column)
        if not isinstance(text, str) or not text:
            skipped_documents += 1
            continue
        byte_count = len(text.encode("utf-8"))
        if args.max_documents and processed_documents >= args.max_documents:
            break
        if args.max_bytes and processed_bytes + byte_count > args.max_bytes:
            break

        if single_batcher is not None and single_tokenizer is not None and single_block_size is not None:
            single_batcher.add(encode_document(single_tokenizer, text), single_block_size)
            single_batcher.metrics.documents += 1
            single_batcher.metrics.source_bytes += byte_count
        if multi_batchers is not None and multi_tokenizers is not None and multi_block_size is not None:
            endpoint_id, multi_token_ids = route_shortest(multi_tokenizers, text)
            multi_batchers[endpoint_id].add(multi_token_ids, multi_block_size)
            endpoint_documents[endpoint_id] += 1
            endpoint_bytes[endpoint_id] += byte_count
        processed_documents += 1
        processed_bytes += byte_count
        if processed_documents % 1_000 == 0:
            print(f"Processed {processed_documents:,} documents ({processed_bytes:,} UTF-8 bytes)", flush=True)

    if single_batcher is not None:
        single_batcher.flush()

    multi_total = None
    endpoint_results = None
    if multi_batchers is not None:
        for batcher in multi_batchers:
            batcher.flush()
        multi_total = Metrics(documents=processed_documents, source_bytes=processed_bytes)
        endpoint_results = {}
        for endpoint_id, batcher in enumerate(multi_batchers):
            batcher.metrics.documents = endpoint_documents[endpoint_id]
            batcher.metrics.source_bytes = endpoint_bytes[endpoint_id]
            multi_total.add_loss(batcher.metrics.nll_sum, batcher.metrics.predicted_tokens)
            endpoint_results[str(endpoint_id)] = batcher.metrics.as_dict()

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "text_column": args.text_column,
        "streaming": args.streaming,
        "documents_evaluated": processed_documents,
        "documents_skipped_empty_or_missing_text": skipped_documents,
        "source_bytes": processed_bytes,
        "device": str(device),
        "block_size": {name: size for name, size in (("single", single_block_size), ("multi", multi_block_size)) if size is not None},
    }
    if multi_batchers is not None:
        result["routing"] = {
            "policy": "shortest add_special_tokens=False encoding; ties use the lowest endpoint ID",
            "documents_by_endpoint": {str(key): endpoint_documents[key] for key in range(len(multi_batchers))},
        }
    if single_batcher is not None and args.single_model is not None:
        result["single_tokenizer_lm"] = {"model": str(args.single_model.resolve()), **single_batcher.metrics.as_dict()}
    if multi_total is not None and endpoint_results is not None and args.multi_model is not None:
        result["multi_tokenizer_lm"] = {
            "model": str(args.multi_model.resolve()),
            **multi_total.as_dict(),
            "endpoints": endpoint_results,
        }
    if single_batcher is not None and multi_total is not None:
        result["comparison_note"] = "Compare nats_per_byte or bits_per_byte across models. nats_per_token depends on each model's tokenizer."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Wrote metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
