#!/usr/bin/env python3
"""Sample a reproducible DCLM + The Stack mixture and optionally publish it.

The train and evaluation commands must use the same ``--partition-salt`` and
``--eval-fraction``. Every source document is assigned to exactly one split by
hashing its source label and full text before any token-budget truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("curate_corpus")
DEFAULT_SOURCES = [
    "mlfoundations/dclm-baseline-1.0-parquet",
    "bigcode/the-stack",
]


@dataclass(frozen=True)
class Source:
    dataset: str
    config: str | None
    split: str
    text_field: str

    @property
    def label(self) -> str:
        return self.dataset if self.config is None else f"{self.dataset}::{self.config}"


def parse_source(value: str) -> Source:
    """Parse DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]]."""
    parts = value.split("::")
    if not 1 <= len(parts) <= 4 or not parts[0]:
        raise ValueError("Sources must be DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]]")
    parts += [""] * (4 - len(parts))
    dataset, config, split, text_field = parts
    return Source(dataset, config or None, split or "train", text_field or "text")


def document_id(source: Source, text: str, salt: str) -> str:
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source.label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def is_in_partition(doc_id: str, partition: str, eval_fraction: float) -> bool:
    # The first 64 hash bits are enough for a stable, uniform partition rule.
    value = int(doc_id[:16], 16) / 2**64
    return value < eval_fraction if partition == "eval" else value >= eval_fraction


def load_stream(source: Source, seed: int, shuffle_buffer: int):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": source.dataset, "split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    LOGGER.info("Opening %s[%s]", source.label, source.split)
    stream = load_dataset(**kwargs)
    if shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return iter(stream)


def get_text(row: dict[str, Any], source: Source) -> str | None:
    value = row.get(source.text_field)
    # The Stack exposes content on common configurations; DCLM exposes text.
    if not isinstance(value, str) and source.text_field == "text":
        value = row.get("content")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def next_partition_batch(state: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, str]]:
    """Read a batch of eligible documents before calling the tokenizer once."""
    batch: list[tuple[str, str]] = []
    while len(batch) < args.tokenizer_batch_size:
        try:
            row = next(state["rows"])
        except StopIteration:
            state["exhausted"] = True
            break
        state["rows_seen"] += 1
        text = get_text(row, state["source"])
        if text is None or len(text.encode("utf-8")) < args.min_document_bytes:
            continue
        original_id = document_id(state["source"], text, args.partition_salt)
        if is_in_partition(original_id, args.partition, args.eval_fraction):
            batch.append((text, original_id))
    return batch


def batched_token_counts(tokenizer: Any, texts: list[str]) -> list[int]:
    """Tokenize a list in one fast-tokenizer call instead of one call per document."""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def truncate_to_budget(tokenizer: Any, text: str, budget: int) -> tuple[str, int]:
    """Keep a deterministic prefix that fits the remaining token budget."""
    low, high, best_text, best_count = 0, len(text), "", 0
    while low <= high:
        midpoint = (low + high) // 2
        candidate = text[:midpoint]
        count = token_count(tokenizer, candidate)
        if count <= budget:
            best_text, best_count = candidate, count
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_text.strip(), best_count


def source_quotas(total: int, weights: list[float]) -> list[int]:
    denominator = sum(weights)
    quotas = [int(total * weight / denominator) for weight in weights]
    quotas[-1] += total - sum(quotas)
    return quotas


def sample(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    from transformers import AutoTokenizer

    sources = [parse_source(value) for value in args.source]
    quotas = source_quotas(args.token_budget, args.source_weight)
    tokenizer = AutoTokenizer.from_pretrained(args.budget_tokenizer, use_fast=True)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    states = [
        {
            "source": source,
            "rows": load_stream(source, args.seed + index, args.shuffle_buffer),
            "tokens": 0,
            "documents": 0,
            "bytes": 0,
            "rows_seen": 0,
            "exhausted": False,
        }
        for index, source in enumerate(sources)
    ]
    active = list(range(len(states)))
    written_tokens = 0
    written_documents = 0
    shards: list[dict[str, Any]] = []
    shard_index = 0
    shard_tokens = 0
    shard_documents = 0
    output = None

    def open_shard() -> None:
        nonlocal output, shard_index, shard_tokens, shard_documents
        path = args.output_dir / f"data-{shard_index:05d}.jsonl"
        output = path.open("w", encoding="utf-8")
        shards.append({"path": path.name, "tokens": 0, "documents": 0})
        shard_tokens = 0
        shard_documents = 0

    def close_shard() -> None:
        nonlocal output
        if output is not None:
            output.close()
            output = None

    try:
        open_shard()
        while active:
            for index in active.copy():
                state = states[index]
                remaining = quotas[index] - state["tokens"]
                if remaining <= 0:
                    active.remove(index)
                    continue
                batch = next_partition_batch(state, args)
                if not batch:
                    if state["exhausted"]:
                        LOGGER.warning("%s exhausted at %d/%d tokens after scanning %d rows", state["source"].label, state["tokens"], quotas[index], state["rows_seen"])
                        active.remove(index)
                    continue
                texts = [text for text, _ in batch]
                for (text, original_id), count in zip(batch, batched_token_counts(tokenizer, texts)):
                    remaining = quotas[index] - state["tokens"]
                    if remaining <= 0:
                        break
                    if count > remaining:
                        text, count = truncate_to_budget(tokenizer, text, remaining)
                    if count == 0 or len(text.encode("utf-8")) < args.min_document_bytes:
                        continue
                    if shard_tokens and shard_tokens + count > args.shard_token_budget:
                        close_shard()
                        shard_index += 1
                        open_shard()
                    record = {
                        "text": text,
                        "source": state["source"].label,
                        "document_id": original_id,
                        "token_count": count,
                        "partition": args.partition,
                    }
                    assert output is not None
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    shard_tokens += count
                    shard_documents += 1
                    shards[-1]["tokens"] = shard_tokens
                    shards[-1]["documents"] = shard_documents
                    state["tokens"] += count
                    state["documents"] += 1
                    state["bytes"] += len(text.encode("utf-8"))
                    written_tokens += count
                    written_documents += 1
                    if written_documents % 10_000 == 0:
                        LOGGER.info("Wrote %d documents / %d tokens across %d shard(s)", written_documents, written_tokens, len(shards))
    finally:
        close_shard()

    metadata = {
        "format_version": 1,
        "partition": args.partition,
        "partition_salt": args.partition_salt,
        "eval_fraction": args.eval_fraction,
        "budget_tokenizer": args.budget_tokenizer,
        "requested_token_budget": args.token_budget,
        "written_tokens": written_tokens,
        "written_documents": written_documents,
        "shard_token_budget": args.shard_token_budget,
        "shards": shards,
        "sources": [
            {**asdict(state["source"]), "token_quota": quota, "written_tokens": state["tokens"],
             "written_documents": state["documents"], "written_bytes": state["bytes"], "rows_seen": state["rows_seen"]}
            for state, quota in zip(states, quotas)
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return args.output_dir, metadata


def push(output_dir: Path, args: argparse.Namespace) -> None:
    from datasets import load_dataset

    data_files = sorted(str(path) for path in output_dir.glob("data-*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"No data shards found in {output_dir}")
    dataset = load_dataset("json", data_files=data_files, split="train")
    LOGGER.info("Pushing %d rows to %s (%s split)", len(dataset), args.repo_id, args.hub_split)
    dataset.push_to_hub(args.repo_id, config_name=args.hub_config, split=args.hub_split, private=args.private)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("train", "eval"), required=True)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", default=None, help="Repeatable DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]] source")
    parser.add_argument("--source-weight", action="append", type=float, default=None)
    parser.add_argument("--budget-tokenizer", default="apple/DCLM-7B")
    parser.add_argument("--eval-fraction", type=float, default=0.01)
    parser.add_argument("--partition-salt", default="dclm-stack-v1")
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--shard-token-budget", type=int, default=100_000_000)
    parser.add_argument("--min-document-bytes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", help="Optional Hugging Face dataset repository, e.g. org/dclm-stack")
    parser.add_argument("--hub-config", default="default")
    parser.add_argument("--hub-split", help="Hub split name; defaults to --partition")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    args.source = args.source or DEFAULT_SOURCES
    args.source_weight = args.source_weight or [1.0] * len(args.source)
    args.hub_split = args.hub_split or args.partition
    if args.token_budget <= 0:
        parser.error("--token-budget must be positive")
    if args.tokenizer_batch_size <= 0 or args.shard_token_budget <= 0:
        parser.error("--tokenizer-batch-size and --shard-token-budget must be positive")
    if len(args.source_weight) != len(args.source) or any(weight <= 0 for weight in args.source_weight):
        parser.error("Provide one positive --source-weight for every --source")
    if not 0 < args.eval_fraction < 1:
        parser.error("--eval-fraction must be between 0 and 1")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    output_dir, _ = sample(args)
    if args.repo_id:
        push(output_dir, args)


if __name__ == "__main__":
    main()
