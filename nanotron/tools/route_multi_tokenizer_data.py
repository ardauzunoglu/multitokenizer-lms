#!/usr/bin/env python3
"""Route a Hugging Face text split to the shortest expert-tokenizer encoding.

The output is sharded JSONL. Each record preserves the input ``text`` and adds
``tokenizer_id``, which is required by ``preprocess_multi_tokenizer_data.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def load_registry(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    entries = raw.get("multi_tokenizer", raw)["tokenizers"]
    ids = [entry["id"] for entry in entries]
    if ids != list(range(len(entries))):
        raise ValueError(f"Registry tokenizer IDs must be contiguous from zero; got {ids}")
    for entry in entries:
        tokenizer_path = Path(entry["tokenizer_name_or_path"])
        if not tokenizer_path.is_absolute():
            entry["tokenizer_name_or_path"] = str((path.parent / tokenizer_path).resolve())
    return entries


def route_batch(tokenizers: list[Any], texts: list[str]) -> list[int]:
    """Select the smallest local-ID encoding for each document."""
    lengths = []
    for tokenizer in tokenizers:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append([len(ids) for ids in encoded["input_ids"]])
    return [min(range(len(tokenizers)), key=lambda index: lengths[index][row]) for row in range(len(texts))]


def flush_batch(
    output, tokenizers: list[Any], texts: list[str], metadata: list[dict[str, Any]], counts: Counter
) -> int:
    if not texts:
        return 0
    selected = route_batch(tokenizers, texts)
    for text, extra, tokenizer_id in zip(texts, metadata, selected):
        output.write(json.dumps({"text": text, "tokenizer_id": tokenizer_id, **extra}, ensure_ascii=False) + "\n")
        counts[tokenizer_id] += 1
    texts.clear()
    metadata.clear()
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-documents", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--max-documents", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.shard_documents <= 0 or args.progress_interval <= 0:
        parser.error("--batch-size, --shard-documents, and --progress-interval must be positive")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    registry = load_registry(args.registry)
    tokenizers = [
        AutoTokenizer.from_pretrained(entry["tokenizer_name_or_path"], trust_remote_code=False)
        for entry in registry
    ]
    stream = load_dataset(args.dataset, split=args.split, streaming=True)
    counts: Counter[int] = Counter()
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    shard_index = 0
    documents = 0
    shard_documents = 0
    next_progress = args.progress_interval
    output = (args.output_dir / f"routed-{shard_index:05d}.jsonl").open("w", encoding="utf-8")
    print(
        f"Routing {args.dataset}[{args.split}] with {len(tokenizers)} tokenizers; "
        f"batch_size={args.batch_size}, shard_documents={args.shard_documents}",
        flush=True,
    )

    try:
        for row_index, row in enumerate(stream):
            text = row.get(args.text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            texts.append(text)
            metadata.append({"source_document_index": row_index})
            if len(texts) < args.batch_size:
                continue
            written = flush_batch(output, tokenizers, texts, metadata, counts)
            documents += written
            shard_documents += written
            if shard_documents >= args.shard_documents:
                output.close()
                shard_index += 1
                shard_documents = 0
                output = (args.output_dir / f"routed-{shard_index:05d}.jsonl").open("w", encoding="utf-8")
            if documents >= next_progress:
                print(f"Routed {documents:,} documents; assignments={dict(sorted(counts.items()))}", flush=True)
                while next_progress <= documents:
                    next_progress += args.progress_interval
            if args.max_documents is not None and documents >= args.max_documents:
                break
        written = flush_batch(output, tokenizers, texts, metadata, counts)
        documents += written
    finally:
        output.close()

    manifest = {
        "format_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "text_column": args.text_column,
        "registry": str(args.registry.resolve()),
        "documents": documents,
        "assignments": {str(key): value for key, value in sorted(counts.items())},
        "shards": sorted(path.name for path in args.output_dir.glob("routed-*.jsonl") if path.stat().st_size),
    }
    (args.output_dir / "routing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Routed {documents:,} documents into {len(manifest['shards'])} shard(s): {args.output_dir}")


if __name__ == "__main__":
    main()
