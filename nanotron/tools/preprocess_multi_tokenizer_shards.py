#!/usr/bin/env python3
"""Stream routed JSONL shards into packed, disjoint multi-tokenizer datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from transformers import AutoTokenizer


TOKENIZER_ARTIFACTS = (
    "tokenizer.json", "tokenizer.model", "added_tokens.json", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
)


def fingerprint_tokenizer(path: str, revision: str | None) -> str:
    root = Path(path)
    digest = hashlib.sha256((revision or "").encode())
    for candidate in [root / name for name in TOKENIZER_ARTIFACTS]:
        if candidate.is_file():
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def load_registry(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    raw = raw.get("multi_tokenizer", raw)
    for entry in raw["tokenizers"]:
        tokenizer_path = Path(entry["tokenizer_name_or_path"])
        if not tokenizer_path.is_absolute():
            entry["tokenizer_name_or_path"] = str((path.parent / tokenizer_path).resolve())
    entries = raw["tokenizers"]
    if [entry["id"] for entry in entries] != list(range(len(entries))):
        raise ValueError("Registry tokenizer IDs must be contiguous and ordered from zero")
    return entries


class TokenWriter:
    def __init__(self, directory: Path, dtype: np.dtype, sequence_length: int, tokens_per_file: int, flush_tokens: int):
        self.directory = directory
        self.dtype = dtype
        self.chunk_length = sequence_length + 1
        self.tokens_per_file = tokens_per_file // self.chunk_length * self.chunk_length
        self.flush_tokens = max(flush_tokens, self.chunk_length)
        self.pending: list[int] = []
        self.offset = 0
        self.file_index = 0
        self.file_tokens = 0
        self.total_tokens = 0
        self.stream = None
        directory.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.stream is None:
            path = self.directory / f"data-{self.file_index:05d}.ds"
            self.stream = path.open("wb")

    def _rotate(self) -> None:
        assert self.stream is not None
        self.stream.close()
        self.stream = None
        self.file_index += 1
        self.file_tokens = 0

    def add(self, token_ids: list[int]) -> None:
        self.pending.extend(token_ids)
        if len(self.pending) - self.offset >= self.flush_tokens:
            self.flush(final=False)

    def flush(self, *, final: bool) -> None:
        available = len(self.pending) - self.offset
        writable = available // self.chunk_length * self.chunk_length
        while writable:
            self._open()
            capacity = self.tokens_per_file - self.file_tokens
            count = min(writable, capacity)
            # Both capacity and writable are multiples of the context length.
            np.asarray(self.pending[self.offset : self.offset + count], dtype=self.dtype).tofile(self.stream)
            self.offset += count
            self.file_tokens += count
            self.total_tokens += count
            writable -= count
            if self.file_tokens == self.tokens_per_file:
                self._rotate()
        if self.offset and (self.offset >= self.flush_tokens or final):
            del self.pending[: self.offset]
            self.offset = 0
        if final and self.pending:
            # A final incomplete context cannot be consumed by Nanotron.
            self.pending.clear()
        if final and self.stream is not None:
            self.stream.close()
            self.stream = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-glob", default="routed-*.jsonl")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tokens-per-output-shard", type=int, default=100_000_000)
    parser.add_argument("--write-buffer-tokens", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.tokens_per_output_shard, args.write_buffer_tokens) <= 0:
        parser.error("sequence length, batch size, and shard/buffer token counts must be positive")
    if args.tokens_per_output_shard < args.sequence_length + 1:
        parser.error("--tokens-per-output-shard must hold at least one full sequence")
    input_files = sorted(args.input_dir.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No {args.input_glob!r} files found in {args.input_dir}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)

    registry = load_registry(args.registry)
    tokenizers = {entry["id"]: AutoTokenizer.from_pretrained(entry["tokenizer_name_or_path"], trust_remote_code=False) for entry in registry}
    vocabularies = []
    for entry in registry:
        tokenizer = tokenizers[entry["id"]]
        vocab_size = len(tokenizer)
        if entry.get("vocab_size") is not None and entry["vocab_size"] != vocab_size:
            raise ValueError(f"Tokenizer {entry['name']!r} declares {entry['vocab_size']} tokens but contains {vocab_size}")
        vocabularies.append({
            "id": entry["id"], "name": entry["name"], "vocab_size": vocab_size,
            "fingerprint": entry.get("fingerprint") or fingerprint_tokenizer(entry["tokenizer_name_or_path"], entry.get("tokenizer_revision")),
            "bos_token_id": entry.get("bos_token_id", tokenizer.bos_token_id),
            "eos_token_id": entry.get("eos_token_id", tokenizer.eos_token_id),
            "pad_token_id": entry.get("pad_token_id", tokenizer.pad_token_id),
        })
    writers = {
        vocabulary["id"]: TokenWriter(
            args.output_dir / f"{vocabulary['id']:03d}-{vocabulary['name']}",
            np.uint16 if vocabulary["vocab_size"] <= np.iinfo(np.uint16).max else np.uint32,
            args.sequence_length,
            args.tokens_per_output_shard,
            args.write_buffer_tokens,
        )
        for vocabulary in vocabularies
    }
    documents: dict[int, int] = defaultdict(int)
    source_bytes: dict[int, int] = defaultdict(int)
    routing_digest = hashlib.sha256()
    processed = 0
    next_progress = 100_000

    def process_batch(records: list[tuple[str, int]]) -> None:
        nonlocal processed, next_progress
        grouped: dict[int, list[str]] = defaultdict(list)
        for text, tokenizer_id in records:
            grouped[tokenizer_id].append(text)
        for tokenizer_id, texts in grouped.items():
            tokenizer = tokenizers.get(tokenizer_id)
            if tokenizer is None:
                raise ValueError(f"Unknown tokenizer_id={tokenizer_id}")
            if tokenizer.eos_token_id is None:
                raise ValueError(f"Tokenizer {tokenizer_id} has no EOS token")
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            for text, ids in zip(texts, encoded["input_ids"]):
                writers[tokenizer_id].add(ids + [tokenizer.eos_token_id])
                documents[tokenizer_id] += 1
                source_bytes[tokenizer_id] += len(text.encode("utf-8"))
                processed += 1

    batch: list[tuple[str, int]] = []
    for input_path in input_files:
        with input_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                routing_digest.update(line.encode())
                record = json.loads(line)
                text = record.get("text")
                tokenizer_id = record.get("tokenizer_id")
                if not isinstance(text, str) or not isinstance(tokenizer_id, int):
                    raise ValueError(f"{input_path}:{line_number} requires string text and integer tokenizer_id")
                batch.append((text, tokenizer_id))
                if len(batch) >= args.batch_size:
                    process_batch(batch)
                    batch.clear()
                    if processed >= next_progress:
                        print(f"Preprocessed {processed:,} documents", flush=True)
                        while next_progress <= processed:
                            next_progress += 100_000
    process_batch(batch)
    for writer in writers.values():
        writer.flush(final=True)

    routing_fingerprint = f"sha256:{routing_digest.hexdigest()}"
    top_manifest = {"format_version": 1, "sources": []}
    for entry, vocabulary in zip(registry, vocabularies):
        tokenizer_id = vocabulary["id"]
        folder = args.output_dir / f"{tokenizer_id:03d}-{vocabulary['name']}"
        manifest = {
            "format_version": 1, "tokenizer_id": tokenizer_id, "tokenizer_name": vocabulary["name"],
            "tokenizer_name_or_path": entry["tokenizer_name_or_path"], "tokenizer_revision": entry.get("tokenizer_revision"),
            "tokenizer_fingerprint": vocabulary["fingerprint"], "original_vocab_size": vocabulary["vocab_size"],
            "token_size_in_bytes": np.dtype(writers[tokenizer_id].dtype).itemsize,
            "bos_token_id": vocabulary["bos_token_id"], "eos_token_id": vocabulary["eos_token_id"],
            "pad_token_id": vocabulary["pad_token_id"], "routing_fingerprint": routing_fingerprint,
            "documents": documents[tokenizer_id], "tokens": writers[tokenizer_id].total_tokens,
            "source_bytes": source_bytes[tokenizer_id],
        }
        (folder / "multi_tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        top_manifest["sources"].append({"path": str(folder), **manifest})
    with (args.output_dir / "multi_tokenizer_manifest.json").open("w") as stream:
        json.dump(top_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Preprocessed {processed:,} documents into {args.output_dir}")


if __name__ == "__main__":
    main()
