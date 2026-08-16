#!/usr/bin/env python3
"""Pre-tokenize routed JSONL into disjoint TokenizedBytes folders.

Input records must contain ``text`` and ``tokenizer_id``.  Output sequences
never cross tokenizer boundaries and are directly readable as ``*.ds`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

from nanotron.config.config import (
    MultiTokenizerArgs,
    TokenizerScheduleArgs,
    TokenizerSpec,
)
from nanotron.config.vocabulary import resolve_multi_vocabulary
from nanotron.data.multi_tokenizer import MultiTokenizerDatasetManifest


def _load_registry(path: Path) -> MultiTokenizerArgs:
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    raw = raw.get("multi_tokenizer", raw)
    return MultiTokenizerArgs(
        tokenizers=[TokenizerSpec(**entry) for entry in raw["tokenizers"]],
        schedule=TokenizerScheduleArgs(**raw.get("schedule", {})),
        batching=raw.get("batching", "step_homogeneous"),
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
        mask_padded_vocab_logits=raw.get("mask_padded_vocab_logits", True),
    )


def preprocess(
    input_path: Path, output_dir: Path, registry_path: Path, sequence_length: int
) -> None:
    registry = _load_registry(registry_path)
    resolved = resolve_multi_vocabulary(
        registry, tp_size=1, make_vocab_size_divisible_by=1
    )
    tokenizers = {
        spec.id: AutoTokenizer.from_pretrained(
            spec.tokenizer_name_or_path,
            revision=spec.tokenizer_revision,
            trust_remote_code=False,
        )
        for spec in registry.tokenizers
    }
    tokens = {spec.id: [] for spec in registry.tokenizers}
    documents = {spec.id: 0 for spec in registry.tokenizers}
    source_bytes = {spec.id: 0 for spec in registry.tokenizers}
    routing_digest = hashlib.sha256()

    with input_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            routing_digest.update(line.encode())
            record = json.loads(line)
            if "text" not in record or "tokenizer_id" not in record:
                raise ValueError(
                    f"Line {line_number} must contain text and tokenizer_id"
                )
            tokenizer_id = int(record["tokenizer_id"])
            if tokenizer_id not in tokenizers:
                raise ValueError(
                    f"Line {line_number} references unknown tokenizer_id={tokenizer_id}"
                )
            text = record["text"]
            if not isinstance(text, str):
                raise TypeError(f"Line {line_number} text must be a string")
            tokenizer = tokenizers[tokenizer_id]
            document_tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer {tokenizer_id} has no EOS token for document separation"
                )
            document_tokens.append(tokenizer.eos_token_id)
            tokens[tokenizer_id].extend(document_tokens)
            documents[tokenizer_id] += 1
            source_bytes[tokenizer_id] += len(text.encode("utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    top_manifest = {"format_version": 1, "sources": []}
    chunk_length = sequence_length + 1
    routing_fingerprint = f"sha256:{routing_digest.hexdigest()}"
    for spec, vocabulary in zip(registry.tokenizers, resolved.vocabularies):
        folder = output_dir / f"{vocabulary.id:03d}-{vocabulary.name}"
        folder.mkdir(parents=True, exist_ok=True)
        usable = len(tokens[vocabulary.id]) // chunk_length * chunk_length
        if usable == 0:
            raise ValueError(
                f"Tokenizer {vocabulary.name!r} has no complete sequence of {chunk_length} tokens"
            )
        dtype = (
            np.uint16
            if vocabulary.original_vocab_size <= np.iinfo(np.uint16).max
            else np.uint32
        )
        np.asarray(tokens[vocabulary.id][:usable], dtype=dtype).tofile(
            folder / "data.ds"
        )
        manifest = MultiTokenizerDatasetManifest(
            format_version=1,
            tokenizer_id=vocabulary.id,
            tokenizer_name=vocabulary.name,
            tokenizer_name_or_path=spec.tokenizer_name_or_path,
            tokenizer_revision=spec.tokenizer_revision,
            tokenizer_fingerprint=vocabulary.fingerprint,
            original_vocab_size=vocabulary.original_vocab_size,
            token_size_in_bytes=np.dtype(dtype).itemsize,
            bos_token_id=vocabulary.special_token_ids.bos,
            eos_token_id=vocabulary.special_token_ids.eos,
            pad_token_id=vocabulary.special_token_ids.pad,
            routing_fingerprint=routing_fingerprint,
            documents=documents[vocabulary.id],
            tokens=usable,
            source_bytes=source_bytes[vocabulary.id],
        )
        manifest.save(folder / "multi_tokenizer_manifest.json")
        top_manifest["sources"].append({"path": str(folder), **asdict(manifest)})
    with (output_dir / "multi_tokenizer_manifest.json").open("w") as stream:
        json.dump(top_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Routed JSONL input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="YAML file containing multi_tokenizer",
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    preprocess(args.input, args.output_dir, args.registry, args.sequence_length)


if __name__ == "__main__":
    main()
