#!/usr/bin/env python3
"""Route local JSONL corpus shards to multi-tokenizer experts in parallel.

Each worker owns a stable subset of source shards, loads the expert tokenizers
once, and writes an atomic ``routed-*.jsonl`` output per input shard. Finished
shards are skipped on later runs, making the operation resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_registry(path: str) -> list[dict[str, Any]]:
    registry_path = Path(path)
    with registry_path.open() as stream:
        entries = yaml.safe_load(stream).get("multi_tokenizer", {})["tokenizers"]
    if [entry["id"] for entry in entries] != list(range(len(entries))):
        raise ValueError("Tokenizer IDs must be contiguous and start at zero")
    for entry in entries:
        tokenizer_path = Path(entry["tokenizer_name_or_path"])
        if not tokenizer_path.is_absolute():
            entry["tokenizer_name_or_path"] = str((registry_path.parent / tokenizer_path).resolve())
    return entries


def route_files(files: list[str], registry_path: str, output_dir: str, batch_size: int) -> dict[str, Any]:
    # Process-level parallelism is more predictable than each Rust tokenizer
    # spawning its own large thread pool.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer

    registry = load_registry(registry_path)
    tokenizers = [AutoTokenizer.from_pretrained(entry["tokenizer_name_or_path"], trust_remote_code=False) for entry in registry]
    output_root = Path(output_dir)
    assignments: Counter[int] = Counter()
    documents = 0
    completed: list[str] = []

    def flush(output, texts: list[str], extras: list[dict[str, Any]]) -> int:
        if not texts:
            return 0
        lengths = []
        for tokenizer in tokenizers:
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            lengths.append([len(ids) for ids in encoded["input_ids"]])
        for row, (text, extra) in enumerate(zip(texts, extras)):
            tokenizer_id = min(range(len(tokenizers)), key=lambda index: lengths[index][row])
            output.write(json.dumps({"text": text, "tokenizer_id": tokenizer_id, **extra}, ensure_ascii=False) + "\n")
            assignments[tokenizer_id] += 1
        count = len(texts)
        texts.clear()
        extras.clear()
        return count

    for source_name in files:
        source = Path(source_name)
        destination = output_root / source.name.replace("data-", "routed-", 1)
        if destination.exists() and destination.stat().st_size:
            completed.append(destination.name)
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        texts: list[str] = []
        extras: list[dict[str, Any]] = []
        written = 0
        next_progress = 10_000
        with source.open(encoding="utf-8") as input_stream, temporary.open("w", encoding="utf-8") as output:
            for record_index, line in enumerate(input_stream):
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text:
                    continue
                texts.append(text)
                extras.append({"source_document_index": record_index})
                if len(texts) >= batch_size:
                    written += flush(output, texts, extras)
                    if written >= next_progress:
                        print(f"{source.name}: routed {written:,} documents", flush=True)
                        while next_progress <= written:
                            next_progress += 10_000
            written += flush(output, texts, extras)
        temporary.replace(destination)
        documents += written
        completed.append(destination.name)
    return {"documents": documents, "assignments": dict(assignments), "shards": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-glob", default="data-*.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("--workers and --batch-size must be positive")
    source_files = sorted(args.input_dir.glob(args.input_glob))
    if not source_files:
        raise FileNotFoundError(f"No {args.input_glob!r} files under {args.input_dir}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (
        args.output_dir.exists()
        and not args.overwrite
        and any(args.output_dir.glob("routed-*.jsonl"))
        and not (args.output_dir / "routing_manifest.json").exists()
    ):
        raise FileExistsError(
            f"{args.output_dir} contains incomplete routed shards. "
            "Pass --overwrite to discard them, or restore a completed routing_manifest.json."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = [path for path in source_files if not (args.output_dir / path.name.replace("data-", "routed-", 1)).exists()]
    if not pending:
        print(f"All {len(source_files)} routed shards already exist in {args.output_dir}")
        return
    worker_count = min(args.workers, len(pending))
    assignments = [[] for _ in range(worker_count)]
    for index, path in enumerate(pending):
        assignments[index % worker_count].append(str(path))
    print(f"Routing {len(pending)} shard(s) locally with {worker_count} worker process(es)", flush=True)
    totals: Counter[int] = Counter()
    documents = 0
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(route_files, group, str(args.registry), str(args.output_dir), args.batch_size)
            for group in assignments
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            documents += result["documents"]
            totals.update({int(key): value for key, value in result["assignments"].items()})
            completed += len(result["shards"])
            print(f"Completed {completed}/{len(pending)} shard(s); routed {documents:,} documents", flush=True)
    manifest = {
        "format_version": 1,
        "input_dir": str(args.input_dir.resolve()),
        "registry": str(args.registry.resolve()),
        "newly_routed_documents": documents,
        "new_assignments": {str(key): value for key, value in sorted(totals.items())},
        "shards": sorted(path.name for path in args.output_dir.glob("routed-*.jsonl")),
    }
    (args.output_dir / "routing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
