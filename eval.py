#!/usr/bin/env python3
"""Compression-only evaluation for a latent disjoint-tokenizer run.

This script deliberately does not train or score a language model. It trains
two single-tokenizer BPE baselines on the exact discovery sample:

* single-small: the same target vocabulary size as one active expert;
* single-large: K times that size, matching K disjoint expert vocabularies.

It then evaluates bytes/token on byte-budgeted held-out samples from every
training source. The disjoint tokenizers are reported with both a learned
raw-text router and an oracle router that chooses the shortest encoding for
each document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import create_domains as latent

LOGGER = logging.getLogger("evaluate_tokenizer_compression")


@dataclass
class EvaluationDocument:
    doc_id: int
    text: str
    raw_bytes: int
    digest: str
    source: str = ""


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_utf8(text: str, byte_budget: int) -> str:
    if byte_budget <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_budget:
        return text
    return encoded[:byte_budget].decode("utf-8", errors="ignore")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe_namespace(namespace: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse values such as Path into JSON-serializable values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }


def load_training_documents(path: Path) -> list[latent.Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training sample not found: {path}. The discovery run must use --save-sample "
            "so the matched baseline tokenizers can be trained on identical text."
        )
    documents: list[latent.Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            documents.append(
                latent.Document(
                    doc_id=int(item["doc_id"]),
                    text=str(item["text"]),
                    raw_bytes=int(item["raw_bytes"]),
                    budget_tokens=int(item["budget_tokens"]),
                )
            )
    if not documents:
        raise RuntimeError(f"No training documents found in {path}")
    return documents


def load_training_labels(path: Path, expected_documents: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    labels = np.full(expected_documents, -1, dtype=np.int64)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            doc_id = int(row["doc_id"])
            if not 0 <= doc_id < expected_documents:
                raise ValueError(f"Unexpected doc_id {doc_id} in {path}")
            labels[doc_id] = int(row["expert_id"])
    if np.any(labels < 0):
        missing = np.where(labels < 0)[0][:10].tolist()
        raise ValueError(f"Missing assignment rows for document IDs {missing}")
    return labels


def load_disjoint_tokenizers(run_dir: Path) -> list[Any]:
    tokenizers = latent.require_module("tokenizers", "tokenizers")
    paths = sorted((run_dir / "tokenizers").glob("expert_*/tokenizer.json"))
    if not paths:
        raise FileNotFoundError(
            f"No final expert tokenizers found under {run_dir / 'tokenizers'}"
        )
    return [tokenizers.Tokenizer.from_file(str(path)) for path in paths]


def resolve_source_args(
    args: argparse.Namespace,
    original: dict[str, Any],
) -> list[argparse.Namespace]:
    original_source = str(original.get("source", "dclm"))
    source = original_source if args.eval_source == "auto" else args.eval_source
    if source == "jsonl" and args.eval_jsonl is None:
        raise ValueError(
            "A discovery run based on JSONL requires a separate --eval-jsonl file "
            "for held-out evaluation"
        )
    default_text_field = args.text_field or str(original.get("text_field", "text"))
    default_split = args.split or str(original.get("split", "train"))

    # An explicit evaluation override selects one source. Otherwise, reproduce
    # every Hugging Face source recorded by multi-source domain discovery.
    use_original_sources = (
        args.eval_source == "auto"
        and args.dataset is None
        and args.dataset_config is None
        and args.text_field is None
        and args.split is None
    )
    hf_values = original.get("hf_source") if use_original_sources else None
    if source in {"dclm", "hf"} and hf_values:
        specs = [
            latent.parse_hf_source(value, default_split, default_text_field)
            for value in hf_values
        ]
        return [
            argparse.Namespace(
                source="hf",
                source_label=spec.label,
                hf_source=None,
                input_jsonl=None,
                text_field=spec.text_field,
                dataset=spec.dataset,
                dataset_config=spec.config,
                split=spec.split,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.eval_seed + index,
                synthetic_documents=args.synthetic_documents,
            )
            for index, spec in enumerate(specs)
        ]

    dataset = args.dataset or str(
        original.get("dataset", "mlfoundations/dclm-baseline-1.0-parquet")
    )
    dataset_config = (
        args.dataset_config
        if args.dataset_config is not None
        else original.get("dataset_config")
    )
    if source in {"dclm", "hf"}:
        source_label = (
            dataset if dataset_config is None else f"{dataset}::{dataset_config}"
        )
    elif source == "jsonl":
        source_label = str(args.eval_jsonl)
    else:
        source_label = source
    return [
        argparse.Namespace(
            source=source,
            source_label=source_label,
            hf_source=None,
            input_jsonl=args.eval_jsonl,
            text_field=default_text_field,
            dataset=dataset,
            dataset_config=dataset_config,
            split=default_split,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.eval_seed,
            synthetic_documents=args.synthetic_documents,
        )
    ]


def collect_heldout_documents(
    rows: Iterable[dict[str, Any]],
    text_field: str,
    target_bytes: int,
    excluded_hashes: set[str],
    min_document_bytes: int,
    max_documents: int,
    source_label: str = "",
) -> tuple[list[EvaluationDocument], dict[str, int]]:
    documents: list[EvaluationDocument] = []
    seen = set(excluded_hashes)
    total_bytes = 0
    scanned = 0
    excluded = 0
    duplicate = 0

    for row in rows:
        scanned += 1
        value = row.get(text_field)
        if not isinstance(value, str) and text_field == "text":
            value = row.get("content")
        if not isinstance(value, str):
            continue
        text = value.strip()
        raw_bytes = len(text.encode("utf-8"))
        if raw_bytes < min_document_bytes:
            continue
        digest = text_digest(text)
        if digest in excluded_hashes:
            excluded += 1
            continue
        if digest in seen:
            duplicate += 1
            continue

        remaining = target_bytes - total_bytes
        if remaining <= 0:
            break
        if raw_bytes > remaining:
            text = truncate_utf8(text, remaining)
            raw_bytes = len(text.encode("utf-8"))
            digest = text_digest(text)
        if raw_bytes < min_document_bytes:
            break

        documents.append(
            EvaluationDocument(
                len(documents), text, raw_bytes, digest, source=source_label
            )
        )
        seen.add(digest)
        total_bytes += raw_bytes
        if len(documents) % 1_000 == 0:
            LOGGER.info(
                "Held out %s documents, %.2f/%.2f MiB",
                f"{len(documents):,}",
                total_bytes / 2**20,
                target_bytes / 2**20,
            )
        if max_documents and len(documents) >= max_documents:
            break
        if total_bytes >= target_bytes:
            break

    if not documents:
        raise RuntimeError("No held-out documents were collected")
    if total_bytes < target_bytes:
        LOGGER.warning(
            "Held-out stream ended or hit --max-documents at %.2f MiB, below the %.2f MiB target",
            total_bytes / 2**20,
            target_bytes / 2**20,
        )
    return documents, {
        "scanned_rows": scanned,
        "excluded_training_duplicates": excluded,
        "excluded_heldout_duplicates": duplicate,
    }


def tokenizer_cost_matrix(
    documents: Sequence[EvaluationDocument] | Sequence[latent.Document],
    tokenizers: Sequence[Any],
    batch_size: int,
) -> np.ndarray:
    texts = [document.text for document in documents]
    costs = np.zeros((len(texts), len(tokenizers)), dtype=np.int64)
    for column, tokenizer in enumerate(tokenizers):
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer.encode_batch(texts[start : start + batch_size])
            costs[start : start + len(encoded), column] = [
                len(item.ids) for item in encoded
            ]
    return costs


def train_router(
    documents: Sequence[latent.Document],
    labels: np.ndarray,
    feature_dimensions: int,
    ngram_max: int,
    seed: int,
):
    feature_extraction = latent.require_module(
        "sklearn.feature_extraction.text", "scikit-learn"
    )
    linear_model = latent.require_module("sklearn.linear_model", "scikit-learn")
    vectorizer = feature_extraction.HashingVectorizer(
        analyzer="char",
        ngram_range=(3, ngram_max),
        n_features=feature_dimensions,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    classifier = linear_model.SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1_000,
        tol=1e-4,
        random_state=seed,
    )
    features = vectorizer.transform(document.text for document in documents)
    classifier.fit(features, labels)
    train_accuracy = float(np.mean(classifier.predict(features) == labels))
    return vectorizer, classifier, train_accuracy


def bootstrap_interval(values: Sequence[float], confidence: float) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def bootstrap_metrics(
    raw_bytes: np.ndarray,
    methods: dict[str, np.ndarray],
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    bpt_samples: dict[str, list[float]] = {name: [] for name in methods}
    small_reductions: dict[str, list[float]] = {name: [] for name in methods}
    large_reductions: dict[str, list[float]] = {name: [] for name in methods}

    for _ in range(repetitions):
        indices = rng.integers(0, len(raw_bytes), size=len(raw_bytes))
        byte_total = float(raw_bytes[indices].sum())
        totals = {
            name: float(tokens[indices].sum()) for name, tokens in methods.items()
        }
        for name, token_total in totals.items():
            bpt_samples[name].append(byte_total / max(token_total, 1.0))
            small_reductions[name].append(
                1.0 - token_total / max(totals["single_small"], 1.0)
            )
            large_reductions[name].append(
                1.0 - token_total / max(totals["single_large"], 1.0)
            )

    return {
        name: {
            "bytes_per_token": bootstrap_interval(bpt_samples[name], confidence),
            "reduction_vs_single_small": bootstrap_interval(
                small_reductions[name], confidence
            ),
            "reduction_vs_single_large": bootstrap_interval(
                large_reductions[name], confidence
            ),
        }
        for name in methods
    }


def method_metrics(
    raw_bytes: np.ndarray,
    tokens: np.ndarray,
    small_tokens: np.ndarray,
    large_tokens: np.ndarray,
) -> dict[str, float | int]:
    total_bytes = int(raw_bytes.sum())
    total_tokens = int(tokens.sum())
    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "tokens_per_kib": total_tokens / max(total_bytes / 1024.0, 1e-12),
        "reduction_vs_single_small": 1.0
        - total_tokens / max(int(small_tokens.sum()), 1),
        "reduction_vs_single_large": 1.0
        - total_tokens / max(int(large_tokens.sum()), 1),
        "document_win_rate_vs_small": float(np.mean(tokens < small_tokens)),
        "document_win_rate_vs_large": float(np.mean(tokens < large_tokens)),
        "document_tie_rate_vs_small": float(np.mean(tokens == small_tokens)),
        "document_tie_rate_vs_large": float(np.mean(tokens == large_tokens)),
    }


def format_percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def print_results(results: dict[str, dict[str, Any]]) -> None:
    headers = [
        "method",
        "active vocab",
        "tokens",
        "bytes/token",
        "vs small",
        "vs large",
    ]
    rows = []
    for name, item in results.items():
        rows.append(
            [
                name,
                f"{item['active_vocab']:,}",
                f"{item['total_tokens']:,}",
                f"{item['bytes_per_token']:.4f}",
                format_percent(float(item["reduction_vs_single_small"])),
                format_percent(float(item["reduction_vs_single_large"])),
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def write_outputs(
    output_dir: Path,
    documents: Sequence[EvaluationDocument],
    methods: dict[str, np.ndarray],
    learned_routes: np.ndarray,
    oracle_routes: np.ndarray,
    report: dict[str, Any],
) -> None:
    (output_dir / "compression_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "compression_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "method",
            "active_vocab",
            "stored_vocab",
            "total_bytes",
            "total_tokens",
            "bytes_per_token",
            "tokens_per_kib",
            "reduction_vs_single_small",
            "reduction_vs_single_large",
            "document_win_rate_vs_small",
            "document_win_rate_vs_large",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in report["methods"].items():
            writer.writerow(
                {
                    "method": name,
                    **{field: values[field] for field in fields if field != "method"},
                }
            )

    with (output_dir / "compression_results_by_dataset.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        dataset_fields = ["dataset", *fields]
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        for dataset, dataset_values in report["datasets"].items():
            for name, values in dataset_values["methods"].items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": name,
                        **{
                            field: values[field]
                            for field in fields
                            if field != "method"
                        },
                    }
                )

    with (output_dir / "heldout_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "doc_id",
                "dataset",
                "sha256_prefix",
                "raw_bytes",
                *[f"tokens_{name}" for name in methods],
                "learned_expert",
                "oracle_expert",
            ]
        )
        for index, document in enumerate(documents):
            writer.writerow(
                [
                    document.doc_id,
                    document.source,
                    document.digest[:16],
                    document.raw_bytes,
                    *[int(methods[name][index]) for name in methods],
                    int(learned_routes[index]),
                    int(oracle_routes[index]),
                ]
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train matched single-tokenizer baselines and compare held-out bytes/token.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-sample", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--small-vocab-size", type=int, default=None)
    parser.add_argument("--large-vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)

    source = parser.add_argument_group("held-out source")
    source.add_argument(
        "--eval-source",
        choices=["auto", "dclm", "hf", "jsonl", "synthetic"],
        default="auto",
    )
    source.add_argument("--eval-jsonl", type=Path, default=None)
    source.add_argument("--dataset", default=None)
    source.add_argument("--dataset-config", default=None)
    source.add_argument("--split", default=None)
    source.add_argument("--text-field", default=None)
    source.add_argument(
        "--eval-bytes",
        type=int,
        default=50_000_000,
        help="held-out byte budget for each evaluated dataset",
    )
    source.add_argument("--eval-seed", type=int, default=10_017)
    source.add_argument("--shuffle-buffer", type=int, default=10_000)
    source.add_argument("--max-eval-documents", type=int, default=0)
    source.add_argument("--min-document-bytes", type=int, default=None)
    source.add_argument("--synthetic-documents", type=int, default=5_000)

    routing = parser.add_argument_group("router")
    routing.add_argument("--router-feature-dimensions", type=int, default=262_144)
    routing.add_argument("--router-ngram-max", type=int, default=5)

    reporting = parser.add_argument_group("reporting")
    reporting.add_argument("--score-batch-size", type=int, default=256)
    reporting.add_argument("--bootstrap", type=int, default=500)
    reporting.add_argument("--confidence", type=float, default=0.95)
    reporting.add_argument("--overwrite", action="store_true")
    reporting.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_bytes <= 0:
        parser.error("--eval-bytes must be positive")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between zero and one")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    started = time.time()
    run_dir = args.run_dir.resolve()
    original_args = load_json(run_dir / "arguments.json")
    discovery_summary = load_json(run_dir / "summary.json")
    train_sample_path = (args.train_sample or run_dir / "sample.jsonl").resolve()
    training_documents = load_training_documents(train_sample_path)
    training_labels = load_training_labels(
        run_dir / "assignments.csv", len(training_documents)
    )
    disjoint_tokenizers = load_disjoint_tokenizers(run_dir)

    if int(training_labels.max()) >= len(disjoint_tokenizers):
        raise ValueError(
            "assignments.csv refers to an expert tokenizer that does not exist"
        )
    expert_count = len(disjoint_tokenizers)
    small_vocab_target = args.small_vocab_size or int(
        discovery_summary["vocab_size_target"]
    )
    large_vocab_target = args.large_vocab_size or small_vocab_target * expert_count
    min_frequency = args.min_frequency or int(original_args.get("min_frequency", 2))

    output_dir = (args.output_dir or run_dir / "compression_eval").resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Training matched baselines on %s documents: V=%s and K*V=%s",
        f"{len(training_documents):,}",
        f"{small_vocab_target:,}",
        f"{large_vocab_target:,}",
    )
    train_texts = [document.text for document in training_documents]
    small_tokenizer = latent.train_bpe(
        train_texts,
        output_dir / "baselines" / "single_small",
        small_vocab_target,
        min_frequency,
    )
    large_tokenizer = latent.train_bpe(
        train_texts,
        output_dir / "baselines" / "single_large",
        large_vocab_target,
        min_frequency,
    )

    source_args_list = resolve_source_args(args, original_args)
    excluded_hashes = {text_digest(document.text) for document in training_documents}
    min_document_bytes = args.min_document_bytes or int(
        original_args.get("min_document_bytes", 128)
    )
    heldout: list[EvaluationDocument] = []
    source_slices: dict[str, slice] = {}
    sampling_by_source: dict[str, dict[str, int]] = {}
    for source_args in source_args_list:
        LOGGER.info(
            "Collecting %.2f MiB of held-out text from %s",
            args.eval_bytes / 2**20,
            source_args.source_label,
        )
        source_documents, sampling_stats = collect_heldout_documents(
            latent.iter_source_rows(source_args),
            source_args.text_field,
            args.eval_bytes,
            excluded_hashes,
            min_document_bytes,
            args.max_eval_documents,
            source_args.source_label,
        )
        start = len(heldout)
        for document in source_documents:
            document.doc_id = len(heldout)
            heldout.append(document)
        source_slices[source_args.source_label] = slice(start, len(heldout))
        sampling_by_source[source_args.source_label] = sampling_stats
    raw_bytes = np.asarray([document.raw_bytes for document in heldout], dtype=np.int64)

    LOGGER.info(
        "Tokenizing held-out text with two baselines and %d experts", expert_count
    )
    baseline_costs = tokenizer_cost_matrix(
        heldout,
        [small_tokenizer, large_tokenizer],
        args.score_batch_size,
    )
    expert_costs = tokenizer_cost_matrix(
        heldout, disjoint_tokenizers, args.score_batch_size
    )
    oracle_routes = np.argmin(expert_costs, axis=1).astype(np.int64)
    oracle_tokens = expert_costs[np.arange(len(heldout)), oracle_routes]

    if expert_count == 1:
        learned_routes = np.zeros(len(heldout), dtype=np.int64)
        router_train_accuracy = 1.0
    else:
        vectorizer, router, router_train_accuracy = train_router(
            training_documents,
            training_labels,
            args.router_feature_dimensions,
            args.router_ngram_max,
            args.eval_seed,
        )
        heldout_features = vectorizer.transform(document.text for document in heldout)
        learned_routes = router.predict(heldout_features).astype(np.int64)
    learned_tokens = expert_costs[np.arange(len(heldout)), learned_routes]

    methods = {
        "single_small": baseline_costs[:, 0],
        "single_large": baseline_costs[:, 1],
        "disjoint_learned": learned_tokens,
        "disjoint_oracle": oracle_tokens,
    }
    bootstrap = bootstrap_metrics(
        raw_bytes,
        methods,
        args.bootstrap,
        args.confidence,
        args.eval_seed,
    )
    active_vocabs = {
        "single_small": small_tokenizer.get_vocab_size(),
        "single_large": large_tokenizer.get_vocab_size(),
        "disjoint_learned": max(
            tokenizer.get_vocab_size() for tokenizer in disjoint_tokenizers
        ),
        "disjoint_oracle": max(
            tokenizer.get_vocab_size() for tokenizer in disjoint_tokenizers
        ),
    }
    stored_vocabs = {
        "single_small": small_tokenizer.get_vocab_size(),
        "single_large": large_tokenizer.get_vocab_size(),
        "disjoint_learned": sum(
            tokenizer.get_vocab_size() for tokenizer in disjoint_tokenizers
        ),
        "disjoint_oracle": sum(
            tokenizer.get_vocab_size() for tokenizer in disjoint_tokenizers
        ),
    }

    method_results: dict[str, dict[str, Any]] = {}
    for name, token_counts in methods.items():
        method_results[name] = {
            "active_vocab": int(active_vocabs[name]),
            "stored_vocab": int(stored_vocabs[name]),
            **method_metrics(
                raw_bytes,
                token_counts,
                methods["single_small"],
                methods["single_large"],
            ),
            "bootstrap_intervals": bootstrap.get(name),
        }

    dataset_results: dict[str, dict[str, Any]] = {}
    for dataset_index, (source_label, source_slice) in enumerate(source_slices.items()):
        source_bytes = raw_bytes[source_slice]
        source_methods = {
            name: token_counts[source_slice] for name, token_counts in methods.items()
        }
        source_bootstrap = bootstrap_metrics(
            source_bytes,
            source_methods,
            args.bootstrap,
            args.confidence,
            args.eval_seed + dataset_index + 1,
        )
        source_method_results: dict[str, dict[str, Any]] = {}
        for name, token_counts in source_methods.items():
            source_method_results[name] = {
                "active_vocab": int(active_vocabs[name]),
                "stored_vocab": int(stored_vocabs[name]),
                **method_metrics(
                    source_bytes,
                    token_counts,
                    source_methods["single_small"],
                    source_methods["single_large"],
                ),
                "bootstrap_intervals": source_bootstrap.get(name),
            }

        source_learned_routes = learned_routes[source_slice]
        source_oracle_routes = oracle_routes[source_slice]
        source_oracle_tokens = oracle_tokens[source_slice]
        source_learned_tokens = learned_tokens[source_slice]
        dataset_results[source_label] = {
            "heldout_documents": len(source_bytes),
            "heldout_bytes": int(source_bytes.sum()),
            "sampling": sampling_by_source[source_label],
            "router": {
                "heldout_oracle_agreement": float(
                    np.mean(source_learned_routes == source_oracle_routes)
                ),
                "token_regret_vs_oracle": int(source_learned_tokens.sum())
                / max(int(source_oracle_tokens.sum()), 1)
                - 1.0,
                "learned_usage": np.bincount(
                    source_learned_routes, minlength=expert_count
                ).tolist(),
                "oracle_usage": np.bincount(
                    source_oracle_routes, minlength=expert_count
                ).tolist(),
            },
            "methods": source_method_results,
        }

    oracle_agreement = float(np.mean(learned_routes == oracle_routes))
    oracle_total = int(oracle_tokens.sum())
    learned_total = int(learned_tokens.sum())
    report = {
        "run_dir": str(run_dir),
        "training_sample": str(train_sample_path),
        "training_documents": len(training_documents),
        "heldout_documents": len(heldout),
        "heldout_bytes": int(raw_bytes.sum()),
        "heldout_source": (
            json_safe_namespace(source_args_list[0])
            if len(source_args_list) == 1
            else {"source": "multiple"}
        ),
        "heldout_sources": [
            json_safe_namespace(source_args) for source_args in source_args_list
        ],
        "eval_bytes_per_source": args.eval_bytes,
        "sampling": sampling_by_source,
        "expert_count": expert_count,
        "target_vocab_sizes": {
            "single_small": small_vocab_target,
            "single_large": large_vocab_target,
            "each_disjoint_expert": int(discovery_summary["vocab_size_target"]),
        },
        "actual_disjoint_vocab_sizes": [
            tokenizer.get_vocab_size() for tokenizer in disjoint_tokenizers
        ],
        "router": {
            "training_assignment_accuracy": router_train_accuracy,
            "heldout_oracle_agreement": oracle_agreement,
            "token_regret_vs_oracle": learned_total / max(oracle_total, 1) - 1.0,
            "learned_usage": np.bincount(
                learned_routes, minlength=expert_count
            ).tolist(),
            "oracle_usage": np.bincount(oracle_routes, minlength=expert_count).tolist(),
        },
        "bootstrap": {
            "repetitions": args.bootstrap,
            "confidence": args.confidence,
        },
        "elapsed_seconds": time.time() - started,
        "methods": method_results,
        "datasets": dataset_results,
    }
    write_outputs(output_dir, heldout, methods, learned_routes, oracle_routes, report)
    for source_label, source_results in dataset_results.items():
        print(f"\nDataset: {source_label}")
        print_results(source_results["methods"])
        source_router = source_results["router"]
        print(
            f"Router oracle agreement: "
            f"{100 * source_router['heldout_oracle_agreement']:.2f}% | "
            f"token regret: {100 * source_router['token_regret_vs_oracle']:.2f}%"
        )
    if len(dataset_results) > 1:
        print("\nAggregate")
        print_results(method_results)
    print(
        f"\nRouter oracle agreement: {100 * oracle_agreement:.2f}% | "
        f"token regret: {100 * report['router']['token_regret_vs_oracle']:.2f}%"
    )
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Evaluation failed")
        raise SystemExit(1)
