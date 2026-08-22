#!/usr/bin/env python3
"""Discover disjoint tokenizer domains in streamed text datasets.

The script implements a practical, compression-first approximation to latent
domain discovery for a multi-tokenizer language model:

1. Stream a token-budgeted sample from one or more Hugging Face datasets (or
   a local JSONL file).
2. Initialize latent domains from compression-residual features distilled from
   matched small and union tokenizers.
3. Alternate weighted, overlapping BPE fitting and hard document reassignment.
4. Prune tiny experts and greedily accept MDL-improving splits/merges.
5. Optionally train a small shared causal Transformer with disjoint, tied
   embedding/LM-head tables.

The discovery objective is measured in canonical BPE token count plus a fixed
penalty for each active expert. This directly optimizes the motivating goal:
small per-domain vocabularies whose tokens are information-dense.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import logging
import math
import random
import re
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("dclm_latent_domains")
SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>"]


@dataclass
class Document:
    doc_id: int
    text: str
    raw_bytes: int
    budget_tokens: int
    source: str = ""


@dataclass(frozen=True)
class DatasetSource:
    dataset: str
    config: str | None
    split: str
    text_field: str

    @property
    def label(self) -> str:
        return self.dataset if self.config is None else f"{self.dataset}::{self.config}"


@dataclass
class Expert:
    expert_id: int
    tokenizer: Any
    directory: Path

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())


@dataclass
class DiscoveryResult:
    labels: np.ndarray
    experts: list[Expert]
    costs: np.ndarray
    objective: float
    expert_penalty: float
    responsibilities: np.ndarray | None = None
    converged: bool = False
    em_rounds: int = 0


class WhitespaceBudgetTokenizer:
    """Dependency-free token counter used for local/synthetic smoke tests."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return self._pattern.findall(text)


def require_module(module_name: str, install_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = install_hint or module_name
        raise RuntimeError(
            f"Missing optional dependency {module_name!r}. Install with: "
            f"python -m pip install {hint}"
        ) from exc


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_jsonl(path: Path, text_field: str) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if text_field in item:
                yield item


def synthetic_rows(seed: int, count: int = 10_000) -> Iterator[dict[str, str]]:
    """Generate three unlabeled lexical regimes for a fast end-to-end test."""
    rng = random.Random(seed)
    prose_subjects = ["researchers", "engineers", "students", "gardeners"]
    prose_verbs = ["examined", "described", "improved", "questioned"]
    code_names = ["buffer", "request", "token_count", "result", "node"]
    math_names = ["alpha", "beta", "theta", "lambda", "sigma"]

    for index in range(count):
        regime = index % 3
        if regime == 0:
            text = " ".join(
                f"The {rng.choice(prose_subjects)} {rng.choice(prose_verbs)} "
                f"the system in chapter {rng.randint(1, 40)}."
                for _ in range(rng.randint(4, 12))
            )
        elif regime == 1:
            name = rng.choice(code_names)
            text = "\n".join(
                [
                    f"def process_{name}({name}: list[int]) -> dict[str, int]:",
                    f"    {name}_map = {{str(i): value for i, value in enumerate({name})}}",
                    f"    return {{key: value + {rng.randint(1, 9)} for key, value in {name}_map.items()}}",
                ]
                * rng.randint(2, 5)
            )
        else:
            a, b = rng.sample(math_names, 2)
            text = "\n".join(
                f"Let {a}_{j} = \\sum_{{i=1}}^n {b}_i / ({j}+i). "
                f"Therefore, \\mathbb{{E}}[{a}_{j}] \\leq {rng.randint(2, 20)}\\epsilon."
                for j in range(1, rng.randint(5, 14))
            )
        yield {"text": text}


def parse_hf_source(
    value: str, default_split: str, default_text_field: str
) -> DatasetSource:
    """Parse DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]]."""
    parts = value.split("::")
    if not 1 <= len(parts) <= 4 or not parts[0]:
        raise ValueError(
            f"Invalid --hf-source {value!r}; expected "
            "DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]]"
        )
    parts += [""] * (4 - len(parts))
    dataset, config, split, text_field = parts
    return DatasetSource(
        dataset=dataset,
        config=config or None,
        split=split or default_split,
        text_field=text_field or default_text_field,
    )


def resolve_hf_sources(args: argparse.Namespace) -> list[DatasetSource]:
    values = getattr(args, "hf_source", None)
    if values:
        return [parse_hf_source(value, args.split, args.text_field) for value in values]
    return [
        DatasetSource(args.dataset, args.dataset_config, args.split, args.text_field)
    ]


def iter_hf_rows(
    args: argparse.Namespace, source: DatasetSource
) -> Iterable[dict[str, Any]]:
    datasets = require_module("datasets", "datasets")
    load_kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config:
        load_kwargs["name"] = source.config

    LOGGER.info("Opening streaming dataset %s[%s]", source.label, source.split)
    stream = datasets.load_dataset(**load_kwargs)
    if args.shuffle_buffer > 0:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return stream


def iter_source_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.source == "synthetic":
        return synthetic_rows(args.seed, args.synthetic_documents)

    if args.source == "jsonl":
        if args.input_jsonl is None:
            raise ValueError("--input-jsonl is required when --source=jsonl")
        return iter_jsonl(args.input_jsonl, args.text_field)

    sources = resolve_hf_sources(args)
    if len(sources) != 1:
        raise ValueError(
            "iter_source_rows() only supports one source; use sample_documents()"
        )
    return iter_hf_rows(args, sources[0])


def load_budget_tokenizer(args: argparse.Namespace) -> Any:
    if args.budget_tokenizer == "whitespace":
        return WhitespaceBudgetTokenizer()

    transformers = require_module("transformers", "transformers sentencepiece")
    LOGGER.info("Loading token-budget tokenizer %s", args.budget_tokenizer)
    return transformers.AutoTokenizer.from_pretrained(
        args.budget_tokenizer,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )


def count_budget_tokens(tokenizer: Any, text: str) -> int:
    result = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(result, "ids"):
        return len(result.ids)
    return len(result)


def truncate_to_token_budget(tokenizer: Any, text: str, budget: int) -> tuple[str, int]:
    """Return the longest character prefix whose token count is <= budget."""
    if budget <= 0 or not text:
        return "", 0
    full_count = count_budget_tokens(tokenizer, text)
    if full_count <= budget:
        return text, full_count

    low, high = 0, len(text)
    best_text, best_count = "", 0
    while low <= high:
        midpoint = (low + high) // 2
        candidate = text[:midpoint]
        count = count_budget_tokens(tokenizer, candidate)
        if count <= budget:
            best_text, best_count = candidate, count
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_text, best_count


def sample_documents(args: argparse.Namespace) -> list[Document]:
    tokenizer = load_budget_tokenizer(args)
    documents: list[Document] = []
    sampled_tokens = 0
    sampled_bytes = 0

    if args.source in {"dclm", "hf"}:
        sources = resolve_hf_sources(args)
        weights = args.source_weight or [1.0] * len(sources)
        weight_sum = sum(weights)
        quotas = [int(args.sample_tokens * weight / weight_sum) for weight in weights]
        quotas[-1] += args.sample_tokens - sum(quotas)
        states = [
            {"source": source, "rows": iter(iter_hf_rows(args, source)), "tokens": 0}
            for source in sources
        ]
    else:
        label = args.source if args.source != "jsonl" else str(args.input_jsonl)
        sources = [DatasetSource(label, None, "", args.text_field)]
        quotas = [args.sample_tokens]
        states = [
            {"source": sources[0], "rows": iter(iter_source_rows(args)), "tokens": 0}
        ]

    active = list(range(len(states)))
    while active and sampled_tokens < args.sample_tokens:
        for source_index in active.copy():
            state = states[source_index]
            remaining = quotas[source_index] - int(state["tokens"])
            if remaining <= 0:
                active.remove(source_index)
                continue
            try:
                row = next(state["rows"])
            except StopIteration:
                LOGGER.warning(
                    "Source %s exhausted after %s/%s tokens",
                    state["source"].label,
                    f'{int(state["tokens"]):,}',
                    f"{quotas[source_index]:,}",
                )
                active.remove(source_index)
                continue

            value = row.get(state["source"].text_field)
            # Most text corpora use "text", while code corpora such as The
            # Stack use "content". Make the common mixed-corpus case work
            # without forcing an otherwise-empty config/split specification.
            if (
                not isinstance(value, str)
                and state["source"].text_field == args.text_field == "text"
            ):
                value = row.get("content")
            if not isinstance(value, str):
                continue
            text = value.strip()
            if len(text.encode("utf-8")) < args.min_document_bytes:
                continue
            token_count = count_budget_tokens(tokenizer, text)
            if token_count > remaining:
                text, token_count = truncate_to_token_budget(tokenizer, text, remaining)
            raw_bytes = len(text.encode("utf-8"))
            if token_count == 0 or raw_bytes < args.min_document_bytes:
                active.remove(source_index)
                continue

            documents.append(
                Document(
                    doc_id=len(documents),
                    text=text,
                    raw_bytes=raw_bytes,
                    budget_tokens=token_count,
                    source=state["source"].label,
                )
            )
            state["tokens"] = int(state["tokens"]) + token_count
            sampled_tokens += token_count
            sampled_bytes += raw_bytes

            if len(documents) % 1_000 == 0:
                LOGGER.info(
                    "Sampled %s documents, %s/%s budget tokens",
                    f"{len(documents):,}",
                    f"{sampled_tokens:,}",
                    f"{args.sample_tokens:,}",
                )
            if args.max_documents and len(documents) >= args.max_documents:
                active.clear()
                break

    if len(documents) < 2:
        raise RuntimeError("Sampling produced fewer than two usable documents")

    LOGGER.info(
        "Sample complete: %s documents, %s tokens, %s bytes",
        f"{len(documents):,}",
        f"{sampled_tokens:,}",
        f"{sampled_bytes:,}",
    )
    return documents


def save_sample(path: Path, documents: Sequence[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def document_features(documents: Sequence[Document], dimensions: int, ngram_max: int):
    feature_extraction = require_module(
        "sklearn.feature_extraction.text", "scikit-learn"
    )
    vectorizer = feature_extraction.HashingVectorizer(
        analyzer="char",
        ngram_range=(3, ngram_max),
        n_features=dimensions,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    return vectorizer.transform(document.text for document in documents)


def compression_residual_features(
    documents: Sequence[Document],
    small_tokenizer: Any,
    union_tokenizer: Any,
    batch_size: int,
    max_features_per_document: int,
):
    """Represent documents by the union-token merges that save small-tokenizer tokens.

    The feature for union token ``t`` is its document frequency multiplied by
    the number of small-tokenizer pieces needed to encode ``t`` in isolation.
    Rows are L2 normalized so clustering follows *which* merges are useful,
    rather than merely document length. Only the strongest features in each
    document are retained to keep the sparse matrix bounded on large samples.
    """

    sparse = require_module("scipy.sparse", "scipy")
    preprocessing = require_module("sklearn.preprocessing", "scikit-learn")
    union_vocab_size = int(union_tokenizer.get_vocab_size())

    merge_savings = np.full(union_vocab_size, 0.25, dtype=np.float32)
    for token_id in range(union_vocab_size):
        try:
            piece = union_tokenizer.decode([token_id], skip_special_tokens=False)
            if piece:
                small_pieces = len(small_tokenizer.encode(piece).ids)
                merge_savings[token_id] = max(float(small_pieces - 1), 0.25)
        except Exception:
            # Individual byte-level tokens are not always independently
            # decodable. They still carry a small identity feature.
            continue

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    small_costs = np.zeros(len(documents), dtype=np.int64)
    union_costs = np.zeros(len(documents), dtype=np.int64)
    texts = [document.text for document in documents]

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        small_encoded = small_tokenizer.encode_batch(batch)
        union_encoded = union_tokenizer.encode_batch(batch)
        for offset, (small_item, union_item) in enumerate(
            zip(small_encoded, union_encoded)
        ):
            row = start + offset
            small_costs[row] = len(small_item.ids)
            union_costs[row] = len(union_item.ids)
            counts = Counter(int(token_id) for token_id in union_item.ids)
            weighted = [
                (count * float(merge_savings[token_id]), token_id)
                for token_id, count in counts.items()
            ]
            if max_features_per_document > 0:
                weighted.sort(reverse=True)
                weighted = weighted[:max_features_per_document]
            for value, token_id in weighted:
                rows.append(row)
                columns.append(token_id)
                values.append(value)

    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(documents), union_vocab_size),
        dtype=np.float32,
    )
    matrix = preprocessing.normalize(matrix, norm="l2", copy=False)
    residual_gaps = np.maximum(small_costs - union_costs, 0)
    LOGGER.info(
        "Residual features: %s nonzeros, union saves %s tokens over small teacher",
        f"{matrix.nnz:,}",
        f"{int(residual_gaps.sum()):,}",
    )
    return matrix, residual_gaps


def build_discovery_features(
    documents: Sequence[Document],
    initial_k: int,
    output_dir: Path,
    args: argparse.Namespace,
):
    initialization = getattr(args, "initialization", "residual")
    if initialization == "char":
        LOGGER.info("Building tokenizer-independent character features")
        return (
            document_features(
                documents, args.feature_dimensions, args.feature_ngram_max
            ),
            None,
        )

    teacher_dir = output_dir / "residual_teachers"
    texts = [document.text for document in documents]
    requested_union_vocab = getattr(args, "residual_union_vocab_size", None)
    union_vocab_size = requested_union_vocab or initial_k * args.vocab_size
    union_vocab_size = max(args.vocab_size, int(union_vocab_size))
    LOGGER.info(
        "Training residual teachers with vocabularies %s and %s",
        f"{args.vocab_size:,}",
        f"{union_vocab_size:,}",
    )
    small_tokenizer = train_bpe(
        texts,
        teacher_dir / "single_small",
        args.vocab_size,
        args.min_frequency,
    )
    union_tokenizer = train_bpe(
        texts,
        teacher_dir / "single_union",
        union_vocab_size,
        args.min_frequency,
    )
    return compression_residual_features(
        documents,
        small_tokenizer,
        union_tokenizer,
        args.score_batch_size,
        getattr(args, "residual_features_per_document", 256),
    )


def initialize_labels(features: Any, k: int, seed: int) -> np.ndarray:
    cluster = require_module("sklearn.cluster", "scikit-learn")
    k = max(1, min(k, features.shape[0]))
    if k == 1:
        return np.zeros(features.shape[0], dtype=np.int64)
    model = cluster.MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=5,
        batch_size=min(max(256, k * 16), features.shape[0]),
        reassignment_ratio=0.01,
    )
    return model.fit_predict(features).astype(np.int64)


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    mapping = {old: new for new, old in enumerate(sorted(np.unique(labels).tolist()))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def one_hot_responsibilities(
    labels: np.ndarray, expert_count: int | None = None
) -> np.ndarray:
    labels = relabel_contiguous(labels)
    k = expert_count or int(labels.max()) + 1
    responsibilities = np.zeros((len(labels), k), dtype=np.float64)
    responsibilities[np.arange(len(labels)), labels] = 1.0
    return responsibilities


def weighted_document_counts(
    weights: np.ndarray, oversample: float, seed: int
) -> np.ndarray:
    """Low-variance integer approximation to fractional document weights."""

    expected = np.asarray(weights, dtype=np.float64) * oversample
    counts = np.floor(expected).astype(np.int64)
    residual = expected - counts
    remaining = max(0, int(round(float(expected.sum()))) - int(counts.sum()))
    positive = np.flatnonzero(residual > 0)
    if remaining and positive.size:
        rng = np.random.default_rng(seed)
        remaining = min(remaining, int(positive.size))
        probabilities = residual[positive]
        probabilities /= probabilities.sum()
        selected = rng.choice(
            positive, size=remaining, replace=False, p=probabilities
        )
        counts[selected] += 1
    return counts


def weighted_training_texts(
    documents: Sequence[Document],
    weights: np.ndarray,
    oversample: float,
    seed: int,
) -> list[str]:
    counts = weighted_document_counts(weights, oversample, seed)
    texts = [
        document.text
        for document, repetitions in zip(documents, counts)
        for _ in range(int(repetitions))
    ]
    if texts:
        return texts
    # A positive-mass expert must always have at least one training document.
    return [documents[int(np.argmax(weights))].text]


def train_bpe(
    texts: Sequence[str], output_dir: Path, vocab_size: int, min_frequency: int
):
    tokenizers = require_module("tokenizers", "tokenizers")
    models = require_module("tokenizers.models", "tokenizers")
    trainers = require_module("tokenizers.trainers", "tokenizers")
    pre_tokenizers = require_module("tokenizers.pre_tokenizers", "tokenizers")
    decoders = require_module("tokenizers.decoders", "tokenizers")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tokenizers.Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))
    tokenizer.save(str(output_dir / "tokenizer.json"))
    return tokenizer


def fit_experts(
    documents: Sequence[Document],
    labels: np.ndarray,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
    responsibilities: np.ndarray | None = None,
    soft_training_oversample: float = 1.0,
    seed: int = 0,
) -> list[Expert]:
    labels = relabel_contiguous(labels)
    if responsibilities is None:
        responsibilities = one_hot_responsibilities(labels)
    if responsibilities.shape[0] != len(documents):
        raise ValueError("Responsibilities and documents have different lengths")
    if responsibilities.shape[1] != int(labels.max()) + 1:
        raise ValueError("Responsibilities do not match the active expert count")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("expert_*"):
        if stale.is_dir():
            shutil.rmtree(stale)

    experts: list[Expert] = []
    for expert_id in range(responsibilities.shape[1]):
        weights = responsibilities[:, expert_id]
        texts = weighted_training_texts(
            documents,
            weights,
            soft_training_oversample,
            seed + expert_id,
        )
        directory = output_dir / f"expert_{expert_id:02d}"
        LOGGER.info(
            "Training expert %d BPE on %s weighted draws (mass %.1f; %s hard docs)",
            expert_id,
            f"{len(texts):,}",
            float(weights.sum()),
            f"{int(np.sum(labels == expert_id)):,}",
        )
        effective_min_frequency = max(
            1, int(round(min_frequency * soft_training_oversample))
        )
        tokenizer = train_bpe(
            texts, directory, vocab_size, effective_min_frequency
        )
        experts.append(Expert(expert_id, tokenizer, directory))
    return experts


def token_costs(
    documents: Sequence[Document],
    experts: Sequence[Expert],
    batch_size: int,
) -> np.ndarray:
    costs = np.zeros((len(documents), len(experts)), dtype=np.int64)
    texts = [document.text for document in documents]
    for column, expert in enumerate(experts):
        for start in range(0, len(texts), batch_size):
            encoded = expert.tokenizer.encode_batch(texts[start : start + batch_size])
            costs[start : start + len(encoded), column] = [
                len(item.ids) for item in encoded
            ]
    return costs


def assignment_objective(
    costs: np.ndarray, labels: np.ndarray, penalty: float
) -> float:
    rows = np.arange(costs.shape[0])
    compression = float(costs[rows, labels].sum())
    return compression + penalty * len(np.unique(labels))


def prune_and_assign(
    documents: Sequence[Document],
    costs: np.ndarray,
    old_labels: np.ndarray,
    prior_weight: float,
    min_expert_fraction: float,
) -> np.ndarray:
    k = costs.shape[1]
    total_bytes = sum(document.raw_bytes for document in documents)
    byte_usage = np.bincount(
        old_labels,
        weights=np.asarray([document.raw_bytes for document in documents]),
        minlength=k,
    )
    priors = (byte_usage + 1.0) / (byte_usage.sum() + k)
    active = np.where(byte_usage / max(total_bytes, 1) >= min_expert_fraction)[0]
    if active.size == 0:
        active = np.asarray([int(np.argmax(byte_usage))])

    adjusted = costs[:, active].astype(np.float64)
    if prior_weight > 0:
        adjusted += prior_weight * (-np.log(priors[active]))[None, :]
    labels = active[np.argmin(adjusted, axis=1)]
    return relabel_contiguous(labels)


def soft_prune_and_assign(
    documents: Sequence[Document],
    costs: np.ndarray,
    old_labels: np.ndarray,
    prior_weight: float,
    min_expert_fraction: float,
    temperature: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard routes plus soft top-k training responsibilities.

    Costs are converted to relative regret before applying the temperature, so
    a setting such as 0.02 has the same interpretation for short and long
    documents: approximately a two-percent token-cost temperature.
    """

    k = costs.shape[1]
    total_bytes = sum(document.raw_bytes for document in documents)
    byte_usage = np.bincount(
        old_labels,
        weights=np.asarray([document.raw_bytes for document in documents]),
        minlength=k,
    )
    priors = (byte_usage + 1.0) / (byte_usage.sum() + k)
    active = np.where(byte_usage / max(total_bytes, 1) >= min_expert_fraction)[0]
    if active.size == 0:
        active = np.asarray([int(np.argmax(byte_usage))])

    adjusted = costs[:, active].astype(np.float64)
    if prior_weight > 0:
        adjusted += prior_weight * (-np.log(priors[active]))[None, :]

    labels = np.argmin(adjusted, axis=1).astype(np.int64)
    responsibilities = np.zeros_like(adjusted, dtype=np.float64)
    if temperature <= 0 or adjusted.shape[1] == 1:
        responsibilities[np.arange(len(labels)), labels] = 1.0
        return labels, responsibilities, active

    minimum = adjusted.min(axis=1, keepdims=True)
    relative_regret = (adjusted - minimum) / np.maximum(minimum, 1.0)
    logits = -relative_regret / temperature
    keep = min(max(1, top_k), adjusted.shape[1])
    if keep < adjusted.shape[1]:
        retained = np.argpartition(logits, -keep, axis=1)[:, -keep:]
        mask = np.zeros_like(logits, dtype=bool)
        mask[np.arange(len(labels))[:, None], retained] = True
        logits = np.where(mask, logits, -np.inf)
    logits -= np.max(logits, axis=1, keepdims=True)
    responsibilities = np.exp(logits)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return labels, responsibilities, active


def materialize_experts(
    experts: Sequence[Expert], output_dir: Path
) -> list[Expert]:
    """Copy a coherent EM snapshot to stable ``expert_XX`` directories."""

    tokenizers = require_module("tokenizers", "tokenizers")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("expert_*"):
        if stale.is_dir():
            shutil.rmtree(stale)

    materialized: list[Expert] = []
    for expert_id, expert in enumerate(experts):
        destination = output_dir / f"expert_{expert_id:02d}"
        shutil.copytree(expert.directory, destination)
        tokenizer = tokenizers.Tokenizer.from_file(
            str(destination / "tokenizer.json")
        )
        materialized.append(Expert(expert_id, tokenizer, destination))
    return materialized


def run_reassignment_rounds(
    documents: Sequence[Document],
    initial_labels: np.ndarray,
    work_dir: Path,
    args: argparse.Namespace,
    expert_penalty: float,
) -> DiscoveryResult:
    return fit_experts_to_convergence(
        documents,
        initial_labels,
        work_dir,
        work_dir / "pre_split_merge",
        args,
        expert_penalty,
    )


def fit_experts_to_convergence(
    documents: Sequence[Document],
    initial_labels: np.ndarray,
    work_dir: Path,
    final_dir: Path,
    args: argparse.Namespace,
    expert_penalty: float,
) -> DiscoveryResult:
    """Run weighted tokenizer EM and save one internally coherent snapshot.

    Unlike the previous refit path, this routine never trains once, changes the
    assignments, and silently returns the stale tokenizers. Each round retains
    its tokenizer files. A converged snapshot is preferred; otherwise the best
    complete snapshot is materialized and explicitly reported as unconverged.
    """

    labels = relabel_contiguous(initial_labels)
    responsibilities = one_hot_responsibilities(labels)
    previous_objective = math.inf
    max_rounds = max(1, int(args.assignment_rounds))
    # Pruning changes the parameterization and needs a fresh fit; do not let a
    # sequence of pruning events consume the entire requested EM budget.
    round_limit = max_rounds + int(labels.max()) + 1
    best: tuple[
        float,
        np.ndarray,
        np.ndarray,
        list[Expert],
        np.ndarray,
        int,
    ] | None = None
    converged_snapshot = None

    for round_index in range(round_limit):
        round_dir = work_dir / f"round_{round_index:02d}"
        experts = fit_experts(
            documents,
            labels,
            round_dir,
            args.vocab_size,
            args.min_frequency,
            responsibilities=responsibilities,
            soft_training_oversample=getattr(
                args, "soft_training_oversample", 2.0
            ),
            seed=args.seed + 10_000 * round_index,
        )
        costs = token_costs(documents, experts, args.score_batch_size)
        temperature = getattr(args, "soft_assignment_temperature", 0.02) * (
            getattr(args, "soft_temperature_decay", 0.7) ** round_index
        )
        new_labels, new_responsibilities, active = soft_prune_and_assign(
            documents,
            costs,
            labels,
            args.assignment_prior_weight,
            args.min_expert_fraction,
            temperature,
            getattr(args, "soft_top_k", 2),
        )

        if len(active) != len(experts):
            LOGGER.info(
                "Round %d pruned %d expert(s); refitting the reduced family",
                round_index,
                len(experts) - len(active),
            )
            labels = new_labels
            responsibilities = new_responsibilities
            continue

        used = np.unique(new_labels)
        if len(used) != len(experts):
            LOGGER.info(
                "Round %d produced %d empty hard-route expert(s); refitting",
                round_index,
                len(experts) - len(used),
            )
            new_responsibilities = new_responsibilities[:, used]
            new_responsibilities /= new_responsibilities.sum(
                axis=1, keepdims=True
            )
            labels = relabel_contiguous(new_labels)
            responsibilities = new_responsibilities
            continue

        changed = float(np.mean(new_labels != labels))
        responsibility_delta = float(
            np.max(np.abs(new_responsibilities - responsibilities))
        )
        objective = assignment_objective(costs, new_labels, expert_penalty)
        snapshot = (
            objective,
            new_labels.copy(),
            # These are the weights that actually trained this tokenizer
            # snapshot. The newly inferred responsibilities seed the next
            # round, but must not be reported as if they trained this one.
            responsibilities.copy(),
            experts,
            costs.copy(),
            round_index + 1,
        )
        if best is None or objective < best[0]:
            best = snapshot

        LOGGER.info(
            "Round %d: K=%d, changed=%.2f%%, responsibility delta=%.5f, "
            "objective=%.1f",
            round_index,
            len(experts),
            100 * changed,
            responsibility_delta,
            objective,
        )

        stable = (
            changed <= args.assignment_tolerance
            and responsibility_delta
            <= getattr(args, "responsibility_tolerance", 0.01)
            and abs(previous_objective - objective) < 1.0
        )
        labels = new_labels
        responsibilities = new_responsibilities
        previous_objective = objective
        if stable:
            converged_snapshot = snapshot
            break
        if round_index + 1 >= max_rounds and best is not None:
            # Finish the requested rounds unless pruning required extra fits.
            break

    chosen = converged_snapshot or best
    if chosen is None:
        raise RuntimeError("Tokenizer EM produced no complete expert snapshot")
    objective, labels, responsibilities, experts, _, rounds = chosen
    converged = converged_snapshot is not None
    if not converged:
        LOGGER.warning(
            "Tokenizer EM did not reach the configured tolerances; "
            "materializing the best complete round"
        )

    experts = materialize_experts(experts, final_dir)
    costs = token_costs(documents, experts, args.score_batch_size)
    objective = assignment_objective(costs, labels, expert_penalty)
    return DiscoveryResult(
        labels=labels,
        experts=experts,
        costs=costs,
        objective=objective,
        expert_penalty=expert_penalty,
        responsibilities=responsibilities,
        converged=converged,
        em_rounds=rounds,
    )


def split_proposals(
    documents: Sequence[Document],
    features: Any,
    residual_gaps: np.ndarray | None,
    result: DiscoveryResult,
    work_dir: Path,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.split_proposals <= 0:
        return None

    cluster = require_module("sklearn.cluster", "scikit-learn")
    rows = np.arange(len(documents))
    assigned_cost = result.costs[rows, result.labels]
    candidates: list[tuple[float, int]] = []
    for expert_id in np.unique(result.labels):
        mask = result.labels == expert_id
        if int(mask.sum()) < 2 * args.min_split_documents:
            continue
        if residual_gaps is not None and float(residual_gaps[mask].sum()) > 0:
            # Prefer clusters responsible for the largest absolute gap to the
            # union teacher, not merely clusters with a poor token/byte ratio.
            priority = float(residual_gaps[mask].sum())
        else:
            bytes_total = sum(
                document.raw_bytes
                for document, keep in zip(documents, mask)
                if keep
            )
            priority = float(assigned_cost[mask].sum()) / max(bytes_total, 1)
        candidates.append((priority, int(expert_id)))
    candidates.sort(reverse=True)

    best: tuple[float, np.ndarray] | None = None
    for _, expert_id in candidates[: args.split_proposals]:
        indices = np.where(result.labels == expert_id)[0]
        splitter = cluster.MiniBatchKMeans(
            n_clusters=2,
            random_state=args.seed + expert_id,
            n_init=5,
            batch_size=min(512, len(indices)),
        )
        local_labels = splitter.fit_predict(features[indices])
        if min(np.bincount(local_labels, minlength=2)) < args.min_split_documents:
            continue

        tokenizers = []
        for side in (0, 1):
            texts = [
                documents[index].text
                for index, local in zip(indices, local_labels)
                if local == side
            ]
            tokenizer = train_bpe(
                texts,
                work_dir / f"split_{expert_id:02d}_{side}",
                args.vocab_size,
                args.min_frequency,
            )
            tokenizers.append(tokenizer)

        side_costs = np.zeros((len(indices), 2), dtype=np.int64)
        texts = [documents[index].text for index in indices]
        for side, tokenizer in enumerate(tokenizers):
            side_costs[:, side] = [
                len(item.ids) for item in tokenizer.encode_batch(texts)
            ]
        chosen = np.argmin(side_costs, axis=1)
        if min(np.bincount(chosen, minlength=2)) < args.min_split_documents:
            continue

        old_tokens = float(assigned_cost[indices].sum())
        new_tokens = float(side_costs[np.arange(len(indices)), chosen].sum())
        gain = old_tokens - new_tokens - result.expert_penalty
        LOGGER.info("Split proposal expert %d: MDL gain %.1f", expert_id, gain)
        if gain > 0 and (best is None or gain > best[0]):
            proposed = result.labels.copy()
            new_id = int(result.labels.max()) + 1
            proposed[indices[chosen == 1]] = new_id
            best = (gain, relabel_contiguous(proposed))
    return None if best is None else best[1]


def merge_proposals(
    documents: Sequence[Document],
    result: DiscoveryResult,
    work_dir: Path,
    args: argparse.Namespace,
) -> np.ndarray | None:
    k = len(result.experts)
    if k <= 1 or args.merge_candidates <= 0:
        return None

    # Cross-tokenizer regret provides a cheap similarity proxy. Only the most
    # promising pairs pay the cost of training a temporary merged BPE.
    pair_proxies: list[tuple[float, int, int]] = []
    for a in range(k):
        for b in range(a + 1, k):
            ia = np.where(result.labels == a)[0]
            ib = np.where(result.labels == b)[0]
            if ia.size == 0 or ib.size == 0:
                continue
            proxy = float(
                (result.costs[ia, b] - result.costs[ia, a]).sum()
                + (result.costs[ib, a] - result.costs[ib, b]).sum()
            )
            pair_proxies.append((proxy, a, b))
    pair_proxies.sort()

    rows = np.arange(len(documents))
    assigned_cost = result.costs[rows, result.labels]
    best: tuple[float, np.ndarray] | None = None
    for _, a, b in pair_proxies[: args.merge_candidates]:
        indices = np.where((result.labels == a) | (result.labels == b))[0]
        texts = [documents[index].text for index in indices]
        merged = train_bpe(
            texts,
            work_dir / f"merge_{a:02d}_{b:02d}",
            args.vocab_size,
            args.min_frequency,
        )
        merged_tokens = float(sum(len(item.ids) for item in merged.encode_batch(texts)))
        old_tokens = float(assigned_cost[indices].sum())
        gain = old_tokens + result.expert_penalty - merged_tokens
        LOGGER.info("Merge proposal %d+%d: MDL gain %.1f", a, b, gain)
        if gain > 0 and (best is None or gain > best[0]):
            proposed = result.labels.copy()
            proposed[proposed == b] = a
            best = (gain, relabel_contiguous(proposed))
    return None if best is None else best[1]


def refit_result(
    documents: Sequence[Document],
    labels: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    expert_penalty: float,
) -> DiscoveryResult:
    return fit_experts_to_convergence(
        documents,
        labels,
        output_dir / "em_work",
        output_dir,
        args,
        expert_penalty,
    )


def discover_domains(
    documents: Sequence[Document],
    features: Any | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> DiscoveryResult:
    initial_k = min(
        args.max_experts, max(1, len(documents) // args.min_split_documents)
    )
    residual_gaps = None
    if features is None or getattr(args, "initialization", "residual") == "residual":
        features, residual_gaps = build_discovery_features(
            documents, initial_k, output_dir, args
        )
    labels = initialize_labels(features, initial_k, args.seed)
    expert_penalty = args.expert_penalty_tokens
    if expert_penalty is None:
        sampled_budget = sum(document.budget_tokens for document in documents)
        expert_penalty = max(
            float(args.vocab_size),
            sampled_budget * args.expert_penalty_fraction,
        )
    LOGGER.info("Using expert penalty %.1f pseudo-tokens", expert_penalty)

    result = run_reassignment_rounds(
        documents,
        labels,
        output_dir / "assignment",
        args,
        expert_penalty,
    )

    for pass_index in range(args.structure_passes):
        changed = False
        split_labels = split_proposals(
            documents,
            features,
            residual_gaps,
            result,
            output_dir / f"proposals_{pass_index:02d}",
            args,
        )
        if split_labels is not None:
            LOGGER.info(
                "Accepting split; refitting %d experts", len(np.unique(split_labels))
            )
            result = refit_result(
                documents,
                split_labels,
                output_dir / f"after_split_{pass_index:02d}",
                args,
                expert_penalty,
            )
            changed = True

        merge_labels = merge_proposals(
            documents,
            result,
            output_dir / f"proposals_{pass_index:02d}",
            args,
        )
        if merge_labels is not None:
            LOGGER.info(
                "Accepting merge; refitting %d experts", len(np.unique(merge_labels))
            )
            result = refit_result(
                documents,
                merge_labels,
                output_dir / f"after_merge_{pass_index:02d}",
                args,
                expert_penalty,
            )
            changed = True
        if not changed:
            break

    final_dir = output_dir / "tokenizers"
    result = refit_result(documents, result.labels, final_dir, args, expert_penalty)
    LOGGER.info(
        "Discovery complete: K=%d, objective=%.1f",
        len(result.experts),
        result.objective,
    )
    return result


_MARKER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:+-]{2,}|[^\w\s]{2,}", re.UNICODE)


def domain_markers(
    documents: Sequence[Document], labels: np.ndarray, expert_id: int, limit: int
) -> list[str]:
    global_counts: Counter[str] = Counter()
    local_counts: Counter[str] = Counter()
    local_docs = 0
    for document, label in zip(documents, labels):
        markers = {
            marker.lower() for marker in _MARKER_RE.findall(document.text[:20_000])
        }
        global_counts.update(markers)
        if int(label) == expert_id:
            local_counts.update(markers)
            local_docs += 1
    total_docs = len(documents)
    scored = []
    for marker, count in local_counts.items():
        if count < 2:
            continue
        local_rate = (count + 0.5) / (local_docs + 1.0)
        global_rate = (global_counts[marker] + 0.5) / (total_docs + 1.0)
        scored.append((math.log(local_rate / global_rate), count, marker))
    scored.sort(reverse=True)
    return [marker for _, _, marker in scored[:limit]]


def write_reports(
    output_dir: Path,
    documents: Sequence[Document],
    result: DiscoveryResult,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows = np.arange(len(documents))
    assigned_costs = result.costs[rows, result.labels]
    expert_summaries = []
    responsibilities = result.responsibilities
    if responsibilities is None:
        responsibilities = one_hot_responsibilities(
            result.labels, len(result.experts)
        )

    for expert_id, expert in enumerate(result.experts):
        indices = np.where(result.labels == expert_id)[0]
        raw_bytes = sum(documents[index].raw_bytes for index in indices)
        token_count = int(assigned_costs[indices].sum())
        previews = [
            re.sub(r"\s+", " ", documents[index].text)[: args.preview_chars]
            for index in indices[: args.preview_documents]
        ]
        expert_summaries.append(
            {
                "expert_id": expert_id,
                "documents": len(indices),
                "training_responsibility_mass": float(
                    responsibilities[:, expert_id].sum()
                ),
                "overlap_documents": int(
                    np.sum(
                        (responsibilities[:, expert_id] > 0)
                        & (result.labels != expert_id)
                    )
                ),
                "raw_bytes": raw_bytes,
                "tokens": token_count,
                "bytes_per_token": raw_bytes / max(token_count, 1),
                "vocab_size": expert.vocab_size,
                "markers": domain_markers(
                    documents, result.labels, expert_id, args.marker_count
                ),
                "previews": previews,
                "tokenizer_path": str(expert.directory / "tokenizer.json"),
            }
        )

    source_counts = Counter(document.source for document in documents)
    source_tokens: Counter[str] = Counter()
    for document in documents:
        source_tokens[document.source] += document.budget_tokens
    source_summary = [
        {"source": source, "documents": count, "budget_tokens": source_tokens[source]}
        for source, count in source_counts.items()
    ]
    summary = {
        "dataset": (
            args.dataset
            if args.source == "dclm" and not args.hf_source
            else args.source
        ),
        "sources": source_summary,
        "sample_documents": len(documents),
        "sample_budget_tokens": sum(document.budget_tokens for document in documents),
        "sample_raw_bytes": sum(document.raw_bytes for document in documents),
        "budget_tokenizer": args.budget_tokenizer,
        "active_experts": len(result.experts),
        "vocab_size_target": args.vocab_size,
        "expert_penalty_tokens": result.expert_penalty,
        "objective": result.objective,
        "em_converged": result.converged,
        "em_rounds": result.em_rounds,
        "initialization": getattr(args, "initialization", "residual"),
        "elapsed_seconds": elapsed_seconds,
        "experts": expert_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "responsibilities.npz",
        responsibilities=responsibilities.astype(np.float32),
        hard_labels=result.labels.astype(np.int64),
    )

    with (output_dir / "assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "doc_id",
                "source",
                "expert_id",
                "raw_bytes",
                "budget_tokens",
                "expert_tokens",
            ]
        )
        for index, document in enumerate(documents):
            writer.writerow(
                [
                    document.doc_id,
                    document.source,
                    int(result.labels[index]),
                    document.raw_bytes,
                    document.budget_tokens,
                    int(assigned_costs[index]),
                ]
            )

    return summary


def build_lm_chunks(
    documents: Sequence[Document],
    labels: np.ndarray,
    experts: Sequence[Expert],
    sequence_length: int,
) -> list[list[list[int]]]:
    chunks: list[list[list[int]]] = [[] for _ in experts]
    for expert_id, expert in enumerate(experts):
        stream: list[int] = []
        bos = expert.tokenizer.token_to_id("<bos>")
        eos = expert.tokenizer.token_to_id("<eos>")
        for document, label in zip(documents, labels):
            if int(label) != expert_id:
                continue
            stream.append(bos)
            stream.extend(expert.tokenizer.encode(document.text).ids)
            stream.append(eos)
        width = sequence_length + 1
        chunks[expert_id] = [
            stream[start : start + width]
            for start in range(0, len(stream) - width + 1, width)
        ]
    return chunks


def train_shared_backbone(
    documents: Sequence[Document],
    result: DiscoveryResult,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.lm_steps <= 0:
        return None

    torch = require_module("torch", "torch")
    nn = require_module("torch.nn", "torch")
    functional = require_module("torch.nn.functional", "torch")

    if args.model_dim % args.attention_heads != 0:
        raise ValueError("--model-dim must be divisible by --attention-heads")

    class DisjointTokenizerLM(nn.Module):
        def __init__(self, vocab_sizes: Sequence[int]):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(vocab, args.model_dim) for vocab in vocab_sizes]
            )
            self.positions = nn.Embedding(args.sequence_length, args.model_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=args.model_dim,
                nhead=args.attention_heads,
                dim_feedforward=args.ffn_dim,
                dropout=args.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=args.layers)
            self.final_norm = nn.LayerNorm(args.model_dim)

        def forward(self, expert_id: int, input_ids):
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = (
                self.embeddings[expert_id](input_ids)
                + self.positions(positions)[None, :, :]
            )
            causal_mask = torch.triu(
                torch.ones(
                    input_ids.shape[1],
                    input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            hidden = self.backbone(hidden, mask=causal_mask, is_causal=True)
            hidden = self.final_norm(hidden)
            return functional.linear(hidden, self.embeddings[expert_id].weight)

    chunks = build_lm_chunks(
        documents,
        result.labels,
        result.experts,
        args.sequence_length,
    )
    usable = [index for index, expert_chunks in enumerate(chunks) if expert_chunks]
    if not usable:
        raise RuntimeError(
            "No full LM training sequences were produced; lower --sequence-length or sample more tokens"
        )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = DisjointTokenizerLM([expert.vocab_size for expert in result.experts]).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    expert_weights = np.asarray(
        [len(chunks[index]) for index in usable], dtype=np.float64
    )
    expert_weights /= expert_weights.sum()
    np_rng = np.random.default_rng(args.seed)

    model.train()
    running = 0.0
    running_steps = 0
    for step in range(1, args.lm_steps + 1):
        expert_id = int(np_rng.choice(usable, p=expert_weights))
        source = chunks[expert_id]
        batch = [rng.choice(source) for _ in range(args.batch_size)]
        tensor = torch.tensor(batch, dtype=torch.long, device=device)
        logits = model(expert_id, tensor[:, :-1])
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tensor[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        running += float(loss.item())
        running_steps += 1
        if step == 1 or step % args.log_every == 0:
            LOGGER.info(
                "LM step %d/%d | expert=%d | token CE=%.4f",
                step,
                args.lm_steps,
                expert_id,
                running / running_steps,
            )
            running = 0.0
            running_steps = 0

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_sizes": [expert.vocab_size for expert in result.experts],
        "model_config": {
            "model_dim": args.model_dim,
            "ffn_dim": args.ffn_dim,
            "attention_heads": args.attention_heads,
            "layers": args.layers,
            "sequence_length": args.sequence_length,
            "dropout": args.dropout,
        },
        "tokenizer_paths": [
            str(expert.directory / "tokenizer.json") for expert in result.experts
        ],
    }
    checkpoint_path = output_dir / "joint_backbone.pt"
    torch.save(checkpoint, checkpoint_path)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "checkpoint": str(checkpoint_path),
        "parameters": parameters,
        "device": device,
        "steps": args.lm_steps,
        "chunks_per_expert": [len(item) for item in chunks],
    }
    (output_dir / "joint_backbone.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn disjoint tokenizer domains from token-budgeted text datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("data source")
    source.add_argument(
        "--source", choices=["dclm", "hf", "jsonl", "synthetic"], default="dclm"
    )
    source.add_argument("--dataset", default="mlfoundations/dclm-baseline-1.0-parquet")
    source.add_argument("--dataset-config", default=None)
    source.add_argument(
        "--hf-source",
        action="append",
        metavar="DATASET[::CONFIG[::SPLIT[::TEXT_FIELD]]]",
        help="Hugging Face source; repeat to sample multiple datasets",
    )
    source.add_argument(
        "--source-weight",
        action="append",
        type=float,
        help="Token-budget weight; supply once per --hf-source (equal by default)",
    )
    source.add_argument("--split", default="train")
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--text-field", default="text")
    source.add_argument("--shuffle-buffer", type=int, default=10_000)
    source.add_argument("--synthetic-documents", type=int, default=600)
    source.add_argument("--trust-remote-code", action="store_true")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--sample-tokens", type=int, default=1_000_000)
    sampling.add_argument("--budget-tokenizer", default="apple/DCLM-7B")
    sampling.add_argument("--max-documents", type=int, default=0)
    sampling.add_argument("--min-document-bytes", type=int, default=128)
    sampling.add_argument("--save-sample", action="store_true")

    discovery = parser.add_argument_group("domain discovery")
    discovery.add_argument("--max-experts", type=int, default=8)
    discovery.add_argument("--vocab-size", type=int, default=8_192)
    discovery.add_argument("--min-frequency", type=int, default=2)
    discovery.add_argument("--assignment-rounds", type=int, default=12)
    discovery.add_argument("--assignment-prior-weight", type=float, default=8.0)
    discovery.add_argument("--assignment-tolerance", type=float, default=0.005)
    discovery.add_argument("--responsibility-tolerance", type=float, default=0.01)
    discovery.add_argument(
        "--soft-assignment-temperature",
        type=float,
        default=0.02,
        help="Relative token-regret temperature; zero restores hard EM",
    )
    discovery.add_argument("--soft-temperature-decay", type=float, default=0.9)
    discovery.add_argument("--soft-top-k", type=int, default=2)
    discovery.add_argument(
        "--soft-training-oversample",
        type=float,
        default=2.0,
        help="Weighted corpus multiplier; two lets 50/50 documents reach both experts",
    )
    discovery.add_argument("--min-expert-fraction", type=float, default=0.01)
    discovery.add_argument("--expert-penalty-tokens", type=float, default=None)
    discovery.add_argument("--expert-penalty-fraction", type=float, default=0.005)
    discovery.add_argument("--structure-passes", type=int, default=2)
    discovery.add_argument("--split-proposals", type=int, default=2)
    discovery.add_argument("--merge-candidates", type=int, default=4)
    discovery.add_argument("--min-split-documents", type=int, default=20)
    discovery.add_argument(
        "--initialization", choices=["residual", "char"], default="residual"
    )
    discovery.add_argument(
        "--residual-union-vocab-size",
        type=int,
        default=None,
        help="Union-teacher size; defaults to initial expert count times --vocab-size",
    )
    discovery.add_argument(
        "--residual-features-per-document",
        type=int,
        default=256,
        help="Maximum nonzero union-merge features retained per document; zero keeps all",
    )
    discovery.add_argument("--feature-dimensions", type=int, default=16_384)
    discovery.add_argument("--feature-ngram-max", type=int, default=5)
    discovery.add_argument("--score-batch-size", type=int, default=256)

    lm = parser.add_argument_group("optional shared-backbone training")
    lm.add_argument("--lm-steps", type=int, default=0)
    lm.add_argument("--model-dim", type=int, default=256)
    lm.add_argument("--ffn-dim", type=int, default=1_024)
    lm.add_argument("--attention-heads", type=int, default=8)
    lm.add_argument("--layers", type=int, default=4)
    lm.add_argument("--sequence-length", type=int, default=256)
    lm.add_argument("--batch-size", type=int, default=8)
    lm.add_argument("--learning-rate", type=float, default=3e-4)
    lm.add_argument("--weight-decay", type=float, default=0.1)
    lm.add_argument("--dropout", type=float, default=0.0)
    lm.add_argument("--grad-clip", type=float, default=1.0)
    lm.add_argument("--device", default="auto")
    lm.add_argument("--log-every", type=int, default=20)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir", type=Path, default=Path("runs/dclm_latent_domains")
    )
    output.add_argument("--preview-documents", type=int, default=3)
    output.add_argument("--preview-chars", type=int, default=240)
    output.add_argument("--marker-count", type=int, default=20)
    output.add_argument("--seed", type=int, default=17)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.source == "synthetic" and args.budget_tokenizer == "apple/DCLM-7B":
        args.budget_tokenizer = "whitespace"
    if args.sample_tokens <= 0:
        parser.error("--sample-tokens must be positive")
    if args.hf_source and args.source not in {"dclm", "hf"}:
        parser.error("--hf-source requires --source=dclm or --source=hf")
    if args.source_weight and not args.hf_source:
        parser.error("--source-weight requires --hf-source")
    if args.source_weight and any(weight <= 0 for weight in args.source_weight):
        parser.error("--source-weight values must be positive")
    if args.source_weight and len(args.source_weight) != len(args.hf_source):
        parser.error("--source-weight must be supplied once per --hf-source")
    if args.max_experts <= 0 or args.vocab_size < 512:
        parser.error(
            "--max-experts must be positive and --vocab-size must be at least 512"
        )
    if args.min_split_documents < 2:
        parser.error("--min-split-documents must be at least 2")
    if args.assignment_rounds <= 0:
        parser.error("--assignment-rounds must be positive")
    if args.soft_assignment_temperature < 0:
        parser.error("--soft-assignment-temperature must be nonnegative")
    if not 0 < args.soft_temperature_decay <= 1:
        parser.error("--soft-temperature-decay must be in (0, 1]")
    if args.soft_top_k <= 0:
        parser.error("--soft-top-k must be positive")
    if args.soft_training_oversample < 1:
        parser.error("--soft-training-oversample must be at least 1")
    if args.responsibility_tolerance < 0:
        parser.error("--responsibility-tolerance must be nonnegative")
    if (
        args.residual_union_vocab_size is not None
        and args.residual_union_vocab_size < args.vocab_size
    ):
        parser.error("--residual-union-vocab-size must be at least --vocab-size")
    if args.residual_features_per_document < 0:
        parser.error("--residual-features-per-document must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arguments.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    started = time.time()
    documents = sample_documents(args)
    if args.save_sample:
        save_sample(output_dir / "sample.jsonl", documents)

    features = None
    if args.initialization == "char":
        LOGGER.info("Building tokenizer-independent character features")
        features = document_features(
            documents, args.feature_dimensions, args.feature_ngram_max
        )
    result = discover_domains(documents, features, output_dir, args)
    summary = write_reports(output_dir, documents, result, args, time.time() - started)
    lm_metadata = train_shared_backbone(documents, result, output_dir, args)
    if lm_metadata is not None:
        summary["joint_backbone"] = lm_metadata
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    LOGGER.info("Artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Run failed")
        raise SystemExit(1)