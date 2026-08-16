"""Resolved, immutable vocabulary configuration for disjoint tokenizer endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

from transformers import AutoTokenizer

from nanotron.config.config import MultiTokenizerArgs


@dataclass(frozen=True)
class SpecialTokenIds:
    bos: Optional[int]
    eos: Optional[int]
    pad: Optional[int]


@dataclass(frozen=True)
class ResolvedVocabulary:
    id: int
    name: str
    original_vocab_size: int
    padded_vocab_size: int
    special_token_ids: SpecialTokenIds
    fingerprint: str


@dataclass(frozen=True)
class ResolvedVocabularyConfig:
    mode: Literal["single", "multi"]
    vocabularies: tuple[ResolvedVocabulary, ...]
    tie_word_embeddings: bool
    mask_padded_vocab_logits: bool = True

    def __post_init__(self):
        if not self.vocabularies:
            raise ValueError("At least one resolved vocabulary is required")
        if self.mode == "multi" and len(self.vocabularies) < 2:
            raise ValueError("Multi vocabulary mode requires at least two vocabularies")

    def by_id(self, tokenizer_id: int) -> ResolvedVocabulary:
        if tokenizer_id < 0 or tokenizer_id >= len(self.vocabularies):
            raise ValueError(
                f"tokenizer_id {tokenizer_id} is outside [0, {len(self.vocabularies)})"
            )
        vocabulary = self.vocabularies[tokenizer_id]
        if vocabulary.id != tokenizer_id:
            raise RuntimeError("Resolved tokenizer registry is not ordered by ID")
        return vocabulary

    def to_manifest(self) -> dict:
        return {
            "format_version": 1,
            "ordered_tokenizers": [
                asdict(vocabulary) for vocabulary in self.vocabularies
            ],
            "tie_word_embeddings": self.tie_word_embeddings,
            "batching": "step_homogeneous",
            "schedule_version": 1,
        }


_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer.model",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def fingerprint_tokenizer(path: str, revision: Optional[str] = None) -> str:
    """Hash tokenizer artifacts and the pinned revision in a stable order."""
    root = Path(path)
    digest = hashlib.sha256()
    digest.update((revision or "").encode())
    found = False
    if root.exists():
        candidates = (
            [root] if root.is_file() else [root / name for name in _TOKENIZER_ARTIFACTS]
        )
        for candidate in candidates:
            if candidate.is_file():
                found = True
                digest.update(candidate.name.encode())
                digest.update(candidate.read_bytes())
    if not found:
        # Remote repositories are still protected against accidental renaming;
        # production configs should pin a revision and provide an explicit
        # content fingerprint.
        digest.update(path.encode())
    return f"sha256:{digest.hexdigest()}"


def _padded_size(size: int, tp_size: int, make_vocab_size_divisible_by: int) -> int:
    divisor = math.lcm(tp_size, make_vocab_size_divisible_by)
    return ((size + divisor - 1) // divisor) * divisor


def _special_id(spec_value: Optional[int], tokenizer, attribute: str) -> Optional[int]:
    return spec_value if spec_value is not None else getattr(tokenizer, attribute, None)


def resolve_multi_vocabulary(
    args: MultiTokenizerArgs,
    tp_size: int,
    make_vocab_size_divisible_by: int,
    *,
    validate_artifacts: bool = True,
) -> ResolvedVocabularyConfig:
    """Load and validate all artifacts once, before distributed initialization."""
    args.__post_init__()
    resolved = []
    for spec in args.tokenizers:
        tokenizer = None
        if validate_artifacts or spec.vocab_size is None:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.tokenizer_name_or_path,
                revision=spec.tokenizer_revision,
                trust_remote_code=False,
            )
        measured_size = len(tokenizer) if tokenizer is not None else spec.vocab_size
        if measured_size is None or measured_size <= 0:
            raise ValueError(
                f"Tokenizer {spec.name!r} has an invalid vocabulary size: {measured_size}"
            )
        if spec.vocab_size is not None and measured_size != spec.vocab_size:
            raise ValueError(
                f"Tokenizer {spec.name!r} declares vocab_size={spec.vocab_size}, but its artifact contains {measured_size} tokens"
            )
        special_ids = SpecialTokenIds(
            bos=_special_id(spec.bos_token_id, tokenizer, "bos_token_id"),
            eos=_special_id(spec.eos_token_id, tokenizer, "eos_token_id"),
            pad=_special_id(spec.pad_token_id, tokenizer, "pad_token_id"),
        )
        for kind, token_id in asdict(special_ids).items():
            if token_id is not None and not 0 <= token_id < measured_size:
                raise ValueError(
                    f"Tokenizer {spec.name!r} has out-of-range {kind}_token_id={token_id}"
                )
        measured_fingerprint = fingerprint_tokenizer(
            spec.tokenizer_name_or_path, spec.tokenizer_revision
        )
        fingerprint = spec.fingerprint or measured_fingerprint
        if (
            validate_artifacts
            and spec.fingerprint is not None
            and spec.fingerprint != measured_fingerprint
        ):
            raise ValueError(
                f"Tokenizer {spec.name!r} fingerprint mismatch: expected {spec.fingerprint}, got {measured_fingerprint}"
            )
        resolved.append(
            ResolvedVocabulary(
                id=spec.id,
                name=spec.name,
                original_vocab_size=measured_size,
                padded_vocab_size=_padded_size(
                    measured_size, tp_size, make_vocab_size_divisible_by
                ),
                special_token_ids=special_ids,
                fingerprint=fingerprint,
            )
        )
    return ResolvedVocabularyConfig(
        mode="multi",
        vocabularies=tuple(resolved),
        tie_word_embeddings=args.tie_word_embeddings,
        mask_padded_vocab_logits=args.mask_padded_vocab_logits,
    )


def validate_vocabulary_manifest(
    expected: ResolvedVocabularyConfig, manifest: dict
) -> None:
    actual = expected.to_manifest()
    if manifest != actual:
        raise ValueError(
            "Checkpoint multi-tokenizer manifest does not match the configured registry:\n"
            f"expected={json.dumps(actual, sort_keys=True)}\nactual={json.dumps(manifest, sort_keys=True)}"
        )
