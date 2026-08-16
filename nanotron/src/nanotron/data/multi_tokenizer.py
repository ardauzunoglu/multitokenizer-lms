"""Data primitives for optimizer-step-homogeneous tokenizer routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class MultiTokenizerDatasetManifest:
    format_version: int
    tokenizer_id: int
    tokenizer_name: str
    tokenizer_name_or_path: str
    tokenizer_revision: Optional[str]
    tokenizer_fingerprint: str
    original_vocab_size: int
    token_size_in_bytes: int
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]
    routing_fingerprint: str
    documents: int
    tokens: int
    source_bytes: int

    def __post_init__(self):
        if self.format_version != 1:
            raise ValueError(
                f"Unsupported multi-tokenizer manifest version: {self.format_version}"
            )
        if self.tokenizer_id < 0 or self.original_vocab_size <= 0:
            raise ValueError("Manifest has an invalid tokenizer ID or vocabulary size")
        if self.token_size_in_bytes not in (1, 2, 4, 8):
            raise ValueError(
                f"Unsupported token storage width: {self.token_size_in_bytes}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "MultiTokenizerDatasetManifest":
        with Path(path).open() as stream:
            return cls(**json.load(stream))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as stream:
            json.dump(asdict(self), stream, indent=2, sort_keys=True)
            stream.write("\n")


def validate_source_manifest(
    manifest: MultiTokenizerDatasetManifest, vocabulary
) -> None:
    if manifest.tokenizer_id != vocabulary.id:
        raise ValueError(
            f"Dataset tokenizer ID {manifest.tokenizer_id} does not match registry ID {vocabulary.id}"
        )
    if manifest.tokenizer_fingerprint != vocabulary.fingerprint:
        raise ValueError(
            f"Dataset fingerprint for tokenizer {vocabulary.name!r} does not match the registry"
        )
    if manifest.original_vocab_size != vocabulary.original_vocab_size:
        raise ValueError(
            f"Dataset vocabulary size for tokenizer {vocabulary.name!r} does not match the registry"
        )


class MultiTokenizerDataset:
    """Own one already-packed dataset per tokenizer without forming a union ID space."""

    def __init__(self, datasets: Mapping[int, Sequence[Any]]):
        if not datasets:
            raise ValueError("At least one tokenizer dataset is required")
        ids = sorted(datasets)
        if ids != list(range(len(ids))):
            raise ValueError(
                f"Dataset tokenizer IDs must be contiguous from zero; got {ids}"
            )
        self.datasets = dict(datasets)
        self.consumed_samples = {tokenizer_id: 0 for tokenizer_id in ids}

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def get(self, tokenizer_id: int, sample_index: int) -> dict:
        dataset = self.datasets[tokenizer_id]
        sample = dataset[sample_index % len(dataset)]
        self.consumed_samples[tokenizer_id] += 1
        result = dict(sample)
        result["tokenizer_id"] = tokenizer_id
        return result

    @property
    def lengths(self) -> dict[int, int]:
        return {
            tokenizer_id: len(dataset)
            for tokenizer_id, dataset in self.datasets.items()
        }


class TokenizerTaggedDataset:
    """Add immutable routing metadata to samples from one local-ID dataset."""

    def __init__(self, dataset, tokenizer_id: int):
        self.dataset = dataset
        self.tokenizer_id = tokenizer_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["tokenizer_id"] = self.tokenizer_id
        return sample


def _stable_tie_rank(seed: int, step: int, tokenizer_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{step}:{tokenizer_id}".encode()).digest()


class HomogeneousTokenizerStepSampler:
    """Deterministically select one tokenizer for an entire optimizer step."""

    VERSION = 1

    def __init__(
        self,
        weights: Sequence[float],
        *,
        strategy: str = "weighted_deficit",
        seed: int = 0,
        weight_unit: str = "tokens",
        bytes_per_token: Optional[Sequence[float]] = None,
        start_step: int = 0,
    ):
        if len(weights) < 2 or any(weight <= 0 for weight in weights):
            raise ValueError("At least two positive tokenizer weights are required")
        if strategy not in ("weighted_deficit", "round_robin"):
            raise ValueError(f"Unsupported tokenizer schedule strategy: {strategy}")
        if weight_unit == "source_bytes":
            if (
                bytes_per_token is None
                or len(bytes_per_token) != len(weights)
                or any(value <= 0 for value in bytes_per_token)
            ):
                raise ValueError(
                    "source_bytes weights require one positive bytes_per_token value per tokenizer"
                )
            weights = [
                weight / compression
                for weight, compression in zip(weights, bytes_per_token)
            ]
        elif weight_unit != "tokens":
            raise ValueError(f"Unsupported tokenizer weight unit: {weight_unit}")
        total = float(sum(weights))
        self.weights = tuple(float(weight) / total for weight in weights)
        self.strategy = strategy
        self.seed = int(seed)
        self.step = 0
        self.deficits = [0.0] * len(weights)
        self.active_steps = [0] * len(weights)
        if start_step:
            self.advance(start_step)

    def next_tokenizer(self) -> int:
        if self.strategy == "round_robin":
            selected = (self.step + self.seed) % len(self.weights)
        else:
            self.deficits = [
                deficit + weight for deficit, weight in zip(self.deficits, self.weights)
            ]
            maximum = max(self.deficits)
            candidates = [
                idx
                for idx, deficit in enumerate(self.deficits)
                if abs(deficit - maximum) < 1e-12
            ]
            selected = min(
                candidates, key=lambda idx: _stable_tie_rank(self.seed, self.step, idx)
            )
            self.deficits[selected] -= 1.0
        self.active_steps[selected] += 1
        self.step += 1
        return selected

    def advance(self, count: int) -> list[int]:
        return [self.next_tokenizer() for _ in range(count)]

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "strategy": self.strategy,
            "seed": self.seed,
            "weights": list(self.weights),
            "step": self.step,
            "deficits": list(self.deficits),
            "active_steps": list(self.active_steps),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        immutable = (
            state["version"],
            state["strategy"],
            state["seed"],
            tuple(state["weights"]),
        )
        expected = (self.VERSION, self.strategy, self.seed, self.weights)
        if immutable != expected:
            raise ValueError(
                f"Tokenizer scheduler state is incompatible: expected {expected}, got {immutable}"
            )
        self.step = int(state["step"])
        self.deficits = [float(value) for value in state["deficits"]]
        self.active_steps = [int(value) for value in state["active_steps"]]


class HomogeneousDataLoader:
    """Route all gradient-accumulation microbatches to one dataloader."""

    def __init__(
        self,
        dataloaders: Mapping[int, Any],
        scheduler: HomogeneousTokenizerStepSampler,
        microbatches_per_step: int,
    ):
        if sorted(dataloaders) != list(range(len(dataloaders))):
            raise ValueError("Dataloaders must be keyed by contiguous tokenizer IDs")
        if microbatches_per_step <= 0:
            raise ValueError("microbatches_per_step must be positive")
        self.dataloaders = dict(dataloaders)
        self.iterators = {
            tokenizer_id: iter(loader) for tokenizer_id, loader in dataloaders.items()
        }
        self.scheduler = scheduler
        self.microbatches_per_step = microbatches_per_step
        self.microbatch_in_step = 0
        self.active_tokenizer_id = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.microbatch_in_step == 0:
            self.active_tokenizer_id = self.scheduler.next_tokenizer()
        tokenizer_id = self.active_tokenizer_id
        try:
            batch = next(self.iterators[tokenizer_id])
        except StopIteration:
            self.iterators[tokenizer_id] = iter(self.dataloaders[tokenizer_id])
            batch = next(self.iterators[tokenizer_id])
        self.microbatch_in_step = (
            self.microbatch_in_step + 1
        ) % self.microbatches_per_step
        return batch

    def state_dict(self) -> dict:
        if self.microbatch_in_step != 0:
            raise RuntimeError(
                "Tokenizer dataloader state can only be checkpointed between optimizer steps"
            )
        return self.scheduler.state_dict()


@dataclass
class StepRoutingContext:
    active_tokenizer_id: Optional[int] = None
    microbatches_seen: int = 0
    _expected_microbatches: Optional[int] = field(default=None, repr=False)

    def begin_step(self, tokenizer_id: int, expected_microbatches: int) -> None:
        if self.active_tokenizer_id is not None:
            raise RuntimeError("A tokenizer-routed optimizer step is already active")
        self.active_tokenizer_id = tokenizer_id
        self.microbatches_seen = 0
        self._expected_microbatches = expected_microbatches

    def validate_microbatch(self, tokenizer_id: int) -> None:
        if tokenizer_id != self.active_tokenizer_id:
            raise ValueError(
                f"Microbatch tokenizer {tokenizer_id} does not match scheduled tokenizer {self.active_tokenizer_id}"
            )
        self.microbatches_seen += 1

    def end_step(self) -> int:
        if self.active_tokenizer_id is None:
            raise RuntimeError("No tokenizer-routed optimizer step is active")
        if self.microbatches_seen != self._expected_microbatches:
            raise ValueError(
                f"Expected {self._expected_microbatches} microbatches, observed {self.microbatches_seen}"
            )
        tokenizer_id = self.active_tokenizer_id
        self.active_tokenizer_id = None
        self._expected_microbatches = None
        return tokenizer_id
