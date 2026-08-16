import json

import pytest

from nanotron.config.config import (
    MultiTokenizerArgs,
    TokenizerScheduleArgs,
    TokenizerSpec,
)
from nanotron.config.vocabulary import (
    resolve_multi_vocabulary,
    validate_vocabulary_manifest,
)
from nanotron.data.multi_tokenizer import (
    HomogeneousTokenizerStepSampler,
    MultiTokenizerDataset,
    MultiTokenizerDatasetManifest,
    StepRoutingContext,
)


def registry(weights=(0.7, 0.3)):
    return MultiTokenizerArgs(
        tokenizers=[
            TokenizerSpec(
                0, "web", "/missing/web", vocab_size=5, fingerprint="sha256:web"
            ),
            TokenizerSpec(
                1, "code", "/missing/code", vocab_size=7, fingerprint="sha256:code"
            ),
        ],
        schedule=TokenizerScheduleArgs(weights=list(weights), seed=17),
    )


def test_registry_validation_and_independent_padding():
    resolved = resolve_multi_vocabulary(
        registry(), tp_size=2, make_vocab_size_divisible_by=4, validate_artifacts=False
    )
    assert [
        (v.original_vocab_size, v.padded_vocab_size) for v in resolved.vocabularies
    ] == [(5, 8), (7, 8)]
    with pytest.raises(ValueError, match="contiguous"):
        MultiTokenizerArgs(
            tokenizers=[
                TokenizerSpec(1, "web", "web"),
                TokenizerSpec(0, "code", "code"),
            ]
        )


def test_checkpoint_manifest_rejects_reordered_semantics():
    resolved = resolve_multi_vocabulary(registry(), 1, 1, validate_artifacts=False)
    manifest = resolved.to_manifest()
    validate_vocabulary_manifest(resolved, manifest)
    manifest = json.loads(json.dumps(manifest))
    manifest["ordered_tokenizers"].reverse()
    with pytest.raises(ValueError, match="does not match"):
        validate_vocabulary_manifest(resolved, manifest)


def test_weighted_deficit_is_deterministic_bounded_and_resumable():
    left = HomogeneousTokenizerStepSampler([0.7, 0.2, 0.1], seed=11)
    right = HomogeneousTokenizerStepSampler([0.7, 0.2, 0.1], seed=11)
    sequence = left.advance(100)
    assert sequence == right.advance(100)
    assert [sequence.count(i) for i in range(3)] == [70, 20, 10]

    resumed = HomogeneousTokenizerStepSampler([0.7, 0.2, 0.1], seed=11)
    resumed.load_state_dict(left.state_dict())
    assert resumed.advance(50) == left.advance(50)


def test_source_byte_weights_are_converted_to_token_share():
    sampler = HomogeneousTokenizerStepSampler(
        [0.5, 0.5], weight_unit="source_bytes", bytes_per_token=[1.0, 2.0]
    )
    sequence = sampler.advance(300)
    assert [sequence.count(0), sequence.count(1)] == [200, 100]


def test_dataset_and_step_context_never_mix_tokenizers():
    dataset = MultiTokenizerDataset(
        {0: [{"input_ids": [0, 1]}], 1: [{"input_ids": [2, 3]}]}
    )
    assert dataset.get(1, 0) == {"input_ids": [2, 3], "tokenizer_id": 1}
    context = StepRoutingContext()
    context.begin_step(1, expected_microbatches=2)
    context.validate_microbatch(1)
    with pytest.raises(ValueError, match="does not match"):
        context.validate_microbatch(0)


def test_dataset_manifest_round_trip(tmp_path):
    manifest = MultiTokenizerDatasetManifest(
        format_version=1,
        tokenizer_id=0,
        tokenizer_name="web",
        tokenizer_name_or_path="tokenizers/web",
        tokenizer_revision="abc",
        tokenizer_fingerprint="sha256:web",
        original_vocab_size=100,
        token_size_in_bytes=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        routing_fingerprint="sha256:routing",
        documents=2,
        tokens=32,
        source_bytes=128,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert MultiTokenizerDatasetManifest.load(path) == manifest
