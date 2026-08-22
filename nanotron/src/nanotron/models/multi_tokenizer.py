"""Tokenizer-specific embedding and LM-head banks around a shared backbone."""

from __future__ import annotations

from typing import Union

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.config.vocabulary import ResolvedVocabularyConfig
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
)


def checked_scalar_id(tokenizer_id: Union[int, torch.Tensor], count: int) -> int:
    if isinstance(tokenizer_id, torch.Tensor):
        if tokenizer_id.numel() != 1:
            raise ValueError(
                f"tokenizer_id must be scalar, got shape {tuple(tokenizer_id.shape)}"
            )
        tokenizer_id = int(tokenizer_id.item())
    elif not isinstance(tokenizer_id, int):
        raise TypeError(
            f"tokenizer_id must be an int or scalar tensor, got {type(tokenizer_id).__name__}"
        )
    if tokenizer_id < 0 or tokenizer_id >= count:
        raise ValueError(f"tokenizer_id {tokenizer_id} is outside [0, {count})")
    return tokenizer_id


def _tag_endpoint(module: nn.Module, tokenizer_id: int, kind: str) -> None:
    for parameter in module.parameters():
        if not isinstance(parameter, NanotronParameter):
            raise TypeError(
                "Tokenizer endpoint parameters must be NanotronParameter instances"
            )
        parameter.set_metadata("tokenizer_endpoint_id", tokenizer_id)
        parameter.set_metadata("tokenizer_endpoint_kind", kind)


def _inactive_endpoint_keepalive(
    modules: nn.ModuleList, selected: int, reference: torch.Tensor
) -> torch.Tensor:
    """Link inactive endpoint parameters to the graph with identically zero gradients.

    Step-homogeneous batches exercise one tokenizer endpoint at a time.  DDP's
    ``find_unused_parameters`` path is not compatible with Nanotron's multiple
    backwards per training step, so every endpoint must appear in each graph.
    Touching one element per inactive parameter gives it a dense zero gradient
    without changing the forward result or materializing a full parameter-sized
    zero tensor.
    """
    zero = reference.new_zeros(())
    for endpoint_id, module in enumerate(modules):
        if endpoint_id == selected:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad:
                zero = zero + parameter.reshape(-1)[0].to(dtype=reference.dtype) * 0
    return zero


class DisjointEmbeddingBank(nn.Module):
    def __init__(
        self,
        vocabulary_config: ResolvedVocabularyConfig,
        hidden_size: int,
        padding_idx,
        tp_pg,
        mode,
    ):
        super().__init__()
        self.vocabulary_config = vocabulary_config
        self.embeddings = nn.ModuleList()
        for vocabulary in vocabulary_config.vocabularies:
            global_padding_idx = vocabulary.special_token_ids.pad
            if global_padding_idx is None:
                global_padding_idx = padding_idx
            shard_width = vocabulary.padded_vocab_size // tp_pg.size()
            shard_start = dist.get_rank(tp_pg) * shard_width
            local_padding_idx = (
                global_padding_idx - shard_start
                if global_padding_idx is not None
                and shard_start <= global_padding_idx < shard_start + shard_width
                else None
            )
            embedding = TensorParallelEmbedding(
                num_embeddings=vocabulary.padded_vocab_size,
                embedding_dim=hidden_size,
                padding_idx=local_padding_idx,
                pg=tp_pg,
                mode=mode,
            )
            _tag_endpoint(embedding, vocabulary.id, "embedding")
            self.embeddings.append(embedding)

    def forward(
        self, input_ids: torch.Tensor, tokenizer_id: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        selected = checked_scalar_id(tokenizer_id, len(self.embeddings))
        vocabulary = self.vocabulary_config.by_id(selected)
        if input_ids.numel() and (
            input_ids.min() < 0 or input_ids.max() >= vocabulary.original_vocab_size
        ):
            raise ValueError(
                f"Input token IDs for {vocabulary.name!r} must be in [0, {vocabulary.original_vocab_size})"
            )
        embeddings = self.embeddings[selected](input_ids)
        return embeddings + _inactive_endpoint_keepalive(self.embeddings, selected, embeddings)


class DisjointLMHeadBank(nn.Module):
    def __init__(
        self,
        vocabulary_config: ResolvedVocabularyConfig,
        hidden_size: int,
        tp_pg,
        mode: TensorParallelLinearMode,
        async_communication: bool = False,
        tp_recompute_allgather: bool = True,
    ):
        super().__init__()
        self.vocabulary_config = vocabulary_config
        self.tp_pg = tp_pg
        self.heads = nn.ModuleList()
        for vocabulary in vocabulary_config.vocabularies:
            head = TensorParallelColumnLinear(
                in_features=hidden_size,
                out_features=vocabulary.padded_vocab_size,
                pg=tp_pg,
                bias=False,
                mode=mode,
                async_communication=async_communication,
                tp_recompute_allgather=tp_recompute_allgather,
            )
            _tag_endpoint(head, vocabulary.id, "lm_head")
            self.heads.append(head)

    def forward(
        self, hidden_states: torch.Tensor, tokenizer_id: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        selected = checked_scalar_id(tokenizer_id, len(self.heads))
        vocabulary = self.vocabulary_config.by_id(selected)
        # Attach inactive heads before the selected projection to avoid an extra
        # vocab-sized operation on the logits tensor.
        hidden_states = hidden_states + _inactive_endpoint_keepalive(self.heads, selected, hidden_states)
        logits = self.heads[selected](hidden_states)
        if (
            self.vocabulary_config.mask_padded_vocab_logits
            and vocabulary.padded_vocab_size > vocabulary.original_vocab_size
        ):
            shard_width = vocabulary.padded_vocab_size // self.tp_pg.size()
            global_indices = torch.arange(
                dist.get_rank(self.tp_pg) * shard_width,
                (dist.get_rank(self.tp_pg) + 1) * shard_width,
                device=logits.device,
            )
            logits = logits.masked_fill(
                global_indices >= vocabulary.original_vocab_size, float("-inf")
            )
        return logits
