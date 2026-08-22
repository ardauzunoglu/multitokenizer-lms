"""Hugging Face inference model for Nanotron's disjoint tokenizer endpoints."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel

from .configuration_multitokenizer_llama import MultiTokenizerLlamaConfig


class MultiTokenizerLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """A shared Llama backbone with one embedding/head pair per tokenizer.

    `tokenizer_id` must be an integer or a scalar tensor. A batch may contain
    only one tokenizer, matching the step-homogeneous Nanotron training setup.
    """

    config_class = MultiTokenizerLlamaConfig
    base_model_prefix = "model"

    def __init__(self, config: MultiTokenizerLlamaConfig):
        super().__init__(config)
        if not config.tokenizer_registry:
            raise ValueError("tokenizer_registry must contain at least one tokenizer")

        self.model = LlamaModel(config)
        # The backbone receives `inputs_embeds` from the selected endpoint.
        self.model.embed_tokens = nn.Identity()
        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(int(spec["vocab_size"]), config.hidden_size) for spec in config.tokenizer_registry]
        )
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, int(spec["vocab_size"]), bias=False) for spec in config.tokenizer_registry]
        )
        self.post_init()

    def _endpoint_index(self, tokenizer_id) -> int:
        if isinstance(tokenizer_id, torch.Tensor):
            ids = tokenizer_id.detach().reshape(-1)
            if ids.numel() == 0 or not torch.all(ids == ids[0]):
                raise ValueError("tokenizer_id must be a nonempty scalar or a uniform tensor")
            tokenizer_id = int(ids[0].item())
        if not isinstance(tokenizer_id, int):
            raise TypeError("tokenizer_id must be an int or a scalar tensor")
        if not 0 <= tokenizer_id < len(self.token_embeddings):
            raise ValueError(f"tokenizer_id {tokenizer_id} is outside [0, {len(self.token_embeddings)})")
        return tokenizer_id

    def get_input_embeddings(self):
        return self.token_embeddings[0]

    def get_output_embeddings(self):
        return self.lm_heads[0]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        tokenizer_id=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        endpoint_id = self._endpoint_index(tokenizer_id)
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings[endpoint_id](input_ids)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_heads[endpoint_id](hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, tokenizer_id=0, **kwargs):
        model_inputs = LlamaForCausalLM.prepare_inputs_for_generation(self, *args, **kwargs)
        model_inputs["tokenizer_id"] = tokenizer_id
        return model_inputs

