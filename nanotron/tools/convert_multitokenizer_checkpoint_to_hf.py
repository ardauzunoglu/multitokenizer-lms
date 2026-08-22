#!/usr/bin/env python3
"""Convert a TP=PP=1 Nanotron single- or multi-tokenizer checkpoint to HF."""

from __future__ import annotations

import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaConfig as HFLlamaConfig, LlamaForCausalLM

from nanotron.config import LlamaConfig as NanotronLlamaConfig
from hf_multitokenizer.configuration_multitokenizer_llama import MultiTokenizerLlamaConfig
from hf_multitokenizer.modeling_multitokenizer_llama import MultiTokenizerLlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tokenizer-id", type=int, default=0)
    parser.add_argument(
        "--export-mode",
        choices=("auto", "single", "multi", "endpoint"),
        default="auto",
        help=(
            "auto-detect checkpoint type (default); export a normal single-tokenizer Llama; "
            "export all multi-tokenizer banks; or extract one standard Llama endpoint from a multi checkpoint"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_checkpoint_config(checkpoint_path: Path) -> tuple[NanotronLlamaConfig, dict, dict | None]:
    with (checkpoint_path / "model_config.json").open() as stream:
        model_config = NanotronLlamaConfig(**json.load(stream))
    with (checkpoint_path / "config.yaml").open() as stream:
        training_config = yaml.safe_load(stream)
    manifest_path = checkpoint_path / "multi_tokenizer_manifest.json"
    tokenizer_manifest = None
    if manifest_path.is_file():
        with manifest_path.open() as stream:
            tokenizer_manifest = json.load(stream)
    return model_config, training_config, tokenizer_manifest


def validate_checkpoint_layout(checkpoint_path: Path, training_config: dict) -> None:
    parallelism = training_config.get("parallelism", {})
    if parallelism.get("tp") != 1 or parallelism.get("pp") != 1:
        raise ValueError(
            "This converter currently supports TP=1 and PP=1 checkpoints only; "
            f"got tp={parallelism.get('tp')}, pp={parallelism.get('pp')}"
        )
    if not (checkpoint_path / "model").is_dir():
        raise FileNotFoundError(f"Missing model weights directory: {checkpoint_path / 'model'}")


def resolve_tokenizer(training_config: dict, tokenizer_manifest: dict, tokenizer_id: int) -> tuple[str, dict]:
    specs = training_config.get("multi_tokenizer", {}).get("tokenizers", [])
    spec = next((item for item in specs if item["id"] == tokenizer_id), None)
    if spec is None:
        raise ValueError(f"Tokenizer ID {tokenizer_id} is not present in checkpoint config")
    vocabulary = next((item for item in tokenizer_manifest["ordered_tokenizers"] if item["id"] == tokenizer_id), None)
    if vocabulary is None:
        raise ValueError(f"Tokenizer ID {tokenizer_id} is not present in checkpoint manifest")
    return spec["tokenizer_name_or_path"], vocabulary


def resolve_single_tokenizer(training_config: dict) -> str:
    tokenizer = training_config.get("tokenizer") or {}
    tokenizer_path = tokenizer.get("tokenizer_name_or_path")
    if not tokenizer_path:
        raise ValueError("Single-tokenizer checkpoint config has no tokenizer.tokenizer_name_or_path")
    return tokenizer_path


def hf_config_from_nanotron(config: NanotronLlamaConfig) -> HFLlamaConfig:
    return HFLlamaConfig(
        attention_bias=config.attention_bias,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        hidden_act=config.hidden_act,
        hidden_size=config.hidden_size,
        initializer_range=config.initializer_range,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        pad_token_id=config.pad_token_id,
        pretraining_tp=config.pretraining_tp,
        rms_norm_eps=config.rms_norm_eps,
        rope_scaling=config.rope_scaling,
        rope_theta=config.rope_theta,
        tie_word_embeddings=config.tie_word_embeddings,
        use_cache=config.use_cache,
        vocab_size=config.vocab_size,
    )


def multi_hf_config_from_nanotron(config: NanotronLlamaConfig, tokenizer_manifest: dict) -> MultiTokenizerLlamaConfig:
    registry = [
        {
            "id": item["id"],
            "name": item["name"],
            "vocab_size": item["original_vocab_size"],
            "special_token_ids": item["special_token_ids"],
        }
        for item in tokenizer_manifest["ordered_tokenizers"]
    ]
    hf_config = MultiTokenizerLlamaConfig(
        attention_bias=config.attention_bias,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        hidden_act=config.hidden_act,
        hidden_size=config.hidden_size,
        initializer_range=config.initializer_range,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        pad_token_id=config.pad_token_id,
        pretraining_tp=config.pretraining_tp,
        rms_norm_eps=config.rms_norm_eps,
        rope_scaling=config.rope_scaling,
        rope_theta=config.rope_theta,
        # Endpoint embedding/head weights are distinct modules in HF, but have
        # identical values because Nanotron ties each endpoint pair.
        tie_word_embeddings=False,
        use_cache=config.use_cache,
        vocab_size=config.vocab_size,
        tokenizer_registry=registry,
    )
    hf_config.auto_map = {
        "AutoConfig": "configuration_multitokenizer_llama.MultiTokenizerLlamaConfig",
        "AutoModelForCausalLM": "modeling_multitokenizer_llama.MultiTokenizerLlamaForCausalLM",
    }
    return hf_config


def hf_to_nanotron_weight_mapping(config: NanotronLlamaConfig) -> dict[str, str]:
    mapping = {
        "lm_head.weight": "model.lm_head.pp_block.weight",
        "model.embed_tokens.weight": "model.token_position_embeddings.pp_block.token_embedding.weight",
        "model.norm.weight": "model.final_layer_norm.pp_block.weight",
    }
    for layer_idx in range(config.num_hidden_layers):
        hf_prefix = f"model.layers.{layer_idx}"
        nt_prefix = f"model.decoder.{layer_idx}.pp_block"
        mapping.update(
            {
                f"{hf_prefix}.self_attn.q_proj.weight": f"{nt_prefix}.attn.qkv_proj.weight",
                f"{hf_prefix}.self_attn.k_proj.weight": f"{nt_prefix}.attn.qkv_proj.weight",
                f"{hf_prefix}.self_attn.v_proj.weight": f"{nt_prefix}.attn.qkv_proj.weight",
                f"{hf_prefix}.self_attn.o_proj.weight": f"{nt_prefix}.attn.o_proj.weight",
                f"{hf_prefix}.mlp.gate_proj.weight": f"{nt_prefix}.mlp.gate_up_proj.weight",
                f"{hf_prefix}.mlp.up_proj.weight": f"{nt_prefix}.mlp.gate_up_proj.weight",
                f"{hf_prefix}.mlp.down_proj.weight": f"{nt_prefix}.mlp.down_proj.weight",
                f"{hf_prefix}.input_layernorm.weight": f"{nt_prefix}.input_layernorm.weight",
                f"{hf_prefix}.post_attention_layernorm.weight": f"{nt_prefix}.post_attention_layernorm.weight",
            }
        )
    return mapping


def split_qkv(weight: torch.Tensor, projection: str, config: NanotronLlamaConfig) -> torch.Tensor:
    head_dim = config.hidden_size // config.num_attention_heads
    q_end = config.num_attention_heads * head_dim
    k_end = q_end + config.num_key_value_heads * head_dim
    if projection == "q":
        return weight[:q_end]
    if projection == "k":
        return weight[q_end:k_end]
    if projection == "v":
        return weight[k_end:]
    raise ValueError(f"Unknown attention projection: {projection}")


def source_weight_name(nanotron_name: str, tokenizer_id: int | None, tie_word_embeddings: bool) -> str:
    embedding = "model.token_position_embeddings.pp_block.token_embedding.weight"
    lm_head = "model.lm_head.pp_block.weight"
    if tokenizer_id is not None and nanotron_name in (embedding, lm_head):
        # Word embeddings are tied in this run, so only the embedding is stored.
        return f"model.token_position_embeddings.pp_block.token_embeddings.embeddings.{tokenizer_id}.weight"
    if tokenizer_id is None and tie_word_embeddings and nanotron_name == lm_head:
        # Standard tied single-tokenizer checkpoints likewise store one shared
        # token-embedding weight and omit a separate lm_head checkpoint file.
        return embedding
    return nanotron_name


def load_weights(
    checkpoint_path: Path,
    tokenizer_id: int | None,
    model_config: NanotronLlamaConfig,
    hf_model: LlamaForCausalLM,
) -> None:
    mapping = hf_to_nanotron_weight_mapping(model_config)

    @lru_cache(maxsize=None)
    def load_source_weight(nanotron_name: str) -> torch.Tensor:
        source_name = source_weight_name(nanotron_name, tokenizer_id, model_config.tie_word_embeddings)
        components = source_name.split(".")
        weight_dir = checkpoint_path / "model" / Path(*components[:-1])
        weight_files = sorted(weight_dir.glob("model_weight*.safetensors"))
        if len(weight_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one TP=PP=1 weight file for {source_name}, found {weight_files}"
            )
        with safe_open(weight_files[0], framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            if keys != ["data"]:
                raise ValueError(f"Unexpected safetensors keys in {weight_files[0]}: {keys}")
            return stream.get_tensor("data")

    copied = set()
    for module_name, module in hf_model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            hf_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            nanotron_name = mapping.get(hf_name)
            if nanotron_name is None:
                raise KeyError(f"No Nanotron checkpoint mapping for Hugging Face parameter {hf_name}")
            source = load_source_weight(nanotron_name)
            if "qkv_proj" in nanotron_name:
                source = split_qkv(source, module_name.rsplit(".", 1)[-1][0], model_config)
            elif "gate_up_proj" in nanotron_name:
                half = source.shape[0] // 2
                source = source[:half] if ".gate_proj" in hf_name else source[half:]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {hf_name}: checkpoint {tuple(source.shape)}, HF {tuple(parameter.shape)}"
                )
            with torch.no_grad():
                parameter.copy_(source.to(dtype=parameter.dtype))
            copied.add(hf_name)

    expected = {name for name, _ in hf_model.named_parameters()}
    missing = expected - copied
    if missing:
        raise RuntimeError(f"HF parameters were not copied: {sorted(missing)}")


def load_multi_weights(
    checkpoint_path: Path,
    model_config: NanotronLlamaConfig,
    tokenizer_manifest: dict,
    hf_model: MultiTokenizerLlamaForCausalLM,
) -> None:
    mapping = hf_to_nanotron_weight_mapping(model_config)

    @lru_cache(maxsize=None)
    def load_source_weight(nanotron_name: str) -> torch.Tensor:
        components = nanotron_name.split(".")
        weight_dir = checkpoint_path / "model" / Path(*components[:-1])
        weight_files = sorted(weight_dir.glob("model_weight*.safetensors"))
        if len(weight_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one TP=PP=1 weight file for {nanotron_name}, found {weight_files}"
            )
        with safe_open(weight_files[0], framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            if keys != ["data"]:
                raise ValueError(f"Unexpected safetensors keys in {weight_files[0]}: {keys}")
            return stream.get_tensor("data")

    copied = set()
    for module_name, module in hf_model.model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            hf_name = f"model.{module_name}.{parameter_name}" if module_name else f"model.{parameter_name}"
            nanotron_name = mapping.get(hf_name)
            if nanotron_name is None:
                raise KeyError(f"No Nanotron checkpoint mapping for Hugging Face parameter {hf_name}")
            source = load_source_weight(nanotron_name)
            if "qkv_proj" in nanotron_name:
                source = split_qkv(source, module_name.rsplit(".", 1)[-1][0], model_config)
            elif "gate_up_proj" in nanotron_name:
                half = source.shape[0] // 2
                source = source[:half] if ".gate_proj" in hf_name else source[half:]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {hf_name}: checkpoint {tuple(source.shape)}, HF {tuple(parameter.shape)}"
                )
            with torch.no_grad():
                parameter.copy_(source.to(dtype=parameter.dtype))
            copied.add(hf_name)

    expected = {f"model.{name}" for name, _ in hf_model.model.named_parameters()}
    missing = expected - copied
    if missing:
        raise RuntimeError(f"Shared-backbone HF parameters were not copied: {sorted(missing)}")

    for endpoint_index, vocabulary in enumerate(tokenizer_manifest["ordered_tokenizers"]):
        if vocabulary["id"] != endpoint_index:
            raise ValueError("Tokenizer endpoint IDs must be contiguous and ordered from zero")
        if vocabulary["original_vocab_size"] != model_config.vocab_size:
            raise ValueError(
                f"Endpoint {endpoint_index} has vocabulary size {vocabulary['original_vocab_size']}, "
                f"expected {model_config.vocab_size}"
            )
        source = load_source_weight(
            f"model.token_position_embeddings.pp_block.token_embeddings.embeddings.{endpoint_index}.weight"
        )
        for target in (hf_model.token_embeddings[endpoint_index].weight, hf_model.lm_heads[endpoint_index].weight):
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Endpoint {endpoint_index} shape mismatch: checkpoint {tuple(source.shape)}, HF {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(source.to(dtype=target.dtype))


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    output_path = args.output_path.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    if output_path.exists() and any(output_path.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_path}; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    model_config, training_config, tokenizer_manifest = load_checkpoint_config(checkpoint_path)
    validate_checkpoint_layout(checkpoint_path, training_config)
    if model_config.rope_interleaved:
        raise ValueError("rope_interleaved=True checkpoints are not supported by this converter")

    is_multi_checkpoint = training_config.get("multi_tokenizer") is not None
    if is_multi_checkpoint != (tokenizer_manifest is not None):
        raise ValueError(
            "Checkpoint config and files disagree about multi-tokenizer mode: "
            "both multi_tokenizer config and multi_tokenizer_manifest.json must be present"
        )
    export_mode = args.export_mode
    if export_mode == "auto":
        export_mode = "multi" if is_multi_checkpoint else "single"
    if export_mode in ("multi", "endpoint") and not is_multi_checkpoint:
        raise ValueError(f"--export-mode {export_mode} requires a multi-tokenizer checkpoint")
    if export_mode == "single" and is_multi_checkpoint:
        raise ValueError("Use --export-mode endpoint to export one standard Llama endpoint from a multi-tokenizer checkpoint")

    output_path.mkdir(parents=True, exist_ok=True)
    if export_mode == "single":
        tokenizer_path = resolve_single_tokenizer(training_config)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        if len(tokenizer) != model_config.vocab_size:
            raise ValueError(f"Tokenizer has {len(tokenizer)} tokens, expected {model_config.vocab_size}")
        hf_model = LlamaForCausalLM(hf_config_from_nanotron(model_config))
        load_weights(checkpoint_path, None, model_config, hf_model)
        hf_model.tie_weights()
        hf_model.to(dtype=torch.bfloat16)
        tokenizer.save_pretrained(output_path)
        hf_model.save_pretrained(output_path, safe_serialization=True)
        metadata = {
            "export_mode": "single",
            "source_checkpoint": str(checkpoint_path),
            "tokenizer_path": tokenizer_path,
            "vocab_size": model_config.vocab_size,
        }
    elif export_mode == "endpoint":
        assert tokenizer_manifest is not None
        tokenizer_path, vocabulary = resolve_tokenizer(training_config, tokenizer_manifest, args.tokenizer_id)
        if vocabulary["original_vocab_size"] != model_config.vocab_size:
            raise ValueError(
                "Endpoint vocabulary size does not match model_config.vocab_size: "
                f"{vocabulary['original_vocab_size']} != {model_config.vocab_size}"
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        if len(tokenizer) != model_config.vocab_size:
            raise ValueError(f"Tokenizer has {len(tokenizer)} tokens, expected {model_config.vocab_size}")
        hf_model = LlamaForCausalLM(hf_config_from_nanotron(model_config))
        load_weights(checkpoint_path, args.tokenizer_id, model_config, hf_model)
        hf_model.tie_weights()
        hf_model.to(dtype=torch.bfloat16)
        tokenizer.save_pretrained(output_path)
        hf_model.save_pretrained(output_path, safe_serialization=True)
        metadata = {
            "export_mode": "endpoint",
            "source_checkpoint": str(checkpoint_path),
            "tokenizer_id": args.tokenizer_id,
            "tokenizer_name": vocabulary["name"],
            "tokenizer_path": tokenizer_path,
            "vocab_size": model_config.vocab_size,
        }
    else:  # export_mode == "multi"
        assert tokenizer_manifest is not None
        endpoint_specs = training_config.get("multi_tokenizer", {}).get("tokenizers", [])
        if len(endpoint_specs) != len(tokenizer_manifest["ordered_tokenizers"]):
            raise ValueError("Checkpoint config and tokenizer manifest disagree on endpoint count")
        hf_model = MultiTokenizerLlamaForCausalLM(multi_hf_config_from_nanotron(model_config, tokenizer_manifest))
        load_multi_weights(checkpoint_path, model_config, tokenizer_manifest, hf_model)
        hf_model.to(dtype=torch.bfloat16)
        source_dir = Path(__file__).with_name("hf_multitokenizer")
        shutil.copy2(source_dir / "configuration_multitokenizer_llama.py", output_path)
        shutil.copy2(source_dir / "modeling_multitokenizer_llama.py", output_path)
        for spec in endpoint_specs:
            endpoint_id = spec["id"]
            tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_name_or_path"], trust_remote_code=False)
            vocabulary = next(item for item in tokenizer_manifest["ordered_tokenizers"] if item["id"] == endpoint_id)
            if len(tokenizer) != vocabulary["original_vocab_size"]:
                raise ValueError(f"Tokenizer {endpoint_id} has {len(tokenizer)} tokens, expected {vocabulary['original_vocab_size']}")
            tokenizer.save_pretrained(output_path / "tokenizers" / f"endpoint_{endpoint_id}")
        shutil.copy2(checkpoint_path / "multi_tokenizer_manifest.json", output_path)
        hf_model.save_pretrained(output_path, safe_serialization=True)
        metadata = {
            "export_mode": "multi",
            "source_checkpoint": str(checkpoint_path),
            "tokenizers": [
                {
                    "id": spec["id"],
                    "name": spec["name"],
                    "path": f"tokenizers/endpoint_{spec['id']}",
                    "vocab_size": spec["vocab_size"],
                }
                for spec in endpoint_specs
            ],
        }
    (output_path / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Converted {export_mode} export from checkpoint {checkpoint_path.name} to {output_path}")


if __name__ == "__main__":
    main()
