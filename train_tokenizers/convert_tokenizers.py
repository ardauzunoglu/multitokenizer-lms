#!/usr/bin/env python3
"""Convert raw ``tokenizers`` BPE artifacts to Hugging Face tokenizer folders.

``create_domains.py`` writes one ``tokenizer.json`` per expert.  Nanotron's
multi-tokenizer data preprocessor loads tokenizers with
``AutoTokenizer.from_pretrained``, which requires the Hugging Face metadata
written by this tool.

Example:
    python convert_tokenizers.py \
      --input-dir runs_v1/dclm-stack-50m50m/tokenizers \
      --output-dir runs_v1/dclm-stack-50m50m/hf_tokenizers
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}


@dataclass(frozen=True)
class ConvertedTokenizer:
    name: str
    source: str
    destination: str
    vocab_size: int
    special_token_ids: dict[str, int]


def discover_tokenizers(input_dir: Path) -> list[Path]:
    """Return expert directories in deterministic order."""
    experts = sorted(
        (path for path in input_dir.glob("expert_*") if (path / "tokenizer.json").is_file()),
        key=lambda path: path.name,
    )
    if not experts:
        raise FileNotFoundError(
            f"No expert_*/tokenizer.json artifacts found below {input_dir}"
        )
    return experts


def convert_one(source_dir: Path, output_dir: Path, overwrite: bool) -> ConvertedTokenizer:
    """Convert one raw tokenizer and verify its saved Hugging Face artifact."""
    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError as error:
        raise RuntimeError(
            "Missing dependencies. Install `tokenizers` and `transformers` in the active environment."
        ) from error

    source = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
    missing = [token for token in SPECIAL_TOKENS.values() if source.token_to_id(token) is None]
    if missing:
        raise ValueError(
            f"{source_dir / 'tokenizer.json'} is missing required special token(s): {missing}"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=source, **SPECIAL_TOKENS)
    tokenizer.save_pretrained(output_dir)

    reloaded = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=False)
    source_vocab_size = source.get_vocab_size(with_added_tokens=True)
    if len(reloaded) != source_vocab_size:
        raise RuntimeError(
            f"Vocabulary changed while converting {source_dir.name}: "
            f"source={source_vocab_size}, saved={len(reloaded)}"
        )

    special_token_ids = {
        name: int(getattr(reloaded, f"{name}_id"))
        for name in ("unk_token", "bos_token", "eos_token", "pad_token")
    }
    return ConvertedTokenizer(
        name=source_dir.name,
        source=str(source_dir),
        destination=str(output_dir),
        vocab_size=len(reloaded),
        special_token_ids=special_token_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing expert_*/tokenizer.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for Hugging Face tokenizer folders"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing converted expert directories"
    )
    args = parser.parse_args()

    experts = discover_tokenizers(args.input_dir)
    converted = [
        convert_one(expert, args.output_dir / expert.name, args.overwrite)
        for expert in experts
    ]
    manifest = {
        "format_version": 1,
        "tokenizers": [asdict(entry) for entry in converted],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Converted {len(converted)} tokenizer(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
