"""Command-line defaults and model checkpoint configuration."""

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

QWEN = "Qwen/Qwen3-VL-8B-Instruct"
SHIELDSTRAL = "mistralai/Shieldstral-1.0-3B"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def add_model_args(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--model", default=default, help="Hugging Face repo ID or local checkpoint.")
    parser.add_argument("--revision", help="Hub branch, tag, or commit SHA.")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--seed", type=int, default=42)


def model_source(model: str, revision: str | None, previous: dict) -> dict:
    local = Path(model).expanduser()
    if local.is_dir():
        if revision is not None:
            raise SystemExit("--revision applies to Hub checkpoints")
        return {"model": str(local.resolve()), "commit": None, "requested_revision": None}

    ref = revision or "main"
    if previous.get("model") == model:
        if revision is None or revision in (previous["requested_revision"], previous["commit"]):
            return previous
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        commit = ref.lower()
    else:
        from huggingface_hub import HfApi

        commit = HfApi().model_info(model, revision=ref).sha
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Could not resolve {model} at {ref} to a commit")
    return {"model": model, "commit": commit, "requested_revision": ref}


@dataclass(frozen=True)
class InferenceConfig:
    dtype: str
    attention: str
    seed: int
    batch_size: int
    min_pixels: int
    max_pixels: int
    max_new_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


def gpu_dtype(torch, name: str):
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch build and one visible NVIDIA GPU are required.")
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 is unavailable. Use --dtype fp16 with a new output directory.")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
