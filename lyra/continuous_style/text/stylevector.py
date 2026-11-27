import argparse
import json
import logging
import math
import platform
from pathlib import Path

from ..common import CONFIG_DIR, RUNS_DIR, new_run_directory, write_json


LOGGER = logging.getLogger(__name__)


def chat_prefix(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def last_activation(model, tokenizer, layer, text, device):
    activation = None

    def capture(_module, _inputs, output):
        nonlocal activation
        activation = output[0, -1].float().cpu()

    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    with layer.register_forward_hook(capture):
        model.model(**inputs, use_cache=False)
    if activation is None:
        raise RuntimeError("The requested layer did not produce an activation.")
    return activation


def generate_responses(model, layer, tokenizer, inputs, vector, alpha, max_new_tokens):
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_length = inputs.input_ids.shape[1]
    stats = {"steered_decode_steps": 0, "first_decode_max_abs_change": None}

    def steer(_module, _inputs, output):
        nonlocal prefill
        if prefill:
            prefill = False
            return None
        updated = output + offset
        if stats["steered_decode_steps"] == 0:
            stats["first_decode_max_abs_change"] = (updated - output).abs().max().item()
        stats["steered_decode_steps"] += 1
        return updated

    LOGGER.info("Generating baseline with alpha=0")
    baseline = model.generate(**inputs, generation_config=generation_config)
    LOGGER.info("Generating steered response with alpha=%s", alpha)
    prefill = True
    offset = (alpha * vector).to(device=model.device, dtype=model.dtype)
    with layer.register_forward_hook(steer):
        steered = model.generate(**inputs, generation_config=generation_config)

    return (
        tokenizer.decode(baseline[0, prompt_length:], skip_special_tokens=True).strip(),
        tokenizer.decode(steered[0, prompt_length:], skip_special_tokens=True).strip(),
        stats,
    )


def run(args, config, run_dir):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("StyleVector requires a CUDA device with the matching PyTorch wheel.")

    model_path = args.model_path.expanduser().resolve()
    LOGGER.info("Loading model and tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)
    if model.config.model_type != "qwen3":
        raise ValueError("The local model must be a Qwen3 causal language model.")
    layers = model.model.layers
    layer_index = config["layer"]
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"--layer must be between 0 and {len(layers) - 1}.")
    layer = layers[layer_index]

    with torch.inference_mode():
        differences = []
        LOGGER.info("Extracting %d contrastive pairs at layer %d", len(config["pairs"]), layer_index)
        for pair in config["pairs"]:
            prefix = chat_prefix(tokenizer, pair["prompt"])
            positive = last_activation(model, tokenizer, layer, prefix + pair["positive"], device)
            negative = last_activation(model, tokenizer, layer, prefix + pair["negative"], device)
            differences.append(positive - negative)
        vector = torch.stack(differences).mean(dim=0)
        norm = vector.norm().item()
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError(f"Style vector has invalid L2 norm {norm}.")
        torch.save(vector, run_dir / "style_vector.pt")
        LOGGER.info("Extracted style vector with L2 norm %.6f", norm)

        inputs = tokenizer(
            chat_prefix(tokenizer, config["prompt"]), add_special_tokens=False, return_tensors="pt"
        ).to(device)
        baseline, steered, stats = generate_responses(
            model, layer, tokenizer, inputs, vector, config["alpha"], config["max_new_tokens"]
        )

    (run_dir / "baseline.txt").write_text(baseline + "\n", encoding="utf-8")
    (run_dir / "steered.txt").write_text(steered + "\n", encoding="utf-8")
    LOGGER.info("Steering hook statistics %s", stats)
    change = stats["first_decode_max_abs_change"]
    if change is not None and not math.isfinite(change):
        raise RuntimeError("The steering hook produced nonfinite activation changes.")
    write_json(run_dir / "run.json", {
        "model_path": str(model_path),
        "model_type": model.config.model_type,
        "architecture": type(model).__name__,
        "hidden_size": model.config.hidden_size,
        "num_layers": len(layers),
        "tokenizer": type(tokenizer).__name__,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "gpu": torch.cuda.get_device_name(model.device),
        "compute_capability": torch.cuda.get_device_capability(model.device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "config": config,
        "baseline_alpha": 0.0,
        "vector_norm": norm,
        **stats,
    })
    if not baseline or not steered:
        raise RuntimeError("Generation produced an empty response. See the saved outputs and run log.")
    if change is None or change <= 0:
        raise RuntimeError("The steering hook did not change the first decode activation.")


def main():
    parser = argparse.ArgumentParser(description="Extract a Qwen3 style vector and generate twice.")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "stylevector.json")
    parser.add_argument("--model-path", type=Path, required=True, help="Local Qwen3-8B snapshot")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "stylevector")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, help="Zero-based decoder layer index")
    parser.add_argument("--alpha", type=float, help="Steering strength, default 1.0")
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    if args.layer is not None:
        config["layer"] = args.layer
    if args.alpha is not None:
        config["alpha"] = args.alpha
    if not math.isfinite(config["alpha"]) or config["alpha"] == 0:
        parser.error("The steered alpha must be finite and nonzero. Baseline alpha is always zero.")
    if not config["pairs"] or config["max_new_tokens"] < 2:
        parser.error("Provide at least one contrastive pair and max_new_tokens >= 2.")
    if not args.model_path.expanduser().is_dir():
        parser.error(f"Model snapshot directory does not exist: {args.model_path}")

    run_dir = new_run_directory(args.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "inference.log", encoding="utf-8"), logging.StreamHandler()],
    )
    LOGGER.info("Output directory %s", run_dir)
    try:
        run(args, config, run_dir)
    except Exception:
        LOGGER.exception("StyleVector smoke test failed")
        raise SystemExit(1) from None
    LOGGER.info("StyleVector smoke test complete")


if __name__ == "__main__":
    main()
