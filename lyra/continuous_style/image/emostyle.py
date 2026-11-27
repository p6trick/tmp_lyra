import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ..common import CONFIG_DIR, RUNS_DIR, new_run_directory, write_json


def run(args, config, upstream, image_path, paths, run_dir):
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("EmoStyle requires a CUDA device with the matching PyTorch wheel.")

    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "false",
        "HF_HUB_OFFLINE": "1",
        "FLUX_DEV": str(paths["flux"]),
        "AE": str(paths["ae"]),
        "CLIP": str(paths["clip"]),
        "T5": str(paths["t5"]),
        "LORA": str(paths["lora"]),
        "PROJECTION_MODEL": str(paths["projector"]),
        "SIGLIP_PATH": str(paths["siglip"]),
    }
    with TemporaryDirectory(dir=run_dir) as work_dir:
        command = [
            sys.executable, "-u", "inference_emostyle.py",
            "--prompt", config["emotion"],
            "--emotion", config["emotion"],
            "--image_paths", str(image_path),
            "--mapper_ckpt", str(paths["mapper"]),
            "--save_path", work_dir,
            "--hf_download", "false",
            "--emo_strength", str(config["emo_strength"]),
            "--width", str(config["width"]),
            "--height", str(config["height"]),
            "--num_steps", str(config["num_steps"]),
            "--guidance", str(config["guidance"]),
            "--seed", str(config["seed"]),
            "--offload", str(args.offload).lower(),
        ]
        with (run_dir / "inference.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=upstream, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Official EmoStyle inference exited with code {result.returncode}.")
        outputs = list((Path(work_dir) / "inference").glob("*.png"))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one output PNG from EmoStyle, found {len(outputs)}.")
        official_output = outputs[0]
        with Image.open(official_output) as image:
            image.verify()
            width, height = image.size
        official_output.rename(run_dir / "stylized.png")

    write_json(run_dir / "run.json", {
        "upstream_checkout": str(upstream),
        "upstream_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
        ).strip(),
        "input_image": str(image_path),
        "model_type": "flux-dev",
        "model_paths": {key: str(path) for key, path in paths.items()},
        "emotion": config["emotion"],
        "emo_strength": config["emo_strength"],
        "seed": config["seed"],
        "guidance": config["guidance"],
        "num_steps": config["num_steps"],
        "offload": args.offload,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "official_output_name": official_output.name,
        "output_image": "stylized.png",
        "output_size": [width, height],
    })


def main():
    parser = argparse.ArgumentParser(description="Stylize one image with the official EmoStyle inference CLI.")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "emostyle.json")
    parser.add_argument("--emostyle-repo", type=Path, required=True, help="Patched official EmoStyle checkout")
    parser.add_argument("--models-dir", type=Path, required=True, help="Root of the local model files")
    parser.add_argument("--image", type=Path, default=CONFIG_DIR / "emostyle-example.png")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "emostyle")
    parser.add_argument("--emotion", help="Official target emotion, default sadness")
    parser.add_argument("--emo-strength", type=float, help="Official embedding multiplier, default 1.0")
    parser.add_argument("--offload", action="store_true", help="Use official CPU offloading")
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    if args.emotion is not None:
        config["emotion"] = args.emotion
    if args.emo_strength is not None:
        config["emo_strength"] = args.emo_strength
    if not math.isfinite(config["emo_strength"]):
        parser.error("--emo-strength must be finite.")

    upstream = args.emostyle_repo.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not (upstream / "inference_emostyle.py").is_file():
        parser.error(f"No inference_emostyle.py in {upstream}")
    if not image_path.is_file():
        parser.error(f"Input image does not exist: {image_path}")

    model_root = args.models_dir.expanduser().resolve()
    paths = {
        key: model_root / Path(value.replace("{emotion}", config["emotion"])).expanduser()
        for key, value in config["model_paths"].items()
    }
    # The official HFEmbedder distinguishes CLIP from T5 by looking at the directory name.
    for key, path in paths.items():
        exists = path.is_dir() if key in {"clip", "t5", "siglip"} else path.is_file()
        if not exists:
            parser.error(f"Missing {key} model files: {path}")

    run_dir = new_run_directory(args.output_dir)
    log_path = run_dir / "inference.log"
    print(f"EmoStyle run directory: {run_dir}\nInference log: {log_path}", flush=True)
    log_path.write_text(f"Loading {image_path} for target {config['emotion']} from {upstream}\n", encoding="utf-8")
    try:
        run(args, config, upstream, image_path, paths, run_dir)
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            print(f"EmoStyle smoke test failed: {exc}", file=log)
        parser.exit(1, f"EmoStyle smoke test failed: {exc}\nSee {log_path}\n")
    print(f"EmoStyle smoke test complete: {run_dir / 'stylized.png'}")


if __name__ == "__main__":
    main()
