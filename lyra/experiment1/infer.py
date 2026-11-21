"""Plan or run Experiment 1 target VLM inference."""

import argparse
import json
from pathlib import Path

from .conditions import CONDITIONS, case_specs, make_plan, prompt_cases
from .config import InferenceConfig, QWEN, add_model_args, model_source, positive_int
from .storage import append_record, check_manifest, digest, output_lock, read_json, records


def case_rows(sample, genres, manifest, selected_condition: str | None):
    for case in prompt_cases(sample, genres, selected_condition):
        yield {
            **case,
            "run_id": manifest["run_id"],
            "sample_id": sample.sample_id,
            "dataset": "vlsbench",
            "category": sample.safety_labels["category"],
            "sub_category": sample.safety_labels["sub_category"],
            "image_path": sample.original_metadata["image_path"],
            "original_metadata": sample.original_metadata,
            "original_instruction": sample.instruction,
            "model": manifest["model_source"]["model"],
            "model_revision": manifest["model_source"]["commit"],
            "seed": manifest["settings"]["seed"],
            "chat_prompt": None,
            "response": None,
            "generation_config": manifest["settings"] | {"do_sample": False},
        }


def failure(row: dict, stage: str, error: str) -> dict:
    return row | {"status": "error", "failure_stage": stage, "error": error}


def generate_batch(backend, record_case, image, batch: list[dict]) -> None:
    for row in batch:
        row["generation_config"]["actual_batch_size"] = len(batch)
    try:
        responses = backend.generate(image, [row["chat_prompt"] for row in batch])
        if len(responses) != len(batch):
            raise ValueError("Target returned an unexpected number of responses")
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
    else:
        for row, response in zip(batch, responses):
            row["response"] = response
            if response.strip():
                record_case(row | {"status": "success"})
            else:
                record_case(failure(row, "empty_response", "Target returned no text"))
        return

    backend.clear_cache()
    if len(batch) > 1:
        middle = len(batch) // 2
        generate_batch(backend, record_case, image, batch[:middle])
        generate_batch(backend, record_case, image, batch[middle:])
        return
    record_case(failure(batch[0], "generation", message))


def run_inference(output, plan, samples, genres, args, settings, selected_count: int) -> int:
    from PIL import Image
    from tqdm import tqdm

    from .qwen import QwenBackend

    with output_lock(output / ".inference.lock"):
        manifest_path = output / "run.json"
        result_path = output / "inference.jsonl"
        previous = read_json(manifest_path) if manifest_path.exists() else {}
        if previous and previous["plan"]["prompt_sha256"] != plan["prompt_sha256"]:
            raise SystemExit(f"Prompt hash differs from {manifest_path}. Use a new output directory.")
        if not manifest_path.exists() and result_path.exists() and result_path.stat().st_size:
            raise SystemExit(f"Cannot resume {result_path} without {manifest_path}. Use a new output directory.")
        manifest = {
            "schema_version": 1,
            "plan": plan,
            "settings": settings.as_dict(),
            "model_source": model_source(args.model, args.revision, previous.get("model_source", {})),
        }
        manifest["run_id"] = digest(manifest)
        check_manifest(manifest_path, manifest)
        completed = set()
        for row in records(result_path, repair_tail=True):
            if row["run_id"] != manifest["run_id"]:
                raise SystemExit(f"Conflicting run_id in {result_path}. Use a new output directory.")
            if row["status"] == "success" and (args.condition is None or row["condition"] == args.condition):
                completed.add(row["case_id"])
        condition = args.condition or "all"
        print(f"Planned {selected_count:,} cases ({condition}), already complete {len(completed):,}", flush=True)
        backend = None
        failed = 0
        with (
            result_path.open("ab") as stream,
            tqdm(
                total=selected_count, initial=len(completed), desc=f"Inference ({condition})",
                unit="case", dynamic_ncols=True, disable=None, postfix="errors=0",
            ) as progress,
        ):
            def record_case(row: dict) -> None:
                nonlocal failed
                append_record(stream, row)
                if row["status"] == "error":
                    failed += 1
                    progress.set_postfix_str(f"errors={failed:,}", refresh=False)
                progress.update()

            for index, sample in enumerate(samples, 1):
                pending = [
                    row for row in case_rows(sample, genres, manifest, args.condition)
                    if row["case_id"] not in completed
                ]
                if not pending:
                    continue
                try:
                    if sample.image_path is None:
                        raise FileNotFoundError(sample.original_metadata["image_path"])
                    with Image.open(sample.image_path) as source:
                        image = source.convert("RGB")
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    for row in pending:
                        record_case(failure(row, "image_load", message))
                    continue

                with image:
                    if backend is None:
                        progress.set_description("Loading Qwen")
                        backend = QwenBackend(manifest["model_source"], settings)
                        progress.set_description(f"Inference ({condition})")
                    rendered = []
                    for row in pending:
                        try:
                            row["chat_prompt"] = backend.render(row["target_prompt"])
                        except Exception as error:
                            message = f"{type(error).__name__}: {error}"
                            record_case(failure(row, "chat_template", message))
                            continue
                        row["generation_config"]["effective"] = backend.generation_settings
                        row["runtime"] = backend.runtime
                        rendered.append(row)
                    for start in range(0, len(rendered), settings.batch_size):
                        generate_batch(
                            backend, record_case, image, rendered[start:start + settings.batch_size],
                        )
                if progress.disable and index % 25 == 0:
                    print(f"Processed up to sample {index:,}/{len(samples):,}; errors: {failed:,}", flush=True)
        print(f"Done. Errors in this invocation: {failed:,}. Results: {result_path}")
        return int(failed > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=positive_int, help="Use the first N samples for a pilot.")
    parser.add_argument("--condition", choices=CONDITIONS, help="Select one condition. Default: all.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print the plan without loading a model.")
    mode.add_argument(
        "--preview-sample-id", type=int,
        help="Print both artistic conditions for this VLSBench sample without loading a model.",
    )
    add_model_args(parser, QWEN)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--min-pixels", type=positive_int, default=4096)
    parser.add_argument("--max-pixels", type=positive_int, default=524288)
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    args = parser.parse_args()
    if args.min_pixels > args.max_pixels or args.min_pixels < 1024:
        parser.error("require 1024 <= --min-pixels <= --max-pixels")
    if not args.plan and args.preview_sample_id is None and args.output_dir is None:
        parser.error("--output-dir is required for inference")
    if args.preview_sample_id is not None and args.condition is not None:
        parser.error("--preview-sample-id shows both artistic conditions; omit --condition")

    plan, samples, genres = make_plan(args.data_dir, args.limit)
    cases_per_sample = sum(1 for _ in case_specs(samples[0].sample_id, genres, args.condition))
    selected_count = len(samples) * cases_per_sample
    if args.plan:
        report = {key: value for key, value in plan.items() if key != "samples"}
        if args.condition is not None:
            report.update(
                condition=args.condition, full_case_count=plan["case_count"], case_count=selected_count,
            )
        report["sample_count"] = len(samples)
        report["missing_images"] = sum(sample.image_path is None for sample in samples)
        report["categories"] = sorted({entry["category"] for entry in plan["samples"]}, key=str)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.preview_sample_id is not None:
        sample = next((sample for sample in samples if sample.sample_id == args.preview_sample_id), None)
        if sample is None:
            parser.error(f"VLSBench sample ID {args.preview_sample_id} is not in the selected dataset.")
        report = {
            "sample_id": sample.sample_id,
            "image_path": sample.original_metadata["image_path"],
            "prompt_sha256": plan["prompt_sha256"],
            "prompts": [case for case in prompt_cases(sample, genres) if case["genre"] is not None],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    settings = InferenceConfig(
        args.dtype, args.attention, args.seed, args.batch_size,
        args.min_pixels, args.max_pixels, args.max_new_tokens,
    )
    return run_inference(
        args.output_dir.expanduser(), plan, samples, genres, args, settings, selected_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
