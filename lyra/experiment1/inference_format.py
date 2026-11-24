"""Store compact inference rows with reusable generation and chat profiles."""

from pathlib import Path

from .conditions import ORIGINAL
from .storage import digest, read_json, write_json


def judge_input_row(row: dict) -> dict:
    if "original_instruction" in row:
        return row
    return row | {
        "sample_id": row["id"],
        "original_instruction": row["instruction"],
        "target_prompt": row.get("target_prompt", row["instruction"]),
        "genre": row.get("genre"),
    }


class InferenceProfiles:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "inference_profiles.json"
        self.profiles = read_json(self.path) if self.path.is_file() else {}

    def compact(self, row: dict) -> dict:
        if "id" in row:
            if row["profile"] not in self.profiles:
                raise ValueError(f"Missing profile {row['profile']} in {self.path}")
            return row

        instruction = row["original_instruction"]
        prompt = row["target_prompt"]
        compact = {
            "id": row["sample_id"],
            "instruction": instruction,
            "response": row["response"],
            "image_path": row["image_path"],
            "category": row["category"],
            "sub_category": row["sub_category"],
            "case_id": row["case_id"],
            "run_id": row["run_id"],
            "condition": row["condition"],
            "status": row["status"],
        }
        if row["genre"] is not None:
            compact["genre"] = row["genre"]
        if row["condition"] != ORIGINAL or prompt != instruction:
            compact["target_prompt"] = prompt
        if row["status"] == "error":
            compact.update(failure_stage=row["failure_stage"], error=row["error"])

        generation = dict(row["generation_config"])
        actual_batch_size = generation.pop("actual_batch_size", None)
        if actual_batch_size is not None:
            compact["actual_batch_size"] = actual_batch_size
        profile = {"generation_config": generation, "runtime": row.get("runtime")}
        rendered = row.get("chat_prompt")
        if rendered is not None:
            if prompt and rendered.count(prompt) == 1:
                start, end = rendered.split(prompt, 1)
                profile["chat_wrapper"] = [start, end]
            else:
                compact["chat_prompt"] = rendered
        profile_id = digest(profile)
        if profile_id not in self.profiles:
            self.profiles[profile_id] = profile
            write_json(self.path, self.profiles)
        compact["profile"] = profile_id
        return compact
