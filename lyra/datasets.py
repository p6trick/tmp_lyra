"""Read local dataset metadata and expose the available image paths."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

VLSBENCH_METADATA = Path("data.json")
VLSU_METADATA = Path("data", "VLSU.csv")
VLSBENCH_LABELS = ("category", "sub_category")
VLSU_LABELS = (
    "image_grade",
    "consensus_text_grade",
    "consensus_combined_grade",
    "image_category",
    "text_category",
    "combined_category",
)


@dataclass
class Sample:
    sample_id: int | str
    image_path: Path | None
    instruction: str
    safety_labels: dict
    original_metadata: dict


def read_vlsbench_metadata(root: Path) -> list[dict]:
    with (root / VLSBENCH_METADATA).open(encoding="utf-8") as source:
        return json.load(source)


def read_vlsu_metadata(root: Path) -> list[dict]:
    with (root / VLSU_METADATA).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def vlsbench_image_file(root: Path, row: dict) -> Path:
    relative = Path(row["image_path"])
    if (
        not relative.parts
        or relative.parts[0] != "imgs"
        or ".." in relative.parts
        or not (root / relative).resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Invalid VLSBench image path: {relative}")
    return root / relative


def vlsu_image_file(root: Path, row: dict) -> Path:
    return root / "images" / str(UUID(row["uuid"]))


def has_local_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_vlsbench(data_dir: str | Path) -> list[Sample]:
    """Load VLSBench from the vlsbench/ directory under data_dir."""
    root = Path(data_dir).expanduser().resolve() / "vlsbench"
    samples = []
    for row in read_vlsbench_metadata(root):
        path = vlsbench_image_file(root, row)
        samples.append(
            Sample(
                sample_id=row["instruction_id"],
                image_path=path if has_local_file(path) else None,
                instruction=row["instruction"],
                safety_labels={field: row[field] for field in VLSBENCH_LABELS},
                original_metadata=row,
            )
        )
    return samples


def load_vlsu(data_dir: str | Path) -> list[Sample]:
    """Load VLSU from the vlsu/ directory under data_dir."""
    root = Path(data_dir).expanduser().resolve() / "vlsu"
    samples = []
    for row in read_vlsu_metadata(root):
        path = vlsu_image_file(root, row)
        samples.append(
            Sample(
                sample_id=row["uuid"],
                image_path=path if has_local_file(path) else None,
                instruction=row["prompt"],
                safety_labels={field: row[field] for field in VLSU_LABELS},
                original_metadata=row,
            )
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Read local dataset metadata.")
    parser.add_argument("dataset", choices=("vlsbench", "vlsu"))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    loader = load_vlsbench if args.dataset == "vlsbench" else load_vlsu
    samples = loader(args.data_dir)
    available = sum(sample.image_path is not None for sample in samples)
    print(f"{args.dataset}: {len(samples)} rows, {available} local images")
    example = next((sample for sample in samples if sample.image_path is not None), None)
    if example is None:
        return 1
    print(f"Sample ID: {example.sample_id}")
    print(f"Image path: {example.image_path}")
    print(f"Safety labels: {example.safety_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
