"""Compact existing inference results in place, preserving compressed backups."""

import argparse
import gzip
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .conditions import CONDITIONS
from .inference_format import InferenceProfiles
from .split_results import split_results
from .storage import encode, inference_path, output_lock, read_json, records


def compact_file(
    path: Path, temporary: Path, run_id: str, condition: str, profiles: InferenceProfiles,
) -> None:
    staged = temporary / path.name
    converted = 0
    with staged.open("wb") as stream:
        for row in records(path):
            if row["run_id"] != run_id or row["condition"] != condition:
                raise ValueError(f"Unexpected run or condition in {path}")
            converted += int("id" not in row)
            stream.write(encode(profiles.compact(row)) + b"\n")
    if not converted:
        print(f"{path.name}: already compact")
        return

    backup_dir = path.parent / "inference-backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"{path.stem}.{time.time_ns()}.jsonl.gz"
    staged_backup = temporary / backup.name
    with path.open("rb") as original, gzip.open(staged_backup, "wb") as archive:
        shutil.copyfileobj(original, archive, length=1024 * 1024)
    staged_backup.replace(backup)

    before, after = path.stat().st_size, staged.stat().st_size
    staged.replace(path)
    print(f"{path.name}: {converted:,} rows compacted, {before:,} → {after:,} bytes")
    print(f"Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = run_dir / "run.json"
    if not manifest.is_file():
        parser.error(f"No run.json found in {run_dir}")

    if (run_dir / "inference.jsonl").is_file():
        split_results(run_dir)

    with output_lock(run_dir / ".inference.lock"):
        run_id = read_json(manifest)["run_id"]
        profiles = InferenceProfiles(run_dir)
        with TemporaryDirectory(dir=run_dir, prefix=".compact-inference-") as temporary:
            for name in CONDITIONS:
                path = inference_path(run_dir, name)
                if path.is_file():
                    compact_file(path, Path(temporary), run_id, name, profiles)


if __name__ == "__main__":
    main()
