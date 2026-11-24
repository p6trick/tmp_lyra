"""Move an existing inference.jsonl into condition-specific result files."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .conditions import CONDITIONS
from .storage import encode, inference_path, output_lock, read_json, records


def starts_with_file(path: Path, prefix: Path) -> bool:
    with path.open("rb") as existing, prefix.open("rb") as migrated:
        chunks = iter(lambda: migrated.read(1024 * 1024), b"")
        return all(existing.read(len(chunk)) == chunk for chunk in chunks)


def split_results(run_dir: Path) -> None:
    with output_lock(run_dir / ".inference.lock"):
        legacy = run_dir / "inference.jsonl"
        if not legacy.is_file():
            print(f"No legacy inference.jsonl in {run_dir}")
            return

        run_id = read_json(run_dir / "run.json")["run_id"]
        counts: Counter[str] = Counter()
        with TemporaryDirectory(dir=run_dir, prefix=".split-inference-") as temporary:
            staged = {name: inference_path(Path(temporary), name) for name in CONDITIONS}
            with ExitStack() as files:
                streams = {name: files.enter_context(path.open("wb")) for name, path in staged.items()}
                for row in records(legacy):
                    name = row["condition"]
                    if row["run_id"] != run_id or name not in streams:
                        raise ValueError(f"Unexpected run or condition for case {row['case_id']} in {legacy}")
                    streams[name].write(encode(row) + b"\n")
                    counts[name] += 1

            replacements = []
            for name, source in staged.items():
                if not counts[name]:
                    continue
                destination = inference_path(run_dir, name)
                if destination.is_file():
                    # If an earlier split stopped during publication, this file may already
                    # start with the complete legacy block. Preserve its subsequent rows.
                    if starts_with_file(destination, source):
                        continue
                    with source.open("ab") as stream:
                        for row in records(destination):
                            if row["run_id"] != run_id or row["condition"] != name:
                                raise ValueError(f"Unexpected run or condition in {destination}")
                            stream.write(encode(row) + b"\n")
                replacements.append((source, destination))

            for source, destination in replacements:
                source.replace(destination)

        backup = run_dir / "inference.legacy.bak.jsonl"
        suffix = 1
        while backup.exists():
            backup = run_dir / f"inference.legacy.{suffix}.bak.jsonl"
            suffix += 1
        legacy.rename(backup)

    for name in CONDITIONS:
        if counts[name]:
            print(f"{inference_path(run_dir, name).name}: {counts[name]:,} legacy rows")
    print(f"Legacy backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not (run_dir / "run.json").is_file():
        parser.error(f"No run.json found in {run_dir}")
    split_results(run_dir)


if __name__ == "__main__":
    main()
