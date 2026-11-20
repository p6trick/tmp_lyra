"""Download public data releases into a user-selected directory."""

import argparse
import fcntl
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from .datasets import (
    VLSBENCH_METADATA,
    VLSU_METADATA,
    has_local_file,
    read_vlsbench_metadata,
    read_vlsu_metadata,
    vlsbench_image_file,
    vlsu_image_file,
)

VLSBENCH_REPO = "Foreshhh/vlsbench"
VLSU_REPO = "apple-aiml-research/ml-vlsu"
SOURCES = {
    "vlsbench": f"https://huggingface.co/datasets/{VLSBENCH_REPO}",
    "vlsu": f"https://github.com/{VLSU_REPO}",
}


def atomic_write(
    destination: Path, write: Callable[[BinaryIO], object], *, image: bool = False
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = NamedTemporaryFile(
        dir=destination.parent, prefix=".lyra-", suffix=".part", delete=False
    )
    partial = Path(output.name)
    try:
        with output:
            write(output)
        if image:
            with Image.open(partial):
                pass
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_url(
    url: str, destination: Path, *, attempts: int, image: bool = False
) -> None:
    if not url or not url.lower().startswith(("http://", "https://")):
        raise ValueError("Expected a public HTTP(S) URL")

    def transfer(output: BinaryIO) -> None:
        with requests.get(url, stream=True, timeout=(10, 20)) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Expected HTTP 200, received {response.status_code}")
            for chunk in response.iter_content(chunk_size=256 * 1024):
                output.write(chunk)

    for attempt in range(attempts):
        try:
            atomic_write(destination, transfer, image=image)
            return
        except requests.HTTPError as error:
            status = error.response.status_code
            if status < 500 and status not in (408, 429):
                raise
            if attempt + 1 == attempts:
                raise
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
        time.sleep(min(2 ** min(attempt, 5), 30))


def resolve_commit(dataset: str, revision: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    if dataset == "vlsbench":
        return HfApi().dataset_info(VLSBENCH_REPO, revision=revision, token=False).sha
    ref = quote(revision, safe="")
    url = f"https://api.github.com/repos/{VLSU_REPO}/commits/{ref}"
    response = requests.get(
        url, headers={"Accept": "application/vnd.github+json"}, timeout=(10, 20)
    )
    response.raise_for_status()
    return response.json()["sha"]


def get_commit(dataset: str, root: Path, revision: str | None) -> str:
    provenance = root / "source.json"
    if provenance.is_file():
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if revision is not None and revision not in (source["revision"], source["commit"]):
            raise ValueError(f"{root} is pinned to {source['commit']}. Use a new --data-dir.")
        return source["commit"]

    revision = revision or "main"
    commit = resolve_commit(dataset, revision)
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Could not resolve {revision!r} to a full commit hash")
    source = {"source": SOURCES[dataset], "revision": revision, "commit": commit}
    payload = json.dumps(source, indent=2) + "\n"
    atomic_write(provenance, lambda output: output.write(payload.encode("utf-8")))
    return commit


def download_hub_file(root: Path, relative: Path, commit: str) -> None:
    if not has_local_file(root / relative):
        hf_hub_download(
            repo_id=VLSBENCH_REPO,
            repo_type="dataset",
            revision=commit,
            filename=relative.as_posix(),
            local_dir=root,
            token=False,
        )


def download_images(
    rows: list[dict], root: Path, image_job: Callable[[dict], None], workers: int
) -> int:
    failed = 0
    started_at = datetime.now(timezone.utc).isoformat()
    with (root / "download_failures.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {}
            try:
                for row in rows:
                    jobs[pool.submit(image_job, row)] = row
                for completed, job in enumerate(as_completed(jobs), start=1):
                    try:
                        job.result()
                    except Exception as error:
                        row = jobs[job]
                        failed += 1
                        record = {
                            "run_started_at": started_at,
                            "sample_id": row.get("uuid", row.get("instruction_id")),
                            "source": row.get("web_path", row.get("image_path")),
                            "error": f"{type(error).__name__}: {error}",
                        }
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if completed % 250 == 0:
                        print(f"{completed}/{len(rows)} processed, {failed} failures", flush=True)
            except KeyboardInterrupt:
                for job in jobs:
                    job.cancel()
                raise
    print(f"{len(rows) - failed}/{len(rows)} images available in {root}")
    if failed:
        print(f"{failed} failures. See {root / 'download_failures.jsonl'}", file=sys.stderr)
    return int(failed > 0)


def download_dataset(
    dataset: str, root: Path, commit: str, attempts: int, workers: int
) -> int:
    if dataset == "vlsbench":
        download_hub_file(root, VLSBENCH_METADATA, commit)
        rows = read_vlsbench_metadata(root)

        def image_job(row: dict) -> None:
            relative = vlsbench_image_file(root, row).relative_to(root)
            download_hub_file(root, relative, commit)
    else:
        metadata = root / VLSU_METADATA
        if not has_local_file(metadata):
            url = (
                f"https://raw.githubusercontent.com/{VLSU_REPO}/{commit}/"
                f"{VLSU_METADATA.as_posix()}"
            )
            download_url(url, metadata, attempts=attempts)
        rows = read_vlsu_metadata(root)

        def image_job(row: dict) -> None:
            destination = vlsu_image_file(root, row)
            if not has_local_file(destination):
                download_url(row["web_path"], destination, attempts=attempts, image=True)

    return download_images(rows, root, image_job, workers)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Lyra dataset.")
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--revision", help="Upstream ref or full commit. Defaults to main on first run."
    )
    parser.add_argument(
        "--attempts", type=positive_int, default=3, help="Total VLSU attempts (default 3)."
    )
    parser.add_argument(
        "--workers", type=positive_int, default=8, help="Parallel image downloads (default 8)."
    )
    args = parser.parse_args()

    root = args.data_dir.expanduser().resolve() / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"A download is already running in {root}", file=sys.stderr)
            return 1
        commit = get_commit(args.dataset, root, args.revision)
        print(f"{args.dataset} at {commit} -> {root}", flush=True)
        return download_dataset(args.dataset, root, commit, args.attempts, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
