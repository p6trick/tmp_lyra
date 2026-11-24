"""Store run manifests and append-only result logs."""

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


def encode(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def digest(data) -> str:
    payload = data if isinstance(data, bytes) else encode(data)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes())


def inference_path(directory: Path, condition: str) -> Path:
    return directory / f"inference.{condition}.jsonl"


def inference_paths(directory: Path, conditions: tuple[str, ...]) -> tuple[Path, ...]:
    return (directory / "inference.jsonl", *(inference_path(directory, name) for name in conditions))


@contextmanager
def output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is writing to {path.parent}") from None
        yield


def check_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        if read_json(path) != manifest:
            raise SystemExit(f"Settings differ from {path}. Use a new output directory.")
        return
    write_json(path, manifest)


def write_json(path: Path, data: dict) -> None:
    temp = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(temp.name)
    try:
        with temp:
            temp.write(encode(data) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def records(path: Path, *, repair_tail: bool = False):
    if not path.exists():
        return
    with path.open("r+b" if repair_tail else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair_tail:
                    stream.truncate(offset)
                break
            try:
                yield json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"Invalid JSONL record in {path} at byte {offset}") from error


def append_record(stream, row: dict) -> None:
    stream.write(encode(row) + b"\n")
    stream.flush()
