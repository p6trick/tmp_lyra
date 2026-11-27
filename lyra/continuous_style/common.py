import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "continuous_style"
RUNS_DIR = REPO_ROOT / "runs" / "continuous_style"


def new_run_directory(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(mkdtemp(dir=output_dir, prefix=f"{timestamp}-"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
