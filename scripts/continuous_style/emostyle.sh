#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" exec python -m lyra.continuous_style.image.emostyle "$@"
