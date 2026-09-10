#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/daniuniu/work/llama32-htp-project-memory/scripts/bootstrap.sh "$source_root"
