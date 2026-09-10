#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 /home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py preflight --source-worktree "$source_root"
