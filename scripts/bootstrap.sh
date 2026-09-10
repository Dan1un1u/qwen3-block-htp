#!/usr/bin/env bash
set -euo pipefail
memory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_worktree="${1:?usage: bootstrap.sh SOURCE_WORKTREE}"
[[ "$(git -C "$memory" branch --show-current)" == codex/llama32-project-memory ]] || { echo 'ERROR: wrong memory branch' >&2; exit 1; }
[[ -z "$(git -C "$memory" status --porcelain)" ]] || { echo 'ERROR: dirty memory worktree' >&2; exit 1; }
git -C "$memory" fetch --quiet origin
git -C "$memory" merge --quiet --ff-only origin/codex/llama32-project-memory
python3 "$memory/scripts/project_memory.py" brief --source-worktree "$source_worktree"
for authority in PROJECT_CONTRACT.md PROJECT_STATUS.yaml CONTEXT.md experiments/index.yaml; do
    printf 'AUTHORITY_FILE=%s/%s\n' "$memory" "$authority"
done
printf 'STATEFUL_PREFLIGHT=python3 %s/scripts/project_memory.py preflight --source-worktree %s\n' "$memory" "$source_worktree"
printf 'PROJECT_MEMORY_BOOTSTRAP=verified\n'
