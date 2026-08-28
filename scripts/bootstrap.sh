#!/usr/bin/env bash
set -euo pipefail

MEMORY_WORKTREE=/home/daniuniu/work/qwen3-block-htp-project-memory
MEMORY_BRANCH=codex/qwen3-block-project-memory
EXPECTED_ORIGIN=https://github.com/Dan1un1u/qwen3-block-htp.git
SOURCE_WORKTREE=${1:-}

hard_stop() {
    printf 'PROJECT_MEMORY_HARD_STOP=%s\n' "$*" >&2
    exit 2
}

[[ -d "$MEMORY_WORKTREE" ]] ||
    hard_stop "missing worktree: $MEMORY_WORKTREE"
[[ -f "$MEMORY_WORKTREE/scripts/project_memory.py" ]] ||
    hard_stop "missing validator: $MEMORY_WORKTREE/scripts/project_memory.py"

actual_branch=$(git -C "$MEMORY_WORKTREE" symbolic-ref --quiet --short HEAD) ||
    hard_stop "project-memory is detached"
[[ "$actual_branch" == "$MEMORY_BRANCH" ]] ||
    hard_stop "wrong project-memory branch: $actual_branch"

actual_origin=$(git -C "$MEMORY_WORKTREE" remote get-url origin) ||
    hard_stop "project-memory origin is unavailable"
[[ "${actual_origin%.git}" == "${EXPECTED_ORIGIN%.git}" ]] ||
    hard_stop "unexpected project-memory origin: $actual_origin"

[[ -z "$(git -C "$MEMORY_WORKTREE" status --porcelain=v1 --untracked-files=all)" ]] ||
    hard_stop "project-memory worktree is dirty"

git -C "$MEMORY_WORKTREE" fetch origin \
    "+refs/heads/$MEMORY_BRANCH:refs/remotes/origin/$MEMORY_BRANCH" ||
    hard_stop "fetch failed; stale fallback is forbidden"

local_head=$(git -C "$MEMORY_WORKTREE" rev-parse HEAD) ||
    hard_stop "local project-memory HEAD is unavailable"
remote_head=$(git -C "$MEMORY_WORKTREE" rev-parse "refs/remotes/origin/$MEMORY_BRANCH") ||
    hard_stop "remote project-memory branch is unavailable"

git -C "$MEMORY_WORKTREE" merge-base --is-ancestor "$local_head" "$remote_head" ||
    hard_stop "local and remote project-memory history diverged"

if [[ "$local_head" != "$remote_head" ]]; then
    git -C "$MEMORY_WORKTREE" merge --ff-only \
        "refs/remotes/origin/$MEMORY_BRANCH" ||
        hard_stop "project-memory fast-forward failed"
fi

python3 "$MEMORY_WORKTREE/scripts/project_memory.py" validate --clean ||
    hard_stop "project-memory validation failed"

if [[ -n "$SOURCE_WORKTREE" ]]; then
    python3 "$MEMORY_WORKTREE/scripts/project_memory.py" brief \
        --source-worktree "$SOURCE_WORKTREE" ||
        hard_stop "source identity check failed"
else
    python3 "$MEMORY_WORKTREE/scripts/project_memory.py" brief ||
        hard_stop "brief generation failed"
fi

printf 'AUTHORITY_FILE=%s\n' \
    "$MEMORY_WORKTREE/PROJECT_CONTRACT.md" \
    "$MEMORY_WORKTREE/PROJECT_STATUS.yaml" \
    "$MEMORY_WORKTREE/CONTEXT.md" \
    "$MEMORY_WORKTREE/experiments/index.yaml"
printf 'STATEFUL_PREFLIGHT=python3 %s/scripts/project_memory.py preflight --source-worktree %s\n' \
    "$MEMORY_WORKTREE" "${SOURCE_WORKTREE:-/home/daniuniu/work/qwen3-block-htp}"
printf 'PROJECT_MEMORY_BOOTSTRAP=verified\n'
