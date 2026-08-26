#!/usr/bin/env python3
"""Validate and summarize qwen3-block-htp Project Memory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "PROJECT_STATUS.yaml"
INDEX = ROOT / "experiments" / "index.yaml"
CONTRACT = ROOT / "PROJECT_CONTRACT.md"

EXPECTED_PROJECT = "qwen3-block-htp"
EXPECTED_MEMORY_BRANCH = "codex/qwen3-block-project-memory"


class Failure(RuntimeError):
    pass


def run(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        list(args), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise Failure(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Failure(f"{path.name} must contain a mapping")
    return value


def git_branch(path: Path) -> str:
    return run("git", "branch", "--show-current", cwd=path)


def git_dirty(path: Path) -> str:
    return run("git", "status", "--porcelain", cwd=path)


def require_origin_sync(path: Path, branch: str) -> None:
    local_head = run("git", "rev-parse", "HEAD", cwd=path)
    remote_head = run(
        "git", "rev-parse", f"refs/remotes/origin/{branch}", cwd=path,
    )
    if local_head != remote_head:
        raise Failure(f"{branch} is not synchronized with origin")


def validate(require_clean: bool = False) -> tuple[dict, dict]:
    required = [
        ROOT / "AGENTS.md", CONTRACT, STATUS, ROOT / "CONTEXT.md", INDEX,
        ROOT / "docs" / "adr" / "0001-standalone-runtime-instead-of-llama-fork.md",
        ROOT / "docs" / "experiments" / "EXP-0001.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise Failure(f"missing control files: {missing}")

    status = load_yaml(STATUS)
    index = load_yaml(INDEX)
    project = status.get("project", {})
    governance = status.get("governance", {})

    if project.get("id") != EXPECTED_PROJECT or index.get("project") != EXPECTED_PROJECT:
        raise Failure("project identity mismatch")
    if project.get("memory_branch") != EXPECTED_MEMORY_BRANCH:
        raise Failure("memory branch identity mismatch")
    if Path(project.get("memory_worktree", "")).resolve() != ROOT:
        raise Failure("memory worktree path mismatch")
    if git_branch(ROOT) != EXPECTED_MEMORY_BRANCH:
        raise Failure("Project Memory is checked out on the wrong branch")
    if require_clean and git_dirty(ROOT):
        raise Failure("Project Memory worktree is dirty")

    experiments = index.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise Failure("experiment index is empty")
    ids = [item.get("id") for item in experiments]
    if len(ids) != len(set(ids)):
        raise Failure("duplicate experiment id")
    running = [item for item in experiments if item.get("execution_state") == "running"]
    if len(running) > governance.get("maximum_running_experiments", 0):
        raise Failure("too many running experiments")
    active = governance.get("active_experiment")
    if [item["id"] for item in running] != ([active] if active else []):
        raise Failure("active experiment lock does not match running experiment")

    for item in experiments:
        record = ROOT / item.get("record", "")
        if not record.is_file():
            raise Failure(f"missing experiment record for {item.get('id')}")

    remote_required = governance.get("remote_sync_required")
    remote_state = project.get("remote_state")
    if remote_required:
        if remote_state != "configured":
            raise Failure("remote sync is required but no configured remote is recorded")
        run("git", "remote", "get-url", "origin", cwd=ROOT)
        require_origin_sync(ROOT, EXPECTED_MEMORY_BRANCH)
    elif active not in (None, governance.get("remote_bootstrap_experiment")):
        raise Failure("local bootstrap exception is valid only for its declared experiment")

    return status, index


def preflight(source: Path) -> None:
    status, index = validate(require_clean=True)
    source = source.resolve()
    if source != Path(status["project"]["source_root"]).resolve():
        raise Failure("source worktree path mismatch")
    active = status["governance"]["active_experiment"]
    if active is None:
        raise Failure("no active experiment")
    record = next(item for item in index["experiments"] if item["id"] == active)
    expected_source_branch = record["runtime"]["source_branch"]
    if git_branch(source) != expected_source_branch:
        raise Failure("source branch mismatch")
    if git_dirty(source):
        raise Failure("source worktree is dirty")
    if status["governance"].get("remote_sync_required"):
        require_origin_sync(source, expected_source_branch)
    if record["runtime"]["source_worktree"] != str(source):
        raise Failure("source worktree does not own the active experiment")
    print("PROJECT_MEMORY=verified")
    print(f"EXPERIMENT={active}")
    print(f"SOURCE_BRANCH={git_branch(source)}")
    print(f"SOURCE_HEAD={run('git', 'rev-parse', 'HEAD', cwd=source)}")
    print("REMOTE_SYNC=verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--clean", action="store_true")
    pre = sub.add_parser("preflight")
    pre.add_argument("--source-worktree", type=Path, required=True)
    sub.add_parser("brief")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            status, _ = validate(require_clean=args.clean)
            print("PROJECT_MEMORY=verified")
            print(f"ACTIVE_EXPERIMENT={status['governance']['active_experiment']}")
        elif args.command == "preflight":
            preflight(args.source_worktree)
        else:
            status, _ = validate()
            print("PROJECT_MEMORY=verified")
            print("BACKEND=standalone-fastrpc-dsp")
            print("QNN=none")
            print(f"ACTIVE_EXPERIMENT={status['governance']['active_experiment']}")
    except (Failure, OSError, KeyError, StopIteration, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
