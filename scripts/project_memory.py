#!/usr/bin/env python3
"""Verify the separate Llama project authority and its owning source worktree."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "codex/llama32-project-memory"
AUTHORITY = ("PROJECT_CONTRACT.md", "PROJECT_STATUS.yaml", "CONTEXT.md", "experiments/index.yaml")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    require(result.returncode == 0, result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def load(name):
    value = yaml.safe_load((ROOT / name).read_text())
    require(isinstance(value, dict), name + " must be a mapping")
    return value


def sync(path, branch):
    require(git(path, "rev-parse", "HEAD") == git(path, "rev-parse", "refs/remotes/origin/" + branch), "Unsynchronized branch: " + branch)


def validate():
    for name in (*AUTHORITY, "AGENTS.md"):
        require((ROOT / name).is_file(), "Missing authority file: " + name)
    status, index = load("PROJECT_STATUS.yaml"), load("experiments/index.yaml")
    require(status["schema_version"] == index["schema_version"] == 1, "Schema mismatch")
    require(status["project"]["id"] == index["project"] == "llama32-htp", "Project identity mismatch")
    require(status["project"]["memory_branch"] == BRANCH, "Memory branch identity mismatch")
    require(Path(status["project"]["memory_worktree"]).resolve() == ROOT, "Memory path mismatch")
    require(git(ROOT, "branch", "--show-current") == BRANCH, "Wrong memory branch")
    require(not git(ROOT, "status", "--porcelain"), "Dirty memory worktree")
    require(git(ROOT, "remote", "get-url", "origin").removesuffix(".git") == status["project"]["repository_url"].removesuffix(".git"), "Memory origin mismatch")
    sync(ROOT, BRANCH)
    records = index["experiments"]
    require(len({e["id"] for e in records}) == len(records), "Duplicate experiment ID")
    running = [e["id"] for e in records if e["execution_state"] == "running"]
    active = status["governance"]["active_experiment"]
    require(len(running) <= status["governance"]["maximum_running_experiments"], "Too many running experiments")
    require(running == ([active] if active else []), "Experiment lock mismatch")
    for record in records:
        require((ROOT / record["record"]).is_file(), "Missing experiment record")
    frozen = status["frozen_qwen3"]
    require(git(ROOT, "rev-parse", "refs/heads/" + frozen["branch"]) == frozen["head"], "Frozen Qwen3 local branch changed")
    require(git(ROOT, "rev-parse", "refs/remotes/origin/" + frozen["branch"]) == frozen["head"], "Frozen Qwen3 remote branch changed")
    return status, index


def check_source(source, stateful=False):
    status, index = validate()
    source = source.resolve()
    sources = status["source_worktrees"]
    require(str(source) in sources, "Unregistered source worktree")
    expected = sources[str(source)]
    require(git(source, "branch", "--show-current") == expected["branch"], "Source branch mismatch")
    require(git(source, "remote", "get-url", "origin").removesuffix(".git") == status["project"]["repository_url"].removesuffix(".git"), "Source origin mismatch")
    common = Path(git(source, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = source / common
    require(common.resolve() == Path(status["project"]["git_common_dir"]).resolve(), "Source Git common directory mismatch")
    require(not git(source, "status", "--porcelain"), "Dirty source worktree")
    sync(source, expected["branch"])
    active = status["governance"]["active_experiment"]
    if stateful:
        require(active is not None, "No active approved experiment")
        record = next(e for e in index["experiments"] if e["id"] == active)
        require(record.get("approval_ref"), "Experiment lacks approval reference")
        runtime = record["runtime"]
        require(runtime["source_worktree"] == str(source) and runtime["source_branch"] == expected["branch"], "Source does not own active experiment")
        if runtime.get("expected_source_head"):
            require(git(source, "rev-parse", "HEAD") == runtime["expected_source_head"], "Active source HEAD mismatch")
    elif active is None and expected.get("recorded_head"):
        require(git(source, "rev-parse", "HEAD") == expected["recorded_head"], "Source HEAD differs from recorded closure")
    policy = json.loads((source / "config/branch.json").read_text())
    require(policy["branch"] == expected["branch"] and policy["w4a8_rotation"] == expected["default_w4a8_rotation"], "Branch recipe policy mismatch")
    print("PROJECT_MEMORY=verified")
    print("PROJECT=llama32-htp")
    print("ACTIVE_EXPERIMENT=" + str(active))
    print("SOURCE_WORKTREE=" + str(source))
    print("SOURCE_BRANCH=" + expected["branch"])
    print("SOURCE_HEAD=" + git(source, "rev-parse", "HEAD"))
    print("SOURCE_DIRTY=no")
    print("SOURCE_REMOTE_SYNC=verified")
    print("QWEN3_FREEZE=verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "brief", "preflight"))
    parser.add_argument("--source-worktree", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "validate":
            validate()
            print("PROJECT_MEMORY=verified")
        else:
            require(args.source_worktree is not None, "--source-worktree is required")
            check_source(args.source_worktree, args.action == "preflight")
    except (ValueError, KeyError, OSError, StopIteration, yaml.YAMLError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
