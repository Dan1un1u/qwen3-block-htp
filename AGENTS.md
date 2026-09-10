# Llama 3.2 project-memory instructions

Before project work, run scripts/bootstrap.sh WORKTREE and read authority files
in the printed order. Both source worktrees are listed in PROJECT_STATUS.yaml.
Before stateful source/build/model/device work, register an approved experiment
and run scripts/project_memory.py preflight --source-worktree WORKTREE.
Routine owned fixes proceed autonomously under PROJECT_CONTRACT.md. Preserve
failed evidence and historical hashes. No auto-stash/reset/clean/force push or
unexplained hash acceptance. New Llama work never uses the frozen Qwen3 authority
to authorize device execution. Initial creation is authorized by Qwen3 EXP0266.
