# qwen3-block-htp Project Memory instructions

This orphan branch is the authoritative control plane for the standalone HTP
block laboratory. It never carries implementation source or build products.

- Before any project reading, theory discussion, source work, build, model
  generation, device execution, or profiling, run
  `scripts/bootstrap.sh /home/daniuniu/work/qwen3-block-htp`. Treat any
  bootstrap failure as an immediate hard stop; stale fallback is forbidden.
- Read `PROJECT_CONTRACT.md`, `PROJECT_STATUS.yaml`, `CONTEXT.md`, and
  `experiments/index.yaml` in the order printed by bootstrap before project
  work.
- Run `python3 scripts/project_memory.py validate` before stateful work.
- Run `python3 scripts/project_memory.py preflight --source-worktree <path>`
  immediately before source edits, builds, model generation, device execution,
  or profiling.
- The repository is in an explicitly approved local bootstrap phase until its
  first remote is created. Do not invent or push a remote without user approval.
- Once a remote is configured, fetch and fast-forward only. Any sync or
  validation failure is a hard stop for user discussion.
- Only the user may amend the contract, accept an ADR, accept an experiment,
  promote a baseline, or reopen a rejected direction.
- Source belongs in `/home/daniuniu/work/qwen3-block-htp`; models and retained
  intermediate artifacts belong under `D:\llm_exp\models\qwen3-block-htp`;
  formal results belong under `D:\llm_exp\results\qwen3-block-htp`.
