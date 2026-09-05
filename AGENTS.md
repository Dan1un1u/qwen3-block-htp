# qwen3-block-htp Project Memory instructions

This orphan branch is the authoritative control plane for the standalone HTP
block laboratory. It never carries implementation source or build products.

- Before any project reading, theory discussion, source work, build, model
  generation, device execution, or profiling, run
  `scripts/bootstrap.sh /home/daniuniu/work/qwen3-block-htp`. A failed bootstrap blocks dependent work until repaired and revalidated; apply the recovery policy below. Stale fallback is forbidden.
- Read `PROJECT_CONTRACT.md`, `PROJECT_STATUS.yaml`, `CONTEXT.md`, and
  `experiments/index.yaml` in the order printed by bootstrap before project
  work.
- Run `python3 scripts/project_memory.py validate` before stateful work.
- Run `python3 scripts/project_memory.py preflight --source-worktree <path>`
  immediately before source edits, builds, model generation, device execution,
  or profiling.
- The repository is in an explicitly approved local bootstrap phase until its
  first remote is created. Do not invent or push a remote without user approval.
- Once a remote is configured, fetch and fast-forward only. Sync or validation failures block dependent work but do not automatically require user input. Apply the recovery policy below.
- Only the user may amend the contract, accept an ADR, accept an experiment,
  promote a baseline, or reopen a rejected direction.
- Source belongs in `/home/daniuniu/work/qwen3-block-htp`; models and retained
  intermediate artifacts belong under `D:\llm_exp\models\qwen3-block-htp`;
  formal results belong under `D:\llm_exp\results\qwen3-block-htp`.

## Autonomous recovery policy (user approved 2026-09-05)

- A gate failure pauses only dependent work, not diagnosis and repair. Fix attributable YAML/format/path/CLI mistakes, missing telemetry serialization, backward-compatible report parsing, build errors and transient tool/network failures without requesting permission again. Preserve original evidence and record the cause, fix and successful recheck.
- Use bounded retries and supported reconnection for transient failures. For owned uncommitted changes or unsynced commits, review, commit and push authorized work normally. Never stash/reset/clean or rewrite history to pass a gate. Unknown changes, concurrent ownership or remote divergence require discussion.
- Repair reference/measurement collection defects and recollect under the existing experiment. Never turn missing values into measured zeros, rewrite old hashes, substitute a reference, loosen numerical/physical thresholds, or retroactively change performance gates to obtain a pass.
- Escalate only for ambiguous authority/ownership, unexplained source/artifact hash mismatches, compromised evidence, required numerical/physical contract amendments, destructive recovery, scope expansion, or exhausted bounded recovery. Implementation bugs within approved scope may be fixed autonomously and affected gates rerun.
- Bootstrap/preflight must pass before resuming dependent source/build/model/device work. During authorized Project Memory repair, minimal inspection and repair may proceed despite its failed gate; validate syntax and invariants before committing, synchronize and rerun full bootstrap/preflight. Baseline promotion remains user-only.
