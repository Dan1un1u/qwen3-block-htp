# qwen3-block-htp source instructions

Before any project reading or discussion, and before source edits, builds,
device runs, or profiling, synchronize and verify the independent Project
Memory worktree:

```sh
/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/bootstrap.sh \
  /home/daniuniu/work/qwen3-block-htp
```

Read the authority files in the order printed by bootstrap. Immediately before
a new stateful phase, run the printed `project_memory.py preflight` command.
Any bootstrap, sync, validation, source-identity, or preflight failure is a hard
stop; a stale local memory copy must never be used.

The authoritative contract lives on `codex/qwen3-block-project-memory`. This
source tree must not add a QNN execution dependency. Build products remain on
WSL ext4 and formal artifacts/results use the D-drive locations named by the
contract.
