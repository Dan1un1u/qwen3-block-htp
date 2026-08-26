# qwen3-block-htp source instructions

Before source edits, builds, device runs, or profiling, verify the independent
Project Memory worktree:

```sh
cd /home/daniuniu/work/qwen3-block-htp-project-memory
python3 scripts/project_memory.py preflight \
  --source-worktree /home/daniuniu/work/qwen3-block-htp
```

The authoritative contract lives on `codex/qwen3-block-project-memory`. This
source tree must not add a QNN execution dependency. Build products remain on
WSL ext4 and formal artifacts/results use the D-drive locations named by the
contract.
