# qwen3-block-htp Project Memory

This orphan branch records the authoritative constraints, language, decisions,
and experiments for the standalone Qualcomm HTP block laboratory.

The source project bypasses QNN and does not attempt full-model inference. It
builds fixed execution units whose VTCM layout, DMA movement, HVX work, and HMX
work are controlled by project source.

The initial local bootstrap was explicitly approved before a remote repository
was created. The user subsequently approved the public repository
`https://github.com/Dan1un1u/qwen3-block-htp`; remote synchronization is now
mandatory before stateful work.

The stable entry point for every new or resumed session is:

```bash
/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/bootstrap.sh \
  /home/daniuniu/work/qwen3-block-htp
```

This entry point fetches and fast-forwards Project Memory only, validates its
schema and source identity, and prints the authoritative file-reading order.
It deliberately allows an already-started source worktree to be dirty; the
separate `project_memory.py preflight` command remains mandatory immediately
before a new stateful phase begins.
