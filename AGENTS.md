# Llama 3.2 source instructions

This is a Llama development branch of the standalone HTP runtime. Qwen3 source
is frozen at EXP0265; its old scripts are historical evidence, not Llama tools.

Before reading source or discussing model implementation, run:

```sh
/home/daniuniu/work/llama32-htp-project-memory/scripts/bootstrap.sh "$PWD"
```

Read the four authority files in printed order. Before source edits, builds,
model generation, device use or profiling, register the user-approved experiment
in that authority and run:

```sh
python3 /home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py preflight --source-worktree "$PWD"
```

Initial branch organization is authorized by Qwen3 PC080/EXP0266 until the
recorded handoff is complete; subsequent work uses only the Llama authority.

Both branches retain common W16A16 and W4A16 OPT2. W4A8 rotation is controlled by
config/branch.json. Never silently fall back from a selected rotation or reuse
Qwen-specific weights, calibration, prefix or tokenizer for Llama. Current model
JSON is a porting plan, not implemented inference support. No QNN execution.
Builds stay in the owning WSL worktree; large Llama artifacts go to separate
D:/llm_exp/models/llama32-htp and results/llama32-htp experiment directories.

Routine owned fixes proceed autonomously under the authority recovery policy.
Preserve failed evidence; no gate weakening, unknown hash acceptance, auto-stash,
reset/clean, force push or history rewriting. Do not start new Qwen3 research.
