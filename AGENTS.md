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
Qwen-specific weights, calibration, prefix or tokenizer for Llama. The model JSON and recipe validation records distinguish functional support from
model-quality acceptance. W4A16 is functional but failed its independent PPL
criteria; W4A8 OFF is tracked in L32-0003 with no model-quality threshold.
Use only Llama-specific tools in docs/LLAMA32_W16A16.md,
docs/LLAMA32_W4A16.md and docs/LLAMA32_W4A8.md. L32-0019 adds opt-in dense R3/R4 for FP32 residual+SP2 single-layer validation.
See docs/LLAMA32_FP32_R3_R4.md and docs/LLAMA32_ROTATION_PIPELINE.md (L32-0020); full-model rotation remains unvalidated and
L32-0022 repairs the pre-existing three-layer rotation numerical failure and passes unchanged chain3 plus single-layer speed gates (docs/LLAMA32_ROTATION_NUMERICAL_REPAIR.md). L32-0023 full16 rotation fails the unchanged independent numerical gate; local boundary diagnosis identifies quantization-threshold amplification (docs/LLAMA32_ROTATION_FULLMODEL_VALIDATION.md). L32-0024 opt-in exact dense R4 mode3 restores full16 bit-exact numerical alignment but fails singlelayer speed by orders of magnitude; retain as diagnostic control (docs/LLAMA32_R4_EXACT_REFINEMENT.md). Mode1 full16 remains failed. Frontend/E2E paused for the exact candidate cost gate. Default OFF and other recipes are unchanged. No QNN execution.
Builds stay in the owning WSL worktree; large Llama artifacts go to separate
D:/llm_exp/models/llama32-htp and results/llama32-htp experiment directories.

Routine owned fixes proceed autonomously under the authority recovery policy.
Preserve failed evidence; no gate weakening, unknown hash acceptance, auto-stash,
reset/clean, force push or history rewriting. Do not start new Qwen3 research.
