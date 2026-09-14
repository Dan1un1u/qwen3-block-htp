# User-selected no-rotation paper speed baselines

| Model / configuration | Prefill end-to-end tok/s | Decode tok/s | Measured experiment |
|---|---:|---:|---|
| Qwen3-0.6B W4A8, SP2, integer residual, R3/R4 OFF | 2030.24 | 48.17 | EXP0268 |
| Llama-3.2-1B-Instruct W4A8, SP2, FP32 residual, R3/R4 OFF | 2069.70 | 42.51 | L32-0018 |

These are historical full-model warm M64+15 measurements including embedding,
final norm, LM head, greedy and FastRPC; package loading/tokenizer are excluded.
The models and cache/prefix policies differ: not a paired cross-model comparison.
Selection is speed/paper scope only. No model-quality acceptance is inferred.
See PAPER_NO_ROTATION_BASELINES.json for measured source, exact command evidence,
environment, ledger hashes, shape, and scope. New builds must not inherit old
speed claims merely because they descend from these commits.

## Completeness audit

Qwen EXP0268 already contains the shared U8 prefill scheduling work. SP2mode8
uses its own measured native producer/consumer scheduling; the U8-specific m3
switch is not an additional validated SP2 optimization. No later completed
Qwen implementation existed before the current EXP0269 port.

Llama current0f7d083 versus measured0018e3d065: FP32 residual, SP2 Down,
MLP U8, HVX U8 and native HMX U8 source files are byte-identical. Seven changed
native files add rotation routing, rotation kernels or rotation-only audit
behavior. R3/R4 OFF preserves the selected computational path. L32-0025/0026
rejected arithmetic candidates were restored;0027 changed documentation only.
The older rotation branch has no omitted common optimization: git cherry finds
all its common commits patch-equivalent, apart from its branch-default config.
Audit covers registered branches/completed experiments, not global optimality.

## Current Qwen high-precision residual work

EXP0269 ports the FP32 residual contract and retains native packed W4/SP2,
integer attention/KV and post-Norm A8. Singlelayers0/14/27 prefill/decode,
actual boundaries/KV and repeat10 are exact against an independent reference.
Its final fixed short gate fails: no ten-round formal or full-model execution.
It does not replace the Qwen paper speed row above. New Qwen epilogue/native
padding changes have not been timed on Llama and must not be assigned L32-0018
historical throughput; they are possible future backport candidates only.

Supersedes L32-0027's rotated primary paper working selection for these speed
rows. Rotation evidence and its known numerical failures remain supplementary;
original reports and measured artifacts are preserved unchanged.

## EXP0270 continuation (not a baseline replacement)

Qwen C1 eight-row exact FP32 Norm and C3 reusable raw-output bias tables are retained. Independent selectedlayers exact; short prefill gate fails+14.42%, decode passes+4.13%. No fullmodel speed measured, no Llama backport. User-selected historical paper speeds remain unchanged. See EXP0270_FP32_RESIDUAL_RESULTS.md.
