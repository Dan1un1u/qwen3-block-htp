# EXP-0269: Qwen SP2 high-precision residual and paper baseline audit

User selects Qwen EXP0268 no-rotation SP2 (2030.2384/48.17335 tokens/s) and
Llama L32-0018 no-rotation SP2+FP32 residual (2069.7026/42.51011) as paper speed
baselines. Port and optimize the latter residual contract on Qwen; audit all
later source/evidence for missed applicable optimizations in both recipes.
This explicit instruction reopens Qwen for this scope and supersedes prior
paper-only pause only for no-rotation integration/performance work.

Start codex/exp-0269-sp2-fp32-residual from EXP0268 closure after preflight;
initial parent branch binding is temporary for creation. Preserve historical
Qwen/Llama branches, evidence, binaries and models; no rotations, PPL, training,
new quantization or unrelated W16/W4A16 work. Llama read-only implementation
audit here; its authority owns baseline registration separately.

Contract: preserve original Qwen C64 packed W4, SP2mode8 codebook and scales,
integer attention/KV, Q/K norms and RoPE, norm output A8 and existing A8 LMhead.
Port Llama FP32 input embedding/residual, raw native integer O/Down outputs
with exact zero/SP2 merge, FP32 scale/add, ordered FP32 RMSNorm then A8. Use
original Qwen FP16 embedding as existing high precision carrier, convert to
FP32; no early embedding U8 requantization. Final FP32 norm quantizes once to
existing A8 head input. No silent FP16 matrix fallback or changed W4 weights.
Qwen RMS epsilon and dimensions stay model-specific. References independently
compute Qwen contract; no Llama weights, norms, qparams, tokenizer or caches.

Reuse validated L32-0018 HVX/native norm stores, ordered reduction, threshold
repair, O/Down two-slot HMX/HVX output overlap, SP2 low/high decode row packing.
Retain EXP0268 scheduling and Qwen-specific phase-dead operand reuse. Fixed8MiB
VTCM, one HMX owner, zero timed intermediate tensor DDR/spill, oneRPC/fullstep.
Prove live-buffer lifetimes and signed accumulator bounds before device.

Singlelayers0/14/27 M64+decode1: independent exact FP32 output and KV, finite,
repeat determinism and actual-boundary audit before timing. Consecutive3 then
full28 must pass independent implementation checks; model-quality PPL not a gate
and not requested. Pure source-format fixes within scope use PC037; retain all
failed attempts. Do not apply rotation's known-rounding exception here.

Port the existing optimized contract first; up to three bounded, explained
Qwen-specific scheduling/vector/layout candidates may follow observed bottlenecks.
Freeze each candidate before its measurements. Short diagnostics may select
before a fixed five-short/ten-formal paired campaign; do not optionally resample
formal sets. Repeat10 primary, repeat1 auxiliary; paired bootstrap20000 seed269,
both Host-wall95%CIupper<=1.10 versus originalSP2 required for normal escalation.
Fix attributable issues autonomously; if bounded candidates leave stable>10%
cost, retain work and discuss before fullmodel. After singlelayer eligibility,
full28 M64+15 same-prompt paired5short10formal, fixedcache128. Report module
ledgers and direct E2E, no extrapolation. User selection of old paper baselines
is explicit; new FP32 Qwen speed remains measured candidate until final report.

Audit later Llama0019-27 and Qwen branches by file/function diff and run-path
conditions, not commit age. Record applied, rotation-only, rejected, unmeasured,
and missing-applicable changes with hashes. Do not claim global optimality.
Close source/memory synchronized with evidence/provenance and baseline manifest.
