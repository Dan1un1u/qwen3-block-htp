# Standalone HTP Llama 3.2 project contract

User approved the Qwen3 freeze and two-branch migration plan on 2026-09-10.
This separate authority governs Llama development; Qwen3 EXP0266 records the
handoff. Historical Qwen3 authority, Selected records and failed gates remain
immutable evidence. This is a standalone FastRPC/HVX/HMX runtime, no QNN graph,
context or execution dependency.

## Scope and branches

- Hardware remains PJZ110 / SM8750 / Hexagon HTP V79, isolated Hexagon SDK6.6,
  Tools19.0.07 and Android NDKr26c. Do not infer firmware/clock control.
- Freeze Qwen3 source branch codex/exp-0265-w4u8-r4-fullmodel-e2e at
  48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5. No further Qwen3 research, including SP2.
- Two Llama source branches share runtime code and W16A16 / W4A16 recipes.
  No-rotation defaults W4A8 OFF. Rotation defaults dense R3 OPT2; dense R3+R4
  OPT6 is optional. W4A16 uses exact EXP0260 OPT2 and C64 per-channel method.
- These are user-approved migration starting implementations, not retrospective
  Qwen3 quality acceptance or promotion of historical Selected records. Preserve
  ideal-R3 whole-layer numerical failure and unaccepted R4 quality explicitly.
- Initial organization does not implement Llama inference. Exact checkpoint and
  revision are unselected. Model dimensions, tokenizer/template, normalization,
  RoPE, embedding/head/tied-weight behavior must come from that checkpoint.
- Original Llama tensors must supply new weights and rotations. Qwen3 folded or
  quantized weights, static-A8 calibration, EOS prefix/cache and tokenizer are
  never Llama inputs. W4 preserves signed [-7,7] per-output-channel with one FP32
  scale. Grouped quantization or mixed-precision deployment needs explicit scope.
- R3/R4 use explicit dense matrix multiplication, no butterfly/FWHT.

## Execution and evidence

- Before source work, run bootstrap and read all four authority files in order.
  An approved active experiment with one worktree owner and clean synchronized
  preflight is required before edits/builds/model generation/device/profiling.
- Only one running experiment; future experiment IDs L32-0001 onward. Register
  bounded protocols under experiments/index.yaml before work. Ordinary fixes
  within approved scope proceed autonomously; preserve failed evidence.
- Single-layer correctness precedes consecutive-layer validation and complete
  token-boundary execution. Use an independent implementation reference for
  actual arithmetic; compare to a floating teacher separately for model quality.
- Preserve exact8MiB VTCM budget, one FastRPC per declared execution unit, one HMX
  owner, no timed intermediate DDR/spill, explicit buffer/DMA ownership and
  complete additive Host/DSP ledgers. No silent CPU fallback or unknown fields
  replaced with zeros. Unmeasured evidence is N/A.
- PPL is the primary quality metric, short-task content-aware answers auxiliary.
  Preserve existing overall5% and each language/domain10% PPL-ratio thresholds,
  independent calibration/selection/heldout sets and declared uncertainty rules.
  New model fixtures/masks are frozen before scoring, not fitted on evaluation.
- Performance decisions use fixed formal rotated paired repeat10 complete Host
  wall; both prefill and decode paired-bootstrap95% CI upper <=1.10 for the
  existing10% slowdown gate. Repeat1 is auxiliary and cannot veto. No optional
  stopping, cherry-picking or layer-to-model throughput extrapolation.
- Report real complete prefill and continuous-decode token/s with token counts
  and Host-wall denominators. Separate loading/frontend from warm model timing.
  User profiling presentation retains the stable recipe module table with
  microseconds/Host shares, followed by E2E. Archive complete evidence separately.

## Storage and recovery

- Source worktrees and memory locations are listed in PROJECT_STATUS.yaml. New
  builds stay on their WSL ext4 worktree. Model payloads under
  /mnt/d/llm_exp/models/llama32-htp; results under
  /mnt/d/llm_exp/results/llama32-htp, separated by experiment and recipe.
- Git stores code/configuration/compact evidence only, never models, SDKs,
  binaries or large profiling dumps. Keep original Qwen3 artifact paths intact.
- Read-only inspection and reversible owned repairs require no repeated user
  permission. Fix owned tooling/schema/path/CLI/build/measurement defects;
  preserve originals and rerun affected checks. No weakening numerical or
  physical gates, accepting unexplained hashes, stale fallback, reset/clean,
  auto-stash, force push or history rewriting.
- Unknown ownership, remote divergence, unexplained provenance/hash changes,
  compromised evidence, destructive recovery or material scope/gate changes
  require discussion. New baselines are promoted only by the user.

## L32-0001 authorization (2026-09-10)

User selected Llama-3.2-1B-Instruct, downloaded the original Transformers BF16
checkpoint from ModelScope to /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin,
and approved model adaptation in W16A16 -> W4A16 -> W4A8 order. The original
user-supplied directory is a read-only input exception to the generated-model
root. L32-0001 owns unrotated W16A16 only; later recipes follow separate bounded
protocols. Fix model dimensions, omit Q/K norm entirely, implement llama3 RoPE,
tied embedding/head, tokenizer/EOS and independent reference/export. Validate
single-layer prefill/decode (nonzero positions and historical KV), consecutive
layers, then complete text and heldout PPL. Preserve physical/numerical gates.
No Qwen3 research or weight/calibration reuse. Common model changes may later
be propagated to the rotation branch after the owning worktree is validated.
