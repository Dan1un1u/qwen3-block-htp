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
- Selected checkpoint is original BF16 Llama-3.2-1B-Instruct from ModelScope,
  pinned by six file hashes in source tools/llama_reference.py. L32-0001 validates
  W16A16 functional integration; L32-0002/0003 connect W4A16/W4A8 OFF.
  W4A16 quality fails its retained PPL gate; A8 text is unusable with no quality
  threshold applied. Llama R3/R4 full-model use remains unvalidated; L32-0019 adds experimental
  single-layer support with FP32 residual/SP2. Model dimensions,
  tokenizer/template, normalization, RoPE and tied-head behavior come from it.
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

## Sequential quantized-chain authorization (2026-09-10)

User defers speed optimization and explicitly requests W4A16 then W4A8 functional
integration. W4A16 is expected to have usable model quality; retain its PPL criteria.
W4A8 model quality has no acceptance threshold in this integration phase; record
actual quality without blocking for low PPL/text quality. This explicit exception
does not relax arithmetic correctness, memory, provenance or execution checks.

## L32-0004 speed authorization and closure (2026-09-10)

User explicitly authorized reuse of existing pipelines for speed, superseding
prior optimization deferral. L32-0004 validated shared head64 HVX RoPE, A8 native
KV packing/head workers/valid decode row, and dimension-correct W4A16 OPT2.
Both recipes passed fixed ten paired AB/BA complete-model timing gates and retained
arithmetic/physical checks. Weights/calibration unchanged; no new quality baseline
acceptance or rotation support. Final heads and evidence are recorded in status.

## L32-0005 speed closure (2026-09-10)

Matched M64+7 ten rotated three-arm experiment verified native A8 relative speed
and exact arithmetic/physical gates. Launchers now use validated row4 common/
SwiGLU,HVX softmax,hvx_tree and pooled residual modes. Per-config V LUT and
native tile RoPE gather are common source. No quality promotion or rotation support.
Read current status/context and L32-0005 record for measured scope and evidence.

## L32-0006 speed closure (2026-09-10)

Three bounded A8 scheduling/layout iterations passed exact arithmetic and physical
checks plus fixed ten M64+7 three-arm and ten M64+15 paired performance cycles.
Measured OFF Llama speed is near frozen Qwen OFF historical decode throughput;
cross-model reference remains non-paired. Common code propagated to both Llama
branches, without adding rotation support or promoting quantization quality.
Current status/context and L32-0006 evidence are authoritative for measured scope.

## L32-0008 SP2 prototype authorization and closure (2026-09-13)

User approved Llama SP2 high/low native-W4 feasibility prototype. Isolated
Down integer component passes exact arithmetic; no production recipe change,
SwiGLU producer fusion,final output requant/residual,full-block10%gate,PPL or
E2E measured. Fixed ten component cycles are not a full-block performance
gate. L32-0008 record preserves two DMA failures and an invalid known stale-DSP
attempt after build failure; final builds are sealed and deployment validates
source HEAD plus all binary hashes. Native W4 SP2 feasibility does not promote
any model-quality or performance baseline. Qwen research remains frozen.

## User-confirmed SP2 method and Qwen3 backport (2026-09-13)

User explicitly confirms the L32-0012 method conclusion and requests migration
to Qwen3. Pin the measured mode8 implementation ddbc07f422693534f7d29e554f1c92c1b87cea18
and report closure e03a0f8028b3f123dbc0263394c772483953dc22, evidence ledger
5277594741ac28c3f013ca33191456e9a1952160b30c2e6d616e2ad6f83e8552.
Confirmation covers exact native-W4 SP2 arithmetic and the measured scheduling
result; it does not establish usable A8 text or PPL acceptance. Preserve Llama
source, binaries, weights and sealed evidence. Qwen3 memory owns EXP-0267, on a
new experiment branch. This user instruction supersedes the earlier blanket
Qwen research freeze only for this migration; the historical EXP0265 branch
and its pinned hash remain frozen and independently checked.

## L32-0013 hardware-priority C-RTN/SP2 authorization (2026-09-14)

User authorizes fresh original-weight training/folding/export, using rotation-quant
read-only as method reference. Existing hardware quantization wins conflicts,
including residual/nonlinear/KV and frontend/head boundaries. Add missing pieces
without converting the runtime into the BF16 fakequant reference. Historical
17.6423999759 PPL is not an exact-reproduction gate. Preserve numerical/physical
implementation checks and label all contract differences. Floating controls and
Qwen3 unchanged. See docs/experiments/L32-0013.md.

## L32-0017 speed-first steering and closure (2026-09-14)
User prioritizes speed and authorizes FP32 vectorization/pipeline repair.
This supersedes the L32-0016 suggested next-step priority of PPL recalibration.
Retain arithmetic/physical correctness; defer new calibration/PPL/rotation work.
Fixed ten paired profile passes decode10%gate but fails prefill; no overall
performance acceptance or baseline promotion. See L32-0017 report and current
status. Further work should address remaining FP32 Norm and O overhead.

## L32-0019 rotation resumption (2026-09-14)
User temporarily accepts L32-0018 speed and authorizes R3/R4 integration using
Qwen implemented dense paths as reference. Supersedes0017 rotation deferral for
this bounded integration only. Preserve current FP32 residual/SP2 and frozen
no-rotation measurements. See0019protocol; no deep speed optimization or quality promotion.

## L32-0020 optimization authorization
User explicitly requests rotation pipeline optimization after0019 checkpoint.
This supersedes prior speed-optimization deferral for the bounded0020protocol.

## Fast-path priority (2026-09-14)
User directs incremental precision repairs on the original fast HMX pipeline.
Speed and correct implementation are the current priorities; usable model
quality is secondary and new PPL/text evaluation is deferred. Exact dense
mode3 remains diagnostic only. This does not relax numerical/physical gates,
independent goldens or the10% performance gate. See L32-0025 protocol.

## Paper working implementation and alignment pause (2026-09-14, L32-0027)
User now explicitly accepts the original fast HMX implementation with its known
rounding/full16 divergence for paper preparation. This supersedes the previous
requirement to repair numerical alignment before proceeding with this paper task.
Pause further H512 repair, integer-plane prototypes, calibration/PPL and hardware
work. Preserve original0022 performance and0023 numerical failure at their actual
scopes; rejected0026 candidates remain rejected, exactmode3 remains diagnostic.
Acceptance is a paper/performance working choice, not a numerical gate pass or
model-quality acceptance. Do not rewrite historical evidence or infer fullmodel
E2E for the combined rotated FP32-residual configuration from layer timings.

## No-rotation paper baseline selection (L32-0028)
User selects L32-0018 no-rotation SP2 plus FP32 residual as the primary Llama
paper speed baseline, and EXP0268 no-rotation SP2 for Qwen with FP32 port under
its own EXP0269. This supersedes the0027 rotated primary working selection;
rotation evidence remains supplementary with original numerical caveats.
Only documentation/audit work on Llama in0028; no new native/hardware/PPL.

## User-approved full-model paper ablations (2026-09-15)
User confirms full-model format/pipeline factorial and remaining agreed paper table, explicitly excludes W4-to-S8 and integer-residual arms, and adds log2 versus FP vector softmax. This supersedes0027 hardware pause and0028 documentation-only scope for separately registered bounded phases. Fix FP32 residual,no rotation; original weights/quantization. All numerical/physical/evidence gates remain. Deliberately slower diagnostic controls may complete above10percent, without baseline promotion. No repeat permission needed for within-table registration. Preserve vector kernels and intra-GEMM DMA/HMX double buffering in conventional schedules; no deliberately redundant native/generic roundtrip or scalar controls. Hardware ownership serialized across projects.

## User-approved post-ablation optimization (2026-09-15)
User requests sequential both-model decode native-format diagnosis/optimization (compact decode if objectively faster), Llama log2 optimization versus FP, then bounded latest-version generalization. Later phases may register under this approval; no full factorial rerun. Preserve no-rotation/nativeW4/SP2/FP32residual baseline and numerical/physical/evidence gates; other recipes frozen. Necessary exact shape support allowed for final bounded workloads; no PPL/quality acceptance or automatic baseline promotion. Device ownership serialized with Qwen. Routine repairs autonomous.

## Latest optimized ablation rerun authorization (2026-09-15; L32-0038)

User now explicitly requests rerunning optimized paper ablations and updating conclusions. This supersedes the prior no-factorial-rerun restriction only for registered A0/A1/A2/A3/A5/A6/A9 under docs/experiments/L32-0038.md. Generalization and shape sweeps are supplementary, excluded from this rerun. Preserve vector/conventional baseline fairness, frozen weights, FP32 residual, no rotation and all numerical/physical/evidence gates. A7 cites existing same-FP32 evidence; no W4-to-S8 or integer-residual comparisons. Routine repairs and sequential phase registration need no repeated permission.

## User-approved floating baseline fairness refresh (2026-09-16; L32-0039)

User explicitly requests the annotated W16A16 baseline repairs and formal E2E before discussing SP2 A16 Down. This supersedes earlier F16 freeze only under docs/experiments/L32-0039.md: exact decode conversion/movement reuse, basic FP16 LM-head prefetch, bounded valid-row cleanup. Preserve FP16 math/weights and other recipes, all numerical/physical/evidence gates; no deep tuning, quality claim or automatic promotion. Routine fixes proceed autonomously.

## User-approved uniform INT16 Down control (2026-09-16; L32-0040)

User approves the prior annotated proposal. Implement uniform INT16 Down input on the existing SP2 two-byte native-W4 pipeline, comparing A8/SP2/INT16 with FP32 residual and frozen non-Down boundaries. This supersedes earlier A16 discussion-only and recipe freezes solely within docs/experiments/L32-0040.md. Preserve all numerical/physical/evidence gates; no PPL or automatic promotion. Both-model device work serialized, Llama first.

## L32-0041 3B adaptation authorization (2026-09-17)
User supplies original Llama-3.2-3B-Instruct and authorizes migration of latest no-rotation W4A8 SP2 FP32-residual runtime. This extends model support beyond the original1B scope solely under L32-0041. Preserve1B artifacts, native math/physical gates; defer memory optimization and PPL. Fresh3B weights/calibration only.

## L32-0045 no-performance-gate loop authorization (2026-09-18)
User explicitly requests iterative Llama3.2-3B W4A8-SP2 E2E optimization without gates. For this experiment, supersede prior10% slowdown/external-speed stopping thresholds; numerical correctness, physical constraints, provenance, fair matched measurements and uncertainty reporting remain. Routine iterative candidates under the registered protocol require no repeated permission.

## Missing recipe completion — 2026-09-18
User authorizes ordinary W4A8 derived from latest SP2 by removing only SP2, preserving all shared optimizations and remaining contracts. Other missing recipes receive generic vector/basic DMA overlap, no specialization tuning. User explicitly defers 3B W16A16 because existing uint32 shared-offset ABI cannot hold its FP16 weights; complete 3B W4A16 and ordinary A8 first. No claim of model-quality acceptance from speed checks.

## L32-0048 mapping probe authorization (2026-09-18)
User explicitly authorizes independent delayed mapping feasibility on current device before discussing multisession. Supersedes W16 deferral only for this isolated probe. Preserve existing runtime/recipes, no quality/E2E claim; costs of any eventual per-token mapping must be included.

## L32-0049 segmented FP16 implementation authorization
User approves full generic3B W16A16 implementation after0048 mapping success. Supersedes prior W16 deferral and0048 probe-only restriction for0049. Fresh original FP16 weights, resident segmented single-session mapping, generic existing kernels/basic DMA only; hardware correctness then formal E2E. No specialization or quality promotion.

## L32-0050 long prefill authorization
User approves generic64-token chunked long prefill on latest1B W4A8-SP2. Supersedes short-context-only scope for0050; preserve current arithmetic/weights and all correctness/physical/evidence gates. Target536+46 and741+3 after boundary validation. No model-quality claim or promotion.

## L32-0051 iterative long-prefill speed authorization
User explicitly authorizes a loop targeting M64 single-chunk prefill throughput parity for current1B SP2 longprefill. Supersedes0050 no-tuning scope; preserve arithmetic/physical/evidence constraints and fair M64 control. See0051protocol. No model-quality or automatic baseline promotion.

## User-approved generic long-context speed repair (2026-09-19)
User authorizes L32-0056 under its registered protocol: all remaining models except Qwen0.6, generic A16 layout/vector/DMA repairs and paired10% long/M64 throughput gate. Supersedes prior experiment-specific no-further-optimization notices only within this bounded scope. Numerical/physical gates, frozen M64 fairness and historical failures remain.
