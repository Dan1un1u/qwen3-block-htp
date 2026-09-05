# Standalone HTP Qwen3 Runtime Contract

Changing any item requires explicit user approval and a committed amendment.

## Scope and architecture

- **PC-001 — Independent project.** This is a new standalone project. The
  historical mllm W4A8 project remains immutable evidence and is not its runtime
  or source-control parent.
- **PC-002 — No QNN execution path.** Formal execution must not create or run a
  QNN graph, QNN context, QAIRT model, or QNN custom operator. QNN headers may be
  consulted as hardware-layout documentation only.
- **PC-003 — Standalone DSP service.** Android Host code reaches a project-owned
  cDSP Skeleton through FastRPC and shared rpcmem.
- **PC-004 — Owned physical contract.** The project owns tensor packing, VTCM
  allocation, DDR-to-VTCM movement, DMA ordering, HVX work, HMX work, waits, and
  buffer reuse inside each Execution Unit.
- **PC-005 — Staged runtime scope.** The approved roadmap advances through a
  real single-layer replay prefill/decode gate, a two-to-three consecutive-layer
  vertical slice, and only then the full Qwen3 transformer stack. Tokenizer,
  sampling, and a general graph compiler remain outside scope unless separately
  approved.
- **PC-006 — RPC boundary.** An experiment declares its Execution Unit as one
  replay block step, one consecutive-layer slice, or one full-stack token/pass;
  that unit uses one Host-to-DSP run invocation. Per-tile and per-kernel FastRPC
  calls remain outside the contract.

## Hardware and toolchain

- **PC-007 — Target hardware.** The target is PJZ110 / SM8750 / Hexagon HTP V79.
- **PC-008 — Isolated toolchain.** DSP code is built with an isolated Hexagon
  SDK and Host code with an isolated Android NDK. Existing QAIRT installations
  are not replaced or modified.
- **PC-009 — Bootstrap candidates.** EXP-0001 evaluates the available Hexagon
  SDK 6.6.0.0 / Hexagon Tools 19.0.07 archive and Android NDK r26c. Successful
  compilation and device loading are experimental facts, not assumed support.
- **PC-010 — Resource boundary.** The runtime may use FastRPC, rpcmem,
  libcdsprpc, QuRT, HAP resource APIs, DMA support, HVX intrinsics, and HMX
  assembly. It does not claim control over firmware arbitration, clocks, thermal
  policy, or HMX internals.

## Numerical variants

- **PC-011 — Canonical names.** `F16F16` means FP16 stored weights and FP16
  activations on FP16 HMX. `W4F16` means per-channel W4 stored weights expanded
  for FP16 HMX with FP16 activations. `W4U8` means the same W4 values expanded
  to S8 and asymmetric U8 activations on integer HMX.
- **PC-012 — W4 contract.** Initial W4 variants use signed symmetric
  per-output-channel quantization with values in `[-7, 7]` and one scale per
  output channel. W4F16 and W4U8 must share byte-identical W4 values and scales
  when their comparison claims activation-path attribution.
- **PC-013 — U8 contract.** W4U8 activations use asymmetric U8. The local
  algorithm must account for activation zero point, including precomputed
  per-output-channel weight sums when used.
- **PC-014 — Correctness authority.** Each device path is checked against an
  independent software implementation of that path. Comparison with a higher
  precision teacher is an accuracy diagnostic, not the implementation
  reference.

## Evidence and comparison

- **PC-015 — Self-comparison.** Formal performance conclusions compare project
  variants with one another. Matching or exceeding Qualcomm QNN is not a gate.
- **PC-016 — Primary metric.** Once an Execution Unit owns a real token boundary,
  directly measured full-stack throughput is the primary user-facing summary
  and performance-ranking expression: prefill tokens/s is the measured prompt
  token count divided by complete prefill Host wall, and decode tokens/s is the
  measured generated-token count divided by complete continuous-decode Host
  wall. The exact Host wall used as each denominator remains mandatory beside
  throughput and remains the attribution authority. Throughput may never be
  projected from a block, module, layer, or partial-model measurement. For an
  Execution Unit without a token boundary, equivalent-scope device wall latency
  remains primary. DSP qtimer stages, DMA waits, bytes, and overlap measurements
  explain the result but do not replace these end-to-end measures.
- **PC-017 — Fair tuning.** Final variant comparisons give each variant the
  same schedule-search budget and selection rule. A common canonical plan is
  used first for correctness and diagnosis.
- **PC-018 — Project-owned profiling.** Optrace is not required or expected.
  The runtime records compact DSP timestamp events and Host wall time without
  printing inside hot loops.
- **PC-019 — Defined DDR boundary.** Each experiment declares which tensors are
  legal DDR boundaries. Intermediate tensors may not be hidden in DDR while a
  result is described as VTCM-resident.
- **PC-020 — No silent fallback.** Host computation may implement reference and
  validation logic but may not silently execute a claimed DSP kernel.

## Provenance, storage, and governance

- **PC-021 — Upstream reuse.** `llama.cpp-npu` and `htp-ops-lib` may be copied or
  adapted for this internal research prototype. Reused code records upstream
  repository and commit provenance.
- **PC-022 — Git boundary.** Git stores source, scripts, contracts, and compact
  evidence only. SDKs, NDKs, models, binaries, maps, traces, and profiling
  outputs are not committed.
- **PC-023 — Storage boundary.** Models and retained intermediates use
  `D:\llm_exp\models\qwen3-block-htp`; formal results use
  `D:\llm_exp\results\qwen3-block-htp`; small-file-intensive builds stay on
  WSL ext4.
- **PC-024 — Experiment authority.** Stateful work requires an explicitly
  approved Experiment ID and successful preflight. Only the user may accept an
  experiment or promote a Selected Baseline.
- **PC-025 — Local bootstrap exception.** Before the user authorizes creation of
  a remote, EXP-0001 may run from the committed local source and Project Memory
  branches. No later experiment may start without a configured and synchronized
  remote unless the user explicitly extends this exception.
- **PC-026 — Fixed full-VTCM experiment budget.** Formal single-client device
  comparisons query the architecture-defined total VTCM, require that complete
  amount once per Prepared Runtime Session, and give every compared Project
  Variant the same grant. A run fails instead of silently shrinking to current
  availability. Physical Plans allocate and reuse only the regions they need
  and report requested, granted, and peak-used bytes separately.
- **PC-027 — Complete profiling closure.** Every experiment closure retains a complete profiling comparison in `docs/FULL_PROFILING_REPORT_CONTRACT.md` format, including repeat-one/repeat-ten control and candidate values, additive ledgers, overlapping counters, correctness, physical gates and provenance. The user-facing response presents only the stable three-recipe module overview and ends with directly measured E2E throughput. Detailed diagnostics remain in the retained report. Missing evidence is never silently replaced by zero. A pre-profiling failure records the failure boundary and unavailable sections.
- **PC-028 — Three-variant completion overview.** User-facing profiling results show one stable repeat-ten module table for latest valid `F16F16`, Selected-Baseline `W4F16`, and latest eligible `W4U8`. Each module cell contains microseconds and percent of complete Host wall; the last column is `W4F16_time / W4U8_time - 1`. Historical columns state non-paired provenance. The response ends with directly measured complete prefill and continuous-decode tokens/s, token counts and Host-wall denominators. Incomplete scopes are N/A and never extrapolated. This presentation amendment was explicitly requested by the user on 2026-09-05; the complete archived evidence requirements remain.
- **PC-029 — Recipe-fastest baseline ranking.** Mathematical correctness,
  physical-contract compliance, equivalent execution scope, valid evidence and
  measurement reproducibility are eligibility requirements rather than
  performance-ranking dimensions. Among eligible candidates for one recipe,
  a token-boundary experiment ranks by formal rotated-pair directly measured
  full-stack tokens/s; because token counts and scope are fixed, this ordering
  must be exactly the inverse of its paired complete Host wall, and any mismatch
  is a hard evidence failure. A scope without a token boundary continues to rank
  by formal rotated-pair repeat-ten complete-block Host wall latency. Repeat-one
  latency, module-preservation
  deltas, qtimer work, HMX/HVX/DMA counters and local gates remain mandatory
  diagnostics but do not veto a lower eligible repeat-ten wall result. A raw
  best run is never a ranking result; gains near measured device noise require
  additional paired rounds and remain tied until reproducible.
- **PC-030 — Non-greedy specialization search.** The Selected recipe-fastest
  baseline records the lowest eligible repeat-ten Host wall observed and may
  advance monotonically, but experiments are not required to branch only from
  it. An evidence-valid slower structural candidate may be retained as an
  explicit enabling parent when it tests a new physical hypothesis. Such a
  candidate is not a baseline; any completed combination must still beat the
  current recipe-fastest baseline under PC-029 before promotion. The frozen
  Public Common Baseline and PC-017 fair-comparison rules remain separate.
- **PC-031 — Real layer replay authority.** Before the project owns the full
  preceding stack, a declared full-model teacher supplies the real input hidden
  state and position metadata for the selected layer at each prompt/decode
  position. The DSP runtime must compute and append that layer's K/V itself;
  imported runtime K/V is not a real replay execution.
- **PC-032 — Persistent per-layer cache state.** A Prepared Decode Session owns
  each layer's K/V capacity, valid length, physical format, and storage for the
  lifetime of the session. Prefill and each decode step append exactly once,
  state persists across calls, and repeated timing of one frozen snapshot must
  never be described as continuous decode.
- **PC-033 — Full-stack-compatible state ABI.** New session and cache interfaces
  are layer-indexed and explicitly versioned even while only one layer is
  executed. A one-layer-only cache ABI that must be replaced for the vertical
  slice does not pass the single-layer gate.
- **PC-034 — Vertical-slice gate before full expansion.** After real single-layer
  replay passes, the next integration target is two to three consecutive layers
  whose intermediate hidden state remains inside the DSP Execution Unit. The
  project does not replicate the block across the full transformer stack until
  that slice proves state ownership, layer transitions, and independent
  correctness.
- **PC-035 — Deterministic token-generation boundary.** Beginning with
  EXP-0164, project scope includes Host tokenizer/text decoding and a
  project-owned embedding, final RMSNorm, LM-head, and greedy next-token path.
  One timed accelerator pass accepts token IDs and persistent session state,
  executes the complete transformer and output head, and returns the selected
  token without storing inter-layer hidden tensors or the full logits vector in
  DDR. A separately declared untimed audit mode may expose logits as a legal
  output boundary. W4F16 is the first semantic and implementation anchor;
  W4U8 integration follows only after W4F16 produces the independently
  reproduced token sequence and stable readable text. Top-p/top-k sampling,
  batching, arbitrary prompt lengths, and a general graph compiler remain out
  of scope until separately approved.

- **PC-036 — F16F16 token boundary and lightweight evaluation preparation.** User approval on 2026-09-05 pauses further W4U8 specialization and authorizes EXP-0217 to extend the proven W4F16 Host tokenizer/text decoding, embedding, final RMSNorm, streaming LM head and greedy feedback boundary to F16F16 with FP16 stored weights. Preserve transformer arithmetic and the physical contract. Establish stable readable text and independent implementation validation, then propose one lightweight repeatable speed and quality evaluation for all three recipes. W4F16 readability is not a quality benchmark; W4U8 semantic degradation is diagnostic and is not to be repaired in this integration experiment. No baseline promotion is implied.

- **PC-037 — Autonomous gate recovery.** User amendment on 2026-09-05 distinguishes a failed work gate from a need for user approval. Diagnose and repair attributable tooling, schema, implementation and evidence-collection defects within approved scope, preserve original evidence, and rerun affected checks without asking again. A failed gate blocks dependent execution until repaired. Unknown ownership, unexplained provenance/hash mismatch, compromised evidence, numerical/physical threshold changes, destructive recovery, scope expansion and exhausted bounded recovery require discussion. No silent acceptance, baseline promotion, history rewriting or stale authority fallback is permitted. `AGENTS.md` specifies the workflow.

- **PC-038 — Unified lightweight evaluation.** The user approved implementation and execution of `docs/THREE_RECIPE_LIGHTWEIGHT_EVALUATION_PLAN.md` on 2026-09-05 as EXP-0218. Scope includes frozen samples, BF16 teacher cache, three-recipe device teacher forcing and target log-prob/rank diagnostics, short-task greedy generation, multi-sample loaded-weight reuse and equivalent M64+15 speed measurement. Quality-only phase-dead VTCM scratch and compact scalar outputs are legal; timed speed disables quality work. Explicit logical prompt/mask support is permitted if needed; no silent padding or truncation. W4U8 semantic failure is reported and not repaired. Quality score snapshots do not promote implementation baselines.

## PC-039 — W4F16 offline R1/R2 validation (user approved 2026-09-05)

Only W4F16 may change model weights in EXP-0219: fold residual-stream RMSNorm gamma, fixed normalized Sylvester H2048 global R1 and shared H128 value-head R2, including embedding and final norm/LM head. Preserve per-output-channel symmetric W4 [-7,7], FP16 activations, Q/K norms and RoPE semantics, runtime schedules and physical contracts. Compare original A, identity-fold B and rotated C. Do not introduce learned rotations, seed search, R3/R4, group quantization or online rotation nodes. F16F16 and W4U8 runtime/packages/results remain frozen. Changed W4 bytes are explicitly authorized; no activation-only attribution against frozen W4U8. Verify unquantized invariance and actual DSP quality on unchanged qbh-lite-v1 before claiming effectiveness. Report observed accuracy and speed; do not automatically promote. Routine recovery remains governed by PC-037.
