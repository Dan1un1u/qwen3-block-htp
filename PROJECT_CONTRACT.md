# Standalone HTP Block Laboratory Contract

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
- **PC-005 — Fixed-scope prototype.** The target outcome is a usable Qwen3
  middle block, not tokenizer, sampling, whole-model orchestration, or a general
  graph compiler.
- **PC-006 — RPC boundary.** A tested Execution Unit uses one Host-to-DSP run
  invocation. Per-tile and per-kernel FastRPC calls are outside the contract.

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
- **PC-016 — Primary metric.** Equivalent-scope device wall latency is primary.
  DSP qtimer stages, DMA waits, bytes, and overlap measurements explain the
  result but do not replace wall latency.
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
- **PC-027 — Complete profiling closure.** Every experiment closure must retain
  and present a complete profiling comparison as defined in
  `docs/FULL_PROFILING_REPORT_CONTRACT.md`. The user-facing completion response
  must contain the tables rather than only linking to raw evidence. It must show
  absolute control and candidate values plus deltas for repeat one and repeat
  ten; every defined additive Block Timing Ledger interval; relevant overlapping
  HMX, HVX, DMA, wait, byte, command, VTCM, and residency counters; correctness
  and physical gates; and evidence provenance. Unchanged, zero, or unavailable
  fields remain visible and unavailable fields require a reason. An experiment
  may not be recorded as completed without this report, except when execution
  ended before profiling was possible, in which case the report records the
  failure boundary and marks the unavailable sections explicitly.
