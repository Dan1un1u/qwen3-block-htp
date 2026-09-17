# EXP-0286: inference DRAM peak — Qwen3-1.7B W4A8-SP2

## Result
Observed process-attributable inference DRAM peak: **2780.527 MiB / 2.715359 GiB**.
This is a sampled, coverage-limited measurement, not a certified whole-device maximum.

| Independent process launch | Peak MiB | Peak GiB | Inference snapshots |
|---|---:|---:|---:|
| 1 | 2780.504 | 2.715336 | 78 |
| 2 | 2780.453 | 2.715286 | 77 |
| 3 | 2780.527 | 2.715359 | 78 |

At the worst observed sample:
- Main rpcmem allocation: 2,907,529,216 bytes = 2772.835938 MiB.
- Three additional DMA-BUF objects: 12,288 bytes = 0.011719 MiB.
- Non-DMA ordinary resident memory: 7.679688 MiB.
- Total: 2915594240 bytes. Sum is from one snapshot, not a sum of separate peaks.
- DMA-BUF mapping RSS/PSS: **zero**, despite all backing pages allocated and initialized.
- Range across three launches: 0.074219 MiB.

## Frozen workload and provenance
SM8750 / HTP V79 (PJZ110), Qwen3-1.7B complete28-layer runtime.
W4A8-SP2 mode8, FP32 residual mode2, no online R3/R4, latest shared scheduling,
prefix enabled, batch1, prompt64,15 sequential decode steps, cache capacity128.
Each independent process runs10 fixed trajectories with loaded weights reused.
All480 measured generation profiles pass existing execution checks; all selected
token IDs and logit half-bit codes exactly reproduce the sealed FP32 baseline.
No model-quality/PPL acceptance is inferred. No speed-baseline replacement.

Measured binary: original sealed EXP0284 source eaea89c81dd0e5f3424dfdea9d1e2b47347f8d72.
Host and DSP binaries verified against archived SHA256 and on-device copies.
Source worktree remains at EXP0285 closure4c96e036c80404b3f22da4bf7170b5e58898edb1;
its experimental binary is NOT used. Full command/seal: baseline-protocol.json.

## Measurement method
1. Launch unchanged inference executable and an independent native Android sampler.
   The sampler's allocations are outside the measured process. No LD_PRELOAD,
   kernel edits, DSP changes, root, cache eviction, or unrelated-process termination.
2. Read /proc/PID/smaps and fdinfo, identify /dmabuf: mappings and DMA exporter fds,
   deduplicate backing allocations by inode. Verify sizes with dmabuf_dump PID.
3. For each snapshot compute:
   **unique DMA-BUF bytes + ordinary smaps RSS - RSS already attributed to DMA mappings**.
   This counts the Host/DSP shared physical buffer once. Do not add DSP virtual maps,
   VmSize, VmPeak, or VTCM. Do not sum separately observed component maxima.
4. Start inference classification only after the executable flushes eval_model_ready.
   Model loading snapshots are recorded separately and excluded from the headline.
   Stop when the process exits; cleanup samples may occur but cannot increase the
   recorded large steady inference allocation in these runs.
5. Requested sleep20ms; actual snapshot start interval median~47.9ms, max49.3ms
   in these runs. smaps+fdinfo collection is not atomic (max28.9ms per snapshot).
   Main allocation is fixed before inference and persists throughout. Additional
   startup RPC buffers disappear during teardown. Transient events between samples
   are not bounded by this measurement.

## What is included
Weights as actually resident in the current runtime, including simultaneously
retained physical layouts/packed bundles; KV/cache capacity; all other live shared
package storage; metadata; normal host heap/stack/code/shared-library RSS.
The rpcmem arena is physically zero-initialized before use, so this is not merely
uncommitted virtual address reservation. It includes current harness storage and
layout redundancy; it is not the mathematical minimum W4 model footprint.
FP32 residual live tiles reside in VTCM and are not added to this DRAM metric.

## Coverage limits and paper wording
Global /sys/kernel/dmabuf/buffers is permission-denied; debugfs FastRPC accounting
and per-process dmabuf_rss/HWM are unavailable on this device. dmabuf_dump warns
that kernel-only exported buffers are unavailable; its printed kernel_rss=0 is
**unknown coverage, not proof of zero kernel/DSP memory**.
Hidden DSP/firmware-private DRAM, driver kernel allocations, unassigned service
memory and reclaimable unmapped filesystem page cache are outside this measurement.
Ordinary RSS counts shared code pages fully; PSS is retained as a diagnostic.
VTCM is explicitly excluded. This is not a system MemAvailable before/after delta.

Suggested paper metric: **Peak observed process-attributable DRAM (MiB),
including deduplicated DMA-BUF allocations, excluding VTCM**.
Do not label this as a complete exact HTP+Host physical-memory peak without adding
privileged DSP/driver accounting. Results apply only to M64+15/cache128 and cannot
be copied to longer-context configurations.

## Retained evidence
collect_dram.c / collect_dram: external native collector.
run_three.sh: full, unchanged runtime command and collection protocol.
run1..3/: stdout profiles, raw samples, fdinfo, maps, peak smaps and dmabuf_dump.
pilot/: earlier coarse sampler, retained as preliminary evidence, not mixed into repeats.
summary.json: numerical results and coverage.
No model/runtime source or baseline changes.

## Primary references
- Linux proc/smaps: https://docs.kernel.org/filesystems/proc.html
- Android DMA-BUF per-process accounting:
  https://android.googlesource.com/kernel/common/+/5737a2075bd97dfd4b643fdb33cc49193b2c2d95
- AOSP dmabuf_dump accounting:
  https://android.googlesource.com/platform//system/core/+/41d77405d6e8bde886fab97e788759e37c30b49e%5E%21/

Evidence ledger SHA256: 73190654a1ef34c5356c8e16c6cf2ab3930e289681a79e169946212763e89ea3
