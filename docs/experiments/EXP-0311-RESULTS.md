# EXP-0311 — Are HVX and HMX VTCM bandwidths independent?

Completed 2026-09-23 on PJZ110 / SM8750 / HTP V79.

**Conclusion:** the observed sustained read throughputs are not independent under simultaneous saturation. HMX maintains approximately its solo rate; HVX loses about38–40% with six HVX readers. The units execute concurrently, but their separately measured maxima cannot be added as an attainable simultaneous rate. This establishes empirical asymmetric interference; it does not identify a physical shared bus, bank map, arbiter policy, or internal port topology.

## Matched formal results

Decimal GB/s. Six128-byte HVX contexts. Medians over three sessions × ten retained50ms windows per row and mode. Two additional warmup windows per session discarded. Same binary, power requests, data layout, loop kernels and timing window for solo/concurrent/control. These fresh controls are used instead of comparing directly against EXP0310's differently batched fixed-work timing.

| HMX mode | Layout | HVX solo | HVX concurrent | HVX loss | HMX solo | HMX concurrent | Concurrent total | Sum of solo peaks retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 × FP16 | Disjoint | 809.33 | 491.91 | 39.22% | 1058.98 | 1058.98 | 1550.89 | 83.01% |
| FP16 × FP16 | Shared read-only | 809.33 | 491.87 | 39.23% | 1058.98 | 1058.98 | 1550.84 | 83.01% |
| U8 × S8 | Disjoint | 809.33 | 500.80 | 38.12% | 928.62 | 928.62 | 1429.42 | 82.25% |
| U8 × S8 | Shared read-only | 809.33 | 500.84 | 38.12% | 928.62 | 928.62 | 1429.46 | 82.25% |
| U8 × packed S4 | Disjoint | 809.33 | 489.89 | 39.47% | 773.85 | 773.85 | 1263.74 | 79.82% |
| U8 × packed S4 | Shared read-only | 809.33 | 489.77 | 39.48% | 773.85 | 773.85 | 1263.62 | 79.82% |

HMX rate quantization is approximately0.67GB/s for FP16 and0.42GB/s for W4 per completed batch/window. “Unchanged” means no resolved loss at this measurement granularity, not proof of exactly zero interference.

## Controls and attribution

1. **Thread-count sweep:** scan HVX1/2/4/5/6 contexts, with FP16, U8S8 and U8S4 HMX. Interference already occurs with one HVX context, so it is not exclusively caused by adding a seventh worker alongside six HVX workers. For example, FP16 HMX plus one HVX reader reduces HVX from~134.9 to~74.0GB/s; HMX stays~1059GB/s.
2. **Extra scalar-worker control:** replace the HMX worker by a continuously busy scalar NOP loop, retaining six HVX readers, session resource allocation and common start/deadline. HVX remains~808.7GB/s versus~809.3GB/s solo (about0.07–0.08% loss), far below the actual HMX interference. Merely creating another busy thread does not reproduce the effect. This control does not rule out every HMX-instruction-specific issue or arbitration interaction.
3. **Read-only address controls:** disjoint regions, common read-only source region, and a disjoint layout with per-HVX-source2KiB skews all produce similar losses. No inter-worker producer-consumer dependency or read/write data alias is introduced. Disjoint virtual/VTCM address ranges are not proof of disjoint physical banks.
4. **Clock and synchronization:** cycle/qtimer ratio remains~2112MHz in all modes. Common-window start/tail delays are recorded; all pass predeclared<1%window limits. Tail batches that finish after the common deadline are excluded from bytes, not credited to the interval.
5. **Correctness:** all178 successful RPC invocations (7smoke,99scan,72formal) return zero error/status; formal checks verify8MiB VTCM and hardware HVX count6. FP16 HMX output is the exact expected0.25; integer HMX follows its positive accumulation and saturation contract; final HVX load-register bytes match the known source. Buffers and output locations are nonoverlapping where writable. No model data or default baseline changes.

A simple model of two fully isolated throughput capacities is inconsistent with these results. A single fixed aggregate byte/s ceiling is also not established: concurrent totals depend on HMX operand format. The data are consistent with contention somewhere along the VTCM/HVX/HMX execution and access paths, with an asymmetric observed outcome favoring HMX throughput. Determining exact circuitry or priority would require architectural documentation or appropriate hardware counters.

## Implementation and byte accounting

Use prepared full8MiB VTCM and existing project Turbo_L3 core/bus performance requests. All workers acquire their contexts before timing and wait for a common future start. A shared50,000µs window is fixed independently of completion order. Each worker executes repeated batches until deadline, counting a batch only if its completion barrier is before the deadline. Report each consumer's completed bytes / common50ms; do not sum isolated worker durations, include solo tails, or infer overlap from host RPC timing.

- HMX activation source is VTCM[0,1MiB), weights start at1MiB. FP16 weight extent1MiB, S8 0.5MiB, packedS4 0.25MiB. Stream length32tiles; each batch executes16full operand traversals with accumulator clearing and output store. FP16 operands are exact1/256 values. U8 and S8 values are1; S4 packed bytes0x11.
- Disjoint HVX sources start at2MiB+i×512KiB. Shared-source mode has all HVX threads read the first512KiB of HMX's activation source. Skewed scan adds i×2KiB to disjoint starts. Each HVX batch performs8traversals using the audited32-load hardware-counted read loop.
- HVX verification output is at6MiB+i×2KiB, HMX output at7MiB, bias at7MiB+4KiB. These locations are outside every read range and do not alias each other. Initialization and validation happen outside timed windows.
- HMX counts2048activation+2048FP16weight, or2048+1024S8, or2048+512packedS4 bytes per tile, never expanded/reused logical MAC bytes. HVX counts128bytes per explicit vector load. Small stores, metadata, control overhead remain in elapsed time but are not added to input-byte counts. HMX still executes MACs; this is effective operand throughput, not direct physical-bus counter data.
- The scalar control performs a busy scalar loop without matrix loads or HVX instructions. The controller waits on semaphores during measurement. Shared timing fields are published before go semaphores.

## Evidence and reproduction

Measured source `14bb13c08304ebff8f18fa38a0a93b29680a2c89`, source branch `codex/exp-0311-vtcm-concurrent-bandwidth`.
Evidence directory: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0311`.

- `seal1/manifest.json`, matched device SHA256 and saved disassembly/build caches.
- `results.jsonl`, `formal_*.log`, `formal_cases.json`, `summary.json`: all raw and summarized measurements.
- Battery/temperature snapshots surrounding each formal session; no clock or thermal-policy override.
- Versioned `scripts/exp0311_concurrent_bandwidth.py`, `experiments/exp0311_concurrent_formal_cases.json`.
- Current CLI modes7HVX-only,8HMX-only,9both,10HVX+scalar. `bytes=1048576`, `workers=1..6`, `repeats=window_us`, `stream=32`, `depth=layout1disjoint/2shared/3skew`, `bypass=HMXformat3FP16/4S8/5S4` (this field is format selection only for concurrent modes; no DMA participates). Deploy a matching host/stub/skel build under `/data/local/tmp/qbh_bw0311` and follow authorized project preflight. Use a new results directory when replaying to retain historical logs.

Initial host JSON-quoting compile failure is preserved in build1.log; repaired build2 generated the only measured seal. It did not produce a hardware result. All hardware measurement output checks passed; no acceptance threshold was relaxed.

## Practical use

Model simultaneous HMX and HVX execution with measured interference rather than blindly adding isolated maxima. This does not imply that every real pipeline loses40%: arithmetic-heavy HVX tasks, different memory intensity, access shape and release timing may overlap differently. A runtime pipeline benchmark is still required for end-to-end claims.

Formal maximum start delay: 1.979µs; maximum completion tail: 39.792µs (excluded from useful byte counts after deadline).
