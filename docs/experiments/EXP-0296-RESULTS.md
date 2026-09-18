# EXP-0296: Qwen long-context throughput loop

Frozen W4A8-SP2, no rotation, FP32 residual. Same-binary opt31 controls and opt927 candidates. Numerical/physical checks pass; no automatic baseline or quality promotion.

| Model | Shape | Old long prefill TPS | New prefill TPS | Old long decode TPS | New decode TPS | Long/M64 prefill ratio | Parity |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 536+46 | 2489.19 | 2603.42 | 24.91 | 62.94 | 0.880326 | False |
| Qwen3-0.6B | 741+3 | 2413.07 | 2548.72 | 21.15 | 57.60 | 0.861829 | False |
| Qwen3-1.7B | 536+46 | 1693.33 | 1744.95 | 19.96 | 39.03 | 0.951760 | False |
| Qwen3-1.7B | 741+3 | 1674.31 | 1741.36 | 17.50 | 37.12 | 0.949803 | False |

Both models completed six poisoned independent boundary audits, five short cycles and ten rotated formal repeat10 cycles. Complete Host wall includes long Host staging. Exact8MiB VTCM, one HMX/DMA owner, no timed intermediate DDR/spill; all timed head outputs and additive ledgers validated.

Read per-model RESULTS.md for full latency, paired95% intervals, boundary counts and reference choice; MODULES.md for additive module profiles. ENGINEERING.md preserves failed candidates, including the excluded V matching LUT. Source and weights frozen before formal profiling.

Prefill parity remains a separate criterion from decode speedup. Do not replace the original M64 reference with a slower implementation or reinterpret the predeclared gate.
