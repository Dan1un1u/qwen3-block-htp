# Long-context throughput loop:2026-09-18

All rows are fullmodel W4A8-SP2, no rotation, FP32 residual. Same-binary opt31 controls; fixed token trajectories and frozen weights. These are length-matched project fixtures, not named-dataset scores.

| Model | Shape | Control prefill TPS | Candidate prefill TPS | Prefill gain | Control decode TPS | Candidate decode TPS | Decode gain | Long/M64 prefill ratio | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 536+46 | 2489.19 | 2603.42 | 4.59% | 24.91 | 62.94 | 152.71% | 0.880326 | False |
| Qwen3-0.6B | 741+3 | 2413.07 | 2548.72 | 5.62% | 21.15 | 57.60 | 172.37% | 0.861829 | False |
| Qwen3-1.7B | 536+46 | 1693.33 | 1744.95 | 3.05% | 19.96 | 39.03 | 95.59% | 0.951760 | False |
| Qwen3-1.7B | 741+3 | 1674.31 | 1741.36 | 4.00% | 17.50 | 37.12 | 112.10% | 0.949803 | False |
| Llama3.2-3B | 536+46 | 1114.45 | 1151.42 | 3.32% | 16.54 | 23.79 | 43.84% | 0.974122 | False |
| Llama3.2-3B | 741+3 | 1109.55 | 1168.46 | 5.31% | 14.78 | 23.16 | 56.69% | 0.988544 | False |

All three candidates pass six poisoned independent layer/KV/head checks.5short plus10rotated formal repeat10 cycles per model. Exact8MiB VTCM, one HMX/DMA owner, zero timed intermediate DDR/spill and complete additive ledgers. Weights and old independent oracle artifacts rechecked against their original hashes. No model-quality or selected-baseline promotion.

Long/M64 gate remains paired lower95>=1 against fastest same-binary measured64-token reference; decode guard remains paired upper95 wall ratio<=1.10. Native64 excludes Host input staging, retained as conservative reference. Complete long Host wall includes staging and FastRPC/model execution.

Qwen candidate927 and Llama candidate8095 are separate frozen measurements. The later Llama V LUT/tail Norm/clear candidates have not been measured on Qwen; do not credit their benefits to the Qwen rows. Llama1B prior passing results remain historical and were not rerun.

Next: first test those proven Llama changes on Qwen without changing math. Then prioritize repeated full-prefix K/V packing, probability preparation and producer/consumer overlap rather than changing weights or weakening the M64 reference.

Detailed evidence: Qwen exp0296/{0.6B,1.7B}/RESULTS.md and MODULES.md; Llama l32-0053/RESULTS.md, ENGINEERING.md and MODULES.md.
