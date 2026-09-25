# B/C W16A16 and W4A16 inference DRAM (2026-09-25)
EXP-0316 / L32-0077. Sixteen actual fresh device processes, archived baseline command/sealed binaries and all package hashes verified. B536+46 and C741+3, cache capacity832, ten trajectories per process. Source/runtime unchanged. A and all throughput cells preserved.
Metric: post-ready maximum of unique process DMA-BUF backing bytes plus ordinary RSS excluding DMA-BUF mapped RSS. Decimal MB=10^6 bytes. Excludes loading, VTCM, sampler, hidden driver/DSP-private memory and unmapped filesystem cache. This is sampled process-attributable DRAM, not a whole-device instantaneous peak. Long entry emits generation_startup after prepare; external ready marker feeds the unchanged native collector. smaps/fdinfo reads are not atomic. DMA-BUF totals match dmabuf_dump for all runs.
Historical long_step fields excluding wall time match; output_hash and recorded numerical/physical flags match. Disabled per-layer hashes remain zero and are not independent full-tensor validation. Each B has550 steps and C150 steps, ten complete trajectories. No new model-quality claim. B/C may have nearly equal memory because they reserve the same832-slot cache; each was independently measured.

| Configuration | Peak MB | Peak MiB | Samples |
|---|---:|---:|---:|
| qwen06-B-w16 | 1703.62 | 1624.70 | 548 |
| qwen06-B-w4 | 812.47 | 774.84 | 565 |
| qwen06-C-w16 | 1703.61 | 1624.69 | 144 |
| qwen06-C-w4 | 812.55 | 774.91 | 160 |
| qwen17-B-w16 | 4263.61 | 4066.10 | 938 |
| qwen17-B-w4 | 1688.82 | 1610.58 | 896 |
| qwen17-C-w16 | 4263.76 | 4066.24 | 247 |
| qwen17-C-w4 | 1688.75 | 1610.52 | 263 |
| llama1-B-w16 | 3060.29 | 2918.52 | 685 |
| llama1-B-w4 | 1210.81 | 1154.72 | 624 |
| llama1-C-w16 | 3060.41 | 2918.63 | 177 |
| llama1-C-w4 | 1210.88 | 1154.78 | 174 |
| llama3-B-w16 | 7414.27 | 7070.80 | 1433 |
| llama3-B-w4 | 2599.21 | 2478.80 | 1520 |
| llama3-C-w16 | 7414.17 | 7070.71 | 389 |
| llama3-C-w4 | 2598.96 | 2478.56 | 477 |

Target Paper_Hardware_Tables_ABC.xlsx: B_Persona-Chat and C_DroidTask F7/F8/F12/F13/F17/F18/F22/F23 only.

Evidence SHA256: `6ff3a7776ea67a892c742a78c40f857b472832445720016e4185fd3e7e90aa39`. Workbook SHA256: `42bfbf4f01ec5bf62682cb598ba4217a058d91c1a18d1372ec53c18a34cbdb1c`.
