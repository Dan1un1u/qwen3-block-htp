# L32-0054: paired ordinary W4A8 and W4A8-SP2 for ABC configurations

Only the Down input quantizer/LUT and its existing ordinary-A8 execution selection differ. All W4 codes/scales, non-Down quantization parameters, FP32 residual policy, no-rotation setting and optimized scheduling are frozen. The long-prefill API whitelist now accepts both ordinary A8 and SP2. No weight requantization, new calibration or optimization candidate search was performed.

Independent actual-arithmetic references verify every layer output, KV and selected head output for full sequences and partial chunk boundaries. Frozen reference bytes are checked against authority-pinned parent evidence ledgers. Five short paired rounds precede ten formal paired rounds, with ten replays per formal process and rotated order. All samples are retained.

Short A uses the native M64 path with 42 decode RPCs. B/C use the existing chunked long frontend and cache832. Long Host wall includes Host token/RoPE staging; native short keeps its established RPC timing definition. Cold loading, tokenization and audit writes are excluded. These are fixed project-owned input fixtures matching the requested lengths, not full named-dataset or PPL evaluation.

Llama1B native short KV storage was extended from80 to128 slots for64+42, identically for A8/SP2. This is a capacity adaptation, with the existing arithmetic and scheduling retained. Three-billion-parameter native short already uses128. The first 1B native42 attempt stopped at decode16 because its inherited fixture supplied only15 RoPE steps. A fresh derived package extends both recipes to42 using the verified long RoPE table; all existing30 sine/cosine files match exactly, and54 missing files are added. The failed attempt is preserved. No kernel, weight, scale or tolerance change was needed.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Llama3.2-1B | 64+42 | W4A8 | 2378.88 | 44.89 |
| Llama3.2-1B | 64+42 | W4A8-SP2 | 2343.83 | 44.87 |
| Llama3.2-1B | 536+46 | W4A8 | 2556.54 | 37.72 |
| Llama3.2-1B | 536+46 | W4A8-SP2 | 2503.94 | 37.69 |
| Llama3.2-1B | 741+3 | W4A8 | 2544.45 | 34.17 |
| Llama3.2-1B | 741+3 | W4A8-SP2 | 2506.66 | 34.22 |
| Llama3.2-3B | 64+42 | W4A8 | 1206.28 | 23.69 |
| Llama3.2-3B | 64+42 | W4A8-SP2 | 1183.81 | 23.68 |
| Llama3.2-3B | 536+46 | W4A8 | 1174.79 | 23.79 |
| Llama3.2-3B | 536+46 | W4A8-SP2 | 1149.87 | 23.76 |
| Llama3.2-3B | 741+3 | W4A8 | 1192.46 | 23.16 |
| Llama3.2-3B | 741+3 | W4A8-SP2 | 1169.48 | 23.18 |

| Model | Shape | SP2 prefill wall overhead over A8 | SP2 decode wall overhead over A8 |
|---|---|---:|---:|
| 1B | 64+42 | +1.50% | +0.05% |
| 1B | 536+46 | +2.10% | +0.07% |
| 1B | 741+3 | +1.51% | -0.16% |
| 3B | 64+42 | +1.90% | +0.05% |
| 3B | 536+46 | +2.17% | +0.13% |
| 3B | 741+3 | +1.96% | -0.08% |

Formal verification covers 45200 timed RPCs. Numerical/physical acceptance is separate from long/M64 throughput parity and model quality. The unchanged throughput gate is paired95% CI lower bound >=0.90; per-model reports include every gate and confidence interval. Small-model optimization remains deferred. No automatic Selected-baseline promotion.

Per-model module profiles, complete per-process timing arrays, runtime seals, package manifests and independent checks are retained under each model directory. Parent archives remain unchanged.

## Verification scope

All-layer/KV audits use the chunked frontend at 64+42, 65+3, 128+3, 129+3, 536+46 and 741+3 for A8, and 64+42/741+3 for SP2. Native M64 uses the established short path: all selected head tokens/codes and physical invariants are checked; this is not an additional native full-boundary dump.

Measured source: ab9d7c2d0bf1061d10a945757589fb6df0ef64fc. Full evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0054. Final ledger will be pinned in the experiment closure.
