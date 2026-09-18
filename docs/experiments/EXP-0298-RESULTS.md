# EXP-0298: paired ordinary W4A8 and W4A8-SP2 for ABC configurations

Only the Down input quantizer/LUT and its existing ordinary-A8 execution selection differ. All W4 codes/scales, non-Down quantization parameters, FP32 residual policy, no-rotation setting and optimized scheduling are frozen. The long-prefill API whitelist now accepts both ordinary A8 and SP2. No weight requantization, new calibration or optimization candidate search was performed.

Independent actual-arithmetic references verify every layer output, KV and selected head output for full sequences and partial chunk boundaries. Frozen reference bytes are checked against authority-pinned parent evidence ledgers. Five short paired rounds precede ten formal paired rounds, with ten replays per formal process and rotated order. All samples are retained.

Short A uses the native M64 path with 42 decode RPCs. B/C use the existing chunked long frontend and cache832. Long Host wall includes Host token/RoPE staging; native short keeps its established RPC timing definition. Cold loading, tokenization and audit writes are excluded. These are fixed project-owned input fixtures matching the requested lengths, not full named-dataset or PPL evaluation.

The initial Qwen1.7 A8 long attempt was rejected before computation by the old SP2-only whitelist. It and the prior binary remain archived. After the minimal whitelist change, the registered checks and measurements passed. A Windows hardlink-limit error during package creation was repaired by verified copying of the affected derived files; frozen parents were not edited.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Qwen3-0.6B | 64+42 | W4A8 | 3060.78 | 94.08 |
| Qwen3-0.6B | 64+42 | W4A8-SP2 | 3011.16 | 93.59 |
| Qwen3-0.6B | 536+46 | W4A8 | 2643.23 | 63.04 |
| Qwen3-0.6B | 536+46 | W4A8-SP2 | 2599.09 | 62.83 |
| Qwen3-0.6B | 741+3 | W4A8 | 2580.34 | 57.59 |
| Qwen3-0.6B | 741+3 | W4A8-SP2 | 2546.67 | 57.54 |
| Qwen3-1.7B | 64+42 | W4A8 | 1861.88 | 47.39 |
| Qwen3-1.7B | 64+42 | W4A8-SP2 | 1845.52 | 47.27 |
| Qwen3-1.7B | 536+46 | W4A8 | 1841.23 | 43.31 |
| Qwen3-1.7B | 536+46 | W4A8-SP2 | 1826.32 | 43.18 |
| Qwen3-1.7B | 741+3 | W4A8 | 1871.29 | 41.94 |
| Qwen3-1.7B | 741+3 | W4A8-SP2 | 1857.84 | 41.91 |

| Model | Shape | SP2 prefill wall overhead over A8 | SP2 decode wall overhead over A8 |
|---|---|---:|---:|
| 0.6B | 64+42 | +1.65% | +0.52% |
| 0.6B | 536+46 | +1.70% | +0.33% |
| 0.6B | 741+3 | +1.32% | +0.10% |
| 1.7B | 64+42 | +0.89% | +0.25% |
| 1.7B | 536+46 | +0.82% | +0.30% |
| 1.7B | 741+3 | +0.72% | +0.07% |

Formal verification covers 45200 timed RPCs. Numerical/physical acceptance is separate from long/M64 throughput parity and model quality. The unchanged throughput gate is paired95% CI lower bound >=0.90; per-model reports include every gate and confidence interval. Small-model optimization remains deferred. No automatic Selected-baseline promotion.

Per-model module profiles, complete per-process timing arrays, runtime seals, package manifests and independent checks are retained under each model directory. Parent archives remain unchanged.

## Evidence and verification scope

Evidence root: /mnt/d/llm_exp/results/qwen3-block-htp/exp0298. EVIDENCE_SHA256.json SHA256: ca07e29994a372f198de08a24fdf0a8ddc82de3d1efbe26d3b05640a20d632bb. Source: 19a6861c8f556360a0565116fa716b56fd5712cb. Ledger covers 27075 files, 22064323793 bytes.

Full layer/KV audits use the chunked frontend at 64+42, 65+3, 128+3, 129+3, 536+46 and 741+3 for A8, and 64+42/741+3 for SP2; prior frozen SP2 references are inherited where unchanged. Native M64 timing uses the established short path, independently checked selected head tokens/codes for every step and physical invariants; it is not an additional native full-boundary dump. No model-quality acceptance is inferred.
