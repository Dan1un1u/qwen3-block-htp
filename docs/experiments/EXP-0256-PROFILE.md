# EXP-0256 profiling record

Fixed-parameter integer A8 independent software confirmation only; no new device work was performed. Native EXP0252 binaries unchanged. Repeat1/repeat10 and short/formal timing are unmeasured.

| 模块 | F16A16 | W4A16 | W4A8 EXP0256 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | N/A | N/A | N/A | N/A |
| Input RMSNorm | N/A | N/A | N/A | N/A |
| QKV＋Q/K Norm-RoPE | N/A | N/A | N/A | N/A |
| QK–Softmax–AV | N/A | N/A | N/A | N/A |
| O projection | N/A | N/A | N/A | N/A |
| Post-attention residual＋RMSNorm | N/A | N/A | N/A | N/A |
| Gate/Up＋SwiGLU | N/A | N/A | N/A | N/A |
| Down | N/A | N/A | N/A | N/A |
| Final residual | N/A | N/A | N/A | N/A |
| KV carrier conversion | N/A | N/A | N/A | N/A |
| KV append DMA | N/A | N/A | N/A | N/A |
| Block orchestration | N/A | N/A | N/A | N/A |
| Layer bookkeeping | N/A | N/A | N/A | N/A |
| Stage-boundary bookkeeping | N/A | N/A | N/A | N/A |
| DSP unattributed | N/A | N/A | N/A | N/A |
| Runtime setup/teardown | N/A | N/A | N/A | N/A |
| Embedding | N/A | N/A | N/A | N/A |
| Final model RMSNorm | N/A | N/A | N/A | N/A |
| LM head＋greedy，不含 final norm | N/A | N/A | N/A | N/A |
| Host–DSP 边界 | N/A | N/A | N/A | N/A |
| 完整 Host wall | N/A | N/A | N/A | N/A |

Host/DSP timing, DMA/HMX/HVX overlap, workers, VTCM/spill/lifetime, hardware physical/numerical gates: N/A. No missing value is a measured zero. Software evaluation elapsed time is not device throughput.

E2E prefill/decode tokens/s: N/A (not measured). Original hardware >10percent complete Host-wall slowdown stop rule remains.
