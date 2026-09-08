# EXP-0253 profiling record

Software precision diagnostic only. No device builds/runs or repeat1/repeat10, short/formal timing. Native EXP0252 binaries unchanged. Values below are unmeasured, never measured zeros.

| 模块 | F16A16 | W4A16 | W4A8 EXP0253 | W4A8 相对 W4A16 增速 |
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

Full Host/DSP timing, DMA/HMX/HVX overlap, worker/call counts, VTCM/spill/lifetime, physical and numerical hardware gates: N/A (no hardware work). Evaluation elapsed seconds are not device throughput.

E2E prefill/decode token/s: N/A (not measured). Future hardware retains the >10percent per-layer Host-wall slowdown stop rule.
