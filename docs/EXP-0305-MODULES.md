# EXP-0305 full additive modules

Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness and diagnostics. Four vector contexts; explicit QK/Softmax/AV phase boundaries and GateUp/SwiGLU boundary. Worker work sums are excluded from the additive ledger.

| Module | Prefill | One-step decode |
|---|---:|---:|
| Token embedding | 54.16 (0.07%) | 1.68 (0.00%) |
| Input staging | 0.54 (0.00%) | 0.38 (0.00%) |
| Metadata | 215.14 (0.28%) | 214.51 (0.49%) |
| Input RMSNorm | 2122.80 (2.80%) | 372.39 (0.86%) |
| QKV projection | 23603.33 (31.08%) | 23615.68 (54.27%) |
| Q/K Norm-RoPE (separate tail) | 0.54 (0.00%) | 0.29 (0.00%) |
| QK-Softmax-AV | 5936.27 (7.82%) | 1877.67 (4.31%) |
| O projection | 2100.54 (2.77%) | 1375.65 (3.16%) |
| Post-attention residual | 2045.34 (2.69%) | 353.83 (0.81%) |
| Post-attention RMSNorm (fused) | 1.93 (0.00%) | 1.10 (0.00%) |
| Gate/Up | 6670.93 (8.78%) | 6091.26 (14.00%) |
| SwiGLU | 22215.90 (29.25%) | 1268.73 (2.92%) |
| Down | 3842.78 (5.06%) | 3508.31 (8.06%) |
| Final residual | 2.12 (0.00%) | 1.81 (0.00%) |
| Output staging | 0.00 (0.00%) | 0.00 (0.00%) |
| KV-cache carrier conversion | 138.91 (0.18%) | 149.66 (0.34%) |
| KV-cache append DMA | 287.82 (0.38%) | 102.74 (0.24%) |
| Block orchestration | 36.90 (0.05%) | 29.10 (0.07%) |
| Layer bookkeeping | 22.54 (0.03%) | 17.53 (0.04%) |
| Stage-boundary bookkeeping | 19.53 (0.03%) | 1.80 (0.00%) |
| Final model RMSNorm | 13.92 (0.02%) | 16.25 (0.04%) |
| LM head + greedy selection (excluding final norm) | 5182.01 (6.82%) | 3196.13 (7.34%) |
| Runtime setup | 50.13 (0.07%) | 28.57 (0.07%) |
| Runtime teardown | 37.91 (0.05%) | 25.12 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Host-DSP boundary | 1343.55 (1.77%) | 1267.52 (2.91%) |

E2E: prefill: 842.71 token/s, decode_one_step: 22.98 token/s

Includes embedding, all 28 blocks, final norm, head, greedy and FastRPC. Diagnostic phase costs are not critical-path shares of a production-overlapped schedule. No model-quality or baseline promotion claim.
