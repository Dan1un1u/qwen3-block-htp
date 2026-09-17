# L32-0044 full-model long decode modules

Units µs; parentheses complete Host-wall shares. M64+42/cache128, five short and ten formal paired AB/BA cycles,repeat10. Frozen weights/scales,SP2,FP32 residual,no rotations.

| 模块 | CONTROL Prefill | OPT Prefill | CONTROL Decode | OPT Decode |
|---|---:|---:|---:|---:|
| I/O、metadata | 247.7 (0.44%) | 244.1 (0.43%) | 247.3 (0.53%) | 244.1 (0.55%) |
| Input RMSNorm | 3534.9 (6.24%) | 3533.2 (6.28%) | 2294.1 (4.96%) | 2294.4 (5.17%) |
| QKV＋RoPE | 11125.1 (19.65%) | 11122.6 (19.78%) | 5902.5 (12.77%) | 5901.0 (13.30%) |
| QK–Softmax–AV | 6641.8 (11.73%) | 6648.1 (11.82%) | 8029.8 (17.37%) | 6558.4 (14.78%) |
| O projection | 4114.0 (7.27%) | 4125.5 (7.34%) | 2884.1 (6.24%) | 2884.5 (6.50%) |
| Post-attention residual＋RMSNorm | 3566.0 (6.30%) | 3565.1 (6.34%) | 2301.9 (4.98%) | 2301.9 (5.19%) |
| Gate/Up＋SwiGLU | 13738.1 (24.27%) | 13731.1 (24.42%) | 12435.6 (26.90%) | 12440.0 (28.04%) |
| Down | 7906.2 (13.97%) | 7915.4 (14.08%) | 6855.1 (14.83%) | 6851.7 (15.44%) |
| Final residual | 3.8 (0.01%) | 3.6 (0.01%) | 2.0 (0.00%) | 1.9 (0.00%) |
| KV carrier conversion | 84.2 (0.15%) | 84.1 (0.15%) | 15.7 (0.03%) | 15.6 (0.04%) |
| KV append DMA | 238.0 (0.42%) | 237.2 (0.42%) | 165.8 (0.36%) | 165.4 (0.37%) |
| Block orchestration | 37.4 (0.07%) | 35.6 (0.06%) | 28.7 (0.06%) | 28.4 (0.06%) |
| Layer bookkeeping | 26.9 (0.05%) | 25.3 (0.05%) | 16.2 (0.03%) | 15.8 (0.04%) |
| Stage-boundary bookkeeping | 7.0 (0.01%) | 7.6 (0.01%) | 1.6 (0.00%) | 1.7 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 94.9 (0.17%) | 93.3 (0.17%) | 51.1 (0.11%) | 51.0 (0.11%) |
| Embedding | 72.9 (0.13%) | 71.8 (0.13%) | 2.5 (0.01%) | 2.4 (0.01%) |
| Final model RMSNorm | 85.5 (0.15%) | 85.8 (0.15%) | 83.0 (0.18%) | 83.0 (0.19%) |
| LM head＋greedy（不含 final norm） | 4261.4 (7.53%) | 3928.5 (6.99%) | 4265.5 (9.23%) | 3926.7 (8.85%) |
| Host–DSP 边界 | 818.5 (1.45%) | 766.6 (1.36%) | 642.5 (1.39%) | 601.2 (1.35%) |
| 完整 Host wall | 56604.4 (100.00%) | 56224.4 (100.00%) | 46224.9 (100.00%) | 44369.1 (100.00%) |

| Shape | Arm | Prefill E2E token/s | Decode E2E token/s |
|---|---|---:|---:|
| M64+42 | CONTROL | 1130.66 | 21.63 |
| M64+42 | OPT | 1138.30 | 22.54 |
| M64+63 | CONTROL | 1124.99 | 21.21 |
| M64+63 | OPT | 1132.03 | 22.13 |

63decode is a five-pair supplement, separate from primary ten-pair formal42decode. Cold loading/session setup and external tokenization excluded. Numerical implementation verification is separate from model quality.
