# L32-0045 full-model long decode modules

Units µs; parentheses complete Host-wall shares. M64+42/cache128, five short and ten formal paired AB/BA cycles,repeat10. Frozen weights/scales,SP2,FP32 residual,no rotations.

| 模块 | CONTROL Prefill | OPT Prefill | CONTROL Decode | OPT Decode |
|---|---:|---:|---:|---:|
| I/O、metadata | 247.1 (0.44%) | 247.8 (0.46%) | 248.2 (0.56%) | 247.2 (0.58%) |
| Input RMSNorm | 3533.4 (6.27%) | 3553.4 (6.58%) | 2294.5 (5.15%) | 2309.6 (5.46%) |
| QKV＋RoPE | 11140.5 (19.78%) | 9147.8 (16.93%) | 5928.7 (13.31%) | 3921.8 (9.27%) |
| QK–Softmax–AV | 6650.5 (11.81%) | 6649.0 (12.31%) | 6579.4 (14.77%) | 6580.1 (15.56%) |
| O projection | 4132.1 (7.34%) | 4123.5 (7.63%) | 2894.7 (6.50%) | 2895.4 (6.85%) |
| Post-attention residual＋RMSNorm | 3565.4 (6.33%) | 3567.2 (6.60%) | 2302.0 (5.17%) | 2301.6 (5.44%) |
| Gate/Up＋SwiGLU | 13789.3 (24.49%) | 14029.9 (25.97%) | 12520.7 (28.10%) | 12770.2 (30.19%) |
| Down | 7932.1 (14.09%) | 7366.1 (13.64%) | 6887.4 (15.46%) | 6359.4 (15.03%) |
| Final residual | 3.6 (0.01%) | 3.4 (0.01%) | 1.9 (0.00%) | 1.9 (0.00%) |
| KV carrier conversion | 84.1 (0.15%) | 81.2 (0.15%) | 15.6 (0.04%) | 12.6 (0.03%) |
| KV append DMA | 238.2 (0.42%) | 238.3 (0.44%) | 166.5 (0.37%) | 167.8 (0.40%) |
| Block orchestration | 35.9 (0.06%) | 35.2 (0.07%) | 28.4 (0.06%) | 27.3 (0.06%) |
| Layer bookkeeping | 24.8 (0.04%) | 24.5 (0.05%) | 15.9 (0.04%) | 16.0 (0.04%) |
| Stage-boundary bookkeeping | 7.8 (0.01%) | 7.9 (0.01%) | 1.7 (0.00%) | 1.7 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 93.6 (0.17%) | 93.7 (0.17%) | 51.1 (0.11%) | 51.1 (0.12%) |
| Embedding | 72.0 (0.13%) | 72.7 (0.13%) | 2.5 (0.01%) | 2.5 (0.01%) |
| Final model RMSNorm | 85.7 (0.15%) | 85.7 (0.16%) | 83.0 (0.19%) | 83.1 (0.20%) |
| LM head＋greedy（不含 final norm） | 3927.1 (6.97%) | 3935.5 (7.29%) | 3930.9 (8.82%) | 3941.5 (9.32%) |
| Host–DSP 边界 | 750.5 (1.33%) | 758.6 (1.40%) | 597.7 (1.34%) | 608.8 (1.44%) |
| 完整 Host wall | 56313.6 (100.00%) | 54021.3 (100.00%) | 44550.6 (100.00%) | 42299.7 (100.00%) |

| Shape | Arm | Prefill E2E token/s | Decode E2E token/s |
|---|---|---:|---:|
| M64+42 | CONTROL | 1136.49 | 22.45 |
| M64+42 | OPT | 1184.72 | 23.64 |

63decode is checked for exact arithmetic in untimed boundary-audit runs; no separate formal63 speed claim. Cold loading/session setup and external tokenization excluded. Numerical implementation verification is separate from model quality.
