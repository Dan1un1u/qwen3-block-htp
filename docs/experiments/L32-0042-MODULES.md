# L32-0042: 3B shape-specific pipeline/layout optimization

M64+15, KV capacity80,28layers;5 short and10 formal paired AB/BA cycles,repeat10. Complete Host wall includes embedding,all blocks,final norm,head,greedy and RPC; excludes cold loading/session setup and external tokenization. Frozen original3B weights/scales/SP2mode8,FP32 residual,no rotations.

| 模块 | 原实现 Prefill μs（占比） | 优化 Prefill μs（占比） | 原实现 Decode μs/token（占比） | 优化 Decode μs/token（占比） |
|---|---:|---:|---:|---:|
| I/O、metadata | 250.6 (0.20%) | 250.2 (0.41%) | 249.3 (0.47%) | 247.7 (0.52%) |
| Input RMSNorm | 3535.1 (2.83%) | 3535.0 (5.74%) | 2291.3 (4.28%) | 2291.3 (4.85%) |
| QKV＋RoPE | 49204.7 (39.37%) | 11037.0 (17.91%) | 7833.2 (14.63%) | 5827.7 (12.33%) |
| QK–Softmax–AV | 27664.8 (22.13%) | 11959.3 (19.41%) | 13731.6 (25.65%) | 9528.9 (20.16%) |
| O projection | 4130.9 (3.30%) | 4133.7 (6.71%) | 2854.2 (5.33%) | 2853.4 (6.04%) |
| Post-attention residual＋RMSNorm | 3565.1 (2.85%) | 3564.4 (5.79%) | 2304.1 (4.30%) | 2303.5 (4.87%) |
| Gate/Up＋SwiGLU | 13414.9 (10.73%) | 13439.8 (21.81%) | 12149.2 (22.69%) | 12164.4 (25.73%) |
| Down | 7962.8 (6.37%) | 7975.7 (12.94%) | 6717.4 (12.55%) | 6729.0 (14.23%) |
| Final residual | 3.1 (0.00%) | 3.0 (0.00%) | 1.9 (0.00%) | 1.9 (0.00%) |
| KV carrier conversion | 9491.1 (7.59%) | 84.3 (0.14%) | 159.8 (0.30%) | 15.6 (0.03%) |
| KV append DMA | 348.3 (0.28%) | 240.9 (0.39%) | 187.3 (0.35%) | 168.0 (0.36%) |
| Block orchestration | 36.8 (0.03%) | 35.8 (0.06%) | 28.8 (0.05%) | 28.9 (0.06%) |
| Layer bookkeeping | 24.4 (0.02%) | 26.1 (0.04%) | 16.0 (0.03%) | 15.9 (0.03%) |
| Stage-boundary bookkeeping | 7.3 (0.01%) | 7.6 (0.01%) | 1.7 (0.00%) | 1.7 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 92.6 (0.07%) | 91.8 (0.15%) | 51.5 (0.10%) | 51.2 (0.11%) |
| Embedding | 70.0 (0.06%) | 69.9 (0.11%) | 2.3 (0.00%) | 2.1 (0.00%) |
| Final model RMSNorm | 85.8 (0.07%) | 85.5 (0.14%) | 83.0 (0.16%) | 82.9 (0.18%) |
| LM head＋greedy（不含 final norm） | 4240.9 (3.39%) | 4251.9 (6.90%) | 4245.4 (7.93%) | 4251.8 (8.99%) |
| Host–DSP 边界 | 864.9 (0.69%) | 821.4 (1.33%) | 627.1 (1.17%) | 710.7 (1.50%) |
| 完整 Host wall | 124994.1 (100.00%) | 61613.2 (100.00%) | 53535.0 (100.00%) | 47276.6 (100.00%) |

| 配置 | Prefill E2E token/s | Decode E2E token/s |
|---|---:|---:|
| CONTROL | 512.02 | 18.68 |
| OPT2 | 1038.74 | 21.15 |

Paired-bootstrap95% CIs (resampling matched cycles,20000 samples,seed320042):
- prefill: candidate/control wall 0.49292928, CI[0.49240170066314065, 0.49349670227505377]; throughput +102.869%.
- decode: candidate/control wall 0.88309800, CI[0.8822698085549525, 0.883800038667266]; throughput +13.238%.

All3200 formal additive profiles reconcile; HMX commands,tile work and weight bytes match by token. VTCM peak8360416 bytes; zero intermediate DDR/spill. No quality claim or automatic baseline promotion.
