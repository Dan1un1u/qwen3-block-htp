# L32-0043: 3B shape-specific pipeline/layout optimization

M64+15, KV capacity80,28layers;5 short and10 formal paired AB/BA cycles,repeat10. Complete Host wall includes embedding,all blocks,final norm,head,greedy and RPC; excludes cold loading/session setup and external tokenization. Frozen original3B weights/scales/SP2mode8,FP32 residual,no rotations.

| 模块 | 原实现 Prefill μs（占比） | 优化 Prefill μs（占比） | 原实现 Decode μs/token（占比） | 优化 Decode μs/token（占比） |
|---|---:|---:|---:|---:|
| I/O、metadata | 238.5 (0.39%) | 239.9 (0.43%) | 238.1 (0.50%) | 239.1 (0.53%) |
| Input RMSNorm | 3536.4 (5.75%) | 3535.1 (6.30%) | 2291.3 (4.86%) | 2291.5 (5.10%) |
| QKV＋RoPE | 11026.4 (17.93%) | 11033.5 (19.65%) | 5818.1 (12.33%) | 5818.9 (12.94%) |
| QK–Softmax–AV | 11914.7 (19.37%) | 6644.9 (11.84%) | 9511.8 (20.17%) | 7324.6 (16.29%) |
| O projection | 4151.9 (6.75%) | 4127.1 (7.35%) | 2861.3 (6.07%) | 2859.4 (6.36%) |
| Post-attention residual＋RMSNorm | 3564.2 (5.80%) | 3566.1 (6.35%) | 2303.6 (4.88%) | 2304.1 (5.12%) |
| Gate/Up＋SwiGLU | 13444.6 (21.86%) | 13443.2 (23.95%) | 12168.3 (25.80%) | 12181.8 (27.09%) |
| Down | 8009.0 (13.02%) | 7944.3 (14.15%) | 6719.8 (14.25%) | 6722.4 (14.95%) |
| Final residual | 3.1 (0.01%) | 3.4 (0.01%) | 1.9 (0.00%) | 2.0 (0.00%) |
| KV carrier conversion | 84.2 (0.14%) | 84.0 (0.15%) | 15.6 (0.03%) | 15.6 (0.03%) |
| KV append DMA | 239.8 (0.39%) | 240.5 (0.43%) | 168.0 (0.36%) | 168.6 (0.37%) |
| Block orchestration | 35.6 (0.06%) | 37.2 (0.07%) | 28.9 (0.06%) | 28.8 (0.06%) |
| Layer bookkeeping | 25.8 (0.04%) | 26.1 (0.05%) | 15.9 (0.03%) | 16.2 (0.04%) |
| Stage-boundary bookkeeping | 7.5 (0.01%) | 6.6 (0.01%) | 1.7 (0.00%) | 1.6 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 91.7 (0.15%) | 92.1 (0.16%) | 51.2 (0.11%) | 51.2 (0.11%) |
| Embedding | 69.7 (0.11%) | 70.2 (0.13%) | 2.1 (0.00%) | 2.1 (0.00%) |
| Final model RMSNorm | 85.5 (0.14%) | 85.3 (0.15%) | 82.9 (0.18%) | 82.9 (0.18%) |
| LM head＋greedy（不含 final norm） | 4261.4 (6.93%) | 4243.7 (7.56%) | 4254.6 (9.02%) | 4244.4 (9.44%) |
| Host–DSP 边界 | 707.6 (1.15%) | 718.0 (1.28%) | 633.1 (1.34%) | 620.1 (1.38%) |
| 完整 Host wall | 61497.6 (100.00%) | 56141.2 (100.00%) | 47168.2 (100.00%) | 44975.5 (100.00%) |

| 配置 | Prefill E2E token/s | Decode E2E token/s |
|---|---:|---:|
| CONTROL | 1040.69 | 21.20 |
| OPT3 | 1139.98 | 22.23 |

Paired-bootstrap95% CIs (resampling matched cycles,20000 samples,seed320043):
- prefill: candidate/control wall 0.91290059, CI[0.9108389434088594, 0.9147267253369943]; throughput +9.541%.
- decode: candidate/control wall 0.95351479, CI[0.9522453565588638, 0.9546675228993019]; throughput +4.875%.

All3200 formal additive profiles reconcile; Weight bytes match; prefill reduces3584 HMX submissions, decode removes10752 padding tile pairs per token. Live arithmetic unchanged. VTCM peak8360416 bytes; zero intermediate DDR/spill. No quality claim or automatic baseline promotion.
