# L32-0047 additive modules
Units microseconds (% complete Host wall).
| Module | M64 prefill | Decode per token |
|---|---:|---:|
| I/O、metadata | 417.217 (0.32%) | 483.388 (0.38%) |
| Input RMSNorm | 660.164 (0.51%) | 654.701 (0.52%) |
| QKV＋RoPE | 21403.508 (16.49%) | 22803.809 (18.08%) |
| QK–Softmax–AV | 15598.762 (12.02%) | 6606.894 (5.24%) |
| O projection | 11091.223 (8.55%) | 11706.719 (9.28%) |
| Post-attention residual＋RMSNorm | 636.756 (0.49%) | 632.642 (0.50%) |
| Gate/Up＋SwiGLU | 54528.390 (42.02%) | 54911.053 (43.54%) |
| Down | 15584.367 (12.01%) | 16183.789 (12.83%) |
| Final residual | 208.206 (0.16%) | 207.219 (0.16%) |
| KV carrier conversion | 358.836 (0.28%) | 421.375 (0.33%) |
| KV append DMA | 504.906 (0.39%) | 249.673 (0.20%) |
| Block orchestration | 22.887 (0.02%) | 17.796 (0.01%) |
| Layer bookkeeping | 30.104 (0.02%) | 17.672 (0.01%) |
| Stage-boundary bookkeeping | 7.598 (0.01%) | 1.730 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 106.534 (0.08%) | 63.463 (0.05%) |
| Embedding | 53.554 (0.04%) | 2.178 (0.00%) |
| Final model RMSNorm | 65.298 (0.05%) | 3.111 (0.00%) |
| LM head＋greedy（不含 final norm） | 7780.852 (6.00%) | 9297.843 (7.37%) |
| Host–DSP 边界 | 699.506 (0.54%) | 1841.837 (1.46%) |
| 完整 Host wall | 129758.669 (100.00%) | 126106.892 (100.00%) |
