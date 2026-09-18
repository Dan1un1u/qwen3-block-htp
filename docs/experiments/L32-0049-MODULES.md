# Additive module table
Units microseconds (% complete Host wall). Decode is per token.
| Module | M64 prefill | Decode per token |
|---|---:|---:|
| I/O、metadata | 171.018 (0.11%) | 164.315 (0.11%) |
| Input RMSNorm | 662.342 (0.42%) | 656.750 (0.45%) |
| QKV＋RoPE | 20831.686 (13.34%) | 20692.530 (14.04%) |
| QK–Softmax–AV | 15501.152 (9.92%) | 6313.625 (4.28%) |
| O projection | 12060.069 (7.72%) | 11979.705 (8.13%) |
| Post-attention residual＋RMSNorm | 637.086 (0.41%) | 632.514 (0.43%) |
| Gate/Up＋SwiGLU | 61125.507 (39.14%) | 60800.926 (41.25%) |
| Down | 28065.478 (17.97%) | 27894.440 (18.93%) |
| Final residual | 208.269 (0.13%) | 207.919 (0.14%) |
| KV carrier conversion | 364.769 (0.23%) | 425.187 (0.29%) |
| KV append DMA | 507.904 (0.33%) | 220.904 (0.15%) |
| Block orchestration | 19.527 (0.01%) | 14.892 (0.01%) |
| Layer bookkeeping | 728.782 (0.47%) | 693.573 (0.47%) |
| Stage-boundary bookkeeping | 9.634 (0.01%) | 2.402 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 155.353 (0.10%) | 113.922 (0.08%) |
| Embedding | 70.299 (0.05%) | 2.243 (0.00%) |
| Final model RMSNorm | 4.835 (0.00%) | 3.366 (0.00%) |
| LM head＋greedy（不含 final norm） | 14379.276 (9.21%) | 14400.768 (9.77%) |
| Host–DSP 边界 | 682.032 (0.44%) | 2158.521 (1.46%) |
| 完整 Host wall | 156185.020 (100.00%) | 147378.502 (100.00%) |
