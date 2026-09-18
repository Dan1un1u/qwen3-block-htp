# L32-0046 additive modules
Units microseconds (% complete Host wall). Counters overlap; only additive modules sum to wall.
## prefill
| Module | SP2 | A8 |
|---|---:|---:|
| I/O、metadata | 248.104 (0.46%) | 249.266 (0.47%) |
| Input RMSNorm | 3552.072 (6.59%) | 3579.191 (6.75%) |
| QKV＋RoPE | 9130.689 (16.93%) | 9154.156 (17.26%) |
| QK–Softmax–AV | 6648.597 (12.33%) | 6653.078 (12.54%) |
| O projection | 4115.922 (7.63%) | 4110.253 (7.75%) |
| Post-attention residual＋RMSNorm | 3565.449 (6.61%) | 3560.598 (6.71%) |
| Gate/Up＋SwiGLU | 13978.264 (25.91%) | 13974.454 (26.35%) |
| Down | 7351.695 (13.63%) | 6387.806 (12.04%) |
| Final residual | 3.068 (0.01%) | 3.088 (0.01%) |
| KV carrier conversion | 81.619 (0.15%) | 81.693 (0.15%) |
| KV append DMA | 239.082 (0.44%) | 239.037 (0.45%) |
| Block orchestration | 35.859 (0.07%) | 36.231 (0.07%) |
| Layer bookkeeping | 24.521 (0.05%) | 24.621 (0.05%) |
| Stage-boundary bookkeeping | 7.227 (0.01%) | 7.217 (0.01%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 95.098 (0.18%) | 95.135 (0.18%) |
| Embedding | 73.199 (0.14%) | 73.570 (0.14%) |
| Final model RMSNorm | 85.961 (0.16%) | 88.591 (0.17%) |
| LM head＋greedy（不含 final norm） | 3921.742 (7.27%) | 3933.091 (7.42%) |
| Host–DSP 边界 | 781.763 (1.45%) | 787.173 (1.48%) |
| 完整 Host wall | 53939.930 (100.00%) | 53038.248 (100.00%) |
## decode
| Module | SP2 | A8 |
|---|---:|---:|
| I/O、metadata | 247.297 (0.59%) | 249.236 (0.59%) |
| Input RMSNorm | 2309.544 (5.48%) | 2307.397 (5.47%) |
| QKV＋RoPE | 3903.301 (9.26%) | 3907.551 (9.27%) |
| QK–Softmax–AV | 6566.294 (15.57%) | 6565.901 (15.58%) |
| O projection | 2888.388 (6.85%) | 2882.425 (6.84%) |
| Post-attention residual＋RMSNorm | 2301.616 (5.46%) | 2304.545 (5.47%) |
| Gate/Up＋SwiGLU | 12707.608 (30.14%) | 12699.845 (30.13%) |
| Down | 6334.633 (15.02%) | 6320.304 (15.00%) |
| Final residual | 1.885 (0.00%) | 1.883 (0.00%) |
| KV carrier conversion | 12.801 (0.03%) | 12.813 (0.03%) |
| KV append DMA | 168.212 (0.40%) | 168.513 (0.40%) |
| Block orchestration | 27.745 (0.07%) | 27.516 (0.07%) |
| Layer bookkeeping | 16.155 (0.04%) | 16.170 (0.04%) |
| Stage-boundary bookkeeping | 1.635 (0.00%) | 1.641 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.131 (0.12%) | 51.179 (0.12%) |
| Embedding | 2.500 (0.01%) | 2.504 (0.01%) |
| Final model RMSNorm | 83.088 (0.20%) | 82.605 (0.20%) |
| LM head＋greedy（不含 final norm） | 3922.338 (9.30%) | 3924.550 (9.31%) |
| Host–DSP 边界 | 617.308 (1.46%) | 620.640 (1.47%) |
| 完整 Host wall | 42163.479 (100.00%) | 42147.219 (100.00%) |
