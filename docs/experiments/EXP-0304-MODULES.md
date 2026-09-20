# Complete module ledgers

Microseconds; decode per token. DSP counters are converted at19.2MHz.

## prefill

| Module | SP2 us (% Host) | INT16 us (% Host) |
|---|---:|---:|
| I/O、metadata | 229.318 (1.074%) | 231.618 (1.083%) |
| Input RMSNorm | 1036.606 (4.856%) | 1031.402 (4.824%) |
| QKV＋RoPE | 6660.759 (31.205%) | 6661.333 (31.156%) |
| QK–Softmax–AV | 3347.007 (15.680%) | 3350.351 (15.670%) |
| O projection | 1233.317 (5.778%) | 1225.026 (5.730%) |
| Post-attention residual＋RMSNorm | 1026.227 (4.808%) | 1038.461 (4.857%) |
| Gate/Up＋SwiGLU | 2683.779 (12.573%) | 2698.082 (12.619%) |
| Down | 1599.224 (7.492%) | 1598.879 (7.478%) |
| Final residual | 3.199 (0.015%) | 3.212 (0.015%) |
| KV carrier conversion | 135.130 (0.633%) | 135.068 (0.632%) |
| KV append DMA | 251.717 (1.179%) | 251.782 (1.178%) |
| Block orchestration | 37.541 (0.176%) | 37.462 (0.175%) |
| Layer bookkeeping | 24.312 (0.114%) | 24.614 (0.115%) |
| Stage-boundary bookkeeping | 20.060 (0.094%) | 20.148 (0.094%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 116.423 (0.545%) | 116.999 (0.547%) |
| Embedding | 42.295 (0.198%) | 42.349 (0.198%) |
| Final model RMSNorm | 9.333 (0.044%) | 7.200 (0.034%) |
| LM head＋greedy（不含 final norm） | 2034.340 (9.531%) | 2031.007 (9.499%) |
| Host-DSP boundary | 854.538 (4.003%) | 875.867 (4.096%) |
| Complete Host wall | 21345.126 (100.000%) | 21380.862 (100.000%) |
## decode

| Module | SP2 us (% Host) | INT16 us (% Host) |
|---|---:|---:|
| I/O、metadata | 225.729 (2.105%) | 226.362 (2.112%) |
| Input RMSNorm | 186.996 (1.744%) | 186.234 (1.738%) |
| QKV＋RoPE | 1481.693 (13.820%) | 1482.102 (13.830%) |
| QK–Softmax–AV | 1912.944 (17.842%) | 1913.607 (17.857%) |
| O projection | 770.157 (7.183%) | 765.556 (7.144%) |
| Post-attention residual＋RMSNorm | 191.691 (1.788%) | 192.949 (1.801%) |
| Gate/Up＋SwiGLU | 1762.989 (16.443%) | 1760.915 (16.432%) |
| Down | 1112.862 (10.380%) | 1110.563 (10.363%) |
| Final residual | 2.252 (0.021%) | 2.249 (0.021%) |
| KV carrier conversion | 164.924 (1.538%) | 164.923 (1.539%) |
| KV append DMA | 92.059 (0.859%) | 91.881 (0.857%) |
| Block orchestration | 28.832 (0.269%) | 28.828 (0.269%) |
| Layer bookkeeping | 16.736 (0.156%) | 16.742 (0.156%) |
| Stage-boundary bookkeeping | 1.853 (0.017%) | 1.852 (0.017%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 72.117 (0.673%) | 72.160 (0.673%) |
| Embedding | 1.585 (0.015%) | 1.600 (0.015%) |
| Final model RMSNorm | 7.752 (0.072%) | 7.600 (0.071%) |
| LM head＋greedy（不含 final norm） | 2044.829 (19.072%) | 2039.879 (19.035%) |
| Host-DSP boundary | 643.583 (6.003%) | 650.425 (6.069%) |
| Complete Host wall | 10721.584 (100.000%) | 10716.427 (100.000%) |
