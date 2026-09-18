# EXP0294 additive module comparison

## prefill

| Module | SP2 us (%Host) | A8 us (%Host) |
|---|---:|---:|
| I/O and metadata | 234.636 (1.11%) | 234.594 (1.13%) |
| Input RMSNorm | 1036.531 (4.89%) | 1024.458 (4.92%) |
| QKV + Q/K Norm-RoPE | 6624.323 (31.28%) | 6627.409 (31.84%) |
| QK-Softmax-AV | 3333.720 (15.74%) | 3332.116 (16.01%) |
| O projection | 1213.465 (5.73%) | 1212.523 (5.83%) |
| Post-attention residual + RMSNorm | 1025.311 (4.84%) | 1012.126 (4.86%) |
| Gate/Up + SwiGLU | 2662.286 (12.57%) | 2642.269 (12.69%) |
| Down projection | 1578.047 (7.45%) | 1275.001 (6.13%) |
| Final residual | 3.316 (0.02%) | 3.351 (0.02%) |
| KV-cache carrier conversion | 134.497 (0.64%) | 134.921 (0.65%) |
| KV-cache append DMA | 255.537 (1.21%) | 255.398 (1.23%) |
| Block internal orchestration | 37.831 (0.18%) | 37.964 (0.18%) |
| Layer bookkeeping | 23.340 (0.11%) | 23.356 (0.11%) |
| Stage-boundary bookkeeping | 32.944 (0.16%) | 33.104 (0.16%) |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) |
| DSP runtime setup/teardown | 117.231 (0.55%) | 117.807 (0.57%) |
| Token embedding | 38.933 (0.18%) | 39.004 (0.19%) |
| Final model RMSNorm | 9.704 (0.05%) | 7.576 (0.04%) |
| LM head + greedy selection (excluding final norm) | 1940.477 (9.16%) | 1942.074 (9.33%) |
| True Host-DSP boundary | 873.353 (4.12%) | 858.916 (4.13%) |
| Complete Host wall | 21175.483 (100.00%) | 20813.967 (100.00%) |
## decode

| Module | SP2 us (%Host) | A8 us (%Host) |
|---|---:|---:|
| I/O and metadata | 227.783 (2.12%) | 226.935 (2.12%) |
| Input RMSNorm | 186.919 (1.74%) | 186.897 (1.75%) |
| QKV + Q/K Norm-RoPE | 1480.248 (13.75%) | 1481.398 (13.85%) |
| QK-Softmax-AV | 1916.590 (17.81%) | 1914.894 (17.90%) |
| O projection | 761.580 (7.08%) | 760.868 (7.11%) |
| Post-attention residual + RMSNorm | 191.267 (1.78%) | 190.687 (1.78%) |
| Gate/Up + SwiGLU | 1769.556 (16.44%) | 1767.270 (16.52%) |
| Down projection | 1090.676 (10.13%) | 1043.564 (9.75%) |
| Final residual | 2.134 (0.02%) | 2.029 (0.02%) |
| KV-cache carrier conversion | 309.204 (2.87%) | 309.114 (2.89%) |
| KV-cache append DMA | 109.655 (1.02%) | 109.561 (1.02%) |
| Block internal orchestration | 29.016 (0.27%) | 29.205 (0.27%) |
| Layer bookkeeping | 16.074 (0.15%) | 15.982 (0.15%) |
| Stage-boundary bookkeeping | 1.680 (0.02%) | 1.726 (0.02%) |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) |
| DSP runtime setup/teardown | 71.750 (0.67%) | 71.774 (0.67%) |
| Token embedding | 1.412 (0.01%) | 1.414 (0.01%) |
| Final model RMSNorm | 7.728 (0.07%) | 7.644 (0.07%) |
| LM head + greedy selection (excluding final norm) | 1930.489 (17.94%) | 1929.565 (18.03%) |
| True Host-DSP boundary | 658.415 (6.12%) | 648.592 (6.06%) |
| Complete Host wall | 10762.176 (100.00%) | 10699.118 (100.00%) |
