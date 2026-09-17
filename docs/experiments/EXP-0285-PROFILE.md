# Complete FP32 / FP16 residual module tables

Microseconds; percentage of complete Host wall. Decode is per token.

## prefill

| Module | FP32 residual | FP16 residual |
|---|---:|---:|
| I/O、metadata | 254.76 (0.74%) | 255.42 (0.65%) |
| Input RMSNorm | 2020.75 (5.83%) | 2807.60 (7.18%) |
| QKV＋Q/K Norm-RoPE | 7033.98 (20.30%) | 7035.01 (17.99%) |
| QK–Softmax–AV | 3493.58 (10.08%) | 3487.64 (8.92%) |
| O projection | 2078.27 (6.00%) | 3874.30 (9.91%) |
| Post-attention residual＋RMSNorm | 2114.48 (6.10%) | 2752.76 (7.04%) |
| Gate/Up＋SwiGLU | 7061.97 (20.39%) | 7070.65 (18.08%) |
| Down | 3780.68 (10.91%) | 5053.65 (12.92%) |
| Final residual | 2.82 (0.01%) | 3.08 (0.01%) |
| KV carrier conversion | 137.00 (0.40%) | 136.14 (0.35%) |
| KV append DMA | 293.08 (0.85%) | 294.57 (0.75%) |
| Block orchestration | 35.56 (0.10%) | 35.43 (0.09%) |
| Layer bookkeeping | 28.12 (0.08%) | 26.52 (0.07%) |
| Stage-boundary bookkeeping | 19.73 (0.06%) | 20.99 (0.05%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 117.86 (0.34%) | 117.50 (0.30%) |
| Embedding | 56.17 (0.16%) | 44.40 (0.11%) |
| Final model RMSNorm | 13.78 (0.04%) | 16.43 (0.04%) |
| LM head＋greedy，不含 final norm | 5169.33 (14.92%) | 5148.98 (13.17%) |
| Host-DSP boundary | 930.08 (2.68%) | 926.34 (2.37%) |
| Complete Host wall | 34641.99 (100.00%) | 39107.40 (100.00%) |

## decode

| Module | FP32 residual | FP16 residual |
|---|---:|---:|
| I/O、metadata | 251.90 (1.19%) | 249.74 (1.17%) |
| Input RMSNorm | 368.77 (1.74%) | 420.95 (1.97%) |
| QKV＋Q/K Norm-RoPE | 2511.84 (11.84%) | 2510.79 (11.76%) |
| QK–Softmax–AV | 1925.02 (9.08%) | 1928.59 (9.03%) |
| O projection | 1343.83 (6.34%) | 1407.44 (6.59%) |
| Post-attention residual＋RMSNorm | 369.42 (1.74%) | 426.12 (2.00%) |
| Gate/Up＋SwiGLU | 6351.62 (29.95%) | 6303.10 (29.52%) |
| Down | 3502.27 (16.51%) | 3506.33 (16.42%) |
| Final residual | 1.88 (0.01%) | 2.05 (0.01%) |
| KV carrier conversion | 290.58 (1.37%) | 290.75 (1.36%) |
| KV append DMA | 119.77 (0.56%) | 122.15 (0.57%) |
| Block orchestration | 27.83 (0.13%) | 28.22 (0.13%) |
| Layer bookkeeping | 16.74 (0.08%) | 17.21 (0.08%) |
| Stage-boundary bookkeeping | 1.75 (0.01%) | 1.75 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.47 (0.35%) | 73.66 (0.35%) |
| Embedding | 1.87 (0.01%) | 1.60 (0.01%) |
| Final model RMSNorm | 14.89 (0.07%) | 16.78 (0.08%) |
| LM head＋greedy，不含 final norm | 3234.09 (15.25%) | 3236.52 (15.16%) |
| Host-DSP boundary | 800.63 (3.78%) | 805.46 (3.77%) |
| Complete Host wall | 21208.17 (100.00%) | 21349.20 (100.00%) |
