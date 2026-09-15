# Bounded latest runtime generalization

warm fullmodel includes embedding,alllayers,finalnorm,head,greedy,FastRPC; excludes tokenizer,coldload,ADB; cache128 maxKV127

Two fixed real-text prompts truncated to64 tokenizer tokens; this is numerical/performance generalization, not instruction-following or PPL. Latest and original execute identical quantized arithmetic. Repeat5,5short10formal cyclic paired runs. No factorial rerun.

## A16 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 235.67 (0.68%) | 234.19 (0.67%) |
| Input RMSNorm | 2101.41 (6.06%) | 2101.68 (6.05%) |
| QKV＋Q/K Norm-RoPE | 7039.58 (20.29%) | 7041.61 (20.26%) |
| QK–Softmax–AV | 3488.13 (10.05%) | 3490.34 (10.04%) |
| O projection | 2076.08 (5.98%) | 2081.11 (5.99%) |
| Post-attention residual＋RMSNorm | 2059.92 (5.94%) | 2060.40 (5.93%) |
| Gate/Up＋SwiGLU | 7067.00 (20.37%) | 7071.78 (20.35%) |
| Down | 3809.31 (10.98%) | 3808.96 (10.96%) |
| Final residual | 3.09 (0.01%) | 3.12 (0.01%) |
| KV carrier conversion | 133.95 (0.39%) | 133.94 (0.39%) |
| KV append DMA | 291.05 (0.84%) | 291.65 (0.84%) |
| Block orchestration | 35.68 (0.10%) | 35.67 (0.10%) |
| Layer bookkeeping | 27.92 (0.08%) | 27.78 (0.08%) |
| Stage-boundary bookkeeping | 19.76 (0.06%) | 19.83 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 117.26 (0.34%) | 117.20 (0.34%) |
| Embedding | 64.58 (0.19%) | 64.67 (0.19%) |
| Final model RMSNorm | 13.81 (0.04%) | 13.75 (0.04%) |
| LM head＋greedy，不含 final norm | 5157.35 (14.86%) | 5156.60 (14.84%) |
| Host-DSP boundary | 955.38 (2.75%) | 999.16 (2.87%) |
| Complete Host wall | 34696.94 (100.00%) | 34753.46 (100.00%) |
## A16 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 228.97 (1.08%) | 228.05 (1.08%) |
| Input RMSNorm | 365.73 (1.73%) | 365.78 (1.73%) |
| QKV＋Q/K Norm-RoPE | 2496.04 (11.78%) | 2493.44 (11.82%) |
| QK–Softmax–AV | 1919.10 (9.06%) | 1916.94 (9.09%) |
| O projection | 1340.29 (6.33%) | 1342.85 (6.37%) |
| Post-attention residual＋RMSNorm | 370.06 (1.75%) | 370.06 (1.75%) |
| Gate/Up＋SwiGLU | 6422.14 (30.31%) | 6329.16 (30.00%) |
| Down | 3499.48 (16.52%) | 3496.92 (16.58%) |
| Final residual | 2.03 (0.01%) | 2.03 (0.01%) |
| KV carrier conversion | 291.79 (1.38%) | 291.89 (1.38%) |
| KV append DMA | 118.23 (0.56%) | 118.36 (0.56%) |
| Block orchestration | 27.84 (0.13%) | 27.84 (0.13%) |
| Layer bookkeeping | 16.75 (0.08%) | 16.80 (0.08%) |
| Stage-boundary bookkeeping | 1.73 (0.01%) | 1.73 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.69 (0.35%) | 73.64 (0.35%) |
| Embedding | 1.95 (0.01%) | 1.93 (0.01%) |
| Final model RMSNorm | 14.05 (0.07%) | 14.06 (0.07%) |
| LM head＋greedy，不含 final norm | 3220.87 (15.20%) | 3220.99 (15.27%) |
| Host-DSP boundary | 774.25 (3.65%) | 782.88 (3.71%) |
| Complete Host wall | 21184.98 (100.00%) | 21095.32 (100.00%) |
## A64 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 238.66 (0.69%) | 239.23 (0.69%) |
| Input RMSNorm | 2101.38 (6.04%) | 2100.64 (6.04%) |
| QKV＋Q/K Norm-RoPE | 7041.53 (20.24%) | 7038.89 (20.25%) |
| QK–Softmax–AV | 3493.89 (10.04%) | 3493.77 (10.05%) |
| O projection | 2088.30 (6.00%) | 2088.40 (6.01%) |
| Post-attention residual＋RMSNorm | 2061.77 (5.93%) | 2061.74 (5.93%) |
| Gate/Up＋SwiGLU | 7146.44 (20.54%) | 7134.76 (20.53%) |
| Down | 3829.46 (11.01%) | 3820.25 (10.99%) |
| Final residual | 3.23 (0.01%) | 3.31 (0.01%) |
| KV carrier conversion | 135.16 (0.39%) | 135.45 (0.39%) |
| KV append DMA | 291.40 (0.84%) | 291.34 (0.84%) |
| Block orchestration | 35.69 (0.10%) | 35.66 (0.10%) |
| Layer bookkeeping | 27.58 (0.08%) | 27.53 (0.08%) |
| Stage-boundary bookkeeping | 20.07 (0.06%) | 20.19 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 118.45 (0.34%) | 118.88 (0.34%) |
| Embedding | 66.75 (0.19%) | 67.34 (0.19%) |
| Final model RMSNorm | 13.80 (0.04%) | 13.81 (0.04%) |
| LM head＋greedy，不含 final norm | 5173.66 (14.87%) | 5162.35 (14.85%) |
| Host-DSP boundary | 898.71 (2.58%) | 898.37 (2.59%) |
| Complete Host wall | 34785.91 (100.00%) | 34751.90 (100.00%) |
## A64 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 230.34 (1.09%) | 230.73 (1.09%) |
| Input RMSNorm | 365.39 (1.72%) | 365.39 (1.73%) |
| QKV＋Q/K Norm-RoPE | 2510.72 (11.84%) | 2508.17 (11.90%) |
| QK–Softmax–AV | 1974.81 (9.31%) | 1973.34 (9.36%) |
| O projection | 1347.63 (6.35%) | 1346.88 (6.39%) |
| Post-attention residual＋RMSNorm | 369.17 (1.74%) | 369.20 (1.75%) |
| Gate/Up＋SwiGLU | 6481.60 (30.56%) | 6383.60 (30.29%) |
| Down | 3525.12 (16.62%) | 3519.28 (16.70%) |
| Final residual | 2.03 (0.01%) | 2.04 (0.01%) |
| KV carrier conversion | 303.44 (1.43%) | 303.25 (1.44%) |
| KV append DMA | 122.31 (0.58%) | 122.53 (0.58%) |
| Block orchestration | 27.76 (0.13%) | 27.73 (0.13%) |
| Layer bookkeeping | 16.56 (0.08%) | 16.54 (0.08%) |
| Stage-boundary bookkeeping | 1.72 (0.01%) | 1.72 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.31 (0.35%) | 73.34 (0.35%) |
| Embedding | 2.03 (0.01%) | 2.05 (0.01%) |
| Final model RMSNorm | 14.39 (0.07%) | 14.33 (0.07%) |
| LM head＋greedy，不含 final norm | 3248.34 (15.32%) | 3243.81 (15.39%) |
| Host-DSP boundary | 590.84 (2.79%) | 572.64 (2.72%) |
| Complete Host wall | 21207.52 (100.00%) | 21076.58 (100.00%) |
## B64 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 239.98 (0.69%) | 239.80 (0.69%) |
| Input RMSNorm | 2059.99 (5.90%) | 2059.17 (5.91%) |
| QKV＋Q/K Norm-RoPE | 7043.40 (20.18%) | 7046.52 (20.22%) |
| QK–Softmax–AV | 3497.97 (10.02%) | 3493.25 (10.02%) |
| O projection | 2088.49 (5.98%) | 2093.16 (6.01%) |
| Post-attention residual＋RMSNorm | 2096.88 (6.01%) | 2096.43 (6.02%) |
| Gate/Up＋SwiGLU | 7180.10 (20.58%) | 7175.03 (20.59%) |
| Down | 3841.55 (11.01%) | 3838.22 (11.01%) |
| Final residual | 3.31 (0.01%) | 3.35 (0.01%) |
| KV carrier conversion | 135.76 (0.39%) | 136.95 (0.39%) |
| KV append DMA | 290.39 (0.83%) | 291.30 (0.84%) |
| Block orchestration | 35.37 (0.10%) | 35.67 (0.10%) |
| Layer bookkeeping | 27.55 (0.08%) | 27.29 (0.08%) |
| Stage-boundary bookkeeping | 20.39 (0.06%) | 20.23 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 119.16 (0.34%) | 118.57 (0.34%) |
| Embedding | 75.97 (0.22%) | 75.84 (0.22%) |
| Final model RMSNorm | 13.85 (0.04%) | 13.69 (0.04%) |
| LM head＋greedy，不含 final norm | 5192.40 (14.88%) | 5192.01 (14.90%) |
| Host-DSP boundary | 933.99 (2.68%) | 894.71 (2.57%) |
| Complete Host wall | 34896.51 (100.00%) | 34851.19 (100.00%) |
## B64 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 230.43 (1.08%) | 230.51 (1.09%) |
| Input RMSNorm | 364.09 (1.71%) | 364.04 (1.72%) |
| QKV＋Q/K Norm-RoPE | 2522.94 (11.85%) | 2522.01 (11.90%) |
| QK–Softmax–AV | 1974.95 (9.28%) | 1975.87 (9.32%) |
| O projection | 1350.75 (6.34%) | 1351.61 (6.38%) |
| Post-attention residual＋RMSNorm | 369.79 (1.74%) | 369.83 (1.74%) |
| Gate/Up＋SwiGLU | 6516.74 (30.61%) | 6427.86 (30.32%) |
| Down | 3540.75 (16.63%) | 3539.99 (16.70%) |
| Final residual | 2.03 (0.01%) | 2.04 (0.01%) |
| KV carrier conversion | 303.39 (1.43%) | 303.26 (1.43%) |
| KV append DMA | 121.89 (0.57%) | 122.19 (0.58%) |
| Block orchestration | 27.77 (0.13%) | 27.76 (0.13%) |
| Layer bookkeeping | 16.57 (0.08%) | 16.57 (0.08%) |
| Stage-boundary bookkeeping | 1.73 (0.01%) | 1.72 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.33 (0.34%) | 73.30 (0.35%) |
| Embedding | 2.30 (0.01%) | 2.29 (0.01%) |
| Final model RMSNorm | 14.08 (0.07%) | 14.14 (0.07%) |
| LM head＋greedy，不含 final norm | 3269.37 (15.36%) | 3268.99 (15.42%) |
| Host-DSP boundary | 585.77 (2.75%) | 583.70 (2.75%) |
| Complete Host wall | 21288.68 (100.00%) | 21197.66 (100.00%) |

## Complete-model E2E

| Case | Original prefill tps | Latest prefill tps | Original decode tps | Latest decode tps |
|---|---:|---:|---:|---:|
| A16 | 1844.54 | 1841.54 | 47.20 | 47.40 |
| A64 | 1839.83 | 1841.63 | 47.15 | 47.45 |
| B64 | 1833.99 | 1836.38 | 46.97 | 47.18 |
