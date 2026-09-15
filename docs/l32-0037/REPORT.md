# Bounded latest runtime generalization

warm fullmodel includes embedding,alllayers,finalnorm,head,greedy,FastRPC; excludes tokenizer,coldload,ADB; cache128 maxKV127

Two fixed real-text prompts truncated to64 tokenizer tokens; this is numerical/performance generalization, not instruction-following or PPL. Latest and original execute identical quantized arithmetic. Repeat5,5short10formal cyclic paired runs. No factorial rerun.

## A16 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 121.90 (0.41%) | 122.02 (0.44%) |
| Input RMSNorm | 1276.12 (4.30%) | 1276.10 (4.64%) |
| QKV＋RoPE | 3872.21 (13.05%) | 3873.45 (14.07%) |
| QK–Softmax–AV | 8376.42 (28.24%) | 6246.49 (22.69%) |
| O projection | 1585.79 (5.35%) | 1582.49 (5.75%) |
| Post-attention residual＋RMSNorm | 1290.79 (4.35%) | 1291.39 (4.69%) |
| Gate/Up＋SwiGLU | 5539.18 (18.67%) | 5532.69 (20.10%) |
| Down | 3135.55 (10.57%) | 3140.58 (11.41%) |
| Final residual | 1.76 (0.01%) | 1.75 (0.01%) |
| KV carrier conversion | 26.39 (0.09%) | 26.49 (0.10%) |
| KV append DMA | 114.11 (0.38%) | 114.19 (0.41%) |
| Block orchestration | 22.68 (0.08%) | 22.92 (0.08%) |
| Layer bookkeeping | 14.24 (0.05%) | 14.27 (0.05%) |
| Stage-boundary bookkeeping | 6.34 (0.02%) | 6.45 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 88.09 (0.30%) | 88.70 (0.32%) |
| Embedding | 58.26 (0.20%) | 58.19 (0.21%) |
| Final model RMSNorm | 58.27 (0.20%) | 58.56 (0.21%) |
| LM head＋greedy（不含 final norm） | 3198.34 (10.78%) | 3204.72 (11.64%) |
| Host-DSP boundary | 875.15 (2.95%) | 864.21 (3.14%) |
| Complete Host wall | 29661.59 (100.00%) | 27525.68 (100.00%) |
## A16 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 116.86 (0.53%) | 116.72 (0.53%) |
| Input RMSNorm | 872.59 (3.94%) | 872.67 (3.98%) |
| QKV＋RoPE | 1669.22 (7.54%) | 1669.78 (7.62%) |
| QK–Softmax–AV | 5994.03 (27.07%) | 5780.88 (26.40%) |
| O projection | 884.73 (4.00%) | 884.04 (4.04%) |
| Post-attention residual＋RMSNorm | 877.99 (3.97%) | 878.07 (4.01%) |
| Gate/Up＋SwiGLU | 4905.37 (22.15%) | 4849.90 (22.14%) |
| Down | 2638.35 (11.91%) | 2639.91 (12.05%) |
| Final residual | 1.17 (0.01%) | 1.18 (0.01%) |
| KV carrier conversion | 48.22 (0.22%) | 48.14 (0.22%) |
| KV append DMA | 91.21 (0.41%) | 91.23 (0.42%) |
| Block orchestration | 16.45 (0.07%) | 16.40 (0.07%) |
| Layer bookkeeping | 9.48 (0.04%) | 9.44 (0.04%) |
| Stage-boundary bookkeeping | 1.27 (0.01%) | 1.27 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 50.93 (0.23%) | 50.95 (0.23%) |
| Embedding | 1.66 (0.01%) | 1.65 (0.01%) |
| Final model RMSNorm | 55.64 (0.25%) | 55.63 (0.25%) |
| LM head＋greedy（不含 final norm） | 3194.30 (14.43%) | 3198.98 (14.61%) |
| Host-DSP boundary | 713.82 (3.22%) | 734.17 (3.35%) |
| Complete Host wall | 22143.29 (100.00%) | 21901.02 (100.00%) |
## A64 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 124.35 (0.42%) | 123.84 (0.45%) |
| Input RMSNorm | 1276.47 (4.30%) | 1275.98 (4.63%) |
| QKV＋RoPE | 3883.92 (13.08%) | 3881.17 (14.07%) |
| QK–Softmax–AV | 8374.40 (28.20%) | 6244.91 (22.64%) |
| O projection | 1580.34 (5.32%) | 1584.74 (5.75%) |
| Post-attention residual＋RMSNorm | 1291.94 (4.35%) | 1291.77 (4.68%) |
| Gate/Up＋SwiGLU | 5583.58 (18.80%) | 5566.72 (20.18%) |
| Down | 3138.51 (10.57%) | 3136.98 (11.37%) |
| Final residual | 1.79 (0.01%) | 1.78 (0.01%) |
| KV carrier conversion | 26.50 (0.09%) | 26.56 (0.10%) |
| KV append DMA | 115.85 (0.39%) | 116.07 (0.42%) |
| Block orchestration | 23.27 (0.08%) | 22.98 (0.08%) |
| Layer bookkeeping | 14.17 (0.05%) | 14.30 (0.05%) |
| Stage-boundary bookkeeping | 7.09 (0.02%) | 6.82 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 92.33 (0.31%) | 90.10 (0.33%) |
| Embedding | 63.31 (0.21%) | 62.85 (0.23%) |
| Final model RMSNorm | 58.28 (0.20%) | 58.44 (0.21%) |
| LM head＋greedy（不含 final norm） | 3199.64 (10.78%) | 3198.03 (11.59%) |
| Host-DSP boundary | 836.69 (2.82%) | 879.47 (3.19%) |
| Complete Host wall | 29692.42 (100.00%) | 27583.48 (100.00%) |
## A64 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 118.62 (0.51%) | 118.35 (0.52%) |
| Input RMSNorm | 871.44 (3.78%) | 871.41 (3.83%) |
| QKV＋RoPE | 1678.76 (7.27%) | 1676.53 (7.36%) |
| QK–Softmax–AV | 7012.81 (30.38%) | 6779.76 (29.76%) |
| O projection | 879.91 (3.81%) | 884.95 (3.89%) |
| Post-attention residual＋RMSNorm | 876.82 (3.80%) | 876.87 (3.85%) |
| Gate/Up＋SwiGLU | 4952.71 (21.46%) | 4894.37 (21.49%) |
| Down | 2661.02 (11.53%) | 2662.19 (11.69%) |
| Final residual | 1.17 (0.01%) | 1.17 (0.01%) |
| KV carrier conversion | 48.21 (0.21%) | 48.20 (0.21%) |
| KV append DMA | 90.94 (0.39%) | 90.86 (0.40%) |
| Block orchestration | 16.45 (0.07%) | 16.38 (0.07%) |
| Layer bookkeeping | 9.48 (0.04%) | 9.41 (0.04%) |
| Stage-boundary bookkeeping | 1.26 (0.01%) | 1.26 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 50.81 (0.22%) | 50.79 (0.22%) |
| Embedding | 1.63 (0.01%) | 1.64 (0.01%) |
| Final model RMSNorm | 55.69 (0.24%) | 55.71 (0.24%) |
| LM head＋greedy（不含 final norm） | 3195.32 (13.84%) | 3193.13 (14.02%) |
| Host-DSP boundary | 557.12 (2.41%) | 544.69 (2.39%) |
| Complete Host wall | 23080.18 (100.00%) | 22777.66 (100.00%) |
## B64 prefill

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 124.57 (0.42%) | 124.15 (0.45%) |
| Input RMSNorm | 1307.99 (4.39%) | 1308.54 (4.74%) |
| QKV＋RoPE | 3881.20 (13.03%) | 3879.27 (14.05%) |
| QK–Softmax–AV | 8396.82 (28.18%) | 6265.16 (22.69%) |
| O projection | 1585.99 (5.32%) | 1586.24 (5.74%) |
| Post-attention residual＋RMSNorm | 1265.99 (4.25%) | 1265.33 (4.58%) |
| Gate/Up＋SwiGLU | 5578.79 (18.73%) | 5568.63 (20.16%) |
| Down | 3134.99 (10.52%) | 3142.79 (11.38%) |
| Final residual | 1.84 (0.01%) | 1.87 (0.01%) |
| KV carrier conversion | 26.72 (0.09%) | 26.67 (0.10%) |
| KV append DMA | 115.53 (0.39%) | 115.78 (0.42%) |
| Block orchestration | 23.37 (0.08%) | 23.58 (0.09%) |
| Layer bookkeeping | 14.32 (0.05%) | 14.35 (0.05%) |
| Stage-boundary bookkeeping | 6.99 (0.02%) | 7.26 (0.03%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 93.32 (0.31%) | 93.23 (0.34%) |
| Embedding | 72.19 (0.24%) | 71.96 (0.26%) |
| Final model RMSNorm | 55.60 (0.19%) | 55.58 (0.20%) |
| LM head＋greedy（不含 final norm） | 3203.21 (10.75%) | 3200.39 (11.59%) |
| Host-DSP boundary | 903.15 (3.03%) | 865.55 (3.13%) |
| Complete Host wall | 29792.59 (100.00%) | 27616.29 (100.00%) |
## B64 decode

| Module | Original μs (Host share) | Latest μs (Host share) |
|---|---:|---:|
| I/O、metadata | 118.50 (0.51%) | 118.30 (0.52%) |
| Input RMSNorm | 870.22 (3.76%) | 870.19 (3.81%) |
| QKV＋RoPE | 1683.34 (7.28%) | 1684.01 (7.37%) |
| QK–Softmax–AV | 7016.86 (30.33%) | 6791.01 (29.73%) |
| O projection | 888.05 (3.84%) | 887.00 (3.88%) |
| Post-attention residual＋RMSNorm | 877.56 (3.79%) | 877.58 (3.84%) |
| Gate/Up＋SwiGLU | 4970.61 (21.49%) | 4916.41 (21.52%) |
| Down | 2671.96 (11.55%) | 2672.81 (11.70%) |
| Final residual | 1.17 (0.01%) | 1.17 (0.01%) |
| KV carrier conversion | 48.20 (0.21%) | 48.17 (0.21%) |
| KV append DMA | 90.84 (0.39%) | 90.72 (0.40%) |
| Block orchestration | 16.48 (0.07%) | 16.41 (0.07%) |
| Layer bookkeeping | 9.46 (0.04%) | 9.42 (0.04%) |
| Stage-boundary bookkeeping | 1.26 (0.01%) | 1.26 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 50.81 (0.22%) | 50.79 (0.22%) |
| Embedding | 1.83 (0.01%) | 1.85 (0.01%) |
| Final model RMSNorm | 55.94 (0.24%) | 55.92 (0.24%) |
| LM head＋greedy（不含 final norm） | 3202.07 (13.84%) | 3201.82 (14.02%) |
| Host-DSP boundary | 556.31 (2.41%) | 549.17 (2.40%) |
| Complete Host wall | 23131.48 (100.00%) | 22844.03 (100.00%) |

## Complete-model E2E

| Case | Original prefill tps | Latest prefill tps | Original decode tps | Latest decode tps |
|---|---:|---:|---:|---:|
| A16 | 2157.67 | 2325.10 | 45.16 | 45.66 |
| A64 | 2155.43 | 2320.23 | 43.33 | 43.90 |
| B64 | 2148.18 | 2317.47 | 43.23 | 43.78 |
