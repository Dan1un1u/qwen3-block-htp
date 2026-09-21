# L32-0063 full additive modules

Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness/diagnostics. Softmax worker work is overlapping, excluded from this ledger.

| Module | Pipelined prefill | Operator-boundary prefill |
|---|---:|---:|
| I/O、metadata | 117.29 (0.28%) | 116.92 (0.24%) |
| Input RMSNorm | 1309.16 (3.13%) | 1309.72 (2.72%) |
| QKV＋RoPE | 3851.53 (9.21%) | 3854.36 (8.01%) |
| QK–Softmax–AV | 6920.81 (16.55%) | 9538.93 (19.82%) |
| O projection | 1592.59 (3.81%) | 1587.38 (3.30%) |
| Post-attention residual＋RMSNorm | 1279.57 (3.06%) | 1279.85 (2.66%) |
| Gate/Up＋SwiGLU | 18357.10 (43.90%) | 22062.47 (45.84%) |
| Down | 3146.62 (7.52%) | 3149.42 (6.54%) |
| Final residual | 1.96 (0.00%) | 1.98 (0.00%) |
| KV carrier conversion | 27.79 (0.07%) | 27.74 (0.06%) |
| KV append DMA | 112.32 (0.27%) | 112.05 (0.23%) |
| Block orchestration | 23.79 (0.06%) | 24.44 (0.05%) |
| Layer bookkeeping | 12.92 (0.03%) | 13.24 (0.03%) |
| Stage-boundary bookkeeping | 5.68 (0.01%) | 5.89 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 86.66 (0.21%) | 89.00 (0.18%) |
| Embedding | 56.57 (0.14%) | 56.69 (0.12%) |
| Final model RMSNorm | 58.07 (0.14%) | 58.16 (0.12%) |
| LM head＋greedy（不含 final norm） | 3352.69 (8.02%) | 3327.43 (6.91%) |
| Host-DSP boundary | 1502.81 (3.59%) | 1514.60 (3.15%) |

Both columns include embedding, all 16 blocks, final norm, LM head, greedy and FastRPC. They are diagnostic schedules, not an optimized-vs-baseline speedup experiment.
