# L32-0064 full additive modules

Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness/diagnostics. Softmax worker work is overlapping, excluded from this ledger.

| Module | Operator-boundary prefill |
|---|---:|
| I/O、metadata | 232.07 (0.27%) |
| Input RMSNorm | 3604.87 (4.23%) |
| QKV＋RoPE | 9106.77 (10.69%) |
| QK–Softmax–AV | 7722.57 (9.06%) |
| O projection | 4155.36 (4.88%) |
| Post-attention residual＋RMSNorm | 3467.95 (4.07%) |
| Gate/Up＋SwiGLU | 42984.08 (50.45%) |
| Down | 7983.70 (9.37%) |
| Final residual | 2.65 (0.00%) |
| KV carrier conversion | 80.50 (0.09%) |
| KV append DMA | 240.40 (0.28%) |
| Block orchestration | 37.23 (0.04%) |
| Layer bookkeeping | 23.04 (0.03%) |
| Stage-boundary bookkeeping | 7.06 (0.01%) |
| DSP unattributed | 0.00 (0.00%) |
| Runtime setup/teardown | 88.84 (0.10%) |
| Embedding | 72.03 (0.08%) |
| Final model RMSNorm | 82.31 (0.10%) |
| LM head＋greedy（不含 final norm） | 3812.98 (4.47%) |
| Host-DSP boundary | 1502.65 (1.76%) |

The column includes embedding, all 28 blocks, final norm, LM head, greedy and FastRPC. They are diagnostic schedules, not an optimized-vs-baseline speedup experiment.
