# L32-0041 Llama 3.2 3B W4A8-SP2 native profiling

M64+15, KV capacity80;5 short +10 formal rounds,10 trajectories/round. Timing includes embedding,28 blocks,final RMSNorm,LM head,greedy and FastRPC; excludes cold loading,session preparation and external tokenization. Fixed trajectory from independent greedy oracle. FP32 residual and no rotations. No PPL/text-quality acceptance.

| 模块 | Prefill μs（Host wall 占比） | Decode μs/token（Host wall 占比） |
|---|---:|---:|
| I/O、metadata | 250.4 (0.20%) | 252.0 (0.47%) |
| Input RMSNorm | 3534.9 (2.83%) | 2291.2 (4.27%) |
| QKV＋RoPE | 49196.4 (39.37%) | 7851.0 (14.63%) |
| QK–Softmax–AV | 27555.5 (22.05%) | 13722.6 (25.58%) |
| O projection | 4122.0 (3.30%) | 2862.3 (5.33%) |
| Post-attention residual＋RMSNorm | 3564.7 (2.85%) | 2304.2 (4.29%) |
| Gate/Up＋SwiGLU | 13500.2 (10.80%) | 12234.6 (22.80%) |
| Down | 7940.0 (6.35%) | 6748.8 (12.58%) |
| Final residual | 3.2 (0.00%) | 1.9 (0.00%) |
| KV carrier conversion | 9491.0 (7.59%) | 159.7 (0.30%) |
| KV append DMA | 346.7 (0.28%) | 188.1 (0.35%) |
| Block orchestration | 36.7 (0.03%) | 28.7 (0.05%) |
| Layer bookkeeping | 23.8 (0.02%) | 16.1 (0.03%) |
| Stage-boundary bookkeeping | 7.2 (0.01%) | 1.7 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 92.2 (0.07%) | 51.5 (0.10%) |
| Embedding | 70.2 (0.06%) | 2.3 (0.00%) |
| Final model RMSNorm | 85.9 (0.07%) | 83.0 (0.15%) |
| LM head＋greedy（不含 final norm） | 4207.7 (3.37%) | 4209.8 (7.85%) |
| Host–DSP 边界 | 936.7 (0.75%) | 643.5 (1.20%) |
| 完整 Host wall | 124965.4 (100.00%) | 53653.1 (100.00%) |

E2E: prefill **512.14 token/s**, decode **18.64 token/s**.

All1600 additive ledgers reconcile exactly; no intermediate DDR spill; one invocation per token boundary,28 blocks per invocation. These3B measurements are not a paired speed comparison with historical1B.
