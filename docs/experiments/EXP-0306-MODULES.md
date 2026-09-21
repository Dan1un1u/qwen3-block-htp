# EXP-0306 full additive modules

Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness and diagnostics. Four vector contexts; explicit QK/Softmax/AV phase boundaries and GateUp/SwiGLU boundary. Worker work sums are excluded from the additive ledger.

| Module | Prefill | One-step decode |
|---|---:|---:|
| Token embedding | 41.48 (0.08%) | 1.29 (0.00%) |
| Input staging | 0.42 (0.00%) | 0.30 (0.00%) |
| Metadata | 212.50 (0.43%) | 206.70 (0.65%) |
| Input RMSNorm | 1020.10 (2.06%) | 195.36 (0.61%) |
| QKV projection | 21428.82 (43.34%) | 21382.53 (67.31%) |
| Q/K Norm-RoPE (separate tail) | 0.85 (0.00%) | 0.47 (0.00%) |
| QK-Softmax-AV | 5715.51 (11.56%) | 1825.77 (5.75%) |
| O projection | 1237.02 (2.50%) | 775.57 (2.44%) |
| Post-attention residual | 1021.00 (2.07%) | 184.19 (0.58%) |
| Post-attention RMSNorm (fused) | 1.40 (0.00%) | 1.02 (0.00%) |
| Gate/Up | 1961.64 (3.97%) | 1638.89 (5.16%) |
| SwiGLU | 11147.98 (22.55%) | 638.44 (2.01%) |
| Down | 1594.20 (3.22%) | 1123.11 (3.54%) |
| Final residual | 3.77 (0.01%) | 2.65 (0.01%) |
| Output staging | 0.00 (0.00%) | 0.00 (0.00%) |
| KV-cache carrier conversion | 136.84 (0.28%) | 149.75 (0.47%) |
| KV-cache append DMA | 244.49 (0.49%) | 79.22 (0.25%) |
| Block orchestration | 36.73 (0.07%) | 29.48 (0.09%) |
| Layer bookkeeping | 22.48 (0.05%) | 17.79 (0.06%) |
| Stage-boundary bookkeeping | 20.15 (0.04%) | 2.24 (0.01%) |
| Final model RMSNorm | 7.42 (0.02%) | 7.41 (0.02%) |
| LM head + greedy selection (excluding final norm) | 2018.09 (4.08%) | 2026.24 (6.38%) |
| Runtime setup | 51.57 (0.10%) | 28.39 (0.09%) |
| Runtime teardown | 37.76 (0.08%) | 23.44 (0.07%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) |
| Host-DSP boundary | 1476.73 (2.99%) | 1427.59 (4.49%) |

E2E: prefill: 1294.53 token/s, decode_one_step: 31.48 token/s

Includes embedding, all 28 blocks, final norm, head, greedy and FastRPC. Diagnostic phase costs are not critical-path shares of a production-overlapped schedule. No model-quality or baseline promotion claim.
