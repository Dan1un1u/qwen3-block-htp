# L32-0035 C2 targeted native decode optimization

No rotation; native per-output-channel W4, SP2mode8, FP32 residual1; original weights and qparams. OPT computes only32 valid channels per decode tile and publishes deterministic low0/high128 padding; ORIGINAL sequential128-value gather; NORM_COMPACT compact Norm+C1 dual128 gather mask17. All arms share pipeline. Full16 independent selected0/7/15,chain3,chain16,frontend original token/logit gates pass; each representative LUT exhaustively checked on DSP for65536 Gate/Up pairs including padding. Component mode9 avoids existing rotation diagnostic7/8. All failed compilation/deployment attempts retained; no numeric gate relaxed.

C2 decode Host wall improves0.357% over original, paired95% CI0.187–0.514%. Versus compact Norm it is numerically0.136% faster, CIincludes equality; compact superiority is not established after fixing wasted row work. Retain optimized native decode. Prefill is unchanged within uncertainty. First candidate dual128 alone gave inconclusive gain. No PPL/quality claim or selected-baseline promotion.

## prefill

| Module | OPT | ORIGINAL | NORM_COMPACT |
|---|---:|---:|---:|
| I/O、metadata | 121.31 (0.41%) | 122.13 (0.41%) | 121.32 (0.41%) |
| Input RMSNorm | 1269.15 (4.31%) | 1268.83 (4.31%) | 1318.17 (4.45%) |
| QKV＋RoPE | 3864.12 (13.14%) | 3866.98 (13.13%) | 3867.36 (13.07%) |
| QK–Softmax–AV | 8387.28 (28.51%) | 8390.92 (28.49%) | 8384.81 (28.33%) |
| O projection | 1581.08 (5.37%) | 1579.15 (5.36%) | 1578.23 (5.33%) |
| Post-attention residual＋RMSNorm | 1276.38 (4.34%) | 1276.45 (4.33%) | 1325.72 (4.48%) |
| Gate/Up＋SwiGLU | 5508.93 (18.73%) | 5520.50 (18.75%) | 5524.51 (18.67%) |
| Down | 3126.58 (10.63%) | 3130.42 (10.63%) | 3131.94 (10.58%) |
| Final residual | 1.63 (0.01%) | 1.79 (0.01%) | 1.54 (0.01%) |
| KV carrier conversion | 27.31 (0.09%) | 27.61 (0.09%) | 27.40 (0.09%) |
| KV append DMA | 107.33 (0.36%) | 107.46 (0.36%) | 107.36 (0.36%) |
| Block orchestration | 23.13 (0.08%) | 23.38 (0.08%) | 22.98 (0.08%) |
| Layer bookkeeping | 14.05 (0.05%) | 14.14 (0.05%) | 13.96 (0.05%) |
| Stage-boundary bookkeeping | 6.59 (0.02%) | 6.81 (0.02%) | 6.51 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 88.82 (0.30%) | 91.54 (0.31%) | 89.38 (0.30%) |
| Embedding | 53.95 (0.18%) | 54.29 (0.18%) | 53.99 (0.18%) |
| Final model RMSNorm | 60.43 (0.21%) | 60.41 (0.21%) | 60.24 (0.20%) |
| LM head＋greedy（不含 final norm） | 3203.97 (10.89%) | 3210.98 (10.90%) | 3209.31 (10.84%) |
| Host-DSP boundary | 694.54 (2.36%) | 694.40 (2.36%) | 752.34 (2.54%) |
| Complete Host wall | 29416.58 (100.00%) | 29448.19 (100.00%) | 29597.07 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.998926 | [0.9957411374815226, 1.001596636900956] |
| NORM_COMPACT_over_OPT | 1.006136 | [1.003495507845639, 1.0085242211301657] |
## decode

| Module | OPT | ORIGINAL | NORM_COMPACT |
|---|---:|---:|---:|
| I/O、metadata | 117.76 (0.54%) | 117.80 (0.53%) | 117.56 (0.53%) |
| Input RMSNorm | 871.28 (3.97%) | 871.33 (3.96%) | 872.95 (3.97%) |
| QKV＋RoPE | 1670.41 (7.61%) | 1670.21 (7.58%) | 1669.01 (7.59%) |
| QK–Softmax–AV | 5982.50 (27.26%) | 5987.62 (27.19%) | 5982.92 (27.23%) |
| O projection | 881.27 (4.02%) | 879.66 (3.99%) | 879.53 (4.00%) |
| Post-attention residual＋RMSNorm | 874.58 (3.99%) | 874.55 (3.97%) | 876.20 (3.99%) |
| Gate/Up＋SwiGLU | 4835.66 (22.03%) | 4896.33 (22.23%) | 4853.43 (22.09%) |
| Down | 2634.16 (12.00%) | 2630.53 (11.94%) | 2631.53 (11.97%) |
| Final residual | 1.05 (0.00%) | 1.05 (0.00%) | 1.04 (0.00%) |
| KV carrier conversion | 48.11 (0.22%) | 48.10 (0.22%) | 48.12 (0.22%) |
| KV append DMA | 90.57 (0.41%) | 90.70 (0.41%) | 90.52 (0.41%) |
| Block orchestration | 15.98 (0.07%) | 15.99 (0.07%) | 15.92 (0.07%) |
| Layer bookkeeping | 9.37 (0.04%) | 9.38 (0.04%) | 9.38 (0.04%) |
| Stage-boundary bookkeeping | 1.28 (0.01%) | 1.29 (0.01%) | 1.30 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 51.04 (0.23%) | 51.02 (0.23%) | 50.99 (0.23%) |
| Embedding | 1.43 (0.01%) | 1.43 (0.01%) | 1.42 (0.01%) |
| Final model RMSNorm | 55.32 (0.25%) | 55.32 (0.25%) | 55.34 (0.25%) |
| LM head＋greedy（不含 final norm） | 3193.41 (14.55%) | 3203.86 (14.55%) | 3199.46 (14.56%) |
| Host-DSP boundary | 610.78 (2.78%) | 618.51 (2.81%) | 619.16 (2.82%) |
| Complete Host wall | 21945.97 (100.00%) | 22024.69 (100.00%) | 21975.77 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.996426 | [0.9948638901985706, 0.9981332361325884] |
| NORM_COMPACT_over_OPT | 1.001358 | [0.9999921298853339, 1.0026929901594308] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| OPT | 2175.644 | 45.566 |
| ORIGINAL | 2173.308 | 45.404 |
| NORM_COMPACT | 2162.376 | 45.505 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.
