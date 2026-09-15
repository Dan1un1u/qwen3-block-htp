# L32-0035 C1 targeted native decode optimization

No rotation; native per-channel W4, SP2 mode8, FP32 residual; original weights/qparams. Same pipeline in every arm. OPT uses two independent gathers before loads; ORIGINAL retains sequential decode gathers. NORM_COMPACT changes only Norm interface; SWIGLU_COMPACT changes only SwiGLU interface; COMPACT changes both. Prefill SP2 arithmetic and scheduling are unchanged; compact prefill remains a diagnostic control.

The historical B01/B11 comparison included unequal gather scheduling and cannot identify a pure format effect. This local experiment separates that implementation difference. Independent selected/slice/full gates precede fixed five short and ten formal cyclic paired rounds, repeat10 primary and repeat1 auxiliary. All samples retained. No PPL or baseline promotion.

## prefill

| Module | OPT | ORIGINAL | NORM_COMPACT | SWIGLU_COMPACT | COMPACT |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 126.52 (0.43%) | 126.91 (0.43%) | 127.76 (0.43%) | 127.14 (0.43%) | 126.10 (0.42%) |
| Input RMSNorm | 1268.10 (4.29%) | 1268.56 (4.30%) | 1318.76 (4.45%) | 1268.23 (4.29%) | 1318.08 (4.44%) |
| QKV＋RoPE | 3869.66 (13.09%) | 3868.55 (13.10%) | 3873.08 (13.06%) | 3868.88 (13.08%) | 3872.58 (13.05%) |
| QK–Softmax–AV | 8408.11 (28.44%) | 8392.53 (28.42%) | 8395.11 (28.32%) | 8393.28 (28.37%) | 8398.07 (28.29%) |
| O projection | 1581.48 (5.35%) | 1580.79 (5.35%) | 1579.17 (5.33%) | 1578.58 (5.34%) | 1581.71 (5.33%) |
| Post-attention residual＋RMSNorm | 1276.08 (4.32%) | 1276.15 (4.32%) | 1325.22 (4.47%) | 1276.26 (4.31%) | 1325.57 (4.47%) |
| Gate/Up＋SwiGLU | 5509.15 (18.63%) | 5508.54 (18.65%) | 5519.13 (18.62%) | 5551.38 (18.76%) | 5551.47 (18.70%) |
| Down | 3131.77 (10.59%) | 3130.36 (10.60%) | 3132.07 (10.56%) | 3132.81 (10.59%) | 3134.47 (10.56%) |
| Final residual | 1.66 (0.01%) | 1.63 (0.01%) | 1.80 (0.01%) | 1.70 (0.01%) | 1.65 (0.01%) |
| KV carrier conversion | 27.20 (0.09%) | 27.33 (0.09%) | 27.39 (0.09%) | 27.32 (0.09%) | 27.22 (0.09%) |
| KV append DMA | 107.83 (0.36%) | 107.74 (0.36%) | 107.86 (0.36%) | 107.81 (0.36%) | 107.97 (0.36%) |
| Block orchestration | 23.11 (0.08%) | 23.18 (0.08%) | 23.34 (0.08%) | 23.07 (0.08%) | 23.22 (0.08%) |
| Layer bookkeeping | 14.49 (0.05%) | 14.61 (0.05%) | 14.60 (0.05%) | 14.56 (0.05%) | 14.60 (0.05%) |
| Stage-boundary bookkeeping | 6.86 (0.02%) | 7.00 (0.02%) | 7.11 (0.02%) | 6.81 (0.02%) | 6.88 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 90.67 (0.31%) | 91.64 (0.31%) | 92.01 (0.31%) | 90.64 (0.31%) | 91.06 (0.31%) |
| Embedding | 55.16 (0.19%) | 55.65 (0.19%) | 55.60 (0.19%) | 55.61 (0.19%) | 55.14 (0.19%) |
| Final model RMSNorm | 60.30 (0.20%) | 60.28 (0.20%) | 60.21 (0.20%) | 60.21 (0.20%) | 60.32 (0.20%) |
| LM head＋greedy（不含 final norm） | 3210.42 (10.86%) | 3204.74 (10.85%) | 3207.21 (10.82%) | 3206.14 (10.84%) | 3202.33 (10.79%) |
| Host-DSP boundary | 799.05 (2.70%) | 787.60 (2.67%) | 780.08 (2.63%) | 798.06 (2.70%) | 787.19 (2.65%) |
| Complete Host wall | 29567.62 (100.00%) | 29533.78 (100.00%) | 29647.53 (100.00%) | 29588.47 (100.00%) | 29685.63 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 1.001146 | [0.9984106558431172, 1.0039506765790693] |
| OPT_over_SWIGLU_COMPACT | 0.999295 | [0.9960783751213498, 1.0027034199787233] |
| OPT_over_COMPACT | 0.996025 | [0.9931161839041606, 0.9994797976121345] |
| NORM_COMPACT_over_OPT | 1.002703 | [1.0003515667832896, 1.0051420298830485] |
| SWIGLU_COMPACT_over_OPT | 1.000705 | [0.9973038687964628, 1.0039370645690178] |
## decode

| Module | OPT | ORIGINAL | NORM_COMPACT | SWIGLU_COMPACT | COMPACT |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 121.96 (0.55%) | 122.58 (0.55%) | 123.02 (0.56%) | 122.74 (0.56%) | 121.98 (0.55%) |
| Input RMSNorm | 870.93 (3.94%) | 870.97 (3.94%) | 872.67 (3.96%) | 870.97 (3.94%) | 872.64 (3.96%) |
| QKV＋RoPE | 1671.11 (7.56%) | 1671.26 (7.56%) | 1671.93 (7.58%) | 1671.45 (7.56%) | 1671.00 (7.57%) |
| QK–Softmax–AV | 5997.20 (27.14%) | 5993.04 (27.10%) | 5996.91 (27.18%) | 5993.19 (27.11%) | 5994.62 (27.17%) |
| O projection | 881.22 (3.99%) | 881.68 (3.99%) | 878.71 (3.98%) | 881.71 (3.99%) | 882.49 (4.00%) |
| Post-attention residual＋RMSNorm | 874.65 (3.96%) | 874.64 (3.95%) | 876.30 (3.97%) | 874.68 (3.96%) | 876.31 (3.97%) |
| Gate/Up＋SwiGLU | 4868.92 (22.04%) | 4892.86 (22.12%) | 4847.36 (21.97%) | 4886.99 (22.11%) | 4854.30 (22.00%) |
| Down | 2624.48 (11.88%) | 2625.60 (11.87%) | 2627.36 (11.91%) | 2627.59 (11.89%) | 2624.52 (11.90%) |
| Final residual | 1.04 (0.00%) | 1.04 (0.00%) | 1.04 (0.00%) | 1.04 (0.00%) | 1.04 (0.00%) |
| KV carrier conversion | 48.35 (0.22%) | 48.35 (0.22%) | 48.33 (0.22%) | 48.33 (0.22%) | 48.37 (0.22%) |
| KV append DMA | 90.81 (0.41%) | 90.84 (0.41%) | 90.92 (0.41%) | 90.98 (0.41%) | 90.82 (0.41%) |
| Block orchestration | 16.23 (0.07%) | 16.22 (0.07%) | 16.15 (0.07%) | 16.21 (0.07%) | 16.15 (0.07%) |
| Layer bookkeeping | 9.35 (0.04%) | 9.35 (0.04%) | 9.34 (0.04%) | 9.36 (0.04%) | 9.36 (0.04%) |
| Stage-boundary bookkeeping | 1.23 (0.01%) | 1.23 (0.01%) | 1.23 (0.01%) | 1.23 (0.01%) | 1.23 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 51.03 (0.23%) | 51.00 (0.23%) | 51.04 (0.23%) | 51.02 (0.23%) | 51.01 (0.23%) |
| Embedding | 1.46 (0.01%) | 1.47 (0.01%) | 1.47 (0.01%) | 1.48 (0.01%) | 1.46 (0.01%) |
| Final model RMSNorm | 55.31 (0.25%) | 55.35 (0.25%) | 55.30 (0.25%) | 55.32 (0.25%) | 55.32 (0.25%) |
| LM head＋greedy（不含 final norm） | 3201.35 (14.49%) | 3193.39 (14.44%) | 3198.91 (14.50%) | 3198.32 (14.47%) | 3193.29 (14.47%) |
| Host-DSP boundary | 707.81 (3.20%) | 715.21 (3.23%) | 695.63 (3.15%) | 705.01 (3.19%) | 697.66 (3.16%) |
| Complete Host wall | 22094.43 (100.00%) | 22116.08 (100.00%) | 22063.63 (100.00%) | 22107.62 (100.00%) | 22063.57 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.999021 | [0.996977577888823, 1.0009612535020078] |
| OPT_over_SWIGLU_COMPACT | 0.999404 | [0.9983712791631298, 1.0002967945837193] |
| OPT_over_COMPACT | 1.001399 | [0.9991110966443888, 1.0038187320687284] |
| NORM_COMPACT_over_OPT | 0.998606 | [0.9971081257339593, 1.0002745894353209] |
| SWIGLU_COMPACT_over_OPT | 1.000597 | [0.9997032934771696, 1.0016313778960975] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| OPT | 2164.530 | 45.260 |
| ORIGINAL | 2167.010 | 45.216 |
| NORM_COMPACT | 2158.696 | 45.323 |
| SWIGLU_COMPACT | 2163.004 | 45.233 |
| COMPACT | 2155.925 | 45.324 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.
