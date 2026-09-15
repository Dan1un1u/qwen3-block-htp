# EXP-0280 C1 targeted native decode optimization

No rotation; native per-channel W4, SP2 mode8, FP32 residual; original weights/qparams. Same pipeline in every arm. OPT uses two independent gathers before loads; ORIGINAL retains sequential decode gathers. NORM_COMPACT changes only Norm interface; SWIGLU_COMPACT changes only SwiGLU interface; COMPACT changes both. Prefill SP2 arithmetic and scheduling are unchanged; compact prefill remains a diagnostic control.

The historical B01/B11 comparison included unequal gather scheduling and cannot identify a pure format effect. This local experiment separates that implementation difference. Independent selected/slice/full gates precede fixed five short and ten formal cyclic paired rounds, repeat10 primary and repeat1 auxiliary. All samples retained. No PPL or baseline promotion.

## prefill

| Module | OPT | ORIGINAL | NORM_COMPACT | SWIGLU_COMPACT | COMPACT |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 253.08 (0.73%) | 254.52 (0.74%) | 252.97 (0.73%) | 253.04 (0.73%) | 254.68 (0.73%) |
| Input RMSNorm | 2020.42 (5.83%) | 2020.76 (5.84%) | 2107.95 (6.05%) | 2020.85 (5.83%) | 2108.44 (6.04%) |
| QKV＋Q/K Norm-RoPE | 7042.53 (20.33%) | 7047.25 (20.37%) | 7041.44 (20.23%) | 7046.56 (20.31%) | 7046.34 (20.19%) |
| QK–Softmax–AV | 3493.41 (10.08%) | 3490.88 (10.09%) | 3493.36 (10.03%) | 3488.39 (10.06%) | 3493.60 (10.01%) |
| O projection | 2089.70 (6.03%) | 2089.10 (6.04%) | 2090.02 (6.00%) | 2085.76 (6.01%) | 2090.93 (5.99%) |
| Post-attention residual＋RMSNorm | 2116.51 (6.11%) | 2116.02 (6.11%) | 2201.06 (6.32%) | 2115.06 (6.10%) | 2200.68 (6.31%) |
| Gate/Up＋SwiGLU | 7045.63 (20.34%) | 7034.21 (20.33%) | 7038.98 (20.22%) | 7070.15 (20.38%) | 7076.90 (20.28%) |
| Down | 3781.82 (10.92%) | 3780.19 (10.92%) | 3785.30 (10.87%) | 3784.33 (10.91%) | 3790.10 (10.86%) |
| Final residual | 2.97 (0.01%) | 2.96 (0.01%) | 2.88 (0.01%) | 2.95 (0.01%) | 2.87 (0.01%) |
| KV carrier conversion | 136.96 (0.40%) | 137.08 (0.40%) | 136.93 (0.39%) | 137.36 (0.40%) | 137.05 (0.39%) |
| KV append DMA | 295.29 (0.85%) | 295.42 (0.85%) | 294.15 (0.84%) | 295.40 (0.85%) | 294.40 (0.84%) |
| Block orchestration | 35.88 (0.10%) | 35.91 (0.10%) | 35.41 (0.10%) | 35.96 (0.10%) | 35.75 (0.10%) |
| Layer bookkeeping | 27.39 (0.08%) | 27.27 (0.08%) | 27.40 (0.08%) | 27.24 (0.08%) | 27.36 (0.08%) |
| Stage-boundary bookkeeping | 21.20 (0.06%) | 21.35 (0.06%) | 20.86 (0.06%) | 21.25 (0.06%) | 21.23 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 117.56 (0.34%) | 117.52 (0.34%) | 115.38 (0.33%) | 117.90 (0.34%) | 117.27 (0.34%) |
| Embedding | 55.40 (0.16%) | 55.25 (0.16%) | 54.99 (0.16%) | 54.83 (0.16%) | 55.08 (0.16%) |
| Final model RMSNorm | 13.40 (0.04%) | 13.43 (0.04%) | 13.41 (0.04%) | 13.38 (0.04%) | 13.42 (0.04%) |
| LM head＋greedy，不含 final norm | 5196.51 (15.00%) | 5189.50 (15.00%) | 5191.12 (14.91%) | 5191.89 (14.97%) | 5194.07 (14.88%) |
| Host-DSP boundary | 900.04 (2.60%) | 875.46 (2.53%) | 911.15 (2.62%) | 925.96 (2.67%) | 942.95 (2.70%) |
| Complete Host wall | 34645.71 (100.00%) | 34604.08 (100.00%) | 34814.76 (100.00%) | 34688.25 (100.00%) | 34903.11 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 1.001203 | [0.9989107343928558, 1.0037319115356547] |
| OPT_over_SWIGLU_COMPACT | 0.998774 | [0.9958881809242182, 1.0017572610109982] |
| OPT_over_COMPACT | 0.992625 | [0.9903130824668885, 0.9946345341400135] |
| NORM_COMPACT_over_OPT | 1.004880 | [1.0024251417911065, 1.0071634929344362] |
| SWIGLU_COMPACT_over_OPT | 1.001228 | [0.9982458215384187, 1.0041287959376883] |
## decode

| Module | OPT | ORIGINAL | NORM_COMPACT | SWIGLU_COMPACT | COMPACT |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 250.16 (1.18%) | 250.58 (1.18%) | 249.39 (1.18%) | 249.14 (1.17%) | 250.13 (1.18%) |
| Input RMSNorm | 368.93 (1.74%) | 368.93 (1.74%) | 371.85 (1.76%) | 368.99 (1.74%) | 371.84 (1.75%) |
| QKV＋Q/K Norm-RoPE | 2503.73 (11.80%) | 2502.56 (11.78%) | 2497.74 (11.83%) | 2498.04 (11.78%) | 2502.57 (11.81%) |
| QK–Softmax–AV | 1930.15 (9.10%) | 1929.44 (9.08%) | 1926.33 (9.12%) | 1924.32 (9.07%) | 1929.57 (9.10%) |
| O projection | 1344.18 (6.34%) | 1344.73 (6.33%) | 1344.50 (6.37%) | 1344.95 (6.34%) | 1346.59 (6.35%) |
| Post-attention residual＋RMSNorm | 369.12 (1.74%) | 369.13 (1.74%) | 372.01 (1.76%) | 369.05 (1.74%) | 371.91 (1.75%) |
| Gate/Up＋SwiGLU | 6388.24 (30.11%) | 6409.39 (30.17%) | 6323.18 (29.94%) | 6393.00 (30.14%) | 6357.67 (30.00%) |
| Down | 3494.52 (16.47%) | 3491.90 (16.44%) | 3490.17 (16.52%) | 3490.25 (16.46%) | 3495.27 (16.49%) |
| Final residual | 1.87 (0.01%) | 1.88 (0.01%) | 1.87 (0.01%) | 1.87 (0.01%) | 1.87 (0.01%) |
| KV carrier conversion | 291.71 (1.38%) | 291.59 (1.37%) | 291.70 (1.38%) | 291.63 (1.38%) | 291.71 (1.38%) |
| KV append DMA | 122.48 (0.58%) | 122.94 (0.58%) | 121.76 (0.58%) | 122.30 (0.58%) | 122.61 (0.58%) |
| Block orchestration | 27.81 (0.13%) | 27.81 (0.13%) | 27.84 (0.13%) | 27.84 (0.13%) | 27.87 (0.13%) |
| Layer bookkeeping | 16.72 (0.08%) | 16.70 (0.08%) | 16.73 (0.08%) | 16.70 (0.08%) | 16.71 (0.08%) |
| Stage-boundary bookkeeping | 1.70 (0.01%) | 1.70 (0.01%) | 1.70 (0.01%) | 1.69 (0.01%) | 1.70 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.79 (0.35%) | 73.68 (0.35%) | 73.67 (0.35%) | 73.76 (0.35%) | 73.77 (0.35%) |
| Embedding | 1.79 (0.01%) | 1.78 (0.01%) | 1.79 (0.01%) | 1.78 (0.01%) | 1.77 (0.01%) |
| Final model RMSNorm | 15.33 (0.07%) | 15.34 (0.07%) | 15.23 (0.07%) | 15.22 (0.07%) | 15.37 (0.07%) |
| LM head＋greedy，不含 final norm | 3230.44 (15.23%) | 3227.80 (15.19%) | 3225.26 (15.27%) | 3225.21 (15.21%) | 3229.03 (15.24%) |
| Host-DSP boundary | 781.18 (3.68%) | 794.71 (3.74%) | 768.87 (3.64%) | 792.90 (3.74%) | 786.64 (3.71%) |
| Complete Host wall | 21213.86 (100.00%) | 21242.58 (100.00%) | 21121.62 (100.00%) | 21208.66 (100.00%) | 21194.61 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.998648 | [0.9972770250244527, 0.9999994527842706] |
| OPT_over_SWIGLU_COMPACT | 1.000245 | [0.9986306447840593, 1.0024870472429492] |
| OPT_over_COMPACT | 1.000908 | [0.999598642283361, 1.0022146725282794] |
| NORM_COMPACT_over_OPT | 0.995652 | [0.99295675403091, 0.9981400554638926] |
| SWIGLU_COMPACT_over_OPT | 0.999755 | [0.9975191228158208, 1.0013712329208935] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| OPT | 1847.271 | 47.139 |
| ORIGINAL | 1849.493 | 47.075 |
| NORM_COMPACT | 1838.301 | 47.345 |
| SWIGLU_COMPACT | 1845.005 | 47.151 |
| COMPACT | 1833.647 | 47.182 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.
