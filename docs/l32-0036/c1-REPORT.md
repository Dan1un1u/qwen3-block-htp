# L32-0036 C1 exact log2 softmax optimization

No rotation; per-output-channel native W4, SP2mode8, FP32 residual, latest native-row1 decode. LOG2 retains original clipped-score exact log2; FAST preserves its full integer probabilities; FP uses vendor HVX exp and its frozen own reference, same U8 output/AV. All weights/qparams fixed. Fixed-token primary trajectory,5short10formal cyclic paired rounds, repeat10; repeat1 auxiliary. Greedy and fixed full16 independent teacher gates both pass; selectedlayer7/chain3/chain16 pass; complete8MiB/nointermediateDDR/no spill/exact ledger. No PPL, quality or baseline promotion.

The original C expression uses uint64, but compiler already lowers its16 divisions to unsigned32 helpers. The first candidate reduces16 divisions to one; do not claim a64-to32 instruction-width optimization. Component diagnostics enable numerical telemetry whereas formal full-model runs disable it, so component time is not directly extrapolated.

## prefill

| Module | LOG2 | FAST | FP |
|---|---:|---:|---:|
| I/O、metadata | 125.82 (0.43%) | 126.59 (0.46%) | 126.11 (0.45%) |
| Input RMSNorm | 1268.61 (4.29%) | 1268.73 (4.57%) | 1290.81 (4.59%) |
| QKV＋RoPE | 3866.77 (13.08%) | 3868.91 (13.94%) | 3874.46 (13.77%) |
| QK–Softmax–AV | 8429.36 (28.52%) | 6536.39 (23.56%) | 6909.87 (24.56%) |
| O projection | 1582.31 (5.35%) | 1579.21 (5.69%) | 1579.58 (5.61%) |
| Post-attention residual＋RMSNorm | 1276.19 (4.32%) | 1276.30 (4.60%) | 1274.78 (4.53%) |
| Gate/Up＋SwiGLU | 5511.57 (18.65%) | 5508.69 (19.85%) | 5515.66 (19.60%) |
| Down | 3134.32 (10.61%) | 3125.28 (11.26%) | 3128.04 (11.12%) |
| Final residual | 2.35 (0.01%) | 2.45 (0.01%) | 2.30 (0.01%) |
| KV carrier conversion | 26.12 (0.09%) | 26.30 (0.09%) | 26.72 (0.09%) |
| KV append DMA | 107.67 (0.36%) | 107.78 (0.39%) | 107.49 (0.38%) |
| Block orchestration | 22.47 (0.08%) | 22.81 (0.08%) | 22.51 (0.08%) |
| Layer bookkeeping | 13.51 (0.05%) | 13.45 (0.05%) | 13.40 (0.05%) |
| Stage-boundary bookkeeping | 6.07 (0.02%) | 6.34 (0.02%) | 6.22 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 90.43 (0.31%) | 90.69 (0.33%) | 89.60 (0.32%) |
| Embedding | 55.29 (0.19%) | 55.15 (0.20%) | 55.00 (0.20%) |
| Final model RMSNorm | 60.14 (0.20%) | 60.19 (0.22%) | 55.44 (0.20%) |
| LM head＋greedy（不含 final norm） | 3214.97 (10.88%) | 3221.37 (11.61%) | 3206.25 (11.39%) |
| Host-DSP boundary | 759.72 (2.57%) | 848.93 (3.06%) | 853.73 (3.03%) |
| Complete Host wall | 29553.68 (100.00%) | 27745.57 (100.00%) | 28137.98 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| FAST_over_LOG2 | 0.938820 | [0.935908282534806, 0.9415619747155118] |
| FAST_over_FP | 0.986054 | [0.9831928372707143, 0.9890540521917152] |
| FP_over_LOG2 | 0.952097 | [0.9502291849590059, 0.9543157248563373] |
## decode

| Module | LOG2 | FAST | FP |
|---|---:|---:|---:|
| I/O、metadata | 121.78 (0.55%) | 122.39 (0.56%) | 121.80 (0.56%) |
| Input RMSNorm | 871.23 (3.95%) | 871.28 (3.97%) | 871.26 (3.99%) |
| QKV＋RoPE | 1671.56 (7.58%) | 1673.22 (7.62%) | 1671.08 (7.64%) |
| QK–Softmax–AV | 5990.57 (27.15%) | 5876.75 (26.75%) | 5762.05 (26.36%) |
| O projection | 886.10 (4.02%) | 882.45 (4.02%) | 880.38 (4.03%) |
| Post-attention residual＋RMSNorm | 875.05 (3.97%) | 875.00 (3.98%) | 879.02 (4.02%) |
| Gate/Up＋SwiGLU | 4829.84 (21.89%) | 4831.91 (22.00%) | 4844.10 (22.16%) |
| Down | 2624.18 (11.89%) | 2626.22 (11.96%) | 2628.59 (12.02%) |
| Final residual | 1.09 (0.00%) | 1.09 (0.00%) | 1.09 (0.00%) |
| KV carrier conversion | 48.19 (0.22%) | 48.14 (0.22%) | 48.17 (0.22%) |
| KV append DMA | 90.81 (0.41%) | 90.94 (0.41%) | 90.86 (0.42%) |
| Block orchestration | 15.76 (0.07%) | 15.84 (0.07%) | 15.78 (0.07%) |
| Layer bookkeeping | 9.62 (0.04%) | 9.70 (0.04%) | 9.64 (0.04%) |
| Stage-boundary bookkeeping | 1.24 (0.01%) | 1.25 (0.01%) | 1.25 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 51.11 (0.23%) | 51.12 (0.23%) | 51.06 (0.23%) |
| Embedding | 1.49 (0.01%) | 1.49 (0.01%) | 1.49 (0.01%) |
| Final model RMSNorm | 55.33 (0.25%) | 55.37 (0.25%) | 55.99 (0.26%) |
| LM head＋greedy（不含 final norm） | 3205.16 (14.53%) | 3204.74 (14.59%) | 3201.82 (14.65%) |
| Host-DSP boundary | 715.42 (3.24%) | 727.56 (3.31%) | 727.45 (3.33%) |
| Complete Host wall | 22065.54 (100.00%) | 21966.46 (100.00%) | 21862.86 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| FAST_over_LOG2 | 0.995510 | [0.992640358453942, 0.9987546168566345] |
| FAST_over_FP | 1.004739 | [1.0029324960459287, 1.006209387545521] |
| FP_over_LOG2 | 0.990815 | [0.9887941488736098, 0.9930826514884432] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| LOG2 | 2165.551 | 45.320 |
| FAST | 2306.674 | 45.524 |
| FP | 2274.506 | 45.740 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.

## Matched4-head component, audit telemetry enabled

| Shape | LOG2 us | FAST us | FP us |
|---|---:|---:|---:|
| m64-p0-random | 175.998 | 124.176 | 105.452 |
| m1-p64-random | 3.562 | 2.800 | 2.170 |
