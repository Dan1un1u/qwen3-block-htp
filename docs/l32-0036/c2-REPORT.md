# L32-0036 C2 exact log2 softmax optimization

No rotation; per-output-channel native W4, SP2mode8, FP32 residual, latest native-row1 decode. LOG2 retains original clipped-score exact log2; FAST preserves its full integer probabilities; FP uses vendor HVX exp and its frozen own reference, same U8 output/AV. All weights/qparams fixed. Fixed-token primary trajectory,5short10formal cyclic paired rounds, repeat10; repeat1 auxiliary. Greedy and fixed full16 independent teacher gates both pass; selectedlayer7/chain3/chain16 pass; complete8MiB/nointermediateDDR/no spill/exact ledger. No PPL, quality or baseline promotion.

The original C expression uses uint64, but compiler already lowers its16 divisions to unsigned32 helpers. The first candidate reduces16 divisions to one; do not claim a64-to32 instruction-width optimization. Component diagnostics enable numerical telemetry whereas formal full-model runs disable it, so component time is not directly extrapolated.

C2 obtains the integer quotient using NR64 seed plus one Newton update and integer correction before constructing any probability; it preserves EXACT normalization. Exhaustive8,388,609 real denominators, initial quotient at most1 off and corrected quotient exact. Below32768 exact divide retained. Skip only unused statistics when audit disabled; audit mask counting is vectorized and retained. Native decode publishes every live probability lane and clears scratch, avoiding irrelevant-row tensor clearing. Full numerical gates unchanged.

## prefill

| Module | LOG2 | FAST | FP |
|---|---:|---:|---:|
| I/O、metadata | 126.98 (0.43%) | 127.41 (0.47%) | 128.24 (0.46%) |
| Input RMSNorm | 1268.26 (4.29%) | 1268.37 (4.63%) | 1290.32 (4.60%) |
| QKV＋RoPE | 3866.79 (13.07%) | 3866.15 (14.12%) | 3871.76 (13.79%) |
| QK–Softmax–AV | 8387.31 (28.36%) | 6241.18 (22.80%) | 6914.63 (24.63%) |
| O projection | 1581.78 (5.35%) | 1580.61 (5.77%) | 1583.68 (5.64%) |
| Post-attention residual＋RMSNorm | 1276.43 (4.32%) | 1276.34 (4.66%) | 1274.69 (4.54%) |
| Gate/Up＋SwiGLU | 5511.81 (18.63%) | 5508.65 (20.12%) | 5515.51 (19.65%) |
| Down | 3135.87 (10.60%) | 3136.02 (11.46%) | 3128.26 (11.14%) |
| Final residual | 1.84 (0.01%) | 1.95 (0.01%) | 2.01 (0.01%) |
| KV carrier conversion | 26.23 (0.09%) | 26.16 (0.10%) | 26.60 (0.09%) |
| KV append DMA | 107.59 (0.36%) | 107.77 (0.39%) | 107.51 (0.38%) |
| Block orchestration | 23.32 (0.08%) | 23.09 (0.08%) | 23.02 (0.08%) |
| Layer bookkeeping | 13.79 (0.05%) | 14.12 (0.05%) | 13.93 (0.05%) |
| Stage-boundary bookkeeping | 7.07 (0.02%) | 6.86 (0.03%) | 6.70 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 91.32 (0.31%) | 89.83 (0.33%) | 89.32 (0.32%) |
| Embedding | 54.58 (0.18%) | 54.44 (0.20%) | 55.11 (0.20%) |
| Final model RMSNorm | 60.38 (0.20%) | 60.55 (0.22%) | 55.57 (0.20%) |
| LM head＋greedy（不含 final norm） | 3198.51 (10.81%) | 3199.92 (11.69%) | 3190.68 (11.37%) |
| Host-DSP boundary | 839.25 (2.84%) | 785.81 (2.87%) | 796.88 (2.84%) |
| Complete Host wall | 29579.10 (100.00%) | 27375.25 (100.00%) | 28074.41 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| FAST_over_LOG2 | 0.925493 | [0.9203774225335516, 0.9297203235717207] |
| FAST_over_FP | 0.975096 | [0.9702816395354887, 0.978593428318567] |
| FP_over_LOG2 | 0.949130 | [0.9468118198773631, 0.9515427542332164] |
## decode

| Module | LOG2 | FAST | FP |
|---|---:|---:|---:|
| I/O、metadata | 123.21 (0.56%) | 122.83 (0.56%) | 123.38 (0.56%) |
| Input RMSNorm | 871.01 (3.94%) | 871.06 (3.99%) | 871.09 (3.98%) |
| QKV＋RoPE | 1668.81 (7.56%) | 1669.08 (7.65%) | 1667.96 (7.63%) |
| QK–Softmax–AV | 5989.51 (27.12%) | 5775.69 (26.47%) | 5759.24 (26.34%) |
| O projection | 884.39 (4.00%) | 885.55 (4.06%) | 883.17 (4.04%) |
| Post-attention residual＋RMSNorm | 874.35 (3.96%) | 874.38 (4.01%) | 878.33 (4.02%) |
| Gate/Up＋SwiGLU | 4831.65 (21.88%) | 4831.39 (22.14%) | 4838.44 (22.13%) |
| Down | 2623.32 (11.88%) | 2619.91 (12.01%) | 2627.47 (12.02%) |
| Final residual | 1.18 (0.01%) | 1.17 (0.01%) | 1.17 (0.01%) |
| KV carrier conversion | 48.18 (0.22%) | 48.17 (0.22%) | 48.19 (0.22%) |
| KV append DMA | 90.74 (0.41%) | 90.95 (0.42%) | 90.82 (0.42%) |
| Block orchestration | 16.44 (0.07%) | 16.38 (0.08%) | 16.40 (0.08%) |
| Layer bookkeeping | 9.48 (0.04%) | 9.42 (0.04%) | 9.47 (0.04%) |
| Stage-boundary bookkeeping | 1.25 (0.01%) | 1.25 (0.01%) | 1.25 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 50.96 (0.23%) | 50.98 (0.23%) | 51.01 (0.23%) |
| Embedding | 1.49 (0.01%) | 1.50 (0.01%) | 1.50 (0.01%) |
| Final model RMSNorm | 55.33 (0.25%) | 55.36 (0.25%) | 55.94 (0.26%) |
| LM head＋greedy（不含 final norm） | 3189.72 (14.44%) | 3189.51 (14.62%) | 3187.72 (14.58%) |
| Host-DSP boundary | 751.71 (3.40%) | 707.33 (3.24%) | 752.25 (3.44%) |
| Complete Host wall | 22082.72 (100.00%) | 21821.91 (100.00%) | 21864.80 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| FAST_over_LOG2 | 0.988189 | [0.9841430582434486, 0.9912787372635457] |
| FAST_over_FP | 0.998038 | [0.9948777304857016, 1.0009434126636474] |
| FP_over_LOG2 | 0.990132 | [0.9877373702183398, 0.9924653256287171] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| LOG2 | 2163.690 | 45.284 |
| FAST | 2337.878 | 45.826 |
| FP | 2279.656 | 45.736 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.

## Matched4-head component, audit telemetry enabled

| Shape | LOG2 us | FAST us | FP us |
|---|---:|---:|---:|
| m64-p0-random | 173.092 | 76.794 | 105.451 |
| m1-p64-random | 3.529 | 2.029 | 2.166 |
