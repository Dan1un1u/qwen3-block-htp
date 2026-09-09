# EXP0260 W4F16 decode exact conversion and bulk-transfer optimization

Freeze original C64 per-output-channel W4, all FP16 math and cache semantics. OPT0 retains current control, OPT1 batches exact storage conversion around unchanged scalar expf/ordered FP32 sum/division, OPT2 adds HVX zero/copy for dynamic decode Attention. W4F16 M1 only; other recipes/prefill kernels unchanged. No new quantizer, grouping, rotation, approximation or PPL acceptance.

Initial legacy global-audit option was incompatible with replay, repaired with dedicated component audit; failed logs retained. Exhaustive finite FP16 roundtrip and probability-midpoint tests found V79vcvt drops negativezero sign. Explicit IEEE sign preservation repaired it; final finite63488patterns and46080midpoint-neighbour tests pass, plus oldSoftmax entire probability buffers exact. Sentinel failure was not observed real-model quality regression and no gate was loosened.

Layer/slice live hidden and allstep KV exact, fullmodel control reproduces old sealed EXP0230 selectedtokens/FP16codes. All2970layer plus5280fullmodel timedRPCs validated. Same8MiB VTCM, no timed intermediateDDR/spill/fulllogit export, oneRPC/fullpass. Independent ledgers/statistics and code/provenance hashes retained.

Formal comparisons are rotated paired5short10formal, bothrepeat1/10. Primary is repeat10 fullmodel decode; prefill guard<=10percent. No optional additional sampling used.

| scope | control μs | candidate μs | paired cost change | 95% interval |
|---|---|---|---|---|
| r1_prefill_ns | 63156.120 | 63094.870 | +0.015% | [0.9973888193044861, 1.013590221195162] |
| r1_decode_ns | 92451.670 | 63677.484 | -31.078% | [0.688141557782803, 0.691509192372775] |
| r10_prefill_ns | 61476.721 | 61524.255 | +0.048% | [0.9988243539579311, 1.0040196211828807] |
| r10_decode_ns | 94996.261 | 65375.276 | -31.166% | [0.6875342742974253, 0.688920339570876] |

M64 overview: μs(Host share); F16EXP0218 and W4A8R3EXP0259 historical nonpaired, new W4F16C64 candidate is not Selected EXP0166. W4A8 previous numerical failure unchanged. Per-row medians need not sum; individual raw additive ledgers checked.

| 模块 | F16A16 EXP0218 | W4A16 C64 EXP0260 | W4A8 R3 EXP0259 | A8相对W4A16增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 345.4 (0.56%) | 229.4 (0.61%) | +50.56% |
| Input RMSNorm | 489.7 (0.61%) | 490.8 (0.80%) | 578.5 (1.54%) | -15.18% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11768.4 (19.13%) | 7126.6 (18.97%) | +65.13% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3959.0 (6.43%) | 3442.0 (9.16%) | +15.02% |
| O projection | 5757.7 (7.14%) | 5066.3 (8.23%) | 1257.1 (3.35%) | +303.01% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 475.3 (0.77%) | 658.5 (1.75%) | -27.81% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22392.0 (36.40%) | 14073.6 (37.46%) | +59.11% |
| Down | 13447.9 (16.67%) | 8671.3 (14.09%) | 3416.5 (9.09%) | +153.81% |
| Final residual | 140.1 (0.17%) | 140.4 (0.23%) | 184.6 (0.49%) | -23.92% |
| KV carrier conversion | 174.0 (0.22%) | 148.1 (0.24%) | 135.7 (0.36%) | +9.15% |
| KV append DMA | 343.6 (0.43%) | 331.8 (0.54%) | 282.4 (0.75%) | +17.50% |
| Block orchestration | 16.1 (0.02%) | 19.7 (0.03%) | 34.3 (0.09%) | -42.64% |
| Layer bookkeeping | 23.9 (0.03%) | 25.2 (0.04%) | 23.8 (0.06%) | +6.01% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 6.8 (0.01%) | 20.2 (0.05%) | -66.21% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 103.5 (0.17%) | 117.1 (0.31%) | -11.64% |
| Embedding | 68.1 (0.08%) | 57.6 (0.09%) | 38.7 (0.10%) | +48.72% |
| Final model RMSNorm | 49.7 (0.06%) | 47.6 (0.08%) | 2.9 (0.01%) | +1528.97% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6592.8 (10.72%) | 5235.6 (13.93%) | +25.92% |
| Host–DSP 边界 | 2374.1 (2.94%) | 900.4 (1.46%) | 720.1 (1.92%) | +25.04% |
| 完整 Host wall | 80692.2 (100.00%) | 61524.3 (100.00%) | 37572.8 (100.00%) | +63.75% |

Actual warm28layer E2E,64prefill+15continuousdecode,16outputs. Host includes embedding/norm/head/greedy/FastRPC; generationloop includes deviceHost loop/log serialization, excludes model loading/ADB/externaltokenizer. No layer extrapolation.

| arm/repeat | prefill tok/s | decode tok/s | 16outputs/Host tok/s | generationloop tok/s |
|---|---|---|---|---|
| a0_r1 | 1013.361817 | 10.816462 | 11.036429 | 10.961526 |
| a2_r1 | 1014.345540 | 15.704138 | 15.713132 | 15.556316 |
| a0_r10 | 1041.044458 | 10.526730 | 10.765004 | 10.624772 |
| a2_r10 | 1040.240143 | 15.296303 | 15.352845 | 15.069794 |
