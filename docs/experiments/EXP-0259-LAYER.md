# EXP0259 single-layer R3 implementation ablations

Same layer0, M64 plus eight real replay decode steps, repeat1/repeat10,5short10formal. Constant+vector includes fusedKoperand; stream restores five workers on readiness events and relocates bias buffers. Exact sqrtf and originalHMX retained.108 complete capture/output comparisons and5940timedRPC pass. Original ideal-R3 numerical gate still failed; noPPL acceptance.

| comparison | control_us | candidate_us | paired ratio | CI95 |
|---|---|---|---|---|
| r3_vs_off_r1_prefill_ns | 1709.505 | 5514.661 | 3.226826 | [3.128986909196109, 3.4032105853355823] |
| vector_vs_off_r1_prefill_ns | 1709.505 | 1988.021 | 1.167136 | [1.1292729896232918, 1.2525151734908508] |
| vector_vs_r3_r1_prefill_ns | 5514.661 | 1988.021 | 0.362133 | [0.3591743449626893, 0.36710822085263783] |
| stream_vs_off_r1_prefill_ns | 1709.505 | 1720.130 | 1.010228 | [0.9774429236861178, 1.0638207038221081] |
| stream_vs_r3_r1_prefill_ns | 5514.661 | 1720.130 | 0.312808 | [0.311007568943308, 0.3177507440597784] |
| r3_vs_off_r1_decode_ns | 970.283 | 1432.630 | 1.512891 | [1.2952963719120263, 1.5851736050513847] |
| vector_vs_off_r1_decode_ns | 970.283 | 982.604 | 1.014330 | [0.959034001547852, 1.082717790263442] |
| vector_vs_r3_r1_decode_ns | 1432.630 | 982.604 | 0.683289 | [0.6573950991613288, 0.7274603094615563] |
| stream_vs_off_r1_decode_ns | 970.283 | 1007.269 | 1.055503 | [0.9652120337209757, 1.1095948219702354] |
| stream_vs_r3_r1_decode_ns | 1432.630 | 1007.269 | 0.698137 | [0.6918988218441458, 0.7293602780610967] |
| r3_vs_off_r10_prefill_ns | 1545.740 | 5361.680 | 3.489197 | [3.40218430601202, 3.5654074204008976] |
| vector_vs_off_r10_prefill_ns | 1545.740 | 1802.594 | 1.180378 | [1.1489691661462378, 1.2117606672183638] |
| vector_vs_r3_r10_prefill_ns | 5361.680 | 1802.594 | 0.337088 | [0.3321160849930784, 0.34527436990949445] |
| stream_vs_off_r10_prefill_ns | 1545.740 | 1529.411 | 0.991698 | [0.9766964948198047, 1.0322189169206288] |
| stream_vs_r3_r10_prefill_ns | 5361.680 | 1529.411 | 0.285269 | [0.28212373205920516, 0.2895756910777673] |
| r3_vs_off_r10_decode_ns | 938.373 | 1419.274 | 1.513057 | [1.4743875193586422, 1.5371525500988428] |
| vector_vs_off_r10_decode_ns | 938.373 | 948.533 | 1.019704 | [1.0020489478840646, 1.0360570660017436] |
| vector_vs_r3_r10_decode_ns | 1419.274 | 948.533 | 0.681056 | [0.6581126740356721, 0.6899331799702884] |
| stream_vs_off_r10_decode_ns | 938.373 | 945.500 | 1.017568 | [0.9811467971207115, 1.0396156668565721] |
| stream_vs_r3_r10_decode_ns | 1419.274 | 945.500 | 0.671674 | [0.6608134730622643, 0.6853806086745389] |

Fullmodel escalation eligible: False. Not model quality or baseline acceptance. E2E unmeasured at this layer scope.
