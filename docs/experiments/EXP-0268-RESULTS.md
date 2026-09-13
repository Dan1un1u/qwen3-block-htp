# EXP-0268 Qwen3 U8 prefill scheduling and SP2 net cost

All five arms use the same binary and immutable EXP0267 packages. U8 m0 original; m1 three HVX contexts after Gate/Up; m2 consumes completed Up groups; m3 additionally interleaves Gate/Up batches. SP2 m8 retains the optimized two-plane producer and overlapping Down epilogue. U8 LUT gather/saturating pack and one-pass native Down unchanged. Dedicated middle allocation prevents overwrite of live Up input. Decode algorithm unchanged. SP2 still uses its own previously validated pipelined gather; this is a practical optimized-method comparison, not mathematical proof of globally optimal U8.

Layers0/14/27 and consecutive3 live carriers/KV/output match original U8 exactly. Full28 U8 feedback tokens and selected logit codes match sealed original; SP2 matches sealed SP2. Not full-vocabulary logit or PPL evaluation.5short+10formal five rotated arms, bothrepeat1/10:13200fullmodelRPCs,369600exclusive layer ledgers. Repeat10 primary,repeat1 auxiliary. Pairedbootstrap20000 seed268, no outlier deletion or optional resampling.

## Primary latency effects

| candidate/control | wall change | 95% ratioCI |
|---|---|---|
| r10_m1_over_m0_prefill_ns | -12.254% | [0.874884, 0.879938] |
| r10_m2_over_m0_prefill_ns | -16.856% | [0.829439, 0.833449] |
| r10_m3_over_m0_prefill_ns | -18.273% | [0.814751, 0.819577] |
| r10_m2_over_m1_prefill_ns | -5.245% | [0.944919, 0.949831] |
| r10_m3_over_m2_prefill_ns | -1.704% | [0.979681, 0.986460] |
| r10_m8_over_m0_prefill_ns | -16.058% | [0.836432, 0.842344] |
| r10_m8_over_m3_prefill_ns | +2.710% | [1.023299, 1.030716] |
| r10_m1_over_m0_decode_ns | +0.074% | [0.999838, 1.001768] |
| r10_m2_over_m0_decode_ns | +0.040% | [0.997725, 1.002807] |
| r10_m3_over_m0_decode_ns | -0.005% | [0.997202, 1.002410] |
| r10_m2_over_m1_decode_ns | -0.035% | [0.996896, 1.002384] |
| r10_m3_over_m2_decode_ns | -0.044% | [0.995400, 1.003291] |
| r10_m8_over_m0_decode_ns | +0.814% | [1.005150, 1.010819] |
| r10_m8_over_m3_decode_ns | +0.818% | [1.006323, 1.010089] |

OptimizedU8/originalU8 both phase CI upper<=1.10: True. SP2/optimizedU8 is diagnostic net cost. NoPPL/quality/default promotion.

## M64 modules

| 模块 / μs | U8 原始 | U8 三 HVX | U8 Up 重叠 | U8 交错流水 | SP2 优化流水 |
|---|---|---|---|---|---|
| I/O、metadata | 231.0 (0.62%) | 232.5 (0.71%) | 234.5 (0.75%) | 233.9 (0.76%) | 233.6 (0.74%) |
| Input RMSNorm | 566.4 (1.51%) | 570.0 (1.73%) | 568.2 (1.82%) | 560.7 (1.83%) | 560.8 (1.78%) |
| QKV＋Q/K Norm-RoPE | 7011.6 (18.67%) | 7014.3 (21.29%) | 7017.6 (22.48%) | 7016.7 (22.86%) | 7014.7 (22.25%) |
| QK–Softmax–AV | 3481.6 (9.27%) | 3484.6 (10.57%) | 3481.8 (11.15%) | 3483.4 (11.35%) | 3488.6 (11.07%) |
| O projection | 1251.0 (3.33%) | 1249.1 (3.79%) | 1248.2 (4.00%) | 1251.2 (4.08%) | 1247.6 (3.96%) |
| Post-attention residual＋RMSNorm | 661.5 (1.76%) | 661.7 (2.01%) | 661.6 (2.12%) | 662.1 (2.16%) | 661.8 (2.10%) |
| Gate/Up＋SwiGLU | 13989.3 (37.25%) | 9378.4 (28.46%) | 7664.5 (24.55%) | 7158.9 (23.33%) | 7038.1 (22.33%) |
| Down | 3388.3 (9.02%) | 3375.4 (10.24%) | 3366.7 (10.78%) | 3374.7 (11.00%) | 4398.2 (13.95%) |
| Final residual | 184.8 (0.49%) | 183.6 (0.56%) | 183.9 (0.59%) | 184.9 (0.60%) | 185.7 (0.59%) |
| KV carrier conversion | 134.6 (0.36%) | 135.1 (0.41%) | 135.1 (0.43%) | 135.3 (0.44%) | 135.3 (0.43%) |
| KV append DMA | 283.9 (0.76%) | 284.1 (0.86%) | 284.7 (0.91%) | 285.2 (0.93%) | 284.8 (0.90%) |
| Block orchestration | 37.4 (0.10%) | 36.7 (0.11%) | 36.7 (0.12%) | 36.9 (0.12%) | 37.1 (0.12%) |
| Layer bookkeeping | 29.7 (0.08%) | 29.9 (0.09%) | 30.1 (0.10%) | 29.9 (0.10%) | 30.0 (0.10%) |
| Stage-boundary bookkeeping | 19.8 (0.05%) | 19.8 (0.06%) | 19.8 (0.06%) | 19.9 (0.06%) | 19.7 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 117.0 (0.31%) | 118.1 (0.36%) | 118.2 (0.38%) | 118.2 (0.38%) | 116.4 (0.37%) |
| Embedding | 38.6 (0.10%) | 38.7 (0.12%) | 38.7 (0.12%) | 38.5 (0.13%) | 38.3 (0.12%) |
| Final model RMSNorm | 3.8 (0.01%) | 3.9 (0.01%) | 3.9 (0.01%) | 3.7 (0.01%) | 3.8 (0.01%) |
| LM head＋greedy，不含 final norm | 5260.6 (14.01%) | 5256.9 (15.95%) | 5253.4 (16.82%) | 5259.3 (17.14%) | 5261.0 (16.69%) |
| Host–DSP 边界 | 862.7 (2.30%) | 879.0 (2.67%) | 876.3 (2.81%) | 838.2 (2.73%) | 768.0 (2.44%) |
| 完整 Host wall | 37553.8 (100.00%) | 32952.0 (100.00%) | 31223.7 (100.00%) | 30691.7 (100.00%) | 31523.4 (100.00%) |

## Decode modules pertoken

| 模块 / μs | U8 原始 | U8 三 HVX | U8 Up 重叠 | U8 交错流水 | SP2 优化流水 |
|---|---|---|---|---|---|
| I/O、metadata | 231.5 (1.12%) | 231.8 (1.12%) | 233.0 (1.13%) | 231.4 (1.12%) | 231.6 (1.12%) |
| Input RMSNorm | 150.1 (0.73%) | 150.1 (0.73%) | 150.1 (0.73%) | 150.1 (0.73%) | 150.1 (0.72%) |
| QKV＋Q/K Norm-RoPE | 2495.4 (12.12%) | 2496.4 (12.11%) | 2495.3 (12.11%) | 2497.0 (12.13%) | 2497.5 (12.03%) |
| QK–Softmax–AV | 1905.8 (9.26%) | 1907.4 (9.26%) | 1907.6 (9.26%) | 1906.7 (9.26%) | 1907.9 (9.19%) |
| O projection | 1240.7 (6.03%) | 1242.1 (6.03%) | 1240.2 (6.02%) | 1241.9 (6.03%) | 1240.7 (5.98%) |
| Post-attention residual＋RMSNorm | 199.6 (0.97%) | 199.5 (0.97%) | 199.8 (0.97%) | 199.7 (0.97%) | 199.8 (0.96%) |
| Gate/Up＋SwiGLU | 6515.8 (31.64%) | 6521.9 (31.65%) | 6512.2 (31.61%) | 6520.4 (31.67%) | 6522.0 (31.42%) |
| Down | 3359.3 (16.31%) | 3360.6 (16.31%) | 3356.3 (16.29%) | 3357.9 (16.31%) | 3536.5 (17.04%) |
| Final residual | 42.4 (0.21%) | 42.4 (0.21%) | 42.4 (0.21%) | 42.4 (0.21%) | 42.7 (0.21%) |
| KV carrier conversion | 293.7 (1.43%) | 293.7 (1.43%) | 293.7 (1.43%) | 293.7 (1.43%) | 293.8 (1.42%) |
| KV append DMA | 114.1 (0.55%) | 114.7 (0.56%) | 114.8 (0.56%) | 114.2 (0.55%) | 114.2 (0.55%) |
| Block orchestration | 29.7 (0.14%) | 29.8 (0.14%) | 29.8 (0.14%) | 29.8 (0.14%) | 29.5 (0.14%) |
| Layer bookkeeping | 17.1 (0.08%) | 17.1 (0.08%) | 17.0 (0.08%) | 17.1 (0.08%) | 17.0 (0.08%) |
| Stage-boundary bookkeeping | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.5 (0.36%) | 73.6 (0.36%) | 73.6 (0.36%) | 73.5 (0.36%) | 73.5 (0.35%) |
| Embedding | 1.4 (0.01%) | 1.4 (0.01%) | 1.4 (0.01%) | 1.4 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 2.9 (0.01%) | 2.9 (0.01%) | 2.8 (0.01%) | 2.8 (0.01%) | 2.9 (0.01%) |
| LM head＋greedy，不含 final norm | 3211.9 (15.60%) | 3212.8 (15.59%) | 3209.9 (15.58%) | 3209.0 (15.59%) | 3213.6 (15.48%) |
| Host–DSP 边界 | 704.1 (3.42%) | 706.3 (3.43%) | 717.3 (3.48%) | 699.0 (3.39%) | 681.9 (3.28%) |
| 完整 Host wall | 20590.8 (100.00%) | 20606.1 (100.00%) | 20598.9 (100.00%) | 20589.8 (100.00%) | 20758.4 (100.00%) |

Text diagnostics; fixed16outputs continue afterEOS.

{
  "0": {
    "token_ids": [
      32313,
      11,
      1431,
      358,
      1184,
      311,
      311,
      653,
      419,
      13,
      4695,
      11,
      1128,
      374,
      279,
      6722
    ],
    "text": "Okay, now I need to to do this. Now, what is the capital"
  },
  "1": {
    "token_ids": [
      32313,
      11,
      1431,
      358,
      1184,
      311,
      311,
      653,
      419,
      13,
      4695,
      11,
      1128,
      374,
      279,
      6722
    ],
    "text": "Okay, now I need to to do this. Now, what is the capital"
  },
  "2": {
    "token_ids": [
      32313,
      11,
      1431,
      358,
      1184,
      311,
      311,
      653,
      419,
      13,
      4695,
      11,
      1128,
      374,
      279,
      6722
    ],
    "text": "Okay, now I need to to do this. Now, what is the capital"
  },
  "3": {
    "token_ids": [
      32313,
      11,
      1431,
      358,
      1184,
      311,
      311,
      653,
      419,
      13,
      4695,
      11,
      1128,
      374,
      279,
      6722
    ],
    "text": "Okay, now I need to to do this. Now, what is the capital"
  },
  "8": {
    "token_ids": [
      785,
      6722,
      315,
      9625,
      374,
      12095,
      13,
      151645,
      271,
      151645,
      271,
      334,
      16141,
      66963,
      576,
      6722
    ],
    "text": "The capital of France is Paris.<|im_end|>\n\n<|im_end|>\n\n**Answer:** The capital"
  }
}

## Actual E2E

| arm/repeat | M64 token/s | decode15 token/s | 16outputs/modelHost token/s | 16outputs/generationloop token/s |
|---|---|---|---|---|
| m0_r1 | 1632.831 | 45.612 | 43.471 | 42.511 |
| m1_r1 | 1861.126 | 45.738 | 44.157 | 43.169 |
| m2_r1 | 1964.016 | 45.643 | 44.294 | 43.271 |
| m3_r1 | 1983.224 | 45.762 | 44.437 | 43.428 |
| m8_r1 | 1928.016 | 45.306 | 43.923 | 42.862 |
| m0_r10 | 1704.224 | 48.565 | 46.187 | 44.086 |
| m1_r10 | 1942.218 | 48.529 | 46.778 | 44.623 |
| m2_r10 | 2049.726 | 48.546 | 47.030 | 44.238 |
| m3_r10 | 2085.256 | 48.568 | 47.123 | 44.896 |
| m8_r10 | 2030.238 | 48.173 | 46.661 | 44.609 |

Complete embedding/28layers/finalnorm/head/greedy/FastRPC Host wall. Tokenizer,ADB,coldload excluded from hotinference; generationloop includes deviceHostloop/JSONserialization. No layer extrapolation.
