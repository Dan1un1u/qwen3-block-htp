# EXP-0267 Qwen3 SP2 native-W4 pipeline migration

Frozen Llama L32-0012 method: nearest241-level SP2; sigmoid(Gate)*Up LUT emits low/high native HMX planes. v=L+256H-32768; native packedW4 dot twice forprefill, low/high in separate otherwise-unused spatial rows in one64row tile fordecode. Precomputed weight sums and perchannelQ31 combine exactly before originalresidual. No W4-to-S8 expansion in Down and no groupedscales.

Prefill m4 serialstages; m5 Up+3HVXproducer overlap; m8 earlyinterleavedGate/Up plus two-slot HMX/HVX Down epilogue overlap. Decode same paired-row method in allSP2arms. Qwen-only scratchaliases dead hmx_activation duringMLP, no liveKV/residual/input overlaps. OneHMXowner,8MiB,zero timedintermediateDDR/spills and one fullmodelRPC/token.

Original QwenC64 W4weights, all nonmiddleA8 qparams, Q/Knorms, RoPE, tokenizer, embedding/head and originalOFFprefix frozen. 28freshSP2 LUTs exhaust65536each onindependenthostquantizer and devicegather. OriginalBF16 precedingstack supplies isolatedlayers0/14/27 realinputs. Actualhardware Gate/Up/residual drive independent integerDown/Q31/residual oracle. Threeconsecutive layers additionallyvalidate finalchangedpath and allSP2outputs/KV; decode high/low rows0..7 exact, unusedmiddle rows8..63 not treatedas semantic data. Untouchedupstream/KV exact againstU8 forisolatedlayers. FullSP2arms exact feedbacktokens and selectedlogitcodes, not fullvocabulary logits or floatingteacher equality.

Builda01 declarationfailure and slicevalidator endpoint/metadata/padding mistakes retained; no thresholdrelaxation or resampling. Singlelayer gate passedbeforeconditionalcontinuation. Arena lifetime change required to fit28layers; correctnessrevalidated and fullmodelmeasurements allsamefinalbuild. Llama and historicalQwenbranches preserved.

Statistics: arithmeticmean of ten formalroundmeans, pairedratio ofmeans,20000pairedbootstrap seed267. Repeat10 alone gates; repeat1 auxiliary.5short+10formal fourarms, bothrepeat1/10 =10560 fullmodelRPCs. All295680 perlayer exclusiveledgers and invocationledgers exact. Modulemeans allsamecohort so sumtoHost. Enginework/wait counters overlap and mustnotbeadded.

## Primary latency effects

| candidate/control | wall change | 95% ratioCI |
|---|---|---|
| r10_m4_over_m0_prefill_ns | -6.400% | [0.933737, 0.938120] |
| r10_m5_over_m0_prefill_ns | -10.927% | [0.888610, 0.892743] |
| r10_m8_over_m0_prefill_ns | -15.590% | [0.841501, 0.846744] |
| r10_m5_over_m4_prefill_ns | -4.837% | [0.949438, 0.953680] |
| r10_m8_over_m5_prefill_ns | -5.235% | [0.945529, 0.949779] |
| r10_m8_over_m4_prefill_ns | -9.819% | [0.899637, 0.904058] |
| r10_m4_over_m0_decode_ns | +1.098% | [1.008112, 1.013907] |
| r10_m5_over_m0_decode_ns | +1.131% | [1.008038, 1.014544] |
| r10_m8_over_m0_decode_ns | +1.228% | [1.009786, 1.014703] |
| r10_m5_over_m4_decode_ns | +0.032% | [0.998317, 1.002487] |
| r10_m8_over_m5_decode_ns | +0.096% | [0.999420, 1.002649] |
| r10_m8_over_m4_decode_ns | +0.128% | [0.999496, 1.003378] |

Both fullmodel SP2m8/U8 latencyCI upper<=1.10: True. NoPPL/quality/default promotion.

## M64 modules

| 模块 / μs | U8 | SP2 串行阶段 m4 | SP2 Up 重叠 m5 | SP2 优化流水 m8 |
|---|---|---|---|---|
| I/O、metadata | 239.2 (0.64%) | 238.7 (0.68%) | 239.2 (0.72%) | 242.6 (0.77%) |
| Input RMSNorm | 560.4 (1.49%) | 559.1 (1.59%) | 557.0 (1.67%) | 560.3 (1.77%) |
| QKV＋Q/K Norm-RoPE | 7009.7 (18.68%) | 7009.9 (19.96%) | 7008.5 (20.97%) | 7012.8 (22.14%) |
| QK–Softmax–AV | 3485.8 (9.29%) | 3489.5 (9.93%) | 3491.2 (10.44%) | 3489.0 (11.01%) |
| O projection | 1249.1 (3.33%) | 1248.9 (3.56%) | 1251.1 (3.74%) | 1251.9 (3.95%) |
| Post-attention residual＋RMSNorm | 661.7 (1.76%) | 661.7 (1.88%) | 661.8 (1.98%) | 661.1 (2.09%) |
| Gate/Up＋SwiGLU | 13963.3 (37.21%) | 9433.9 (26.86%) | 7723.9 (23.11%) | 7030.5 (22.19%) |
| Down | 3369.3 (8.98%) | 5412.0 (15.41%) | 5418.1 (16.21%) | 4414.6 (13.94%) |
| Final residual | 184.7 (0.49%) | 184.0 (0.52%) | 184.1 (0.55%) | 185.2 (0.58%) |
| KV carrier conversion | 135.7 (0.36%) | 135.5 (0.39%) | 135.3 (0.40%) | 135.2 (0.43%) |
| KV append DMA | 286.5 (0.76%) | 285.7 (0.81%) | 286.4 (0.86%) | 286.0 (0.90%) |
| Block orchestration | 37.3 (0.10%) | 36.1 (0.10%) | 36.6 (0.11%) | 36.4 (0.11%) |
| Layer bookkeeping | 29.0 (0.08%) | 28.6 (0.08%) | 28.9 (0.09%) | 29.1 (0.09%) |
| Stage-boundary bookkeeping | 20.4 (0.05%) | 20.3 (0.06%) | 20.2 (0.06%) | 20.1 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 117.2 (0.31%) | 115.6 (0.33%) | 115.9 (0.35%) | 114.2 (0.36%) |
| Embedding | 38.6 (0.10%) | 38.0 (0.11%) | 38.2 (0.11%) | 38.1 (0.12%) |
| Final model RMSNorm | 3.3 (0.01%) | 3.2 (0.01%) | 3.3 (0.01%) | 3.1 (0.01%) |
| LM head＋greedy，不含 final norm | 5261.3 (14.02%) | 5252.8 (14.95%) | 5259.0 (15.73%) | 5258.3 (16.60%) |
| Host–DSP 边界 | 877.6 (2.34%) | 974.7 (2.77%) | 970.5 (2.90%) | 910.4 (2.87%) |
| 完整 Host wall | 37529.9 (100.00%) | 35128.1 (100.00%) | 33429.1 (100.00%) | 31678.9 (100.00%) |

## Decode modules pertoken

| 模块 / μs | U8 | SP2 串行阶段 m4 | SP2 Up 重叠 m5 | SP2 优化流水 m8 |
|---|---|---|---|---|
| I/O、metadata | 238.5 (1.16%) | 239.9 (1.15%) | 240.0 (1.15%) | 242.8 (1.16%) |
| Input RMSNorm | 150.2 (0.73%) | 150.3 (0.72%) | 150.2 (0.72%) | 150.2 (0.72%) |
| QKV＋Q/K Norm-RoPE | 2495.1 (12.09%) | 2493.5 (11.95%) | 2498.1 (11.97%) | 2502.2 (11.98%) |
| QK–Softmax–AV | 1906.9 (9.24%) | 1910.8 (9.16%) | 1912.8 (9.17%) | 1914.3 (9.17%) |
| O projection | 1241.0 (6.02%) | 1241.5 (5.95%) | 1245.0 (5.97%) | 1246.0 (5.97%) |
| Post-attention residual＋RMSNorm | 199.9 (0.97%) | 200.3 (0.96%) | 199.9 (0.96%) | 199.8 (0.96%) |
| Gate/Up＋SwiGLU | 6500.7 (31.51%) | 6505.0 (31.19%) | 6513.4 (31.22%) | 6517.3 (31.21%) |
| Down | 3347.2 (16.22%) | 3530.1 (16.92%) | 3533.5 (16.94%) | 3539.4 (16.95%) |
| Final residual | 42.3 (0.21%) | 42.5 (0.20%) | 42.5 (0.20%) | 42.5 (0.20%) |
| KV carrier conversion | 291.4 (1.41%) | 291.4 (1.40%) | 291.5 (1.40%) | 291.5 (1.40%) |
| KV append DMA | 114.8 (0.56%) | 115.6 (0.55%) | 115.7 (0.55%) | 116.1 (0.56%) |
| Block orchestration | 29.3 (0.14%) | 29.0 (0.14%) | 29.0 (0.14%) | 29.0 (0.14%) |
| Layer bookkeeping | 16.7 (0.08%) | 16.9 (0.08%) | 16.8 (0.08%) | 17.0 (0.08%) |
| Stage-boundary bookkeeping | 1.9 (0.01%) | 1.8 (0.01%) | 1.9 (0.01%) | 1.9 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.4 (0.36%) | 73.5 (0.35%) | 73.5 (0.35%) | 73.5 (0.35%) |
| Embedding | 1.5 (0.01%) | 1.6 (0.01%) | 1.6 (0.01%) | 1.6 (0.01%) |
| Final model RMSNorm | 2.9 (0.01%) | 2.9 (0.01%) | 2.9 (0.01%) | 2.9 (0.01%) |
| LM head＋greedy，不含 final norm | 3215.4 (15.58%) | 3215.7 (15.42%) | 3211.7 (15.39%) | 3217.8 (15.41%) |
| Host–DSP 边界 | 762.6 (3.70%) | 796.1 (3.82%) | 785.3 (3.76%) | 779.4 (3.73%) |
| 完整 Host wall | 20631.8 (100.00%) | 20858.4 (100.00%) | 20865.1 (100.00%) | 20885.2 (100.00%) |

Actual text is a diagnostic only; timing continues fixed16outputs including afterEOS.

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
  "4": {
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
  },
  "5": {
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
| m0_r1 | 1642.677 | 45.902 | 43.746 | 42.569 |
| m4_r1 | 1759.100 | 45.238 | 43.482 | 42.413 |
| m5_r1 | 1842.525 | 45.416 | 43.834 | 42.584 |
| m8_r1 | 1947.127 | 45.469 | 44.106 | 43.013 |
| m0_r10 | 1705.305 | 48.469 | 46.109 | 43.337 |
| m4_r10 | 1821.905 | 47.942 | 45.976 | 43.535 |
| m5_r10 | 1914.502 | 47.927 | 46.189 | 43.526 |
| m8_r10 | 2020.269 | 47.881 | 46.383 | 43.434 |

Measured completeembedding/28layers/finalnorm/head/greedy/FastRPC Hostwall. Tokenizer/detokenizer,ADB,modelcoldload excluded from hotinference; generationloop additionallyincludes deviceHostloop and JSONserialization. No perlayer extrapolation.


Build provenance clarification: full-model captures used source17c259f; timed campaign uses497c30a after a Python-only selected-code representation fix. Rebuilds prove all four Android/DSP binary hashes identical (binary_cohort_identity.json). The first successful short U8 sample is retained, not remeasured or filtered. Model arithmetic/layout was last changed at9e3918b, before all successful3layer captures and fullmodel tests. Slice validation refinements exclude only nonsemantic decode middle rows8..63; rows0..7 and all other capture bytes/KV remain exact. See attempts.json for every preserved failed invocation of build/validation orchestration.
