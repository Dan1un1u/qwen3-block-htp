# L32-0011: SP2 Down full-model E2E

Llama 3.2 1B Instruct,16 layers,SM8750/HTP V79. Ten fixed AB/BA pairs,one shared sealed build. M64 prefill plus15 continuous autoregressive decode tokens,capacity80. SP2 mode5 with Up/SwiGLU overlap on every Down; per-channel W4/embedding/head/other qparams unchanged.

## prefill

Microseconds; parentheses: percent of complete Host wall. Decode is per generated token.

| Module | Original U8 | SP2 Down |
|---|---:|---:|
| I/O、metadata | 135.1 (0.43%) | 135.0 (0.39%) |
| Input RMSNorm | 1320.6 (4.19%) | 1320.9 (3.84%) |
| QKV＋RoPE | 3843.2 (12.20%) | 3774.7 (10.97%) |
| QK–Softmax–AV | 8089.1 (25.69%) | 8110.7 (23.56%) |
| O projection | 715.1 (2.27%) | 715.1 (2.08%) |
| Post-attention residual＋RMSNorm | 695.0 (2.21%) | 693.3 (2.01%) |
| Gate/Up＋SwiGLU | 7892.4 (25.06%) | 9792.2 (28.45%) |
| Down | 2494.2 (7.92%) | 3554.8 (10.33%) |
| Final residual | 345.9 (1.10%) | 348.9 (1.01%) |
| KV carrier conversion | 25.1 (0.08%) | 25.4 (0.07%) |
| KV append DMA | 103.5 (0.33%) | 102.7 (0.30%) |
| Block orchestration | 22.9 (0.07%) | 22.1 (0.06%) |
| Layer bookkeeping | 13.0 (0.04%) | 13.1 (0.04%) |
| Stage-boundary bookkeeping | 7.1 (0.02%) | 7.0 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 80.8 (0.26%) | 81.4 (0.24%) |
| Embedding | 43.4 (0.14%) | 43.1 (0.13%) |
| Final model RMSNorm | 3.8 (0.01%) | 3.7 (0.01%) |
| LM head＋greedy（不含 final norm） | 3253.1 (10.33%) | 3226.9 (9.37%) |
| Host–DSP 边界 | 2407.0 (7.64%) | 2452.4 (7.12%) |
| 完整 Host wall | 31490.5 (100.00%) | 34423.3 (100.00%) |

## decode

Microseconds; parentheses: percent of complete Host wall. Decode is per generated token.

| Module | Original U8 | SP2 Down |
|---|---:|---:|
| I/O、metadata | 110.6 (0.51%) | 113.4 (0.52%) |
| Input RMSNorm | 85.0 (0.39%) | 84.9 (0.39%) |
| QKV＋RoPE | 1642.1 (7.58%) | 1632.4 (7.53%) |
| QK–Softmax–AV | 5932.6 (27.37%) | 5938.6 (27.39%) |
| O projection | 709.4 (3.27%) | 711.4 (3.28%) |
| Post-attention residual＋RMSNorm | 169.0 (0.78%) | 168.7 (0.78%) |
| Gate/Up＋SwiGLU | 4864.6 (22.45%) | 4892.4 (22.57%) |
| Down | 2474.4 (11.42%) | 2571.3 (11.86%) |
| Final residual | 81.3 (0.38%) | 80.6 (0.37%) |
| KV carrier conversion | 48.1 (0.22%) | 48.0 (0.22%) |
| KV append DMA | 83.4 (0.38%) | 82.8 (0.38%) |
| Block orchestration | 16.5 (0.08%) | 16.4 (0.08%) |
| Layer bookkeeping | 9.3 (0.04%) | 9.2 (0.04%) |
| Stage-boundary bookkeeping | 1.3 (0.01%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.3 (0.24%) | 51.2 (0.24%) |
| Embedding | 1.4 (0.01%) | 0.9 (0.00%) |
| Final model RMSNorm | 2.8 (0.01%) | 2.9 (0.01%) |
| LM head＋greedy（不含 final norm） | 3249.2 (14.99%) | 3233.2 (14.91%) |
| Host–DSP 边界 | 2139.9 (9.87%) | 2041.0 (9.41%) |
| 完整 Host wall | 21671.9 (100.00%) | 21680.7 (100.00%) |

## E2E

| Mode | U8 token/s | SP2 token/s | Latency increase | Throughput decrease | Latency ratio 95% CI | 10% gate |
|---|---:|---:|---:|---:|---|---|
| prefill | 2032.36 | 1859.20 | +9.31% | +8.52% | [1.08846, 1.09785] | PASS |
| decode | 46.14 | 46.12 | +0.04% | +0.04% | [0.99437, 1.00667] | PASS |

All320 timed full-model boundaries reconcile exactly. Full Host wall includes embedding,16 transformer layers,final norm,LM head,greedy and FastRPC; excludes tokenizer,weight loading and session preparation. One prompt/context shape only; does not establish long-context scaling or PPL/text recovery. Baseline and SP2 use their own independent integer greedy goldens; all16 IDs and selected U8 logit codes agree. Continuous3-layer output and KV exact. SP2 text remains repetitive; no quality promotion.

# L32-0011 completed: full-model SP2 pipeline gate passed

Candidate A01 is the only native candidate and only formal attempt. New mode5 streams prefill SP2 production from completed Up32-tile batches using three existing persistent HVX workers. Low plane moves from q alias to dedicated middle allocation so it cannot overwrite still-live Up input. Each worker owns disjoint tiles/private scratch. Flags publish after HMXwait; abort/join on failure; all workers join before Down. Down and decode arithmetic/schedule remain mode4. Existing genericU8 path unchanged. Native packedW4,241-level frozenSP2 table,alpha,Q31 and original qparams/weights unchanged. No added VTCM allocation.

Historical L32-0010 detailed prefill: Gate/Up5043.48us(U8)/5041.49us(SP2);activation2822.98us/6371.32us. Exposed activation explained the main module increase. New candidate overlaps HVX activation with Up HMX/DMA. New Gate/Up+SwiGLU fullmodel9792.2us vs originalU87892.4us;Down3554.8us vs2494.2us. Previous SP2 combined11412.8us is a non-paired historical reference. No separate proof of bank conflicts or specific microarchitectural stall cause.

Fixed ten AB/BA M64+15 same-build fullmodel pairs: prefillU82032.35998 /SP21859.20319token/s,latencyratio1.093134944,95%CI[1.088463022,1.097847090],PASS. DecodeU846.142635 /SP246.123949token/s,ratio1.000405124,CI[.994373023,1.006668006],PASS. Upper prefilloverhead9.7847% leaves only0.2153percentagepoints under10%; acceptance applies to this frozen prompt/context,not all sequence lengths. No unchanged formal repeats or selection among multiple runs.

Singlelayer0/7/15 outputs/KV exact,399360 outputcodes;consecutive3layers exact;16-step fullmodel greedyIDs/logitcodes match immutable independent integerreference forbotharms. All26successful deviceprocesses /360tokenboundaries (320formal) preserve8MiB,oneHMXowner,zero intermediateDDR/spill and weight expansion. All360ledgers additive. MaxVTCM8212960B. Assembly:producer/streamworker no vectorstack access;Down unchanged256B zero/255constant masks only,no tensors. Physical/algebraic layoutownership audit and all16LUT reconstruction checks retained.

Tested source8331c9ef7496b70e3d71d57505c53b22d1f5f3da. Layer1/3/16binaries,seals,assembly,immutable commands and raw results archived. No build/native failures. One local offline audit import failed using systemPython withoutnumpy,then completed under projectvenv;no device rerun. Models unchanged L32-0009 singlelayer and L32-0010 stack3/frontend;no weights generated. No PPL or quality/default promotion;SP2 remains repetitive in this case. DefaultoriginalU8 and frozenQwen/rotationbranch unchanged. Stop after bothformal gates pass.

Evidence: `/mnt/d/llm_exp/results/llama32-htp/l32-0011`. Opt-in `QBH_LLAMA_SP2=5` requires the frozen SP2 package,4 attention HVX contexts and32-tile Gate/Up batches. Modes3/4 remain historical controls.
