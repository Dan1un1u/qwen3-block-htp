# L32-0044: Llama3.2-3B longer decode optimization

W4A8-SP2mode8, FP32 residual, no rotation. Frozen0041weights/scales and independent numerical arithmetic. Parent0043; no model-quality evaluation or automatic baseline promotion.

## Result

Primary M64+42/cache128: CONTROL 1130.655081/21.633357, OPT 1138.295039/22.538183 prefill/decode token/s.
Decode throughput +4.182553%, wall ratio 0.95985361, paired-bootstrap95%CI [0.9588332331620764, 0.9609113117067898]. Prefill +0.675711%; both10% gates pass.
Supplementary M64+63/cache128: CONTROL 1124.985320/21.210176; OPT 1132.026119/22.126467. Decode +4.320054%.
Five short plus ten formal pairedAB/BA cycles at42decode; five supplementary pairs at63decode, allrepeat10. Supplement is distinct fromprimary formal. All complete Host wall includes embedding,28blocks,finalnorm,head,greedy,FastRPC; excludes coldloading,sessionprepare,external tokenization. No fastest-run selection.

External Quant.npu Table12 SM8750/Llama3.2-3B/HellaSwag64+42:809.15/28.04tps. Our decode remains19.62% below its reported throughput. We matchlength only, notdataset/token trajectory or completequantization/runtime. External aspiration NOT reached.
Source: https://arxiv.org/html/2605.20295v1#S0.T12

## Implemented and rejected

- V recenter: replace two dependent VTCM gathers with exact eight-bank register LUT, preserving native V layout; exact reciprocal+remainder correction replaces repeated scalar divisions when bounds prove equivalence. Formal V preparation3123.26->1646.94us/token.
- Existing double-buffered native-W4 LM head batch8->32;501->126commands/token, same weightbytes and physical tile products. Formal head excludingnorm4265.5->3926.7us.
- Norm vector broadcast removal had no meaningful gain; scalar ordered sum from memory/registers was slower. Both preserve exact arithmetic but are rejected; original Norm restored. Contiguous K transpose also slower and restored. All attempted commits/logs retained.
- Final native implementation784e754; measured full28 build4686765b98939a34fec218af950eeba573087c71. Parentcontrol rebuilt at488e5f7 solely to support boundedlong evaluation, same parentarithmetic. Original16stepcontrol regression passed. Host v2 reader preserves oldv1 format and bounds counts/buffers.

## Correctness and physical evidence

Selectedlayers0/13/27,chain3/full28 exact independentFP32hidden+KV. Controlfixedtrajectory andcandidate truegreedy64steps each matchFP32hidden,tokenIDs,logitcodes andprefillKV againstindependent oracle. 1Bboundedlayer7regressionpasses. 67 validatedruns/20222 tokenboundaries;8600primaryformal and6400supplementary boundaries. No arithmetic failure.
594weight/scale/bias/LUT/qparam files identicalto0041. First16newreference positions matcholdreference. Integerreciprocal exhaustive16,711,680casespass. V lookup compiler audit confirms vlut32 instead ofvgather, unchanged0x300stackframe.
VTCM peak8360416/8388608; no timed intermediateDDR/spill, additiveledgers reconcile. Sameweightbytes andphysicalHMXtilepairs;375fewerheadcommands/token.
Ownedtooling failures: firstlegacyparser TypeError aftersuccessfuldevice run; fixed/reran. One unused-variable Werror; stage correctlyrejected stale seal beforedeployment; rebuilt. No requiredcheck downgraded.

## Exact launch contract and continuation

Use tools/execute_llama32_3b_decode.py with opt2c-l28 and QBH_3B_HEAD_TILES=32; QBH_W4U8_DECODE_AV_REQUANT_ROWS=4 is mandatory. Longfrontend adds expectedcount64 andcache128. QBH_3B_DECODE_COUNT=42 or63 specifies decodecount, nottotaloutputsteps. Use profile_llama32_3b_decode.py forprespecifiedpaired campaign; allraw commands andbinarySHA are in runprotocols.
Remaining primarydecode shares: Gate/Up+SwiGLU28.04%,Down15.44%,attention14.78%,QKV+RoPE13.30%,twoNormboundaries10.36%,head8.85%. K/V fullprefix preparationstillcosts3.78ms/token; consider persistent nativeKV and crossstage/nextlayer weightpreparation overlap. These are followupdirections, notmeasured gains.
Longlengths should remain explicit: candidate42decode first1-15/16-32/33-42windows22.7753/22.5324/22.2012tps; this iswithinrunlengthsensitivity, notmatched formal15token result.

Full module table follows. Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0044.

# L32-0044 full-model long decode modules

Units µs; parentheses complete Host-wall shares. M64+42/cache128, five short and ten formal paired AB/BA cycles,repeat10. Frozen weights/scales,SP2,FP32 residual,no rotations.

| 模块 | CONTROL Prefill | OPT Prefill | CONTROL Decode | OPT Decode |
|---|---:|---:|---:|---:|
| I/O、metadata | 247.7 (0.44%) | 244.1 (0.43%) | 247.3 (0.53%) | 244.1 (0.55%) |
| Input RMSNorm | 3534.9 (6.24%) | 3533.2 (6.28%) | 2294.1 (4.96%) | 2294.4 (5.17%) |
| QKV＋RoPE | 11125.1 (19.65%) | 11122.6 (19.78%) | 5902.5 (12.77%) | 5901.0 (13.30%) |
| QK–Softmax–AV | 6641.8 (11.73%) | 6648.1 (11.82%) | 8029.8 (17.37%) | 6558.4 (14.78%) |
| O projection | 4114.0 (7.27%) | 4125.5 (7.34%) | 2884.1 (6.24%) | 2884.5 (6.50%) |
| Post-attention residual＋RMSNorm | 3566.0 (6.30%) | 3565.1 (6.34%) | 2301.9 (4.98%) | 2301.9 (5.19%) |
| Gate/Up＋SwiGLU | 13738.1 (24.27%) | 13731.1 (24.42%) | 12435.6 (26.90%) | 12440.0 (28.04%) |
| Down | 7906.2 (13.97%) | 7915.4 (14.08%) | 6855.1 (14.83%) | 6851.7 (15.44%) |
| Final residual | 3.8 (0.01%) | 3.6 (0.01%) | 2.0 (0.00%) | 1.9 (0.00%) |
| KV carrier conversion | 84.2 (0.15%) | 84.1 (0.15%) | 15.7 (0.03%) | 15.6 (0.04%) |
| KV append DMA | 238.0 (0.42%) | 237.2 (0.42%) | 165.8 (0.36%) | 165.4 (0.37%) |
| Block orchestration | 37.4 (0.07%) | 35.6 (0.06%) | 28.7 (0.06%) | 28.4 (0.06%) |
| Layer bookkeeping | 26.9 (0.05%) | 25.3 (0.05%) | 16.2 (0.03%) | 15.8 (0.04%) |
| Stage-boundary bookkeeping | 7.0 (0.01%) | 7.6 (0.01%) | 1.6 (0.00%) | 1.7 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 94.9 (0.17%) | 93.3 (0.17%) | 51.1 (0.11%) | 51.0 (0.11%) |
| Embedding | 72.9 (0.13%) | 71.8 (0.13%) | 2.5 (0.01%) | 2.4 (0.01%) |
| Final model RMSNorm | 85.5 (0.15%) | 85.8 (0.15%) | 83.0 (0.18%) | 83.0 (0.19%) |
| LM head＋greedy（不含 final norm） | 4261.4 (7.53%) | 3928.5 (6.99%) | 4265.5 (9.23%) | 3926.7 (8.85%) |
| Host–DSP 边界 | 818.5 (1.45%) | 766.6 (1.36%) | 642.5 (1.39%) | 601.2 (1.35%) |
| 完整 Host wall | 56604.4 (100.00%) | 56224.4 (100.00%) | 46224.9 (100.00%) | 44369.1 (100.00%) |

| Shape | Arm | Prefill E2E token/s | Decode E2E token/s |
|---|---|---:|---:|
| M64+42 | CONTROL | 1130.66 | 21.63 |
| M64+42 | OPT | 1138.30 | 22.54 |
| M64+63 | CONTROL | 1124.99 | 21.21 |
| M64+63 | OPT | 1132.03 | 22.13 |

63decode is a five-pair supplement, separate from primary ten-pair formal42decode. Cold loading/session setup and external tokenization excluded. Numerical implementation verification is separate from model quality.
