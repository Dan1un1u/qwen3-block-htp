# L32-0004 native pipeline profiling

SM8750 / HTP V79; ten fixed AB/BA pairs per recipe. Same sealed original-derived weights/qparams. No baseline quality promotion.

## prefill

Microseconds; parentheses are share of complete Host wall. Decode rows are per generated token.

| Module | W4A16 before | W4A16 after | W4A8 before | W4A8 after |
|---|---:|---:|---:|---:|
| I/O、metadata | 231.2 (0.16%) | 224.0 (0.44%) | 144.3 (0.05%) | 145.8 (0.25%) |
| Input RMSNorm | 281.8 (0.19%) | 281.5 (0.55%) | 1529.6 (0.52%) | 1529.8 (2.58%) |
| QKV＋RoPE | 101572.5 (69.30%) | 5729.8 (11.29%) | 248470.0 (83.82%) | 15266.0 (25.76%) |
| QK–Softmax–AV | 11514.4 (7.86%) | 11542.5 (22.74%) | 15599.5 (5.26%) | 11591.3 (19.56%) |
| O projection | 2864.3 (1.95%) | 2873.6 (5.66%) | 941.3 (0.32%) | 952.0 (1.61%) |
| Post-attention residual＋RMSNorm | 272.3 (0.19%) | 272.4 (0.54%) | 2681.9 (0.90%) | 2681.5 (4.52%) |
| Gate/Up＋SwiGLU | 17325.9 (11.82%) | 17329.5 (34.15%) | 14267.6 (4.81%) | 14327.9 (24.17%) |
| Down | 5850.4 (3.99%) | 5843.5 (11.51%) | 2948.3 (0.99%) | 2971.4 (5.01%) |
| Final residual | 80.6 (0.05%) | 80.4 (0.16%) | 1153.7 (0.39%) | 1154.1 (1.95%) |
| KV carrier conversion | 96.4 (0.07%) | 88.1 (0.17%) | 2740.0 (0.92%) | 2725.5 (4.60%) |
| KV append DMA | 171.8 (0.12%) | 171.1 (0.34%) | 128.2 (0.04%) | 125.6 (0.21%) |
| Block orchestration | 14.1 (0.01%) | 13.3 (0.03%) | 22.4 (0.01%) | 21.2 (0.04%) |
| Layer bookkeeping | 13.6 (0.01%) | 12.7 (0.03%) | 14.1 (0.00%) | 13.3 (0.02%) |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.1 (0.02%) | 7.2 (0.00%) | 7.5 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 96.5 (0.07%) | 92.3 (0.18%) | 81.1 (0.03%) | 80.7 (0.14%) |
| Embedding | 50.8 (0.03%) | 50.9 (0.10%) | 42.2 (0.01%) | 43.0 (0.07%) |
| Final model RMSNorm | 48.7 (0.03%) | 48.6 (0.10%) | 3.9 (0.00%) | 3.6 (0.01%) |
| LM head＋greedy（不含 final norm） | 5329.3 (3.64%) | 5371.8 (10.58%) | 3273.6 (1.10%) | 3305.0 (5.58%) |
| Host–DSP 边界 | 742.6 (0.51%) | 716.7 (1.41%) | 2392.9 (0.81%) | 2327.3 (3.93%) |
| 完整 Host wall | 146565.2 (100.00%) | 50750.7 (100.00%) | 296441.8 (100.00%) | 59272.5 (100.00%) |

## decode

Microseconds; parentheses are share of complete Host wall. Decode rows are per generated token.

| Module | W4A16 before | W4A16 after | W4A8 before | W4A8 after |
|---|---:|---:|---:|---:|
| I/O、metadata | 184.8 (0.12%) | 177.4 (0.42%) | 126.3 (0.03%) | 124.8 (0.21%) |
| Input RMSNorm | 277.2 (0.18%) | 277.3 (0.65%) | 1526.6 (0.39%) | 1526.7 (2.52%) |
| QKV＋RoPE | 100739.5 (64.01%) | 5702.5 (13.37%) | 245421.5 (63.15%) | 2222.7 (3.66%) |
| QK–Softmax–AV | 22715.1 (14.43%) | 3658.4 (8.58%) | 114373.3 (29.43%) | 29667.4 (48.89%) |
| O projection | 2818.0 (1.79%) | 2823.6 (6.62%) | 939.9 (0.24%) | 942.0 (1.55%) |
| Post-attention residual＋RMSNorm | 268.1 (0.17%) | 267.7 (0.63%) | 2678.2 (0.69%) | 2677.0 (4.41%) |
| Gate/Up＋SwiGLU | 17305.0 (11.00%) | 17301.5 (40.56%) | 14230.1 (3.66%) | 14228.5 (23.45%) |
| Down | 5850.4 (3.72%) | 5845.3 (13.70%) | 2940.8 (0.76%) | 2942.6 (4.85%) |
| Final residual | 79.5 (0.05%) | 79.6 (0.19%) | 1152.1 (0.30%) | 1152.3 (1.90%) |
| KV carrier conversion | 136.9 (0.09%) | 136.7 (0.32%) | 49.4 (0.01%) | 49.5 (0.08%) |
| KV append DMA | 99.3 (0.06%) | 100.1 (0.23%) | 82.8 (0.02%) | 84.2 (0.14%) |
| Block orchestration | 9.7 (0.01%) | 8.9 (0.02%) | 16.5 (0.00%) | 15.8 (0.03%) |
| Layer bookkeeping | 9.3 (0.01%) | 9.6 (0.02%) | 9.4 (0.00%) | 9.2 (0.02%) |
| Stage-boundary bookkeeping | 1.5 (0.00%) | 1.5 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 61.3 (0.04%) | 61.3 (0.14%) | 52.2 (0.01%) | 51.4 (0.08%) |
| Embedding | 1.7 (0.00%) | 1.8 (0.00%) | 1.5 (0.00%) | 1.5 (0.00%) |
| Final model RMSNorm | 2.3 (0.00%) | 2.0 (0.00%) | 3.7 (0.00%) | 3.5 (0.01%) |
| LM head＋greedy（不含 final norm） | 5430.4 (3.45%) | 5479.1 (12.85%) | 3265.9 (0.84%) | 3287.3 (5.42%) |
| Host–DSP 边界 | 1382.4 (0.88%) | 721.0 (1.69%) | 1782.6 (0.46%) | 1698.0 (2.80%) |
| 完整 Host wall | 157372.1 (100.00%) | 42655.4 (100.00%) | 388654.0 (100.00%) | 60685.7 (100.00%) |

## E2E

| Recipe / mode | Before token/s | After token/s | After/before Host wall (95% CI) |
|---|---:|---:|---|
| w4a16 / prefill | 436.67 | 1261.07 | 0.3463 [0.3449, 0.3474] |
| w4a16 / decode | 6.35 | 23.44 | 0.2710 [0.2701, 0.2719] |
| w4a8 / prefill | 215.89 | 1079.76 | 0.1999 [0.1993, 0.2007] |
| w4a8 / decode | 2.57 | 16.48 | 0.1561 [0.1555, 0.1566] |

M64 prefill, W4A16 seven continuous decode tokens to EOS; W4A8 fifteen. Host timing includes embedding,16 layers,final norm,LM head,greedy and FastRPC; excludes loading/session preparation and external tokenizer. Different decode lengths are not a matched cross-recipe comparison. All480 timed token-boundary ledgers reconcile exactly.

## Implementation and acceptance limits

L32-0004 uses original-derived sealed Llama weights and A8 calibration unchanged.
Head64 RoPE uses the existing SF32 HVX operations and carrier packing. A8 handles
only its valid decode row, retains scalar divide/round repair near quantization
boundaries, and dispatches prefill heads to the existing idle worker pool. K uses
masked native scatter for64-byte heads. V uses the existing exact recenter LUT
and native delta-packing kernel with correctly scoped scratch and saturation
accounting. No intermediate weight expansion, new rotation or model fitting.
W4A16 restores decode OPT2, adapting its two-head loops to Llama GQA4.

Both recipes pass single/3/16-layer validation. W4A16 three/16-layer outputs are
byte-identical to the sealed baseline and all numerical metrics unchanged; OPT2
conversion/sentinel audit passes. A8 output and KV match the independent integer
reference exactly. Across all formal generation runs, token IDs and selected
FP16/U8 logit encodings equal baseline. All480 timed ledgers reconcile; exact8MiB
VTCM is retained with zero timed intermediate DDR/spill and one HMX owner.

This is speed validation, not new PPL acceptance. Prior W4A16 PPL31.039 vs BF16
26.698 (+16.26%) remains failed; A8 text remains repeated Sleep, no quality gate.
PPL was not rerun for this pipeline experiment. Only tested M64+7/15 and capacity80
are covered; arbitrary-length serving and Llama R3/R4 remain unsupported.

Reproduction requires an approved active experiment and successful Llama memory
preflight. Existing Llama runners now select W4A16 OPT2; A8 uses the new core
with unchanged launch flags. run_llama32_pipeline_profile.py freezes baseline/
candidate binaries and packages, validates generation, then runs tenAB/BA pairs.
report_llama32_pipeline_profile.py produces additive prefill/decode tables and
paired-bootstrap95% intervals. Never overwrite existing attempt directories.
Evidence root: /mnt/d/llm_exp/results/llama32-htp/l32-0004. Native candidate source
2aa7831; later changes are reporting/metadata only. No quality baseline promotion.

L32-0005 matched A8 relative-speed results and current launch paths:  [LLAMA32_A8_RELATIVE_SPEED.md](LLAMA32_A8_RELATIVE_SPEED.md). Prior timings remain historical evidence.
