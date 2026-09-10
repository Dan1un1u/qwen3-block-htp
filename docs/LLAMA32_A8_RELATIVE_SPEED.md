# L32-0005: restore native W4A8 relative speed

Llama W4A8 OFF native W4 multiplication was already faster, but scalar carrier
and nonlinear work hid the gain. This change preserves every weight, qparam,
activation code and HMX ownership rule; no new quantization or rotation.

The matched full16-layer decode subcounters show V recenter/pack18.319 ->2.086ms
and softmax6.761 ->0.526ms. V LUT and saturation tables are now prepared once
per configuration and reused across heads instead of repeated per-head integer
division. Existing HVX tile4 softmax replaces scalar64-row work. Common norm,
residual and SwiGLU reuse valid4-row native paths; SwiGLU6.939 ->0.444ms. Pooled
residual workers also improve prefill. These are internal explanatory counters;
only the additive module table below is summed into complete Host wall.

Prefill RoPE13.760 ->2.347ms by avoiding64 scalar repair tests on ordinary rows,
hoisting affine constants, exact integer centering, and aligned native tile row
gather/merge. Scalar repair at rare rounding edges remains to preserve codes.
The initial endpoint predicate mask failed the device gate and was corrected
with a rotated byte mask. The CPU libnative layout probe did not expose this;
the failed device result remains evidence and is not a selected candidate.

Validation: independent A8 single/3/16 output and persistent KV exact; single-layer
padding poisoning leaves all valid outputs unchanged; eight dynamic HVX softmax
comparisons to scalar have zero mismatches. Common and SwiGLU poison counters
are1 and256. W4A16 full16 regression is byte-identical to L32-0002. W4 OPT2
conversion audit passes. Every full-generation run matches old IDs AND selected
logit codes; A8 separately retains the full15-decode continuation. Eight MiB VTCM,
peak7668960B A8/8330752B W4, zero timed intermediate DDR/spill, single HMX owner;
all112 A8 transformer projections plus head retain native W4, no weight expansion.

Fixed ten rotated ABC/BCA/CAB cycles use one shared64-token prompt and seven
continuous decode tokens in each arm; outputs are recipe dependent. Native
last-change be389d2, profiled source a6f5e61; binaries archived in experiment
baseline-build16/candidate-build16. Both prefill/decode pass paired-bootstrap95
upper<=1.10. No optional stopping or repeat1 gate. All240 timed ledgers reconcile.

A8 prefill1431.835tok/s and decode39.55584tok/s, versus matched W4A16
1260.21580 and23.59130: +13.62%/+67.67%. A8 old1081.55838/16.70891:
+32.39%/+136.74%. This restores a speed advantage, not an assumed universal2x.
Remaining prefill costs: GateUp/SwiGLU32.0%,attention25.9%,KV conversion6.1%;
decode GateUp/SwiGLU30.8%,attention22.8%,head12.6%,Down11.7%,Host boundary9.0%.
Further speed work should target these measured costs before changing matmul.

Quality unchanged/unaccepted: W4A16 prior PPL31.039101 vs BF16teacher26.697999
fails existing quality gate; A8 repeated Sleep remains unusable with no quality
gate. This speed experiment does not rerun PPL or promote a quality baseline.
Capacity80 and testedM64+7/15 remain the scope; no arbitrary-length/rotation claim.

Reproduction under a new approved experiment: build matching1/3/16 layer model,
run_llama32_stack.py (single --a8-audit for padding/softmax), then
run_llama32_a8_relative_profile.py deploy/gate/long/formal and
report_llama32_a8_relative_profile.py. Current destinations are immutable and
must be changed for any new experiment. Authoritative evidence is under
/mnt/d/llm_exp/results/llama32-htp/l32-0005 and project memory.

# L32-0005: native A8 relative speed

SM8750 / HTP V79. Fixed ten ABC/BCA/CAB cycles; identical original M64 prompt and seven continuous decode tokens for all three arms. Quantized recipe outputs differ; input dimensions and KV lengths match.

## prefill

Microseconds, parentheses are complete Host wall shares. Decode is per token.

| Module | W4A16 OPT2 | A8 before | A8 after |
|---|---:|---:|---:|
| I/O、metadata | 218.5 (0.43%) | 143.9 (0.24%) | 136.3 (0.30%) |
| Input RMSNorm | 282.2 (0.56%) | 1530.1 (2.59%) | 1321.6 (2.96%) |
| QKV＋RoPE | 5735.5 (11.29%) | 15267.4 (25.80%) | 3860.5 (8.64%) |
| QK–Softmax–AV | 11504.8 (22.65%) | 11573.7 (19.56%) | 11560.7 (25.86%) |
| O projection | 2872.3 (5.66%) | 948.1 (1.60%) | 953.3 (2.13%) |
| Post-attention residual＋RMSNorm | 273.2 (0.54%) | 2682.1 (4.53%) | 693.4 (1.55%) |
| Gate/Up＋SwiGLU | 17316.9 (34.10%) | 14276.0 (24.13%) | 14318.4 (32.03%) |
| Down | 5848.7 (11.52%) | 2953.1 (4.99%) | 2980.9 (6.67%) |
| Final residual | 80.4 (0.16%) | 1154.2 (1.95%) | 348.3 (0.78%) |
| KV carrier conversion | 88.4 (0.17%) | 2725.6 (4.61%) | 2724.5 (6.10%) |
| KV append DMA | 162.3 (0.32%) | 125.1 (0.21%) | 123.6 (0.28%) |
| Block orchestration | 13.5 (0.03%) | 21.9 (0.04%) | 23.1 (0.05%) |
| Layer bookkeeping | 13.9 (0.03%) | 13.5 (0.02%) | 12.5 (0.03%) |
| Stage-boundary bookkeeping | 7.2 (0.01%) | 7.6 (0.01%) | 6.9 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 92.4 (0.18%) | 79.6 (0.13%) | 81.3 (0.18%) |
| Embedding | 52.0 (0.10%) | 42.9 (0.07%) | 42.9 (0.10%) |
| Final model RMSNorm | 48.4 (0.10%) | 3.9 (0.01%) | 3.9 (0.01%) |
| LM head＋greedy（不含 final norm） | 5250.9 (10.34%) | 3206.2 (5.42%) | 3201.8 (7.16%) |
| Host–DSP 边界 | 923.4 (1.82%) | 2419.0 (4.09%) | 2303.9 (5.15%) |
| 完整 Host wall | 50785.0 (100.00%) | 59173.9 (100.00%) | 44697.9 (100.00%) |

## decode

Microseconds, parentheses are complete Host wall shares. Decode is per token.

| Module | W4A16 OPT2 | A8 before | A8 after |
|---|---:|---:|---:|
| I/O、metadata | 206.3 (0.49%) | 126.6 (0.21%) | 130.1 (0.51%) |
| Input RMSNorm | 277.3 (0.65%) | 1526.9 (2.55%) | 84.8 (0.34%) |
| QKV＋RoPE | 5712.0 (13.48%) | 2234.0 (3.73%) | 1669.5 (6.60%) |
| QK–Softmax–AV | 3640.8 (8.59%) | 28253.5 (47.21%) | 5773.8 (22.84%) |
| O projection | 2826.0 (6.67%) | 947.0 (1.58%) | 948.8 (3.75%) |
| Post-attention residual＋RMSNorm | 268.0 (0.63%) | 2676.7 (4.47%) | 165.0 (0.65%) |
| Gate/Up＋SwiGLU | 17301.0 (40.82%) | 14256.8 (23.82%) | 7778.5 (30.77%) |
| Down | 5848.6 (13.80%) | 2948.2 (4.93%) | 2961.6 (11.71%) |
| Final residual | 79.6 (0.19%) | 1152.3 (1.93%) | 80.5 (0.32%) |
| KV carrier conversion | 136.7 (0.32%) | 49.5 (0.08%) | 49.0 (0.19%) |
| KV append DMA | 97.4 (0.23%) | 83.8 (0.14%) | 79.9 (0.32%) |
| Block orchestration | 8.9 (0.02%) | 15.8 (0.03%) | 16.7 (0.07%) |
| Layer bookkeeping | 9.5 (0.02%) | 9.2 (0.02%) | 9.1 (0.04%) |
| Stage-boundary bookkeeping | 1.5 (0.00%) | 1.2 (0.00%) | 1.3 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 61.7 (0.15%) | 51.8 (0.09%) | 51.8 (0.20%) |
| Embedding | 1.7 (0.00%) | 1.6 (0.00%) | 1.7 (0.01%) |
| Final model RMSNorm | 3.1 (0.01%) | 4.0 (0.01%) | 3.8 (0.02%) |
| LM head＋greedy（不含 final norm） | 5287.8 (12.47%) | 3212.2 (5.37%) | 3192.5 (12.63%) |
| Host–DSP 边界 | 620.5 (1.46%) | 2297.2 (3.84%) | 2282.3 (9.03%) |
| 完整 Host wall | 42388.5 (100.00%) | 59848.3 (100.00%) | 25280.7 (100.00%) |

## E2E

| Mode | W4A16 token/s | A8 before token/s | A8 after token/s |
|---|---:|---:|---:|
| prefill | 1260.22 | 1081.56 | 1431.84 |
| decode | 23.59 | 16.71 | 39.56 |

Complete warm Host wall includes embedding,16 layers,final norm,LM head,greedy and FastRPC. Model loading/session preparation and external tokenizer are excluded. No layer extrapolation. All240 additive ledgers reconcile.

prefill, new A8 / w4a8_baseline Host ratio: 0.7554,95% paired bootstrap CI [0.7516,0.7607].
prefill, new A8 / w4a16_baseline Host ratio: 0.8801,95% paired bootstrap CI [0.8713,0.8873].
decode, new A8 / w4a8_baseline Host ratio: 0.4224,95% paired bootstrap CI [0.4211,0.4237].
decode, new A8 / w4a16_baseline Host ratio: 0.5964,95% paired bootstrap CI [0.5932,0.5990].

Latest L32-0006 three-round speed results: [LLAMA32_PIPELINE_ITERATIONS.md](LLAMA32_PIPELINE_ITERATIONS.md). Earlier results remain historical.
