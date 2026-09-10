# L32-0006: three native A8 pipeline iterations

User requested several further rounds targeting frozen Qwen3 speed. This Llama
W4A8 OFF experiment preserves weights, calibration, activation arithmetic and
quality status; no Qwen research, new rotations, group quantization or mixed precision.

Round1: adapt existing paired SOLE templates and four-row vector transpose to
Llama GQA4. Scores are already requantized, so the reused paired helper uses
identity conversion. The per-group K operand is dead after QK completion and
provides1024B carrier scratch before AV; V stays live and disjoint. No new VTCM
buffer or spill. Prefill SwiGLU uses two idle persistent HVX workers plus main,
partitioned by native output tile, each with its existing private256B gather
scratch. Exact LUT unchanged; join completes before Down starts. The first paired
row-copy candidate remains auxiliary; final uses the four-row carrier path.
The generic const-score softmax API retains its original implementation.

Round2: head64 native KV tiles are transposed four rows at a time using aligned
HVX vector shuffles. Arbitrary unaligned inputs and final partial rows keep the
bounded original copy path. No read beyond valid rows. Prefill cache conversion
falls2.724 ->0.027ms across16 layers, preserving byte-exact persistent KV.

Round3: use nativeW4 GateUp batch32,continuous Gate->Up DMA, O->Gate prefetch and
streamed SwiGLU. Replace fixed six-group readiness arrays/check with model-derived
eight groups for Llama FFN8192 (Qwen dimensions still derive six). Increase O to16
and Down to8 with existing single-DMA paths, QKV launch batch16. Reuse existing
phase-dead bias/weight buffers within8MiB. Per decode token128 SwiGLU groups are
published/consumed with overlap,16 Gate prefetches consumed. No second HMX owner.

Independent integer single-layer gates after each round pass. Final3/16 full
output and KV match the sealed reference exactly. Padding/softmax comparison
passes; W4A16 full16 output is byte-identical to L32-0002 and OPT2 audit passes.
All8/16-step generation IDs AND selected logit codes equal baseline. A8 text
remains repeated Sleep; this speed task has no model-quality threshold and no
PPL rerun/promotion. W4A16 still fails the prior PPL gate. All560 formal additive
ledgers reconcile; exactly8MiB VTCM,peak7668960B A8/8330752B W4,zero timed
intermediate DDR/spill and nativeW4 weights without expansion retained.

Final candidate frozen at cfd9feeb1d9403185aeffeabfa47bbd174c7b70d. Builds
archived in l32-0006/baseline-build16 (L32-0005) and candidate-build16. Fixed10
three-arm ABC/BCA/CAB M64+7 cycles plus10 independent AB/BA M64+15 A8 pairs,
no optional stopping or repeat1 gate. The two shape cohorts were measured
separately; their small timing differences do not imply longer decode is faster.

M64+7 A8 prefill1437.089 ->2015.916tok/s (+40.28%), decode39.5228 ->45.6466
(+15.49%). Candidate/old Host ratio .712871 [.707863,.719768],
.865843 [.861942,.869598]. Matched W4A16 is1261.161/23.604tok/s.
M64+15 A8 prefill1435.857 ->2046.813,decode40.2185 ->46.6757;
Host ratios .701509 [.690540,.709048], .861657 [.855672,.867739].
Both cohorts pass the retained1.10 upper-bound gate in both stages.

Frozen Qwen3 OFF historical M64+15:1705.318 prefill,48.3572 decode. Llama now
has20.03% higher prefill throughput and3.48% lower decode throughput. This is a
historical different-model comparison, not a paired hardware efficiency claim.
Original model size, head dimensions and vocabulary differ; no Qwen weights reused.

Remaining measured M64+7 A8 costs: prefill attention25.5%,GateUp/SwiGLU25.5%,
QKV/RoPE12.2%,head10.2%; decode attention26.4%,GateUp/SwiGLU22.8%,head14.8%,
Down11.6%,Host boundary10.5%. More room remains especially in decode attention
operand preparation and head/Host overhead, but this bounded three-round task
is complete. Serving scope remains capacity80 and testedM64+7/15.

Reproduce only under new active experiment and new immutable destinations:
run_llama32_stack.py with matching1/3/16 build; --a8-audit for padding/reference;
run_llama32_pipeline_iterations_profile.py deploy/gate/long/formal/formal-long;
report_llama32_pipeline_iterations_profile.py. Frozen reports/tools from L32-0005
remain unchanged. Protocols retain all commands,package and binary hashes.

# L32-0006: native A8 relative speed

SM8750 / HTP V79. Fixed ten ABC/BCA/CAB cycles; identical original M64 prompt and seven continuous decode tokens for all three arms. Quantized recipe outputs differ; input dimensions and KV lengths match.

## prefill

Microseconds, parentheses are complete Host wall shares. Decode is per token.

| Module | W4A16 OPT2 | A8 before | A8 after |
|---|---:|---:|---:|
| I/O、metadata | 224.2 (0.44%) | 137.0 (0.31%) | 137.8 (0.43%) |
| Input RMSNorm | 281.8 (0.56%) | 1321.7 (2.97%) | 1320.9 (4.16%) |
| QKV＋RoPE | 5737.3 (11.31%) | 3853.5 (8.65%) | 3871.0 (12.19%) |
| QK–Softmax–AV | 11545.3 (22.75%) | 11605.2 (26.06%) | 8101.4 (25.52%) |
| O projection | 2870.0 (5.66%) | 945.5 (2.12%) | 732.7 (2.31%) |
| Post-attention residual＋RMSNorm | 270.6 (0.53%) | 695.5 (1.56%) | 692.8 (2.18%) |
| Gate/Up＋SwiGLU | 17334.3 (34.16%) | 14268.9 (32.04%) | 8087.6 (25.48%) |
| Down | 5873.0 (11.57%) | 2949.4 (6.62%) | 2570.5 (8.10%) |
| Final residual | 80.6 (0.16%) | 348.4 (0.78%) | 346.2 (1.09%) |
| KV carrier conversion | 89.5 (0.18%) | 2724.3 (6.12%) | 27.4 (0.09%) |
| KV append DMA | 169.3 (0.33%) | 128.3 (0.29%) | 99.4 (0.31%) |
| Block orchestration | 14.2 (0.03%) | 23.3 (0.05%) | 21.7 (0.07%) |
| Layer bookkeeping | 13.5 (0.03%) | 12.6 (0.03%) | 12.8 (0.04%) |
| Stage-boundary bookkeeping | 7.4 (0.01%) | 7.9 (0.02%) | 7.1 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 92.5 (0.18%) | 83.9 (0.19%) | 81.6 (0.26%) |
| Embedding | 52.1 (0.10%) | 43.5 (0.10%) | 43.5 (0.14%) |
| Final model RMSNorm | 48.3 (0.10%) | 4.1 (0.01%) | 3.5 (0.01%) |
| LM head＋greedy（不含 final norm） | 5343.9 (10.53%) | 3250.1 (7.30%) | 3247.1 (10.23%) |
| Host–DSP 边界 | 698.9 (1.38%) | 2131.2 (4.79%) | 2342.5 (7.38%) |
| 完整 Host wall | 50746.9 (100.00%) | 44534.5 (100.00%) | 31747.3 (100.00%) |

## decode

Microseconds, parentheses are complete Host wall shares. Decode is per token.

| Module | W4A16 OPT2 | A8 before | A8 after |
|---|---:|---:|---:|
| I/O、metadata | 179.3 (0.42%) | 116.0 (0.46%) | 129.9 (0.59%) |
| Input RMSNorm | 277.1 (0.65%) | 84.9 (0.34%) | 85.0 (0.39%) |
| QKV＋RoPE | 5708.6 (13.47%) | 1652.6 (6.53%) | 1670.4 (7.62%) |
| QK–Softmax–AV | 3650.2 (8.62%) | 5781.9 (22.85%) | 5773.2 (26.35%) |
| O projection | 2821.5 (6.66%) | 943.2 (3.73%) | 721.0 (3.29%) |
| Post-attention residual＋RMSNorm | 267.8 (0.63%) | 164.9 (0.65%) | 167.9 (0.77%) |
| Gate/Up＋SwiGLU | 17265.5 (40.75%) | 7744.0 (30.61%) | 4986.9 (22.76%) |
| Down | 5852.1 (13.81%) | 2937.5 (11.61%) | 2534.2 (11.57%) |
| Final residual | 79.5 (0.19%) | 80.4 (0.32%) | 81.7 (0.37%) |
| KV carrier conversion | 136.6 (0.32%) | 49.0 (0.19%) | 49.9 (0.23%) |
| KV append DMA | 99.2 (0.23%) | 82.9 (0.33%) | 80.0 (0.37%) |
| Block orchestration | 9.4 (0.02%) | 16.7 (0.07%) | 16.5 (0.08%) |
| Layer bookkeeping | 9.9 (0.02%) | 9.2 (0.04%) | 9.0 (0.04%) |
| Stage-boundary bookkeeping | 1.3 (0.00%) | 1.3 (0.01%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 61.3 (0.14%) | 51.4 (0.20%) | 51.3 (0.23%) |
| Embedding | 1.7 (0.00%) | 1.5 (0.01%) | 1.6 (0.01%) |
| Final model RMSNorm | 2.3 (0.01%) | 3.1 (0.01%) | 3.7 (0.02%) |
| LM head＋greedy（不含 final norm） | 5414.3 (12.78%) | 3241.6 (12.81%) | 3242.6 (14.80%) |
| Host–DSP 边界 | 528.3 (1.25%) | 2339.7 (9.25%) | 2301.4 (10.51%) |
| 完整 Host wall | 42365.7 (100.00%) | 25301.8 (100.00%) | 21907.4 (100.00%) |

## E2E

| Mode | W4A16 token/s | A8 before token/s | A8 after token/s |
|---|---:|---:|---:|
| prefill | 1261.16 | 1437.09 | 2015.92 |
| decode | 23.60 | 39.52 | 45.65 |

Complete warm Host wall includes embedding,16 layers,final norm,LM head,greedy and FastRPC. Model loading/session preparation and external tokenizer are excluded. No layer extrapolation. All560 additive ledgers reconcile.

prefill, new A8 / w4a8_baseline Host ratio: 0.7129,95% paired bootstrap CI [0.7079,0.7198].
prefill, new A8 / w4a16_baseline Host ratio: 0.6256,95% paired bootstrap CI [0.6205,0.6295].
decode, new A8 / w4a8_baseline Host ratio: 0.8658,95% paired bootstrap CI [0.8619,0.8696].
decode, new A8 / w4a16_baseline Host ratio: 0.5171,95% paired bootstrap CI [0.5139,0.5207].

## M64 +15 decode, ten independent AB/BA pairs

| Mode | A8 before token/s | A8 after token/s |
|---|---:|---:|
| prefill | 1435.86 | 2046.81 |
| decode | 40.22 | 46.68 |
