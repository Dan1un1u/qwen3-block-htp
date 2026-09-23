# L32-0071: HVX butterfly R3 plus R4 full-model E2E

Llama-3.2-1B-Instruct, ordinary W4A8, FP32 residual, no SP2/INT16 Down. R4 is the unchanged L32-0070 normalized H8192 HVX path. Candidate adds normalized H64 on every Q/K head after RoPE and before A8 quantization. Both rotations are HVX butterfly; zero rotation HMX calls. Source f291128c946e4e1124f0d8af2288fbb05b27296d.

## Paired full-model timing

Same sealed binary and package. Fixed64+42; five short and ten formal alternating AB/BA pairs, repeat10. Complete Host wall includes embedding/all16blocks/finalnorm/LMhead/greedy/FastRPC; excludes cold loading/external tokenizer/audit dumps. Decode wall is42 steps in total. No performance-based early stopping.

| Phase | R4 wall ms | R3+R4 wall ms | R4 token/s | R3+R4 token/s | Added R3 wall increase | Ratio 95% CI |
|---|---:|---:|---:|---:|---:|---|
| prefill | 31.681663 | 30.482599 | 2020.10 | 2099.56 | -3.785% | [0.9597687665328369, 0.9657757179495202] |
| decode | 919.498826 | 915.824423 | 45.68 | 45.86 | -0.400% | [0.9945200297661206, 0.9979181016614272] |

## Scope and correctness

No additional R4 pipeline optimization. This measures the complete reused R3 boundary, not pure butterfly insertion into an otherwise instruction-identical RoPE kernel. The no-R3 implementation preserves an SF32/division contract with rounding-edge detection and scalar repair; the inherited R3 boundary uses pre/post FP16 values, reciprocal-based vector quantization and grouped native stores. Both match their own independent references, but these boundary arithmetic/implementation differences can offset rotation cost. Do not attribute a negative E2E difference to butterfly arithmetic itself. R3 reuses the existing per-head/row QK schedule and native U8 stores. Prefill uses the existing persistent head workers, decode existing valid-row processing. Six FP32 add/subtract butterfly stages and normalization1/8, retaining pre/post-R3 FP16 boundaries; ordinary A8 quantization remains. Head-private scratch resides in phase-dead Gate VTCM. No dense Hadamard, new thread launches, weight changes or recalibration.

Q and K receive the same orthogonal matrix, preserving their dot product in ideal arithmetic; frozen A8/FP16 rounding can change actual attention. R3 therefore requires no weight folding. The independently sealed0070 R4 package is reused and hashes rechecked locally/on device. This is not a model-quality or equivalence-to-unrotated-model claim.

Independent stagewise reference passes1/3/16-layer43-step hashes and heads; full16 padding-poison repeat also passes. Full16 KV/finalnorm/hidden payload checks are archived separately. Every timed head matches its own reference;8600 formal token invocations pass physical and additive-ledger checks. PeakVTCM [8098272]B within8MiB,zero intermediateDDR/spills; HMX command counts same between arms. No quality/default promotion.

Initial attempt at ff980ca incorrectly selected legacy dense-R3 ring scheduling and was rejected with QKV status-8 before projection execution. Preserved audit1-R3R4 evidence. Selector separated into mode2/opt3 at final source; no numerical gate relaxed.

R3 is fused with RoPE and quantization; the QKV+RoPE module reports the whole path, not an isolated butterfly-core wall. Historical0070 no-rotation2398.61/46.38token/s is nonpaired context, not a third arm in this run. No workbook update requested.

## prefill module profile

Microseconds (share of complete Host wall); decode per step.

| Module | R4 | R3 + R4 |
|---|---:|---:|
| I/O、metadata | 126.137 (0.40%) | 126.990 (0.42%) |
| Input RMSNorm | 1340.500 (4.23%) | 1301.119 (4.27%) |
| QKV + RoPE + optional R3 | 3846.295 (12.14%) | 2639.791 (8.66%) |
| QK–Softmax–AV | 6158.139 (19.44%) | 6152.002 (20.18%) |
| O projection | 1211.558 (3.82%) | 1210.429 (3.97%) |
| Post-attention residual＋RMSNorm | 1306.532 (4.12%) | 1336.006 (4.38%) |
| Gate/Up + SwiGLU + R4 | 10523.578 (33.22%) | 10527.571 (34.54%) |
| Down | 2611.432 (8.24%) | 2611.397 (8.57%) |
| Final residual | 2.662 (0.01%) | 2.751 (0.01%) |
| KV carrier conversion | 29.742 (0.09%) | 27.391 (0.09%) |
| KV append DMA | 116.624 (0.37%) | 116.944 (0.38%) |
| Block orchestration | 25.138 (0.08%) | 25.861 (0.08%) |
| Layer bookkeeping | 13.656 (0.04%) | 13.800 (0.05%) |
| Stage-boundary bookkeeping | 6.432 (0.02%) | 6.915 (0.02%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 93.782 (0.30%) | 97.897 (0.32%) |
| Embedding | 64.630 (0.20%) | 65.193 (0.21%) |
| Final model RMSNorm | 55.565 (0.18%) | 55.685 (0.18%) |
| LM head＋greedy（不含 final norm） | 3290.559 (10.39%) | 3279.934 (10.76%) |
| Host-DSP boundary | 858.703 (2.71%) | 884.926 (2.90%) |
| Complete Host wall | 31681.663 (100.00%) | 30482.599 (100.00%) |

## decode module profile

Microseconds (share of complete Host wall); decode per step.

| Module | R4 | R3 + R4 |
|---|---:|---:|
| I/O、metadata | 124.525 (0.57%) | 124.996 (0.57%) |
| Input RMSNorm | 873.932 (3.99%) | 874.070 (4.01%) |
| QKV + RoPE + optional R3 | 1662.014 (7.59%) | 1592.284 (7.30%) |
| QK–Softmax–AV | 5464.527 (24.96%) | 5467.264 (25.07%) |
| O projection | 883.037 (4.03%) | 884.236 (4.06%) |
| Post-attention residual＋RMSNorm | 880.541 (4.02%) | 880.453 (4.04%) |
| Gate/Up + SwiGLU + R4 | 5132.467 (23.44%) | 5134.832 (23.55%) |
| Down | 2581.126 (11.79%) | 2581.093 (11.84%) |
| Final residual | 1.083 (0.00%) | 1.085 (0.00%) |
| KV carrier conversion | 49.947 (0.23%) | 50.585 (0.23%) |
| KV append DMA | 92.200 (0.42%) | 92.298 (0.42%) |
| Block orchestration | 17.690 (0.08%) | 17.597 (0.08%) |
| Layer bookkeeping | 9.680 (0.04%) | 9.664 (0.04%) |
| Stage-boundary bookkeeping | 1.257 (0.01%) | 1.256 (0.01%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.164 (0.23%) | 51.159 (0.23%) |
| Embedding | 1.565 (0.01%) | 1.576 (0.01%) |
| Final model RMSNorm | 56.061 (0.26%) | 55.801 (0.26%) |
| LM head＋greedy（不含 final norm） | 3290.870 (15.03%) | 3279.455 (15.04%) |
| Host-DSP boundary | 719.140 (3.28%) | 705.639 (3.24%) |
| Complete Host wall | 21892.829 (100.00%) | 21805.343 (100.00%) |

Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0071. SUMMARY.json contains module profiles; BOUNDARY_CHECKS.json and POSTFLIGHT.json record gates. EVIDENCE_LEDGER.json seals result files. Frozen model package remains at models/llama32-htp/l32-0070/R4.
