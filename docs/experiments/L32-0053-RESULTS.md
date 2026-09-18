# L32-0053: Llama-3.2-3B-Instruct long-context throughput loop

Numerical and physical checks pass. Long/M64 prefill parity: FAIL. No automatic baseline promotion or model-quality acceptance.

Frozen W4A8-SP2 mode8, no rotation, FP32 residual and model-specific quantization. Independent arithmetic oracle reused unchanged. Tests use project-owned fixed token trajectories, not named-dataset evaluation.

Measured source: `04673543a6c847809bcdc16621d9db2d5aa5723d`. Candidate `QBH_LONG_OPT=8095`; same-binary previous long path `31`. Five short cycles followed by ten predeclared rotated formal cycles, ten replays per process. All formal samples retained. Warm complete Host wall includes Host input/RoPE staging on the long frontend and FastRPC/model execution; cold loading, external tokenization and untimed audit writes excluded. The original short frontend measures RPC only and is retained as a conservative faster reference.

## Full-model timing

| Configuration | Prefill tokens | Decode steps | Prefill Host ms | Decode total Host ms | Prefill token/s | Decode token/s |
|---|---:|---:|---:|---:|---:|---:|
| native64 | 64 | 3 | 54.145303 | 124.827343 | 1182.004649 | 24.033196 |
| control64 | 64 | 3 | 54.343393 | 124.175377 | 1177.696058 | 24.159379 |
| m64 | 64 | 3 | 54.246940 | 115.979958 | 1179.790046 | 25.866538 |
| control536 | 536 | 46 | 480.953770 | 2781.677930 | 1114.452227 | 16.536781 |
| opt536 | 536 | 46 | 465.513357 | 1933.877207 | 1151.417014 | 23.786412 |
| control741 | 741 | 3 | 667.840019 | 202.953505 | 1109.547165 | 14.781711 |
| opt741 | 741 | 3 | 634.166050 | 129.527987 | 1168.463686 | 23.161018 |

## Fixed gates

| Length | Fastest measured M64 reference | Long/M64 throughput | Paired bootstrap 95% CI | Parity | Decode wall ratio to matched old long path | Decode 95% CI | Decode 10% guard |
|---|---|---:|---|---|---:|---|---|
| 536 | native64 | 0.974122 | [0.9713928893827929, 0.9768725035993149] | False | 0.695220 | [0.6946927179671638, 0.6957916783476007] | True |
| 741 | native64 | 0.988544 | [0.9858774346431152, 0.991121995764326] | False | 0.638215 | [0.6372096064408452, 0.6391135583439437] | True |

Gate is lower95(long/M64 throughput)>=1; decode upper95(candidate/control wall)<=1.10. Twenty thousand paired process-level bootstrap resamples, seed510051. No relaxed thresholds or deliberately slower M64.

## Attribution

| Length | Prefill throughput gain | Decode throughput gain | Attention wall saved per prefill (ms) | Attention wall saved per decode step (ms) |
|---|---:|---:|---:|---:|
| 536 | 3.317% | 43.839% | 11.058271 | 18.567875 |
| 741 | 5.310% | 56.687% | 19.548949 | 24.471394 |

Parallel GQA worker service counters overlap and are not additive wall-time components. MODULES.md uses the enclosing attention elapsed wall; complete module sums and Host-DSP boundaries are verified for every timed invocation.

## Correctness and physical evidence

Final candidate checks cover64,65,128,129,536+46,741+3; all layer outputs, cumulative K/V contents and head token/code are compared with frozen independent references. Future cache padding is poisoned and remains unchanged. Failed engineering evidence is retained outside the eligible set.
Timed verified invocation boundaries: 15200. Exact8MiB acquisition, zero timed intermediate DDR traffic/spill, attention overlay within its allocation, one serialized HMX/DMA owner. Maximum planned VTCM: 8360416 bytes. Maximum attention overlay: 1904640 bytes.

Candidate1 (927) ports exact head128 K transpose, parallel decode admission and causal tile pruning, retaining Llama wide8/exact normalization. Candidate2 (1951) repairs matching V LUT banks with explicit bit6 selection:65536 simulator checks and independent device audit. Candidate3 (3999) retains parallel full64 Norm for partial multirow prefill while only live rows are consumed/appended. Candidate4 (8095) omits redundant K/V plane clearing before a complete aligned DMA overwrite. Four engineering candidates were measured before formal; selection is frozen in engineering-selection.json. Llama1B default opt31 dispatch and all other recipes remain unchanged, not newly benchmarked.

## Remaining direction

Prefill still includes preparation and matrix work over increasing KV lengths. The remaining parity gap should be addressed by reducing repeated prefix K/V preparation and aligning persistent KV producers with matrix consumers, then measuring complete Host wall. Do not infer parity from decode gains or summed worker timings.

Evidence: formal-protocol.json, runtime.json and archived binaries; final validation JSONs; per-run protocol/stdout/stderr/records; profiling_summary.json; MODULES.md; frozen-model-verification.json. All weights and references retain original hashes.
