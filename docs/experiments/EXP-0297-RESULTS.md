# EXP-0297 Qwen3-1.7B long-context decode optimization

Scope: W4A8-SP2, FP32 residual, no rotation. Four bounded engineering candidates; C4 selected by the frozen rule before formal profiling. Other model sizes/recipes, original weights, scales and integer arithmetic are frozen. No model-quality or baseline-promotion claim.

## Formal complete-model E2E

Five short cycles followed by ten rotated paired processes, each replayed ten times; all samples retained. Prefill is prompt tokens divided by complete prompt Host wall; decode is generated steps divided by total decode Host wall. Long-path wall includes input/RoPE staging. Native short RPC-only timing excludes that staging and remains the conservatively faster parity reference. Embedding, all28 layers, final norm, head, greedy and FastRPC are included; cold loading/external tokenization excluded. Fixed project-owned trajectories, not named-dataset quality/throughput evaluations.

| Path | Shape | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| native64 | 64+15 | 1850.732852 | 47.790575 |
| control64 | 64+15 | 1841.707824 | 44.749000 |
| m64 | 64+15 | 1840.239863 | 46.299669 |
| control536 | 536+46 | 1745.746324 | 39.068792 |
| opt536 | 536+46 | 1826.921277 | 43.148784 |
| control741 | 741+3 | 1737.177010 | 37.049159 |
| opt741 | 741+3 | 1860.806220 | 41.960941 |

## Why the D history appeared faster

EXP-0284 D history is M64+15,1846.788402 prefill /47.117789 decode token/s. The prior EXP0296 native64+3 remained47.031907 token/s. Longer536/741-context attention has more KV work; comparing those speeds directly with D history confounds context length and implementation. This experiment measures native64/control64/optimized64 at exactly64+15 under one binary. Historical sealed values remain untouched.

The current untouched native short path measures 47.790575 decode token/s. Optimized long64 measures 46.299669.

| Same-shape comparison | Stage | Wall ratio new/old | Paired95% CI | Throughput change |
|---|---|---:|---|---:|
| 536 control927 to65439 | prefill | 0.955567 | [0.955019,0.956206] | +4.650% |
| 536 control927 to65439 | decode | 0.905444 | [0.904667,0.906201] | +10.443% |
| 741 control927 to65439 | prefill | 0.933561 | [0.930247,0.936467] | +7.117% |
| 741 control927 to65439 | decode | 0.882944 | [0.878227,0.886699] | +13.257% |
| control64_to_m64 | prefill | 1.000798 | [0.998784,1.002795] | -0.080% |
| control64_to_m64 | decode | 0.966508 | [0.965542,0.967400] | +3.465% |
| native64_to_m64 | prefill | 1.005702 | [1.003511,1.007673] | -0.567% |
| native64_to_m64 | decode | 1.032201 | [1.030635,1.033938] | -3.120% |

## Unchanged gates

| Prompt | Fastest M64 reference | Prefill throughput ratio |95% CI | Parity |
|---|---|---:|---|---|
| 536 | native64 | 0.987134 | [0.985157,0.989076] | fail |
| 741 | native64 | 1.005443 | [1.002960,1.008607] | pass |

Numerical/physical correctness and the matched10% prefill/decode guards are recorded independently of the strict long/M64 prefill parity requirement. A statistically unresolved or failing parity result is not relaxed into a pass.

## Implementation and attribution

- C1: use the existing four-physical-row AV requantization helper at decode; remove clears fully overwritten by aligned valid cache DMA. Padding is still explicitly generated.
- C2: load/recenter four V rows once and register-shuffle directly into HMX tiles, reusing prepared LUT banks instead of repeated row loads/rotations.
- C3: express the unchanged K transpose as an unrolled register shuffle network, followed by direct bit-reversed output stores, reducing VTCM intermediate traffic.
- C4: separate DMA ownership from HMX ownership so independent client buffers can overlap transfers and matrix execution. Each resource still has one owner, and completion waits precede buffer reuse. Partial-prefill RMSNorm retains the already available parallel schedule. No persistent cache ABI change, extra KV copy or intermediate DDR was introduced.

Engineering C1..C4 results are diagnostic repeat3 observations, not formal ablations of isolated features. Formal gain applies to the combined candidate. Attention worker subcounters overlap and are service-work totals; only elapsed module wall supports additive attribution. See MODULES.md.

## Correctness and evidence

Selected candidate: 2968 bit-exact per-layer outputs across six lengths64/65/128/129/536/741 plus64+15, with zero signed-zero exceptions. 2225119232 valid K/V elements checked; future storage poison unchanged; head tokens/logit codes exact. Control64+15 independently checked.
Formal timed RPC boundaries checked: 18800. Each requested/acquired exactly8MiB VTCM, all plans within capacity, zero timed intermediate spill/read/write DDR, additive complete elapsed module ledger.
Six frozen packages rehashed against original manifests,1548 inherited oracle artifacts verified against the original EXP0295 ledger. Extended64+15 reference uses the pre-existing independent CPU/SDK arithmetic implementation, not outputs copied from the candidate; its initial64+3 prefix is identical to the old oracle. References, source, binaries and commands are bound by the final evidence ledger.
Measured/source closure: fa7cb9c8e2718c3a965fc75d04544ef79d3a0da3; explicit runtime option65439, old control927. Workspace build remains1.7B/28layers. Native short default path is unchanged.

## Follow-up

The remaining long-decode cost includes repeatedly preparing a longer K/V prefix; native64 also benefits from the older persistent short-cache layout. Investigate a separately versioned incremental native KV cache or tiled cache release only in a new experiment with independent layout verification and the same physical constraints. Do not attribute residual long/short differences wholly to regression. Preserve this validated candidate and historical selected baselines.

## Elapsed decode attention attribution

QKV/RoPE module includes Q/K normalization for Qwen. Parallel service-time counters are not added to these elapsed values.

| Shape | Control attention ms/step | Candidate attention ms/step | Complete Host ms saved/step |
|---|---:|---:|---:|
| 64+15 | 3.166239 | 2.419040 | 0.748442 |
| 536+46 | 6.446685 | 3.994202 | 2.420253 |
| 741+3 | 7.819815 | 4.729803 | 3.159479 |
