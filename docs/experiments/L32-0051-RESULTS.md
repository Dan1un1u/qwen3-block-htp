# L32-0051: long-prefill throughput parity passed

Frozen source e31a286fac5b43cfccd93706bcc21a00400920ce; archive binaries-a10; QBH_LONG_OPT=31. Llama-3.2-1B-Instruct W4A8-SP2, FP32 residual, no rotation. No weight, calibration or arithmetic changes.

Four bounded engineering rounds, then one frozen formal campaign: ten rotated paired cycles, repeat10, five arms, all samples retained. M64 prefill kernels are unchanged; no baseline slowdown. Long inputs are real token counts; padded physical rows are not counted as tokens.

| Configuration (prompt + decode RPC steps) | Prefill token/s | Decode token/s | Prefill Host wall ms | Decode total Host wall ms |
|---|---:|---:|---:|---:|
| M64+3 reference | 2305.96 | 50.29 | 27.754 | 59.652 |
| 536+46 original0050 | 638.32 | 23.01 | 839.706 | 1998.757 |
| 536+46 optimized | 2505.95 | 37.71 | 213.891 | 1219.727 |
| 741+3 original0050 | 518.94 | 19.38 | 1427.898 | 154.789 |
| 741+3 optimized | 2499.13 | 34.30 | 296.503 | 87.470 |

## Gate

Long/M64 complete-prefill throughput ratios; paired-bootstrap20000 samples, fixed seed510051, process-level paired means. Pass requires95%lower>=1 for both lengths.
- 536: 1.086727, 95% CI [1.081447, 1.091901], PASS.
- 741: 1.083769, 95% CI [1.078151, 1.089085], PASS.
Both long-decode original/optimized paired comparisons also pass the unchanged10% slowdown guard.

## Numerical and physical evidence

The exact frozen formal binary passes independent128+3,741+3 poisoned-padding,536+46 audits: all1200 layer-output hashes match bit-for-bit, all595607552 compared KV elements and all head IDs/codes match, future cache remains untouched. Earlier same-family64/65/129 boundary tests and carrier comparison are separately retained; the historical65 case has one signed-zero-only difference. They are not counted as final-binary proofs.
All14,400 timed RPC boundaries have exact additive module ledgers. VTCM peak8,098,272/8,388,608 bytes; maximum parallel overlay2,494,464/3,145,728 bytes, including shared LUT reservation. No timed intermediate DDR/spill. One HMX owner, one full-model RPC perchunk/token. Audit data transfers are outside speed runs.

## Attribution

1. Direct native four-query rows remove score/probability gather/scatter; reductions remain vector-valued across KV tiles.
2. Head64 K uses two-row vector packing/horizontal signed sums. V uses an exact register LUT and vector saturation counts.
3. Four independent GQA clients overlap HVX work on disjoint bounded buffers, serializing HMX/DMA ownership; cached QK RoPE reuses four HVX contexts.
4. Requantized scores and identical immutable V LUTs are reused with explicit lifetime separation.
Parallel attention subcounters are summed overlapping worker service durations; full-model module attribution uses elapsed attention wall. Do not add the subcounters as sequential latencies.

## Limits and scope

Only current1B W4A8-SP2 is tuned here. Existing bounded frontend remains prompt<=768, decode<=63/cache832; tested targets do not imply arbitrary-length throughput parity. No other model/recipe migration, PPL/text-quality claim or automatic paper-baseline promotion.
Complete warm Host wall includes each RPC and host input/RoPE staging. Cold loading/preparation, external tokenization, logging and untimed audits are excluded consistently. +D means D decode RPCs, with the first selected token from prefill. Fixed project-owned trajectory, not named datasets.
Long prompts amortize one final LM-head call, so meeting M64 token/s does not mean attention has no context-length cost.

See IMPLEMENTATION.md, MODULES.md, formal_protocol.json, profiling_summary.json, correctness_summary.json, failed_attempts.json and the archived source patch/history. Failed engineering candidates and compile logs are retained and excluded from formal results.
