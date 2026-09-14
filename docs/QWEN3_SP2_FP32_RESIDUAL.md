# EXP0269: Qwen no-rotation SP2 with FP32 residual

Final tested source `aa55a3ece5a04f40c1cfe135b699433b8f6e4ff2`. The independent single-layer implementation gate passes; the fixed five-pair short speed gate fails. No formal-ten, chain3, full28, E2E or PPL was run. Original paper baselines remain selected.

| Single-layer Host wall | Original SP2 | SP2 + FP32 residual | Ratio | 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| prefill | 1348.53 us | 1554.16 us | 1.152490 | [1.100393, 1.209263] | fail |
| decode | 1053.72 us | 1102.92 us | 1.046689 | [0.957625, 1.148528] | fail |

Five fixed AB/BA rounds, repeat10 primary and repeat1 auxiliary, seed269/20000 paired bootstrap samples. M64 then decode1, past64/cache128. Host RPC/setup overhead is included; no single-layer throughput is presented as E2E. The earlier short set belongs to the rejected prefill-transpose binary and is not merged with these final results.

## Implementation and validation

FP16 embedding expands to FP32 residual; input RMSNorm computes FP32 then quantizes to A8. Native packed-W4 QKV, Qwen Q/K Norm-RoPE, wide-NR64 integer attention and U8 KV are retained. O raw integers are scaled and added once to FP32 residual; post RMSNorm quantizes to A8; original W4 Gate/Up plus fused SP2 SwiGLU feed raw native-W4 Down, which scales/adds once to FP32 residual. Final FP32 RMSNorm feeds the original A8 LM head. Frontend/head code is ported but not yet independently/device validated on full Qwen; no fullmodel completion claim.

Exact selected-layer0/14/27 prefill64/decode1 outputs, Q/K, AV, postnorm, Gate/Up, SP2 low/high live codes and physical prefill KV caches pass against the independent Qwen oracle. All three repeat10 outputs are deterministic. FP32 output checks cover399360 unique values. Every O/Down output channel has a bound proof for raw signed24 stores and signed32 SP2 merge. Qwen epsilon1e-6, dimensions, weights, fixed scales and SP2 LUT are unchanged.

8MiB requested/acquired, selected-layer VTCM peak6682752B. Tensor intermediate DDR/spill counters and additive ledger unattributed ticks are zero. New O/Down epilogue has no stack vmem in final assembly. Original safe VTCM norm transpose is retained; remaining norm stack vectors are constants, not activation tiles.

## Repairs and bounded candidates

1. Initial raw scratch reused normalized at HVX128-byte alignment. HMX retained stores require2048-byte alignment; this damaged prefill rows48..63. Aligning the reused arena fixes exact outputs without increasing peak (moves padding preceding the already-aligned Q arena). Failed a01/a02 are preserved.
2. Scalar decode RMS reduction is exact but slower and was reverted.
3. Explicit named O/Down vector operands remove the indexed-array1024B activation stack materialization.
4. Candidate3 attempted register prefill transpose; final disassembly found256B activation spills. It was rejected and original VTCM transpose restored. Decode keeps ordered live-lane reduction and HVX padding stores, reducing this isolated diagnostic Norm from about56us to15us; these nonpaired diagnostics are not formal speed claims.

The final five-pair short gate runs the repaired binary. Prefill remains the limiting decision; decode mean cost is smaller but its upper CI also exceeds10%. No optional formal resampling and no relaxed numerical/physical gate.

## Stable module table (short gate only)

Microseconds, parentheses share of complete single-layer Host wall. Embedding/head are outside this replay scope.

### prefill

| Module | Original SP2 | FP32 residual |
|---|---:|---:|
| I/O, metadata | 18.7 (1.39%) | 45.9 (2.95%) |
| Input RMSNorm | 20.3 (1.51%) | 77.6 (4.99%) |
| QKV + Q/K Norm-RoPE | 251.7 (18.66%) | 250.8 (16.14%) |
| QK-Softmax-AV | 120.7 (8.95%) | 120.4 (7.75%) |
| O projection | 44.8 (3.32%) | 75.1 (4.83%) |
| Post-attention residual + RMSNorm | 23.7 (1.76%) | 81.7 (5.25%) |
| Gate/Up + SwiGLU | 249.1 (18.47%) | 248.4 (15.98%) |
| Down | 157.3 (11.66%) | 165.1 (10.62%) |
| Final residual | 6.7 (0.49%) | 0.1 (0.01%) |
| KV carrier conversion | 4.8 (0.35%) | 4.8 (0.31%) |
| KV append DMA | 8.9 (0.66%) | 9.0 (0.58%) |
| Block orchestration | 1.9 (0.14%) | 1.9 (0.12%) |
| Layer bookkeeping | 0.8 (0.06%) | 0.9 (0.05%) |
| Stage-boundary bookkeeping | 1.7 (0.13%) | 1.7 (0.11%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 74.6 (5.53%) | 74.7 (4.81%) |
| Embedding | N/A: outside layer scope | N/A: outside layer scope |
| Final model RMSNorm | N/A: outside layer scope | N/A: outside layer scope |
| LM head + greedy | N/A: outside layer scope | N/A: outside layer scope |
| Host-DSP boundary | 362.8 (26.90%) | 396.2 (25.49%) |
| Complete Host wall | 1348.5 (100%) | 1554.2 (100%) |

### decode

| Module | Original SP2 | FP32 residual |
|---|---:|---:|
| I/O, metadata | 21.3 (2.02%) | 57.5 (5.21%) |
| Input RMSNorm | 5.4 (0.51%) | 14.6 (1.32%) |
| QKV + Q/K Norm-RoPE | 86.9 (8.24%) | 87.2 (7.91%) |
| QK-Softmax-AV | 66.1 (6.27%) | 66.3 (6.01%) |
| O projection | 43.5 (4.12%) | 53.6 (4.86%) |
| Post-attention residual + RMSNorm | 7.2 (0.68%) | 15.0 (1.36%) |
| Gate/Up + SwiGLU | 226.8 (21.52%) | 220.6 (20.00%) |
| Down | 124.1 (11.78%) | 124.9 (11.32%) |
| Final residual | 1.6 (0.15%) | 0.1 (0.01%) |
| KV carrier conversion | 10.6 (1.01%) | 10.7 (0.97%) |
| KV append DMA | 3.3 (0.31%) | 3.3 (0.30%) |
| Block orchestration | 1.5 (0.14%) | 1.4 (0.12%) |
| Layer bookkeeping | 0.6 (0.06%) | 0.6 (0.06%) |
| Stage-boundary bookkeeping | 0.5 (0.05%) | 0.5 (0.04%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 71.1 (6.75%) | 70.9 (6.43%) |
| Embedding | N/A: outside layer scope | N/A: outside layer scope |
| Final model RMSNorm | N/A: outside layer scope | N/A: outside layer scope |
| LM head + greedy | N/A: outside layer scope | N/A: outside layer scope |
| Host-DSP boundary | 383.3 (36.38%) | 375.8 (34.08%) |
| Complete Host wall | 1053.7 (100%) | 1102.9 (100%) |

Full-model E2E for this new Qwen FP32 candidate: **N/A, not run**. Historical selected baselines: Qwen SP2 integer residual2030.24/48.17tok/s; Llama SP2 FP32 residual2069.70/42.51tok/s. These are separate historical full-model runs.

## Baseline completeness

See PAPER_NO_ROTATION_BASELINES.md/json. All181 L32-0018 and1511 EXP0268 ledger entries rehashed successfully. Llama0018-to-current five common hot files are identical; later changes are rotation-only, reverted trials or documentation. Rotation-branch common commits are patch-equivalent to current, excluding its branch-default config. No omitted later validated no-rotation optimization found. New Qwen edits are not yet measured/accepted on Llama.

## Next decision

Retain both selected paper rows. Further Qwen work should target strict FP32 prefill RMSNorm and O raw-store/epilogue wall time. The earlier Llama uplift cannot be assumed transferable: its paired SP2 control spent more time in Norm and Gate/Up; the Qwen SP2 parent already has a shorter path there. New fullmodel speed must be measured after eligibility; do not extrapolate from this layer.
