# EXP-0076 — Complete profiling closure

## PC-028 three-variant module overview

EXP-0076 ended at its mandatory pre-build correctness gate, so it does not
produce a new eligible W4U8 result. The W4U8 column therefore reuses EXP-0075;
F16F16 and the Selected-Baseline W4F16 columns reuse their latest valid formal
evidence. Percentages are shares of complete Host wall and positive speed means
W4U8 is faster than W4F16.

| Module | W16A16 | W4A16 | W4A8 latest eligible | A8 vs A16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +81.7% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 19.0 us (0.9%) | +124.4% |
| QKV + Q/K Norm/RoPE | 397.1 us (16.0%) | 439.2 us (19.7%) | 424.6 us (20.9%) | +3.4% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 197.0 us (9.7%) | -28.6% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 170.5 us (8.4%) | +2.0% |
| Post-attn residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 33.6 us (1.7%) | +22.5% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 719.1 us (35.5%) | +35.3% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 359.9 us (17.7%) | -9.0% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 17.4 us (0.9%) | -71.2% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 82.7 us (4.1%) | -3.7% |
| Complete block Host wall | 2488.3 us | 2229.7 us | 2027.7 us | +10.0% |

## Failure boundary

The accepted AV mapping is
`M * round(accumulator / 2^shift) + output_zero_point`, including an
intermediate U8 saturation. A direct HMX affine conversion would instead
implement `round(M * accumulator / 2^shift + output_zero_point)`. For every
group with `M > 1`, these functions have different rounding order and cannot
represent the same global staircase. On the retained real layer-14 tensors,
the accepted software formulation reproduces the archived AV boundary with
zero mismatches and zero maximum LSB error. The one-stage candidate differs in
19,713 of 131,072 bytes (15.04%), with 5 LSB maximum and 0.187 LSB mean absolute
error. Only group 4, whose multiplier is one, is exactly representable.

Consequently EXP-0076 failed before a candidate runtime was built. Device
profiling, paired rounds and candidate physical counters are unavailable by
design rather than missing evidence.

## Repeat-one direct-control closure

| Primary metric | EXP-0075 control | EXP-0076 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| Host wall ns/block | 2,443,542.0 | unavailable | unavailable | unavailable |
| Attention ticks | 3,780.0 | unavailable | unavailable | unavailable |
| Invocation ticks | 38,852.0 | unavailable | unavailable | unavailable |
| Total ticks | 38,122.0 | unavailable | unavailable | unavailable |

| Additive Block Timing Ledger interval | EXP-0075 control | EXP-0076 candidate |
|---|---:|---:|
| Input stage | 50.0 | unavailable |
| Metadata stage | 118.0 | unavailable |
| Input RMSNorm | 373.0 | unavailable |
| QKV projection | 8,215.0 | unavailable |
| Q/K Norm-RoPE handoff | 0.0 | unavailable |
| Attention | 3,780.0 | unavailable |
| O projection | 3,271.0 | unavailable |
| Post-attention residual | 636.0 | unavailable |
| Post-attention norm handoff | 0.0 | unavailable |
| Gate/Up | 13,755.0 | unavailable |
| Activation handoff | 0.0 | unavailable |
| Down | 6,866.0 | unavailable |
| Final residual | 335.0 | unavailable |
| Output stage | 124.0 | unavailable |
| Stage boundary | 29.0 | unavailable |

## Repeat-ten direct-control closure

| Primary metric | EXP-0075 control | EXP-0076 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| Host wall ns/block | 2,027,734.4 | unavailable | unavailable | unavailable |
| Attention ticks | 3,781.6 | unavailable | unavailable | unavailable |
| Invocation ticks | 37,505.9 | unavailable | unavailable | unavailable |
| Total ticks | 37,431.8 | unavailable | unavailable | unavailable |

| Additive Block Timing Ledger interval | EXP-0075 control | EXP-0076 candidate |
|---|---:|---:|
| Input stage | 54.0 | unavailable |
| Metadata stage | 11.8 | unavailable |
| Input RMSNorm | 365.3 | unavailable |
| QKV projection | 8,152.5 | unavailable |
| Q/K Norm-RoPE handoff | 0.1 | unavailable |
| Attention | 3,781.6 | unavailable |
| O projection | 3,273.1 | unavailable |
| Post-attention residual | 645.5 | unavailable |
| Post-attention norm handoff | 0.1 | unavailable |
| Gate/Up | 13,806.4 | unavailable |
| Activation handoff | 0.0 | unavailable |
| Down | 6,909.2 | unavailable |
| Final residual | 333.4 | unavailable |
| Output stage | 12.4 | unavailable |
| Stage boundary | 25.8 | unavailable |

## Relevant overlapping work and physical counters

| Metric | EXP-0075 repeat 1 | EXP-0075 repeat 10 | EXP-0076 candidate |
|---|---:|---:|---:|
| Weight DMA ticks | 10,746.0 | 10,623.8 | unavailable |
| HMX compute ticks | 13,146.0 | 13,137.1 | unavailable |
| HMX ready-wait ticks | 8,067.0 | 8,111.2 | unavailable |
| QK HMX ticks | 437.0 | 429.0 | unavailable |
| Softmax HVX ticks | 14,204.0 | 14,123.7 | unavailable |
| AV HMX ticks | 433.0 | 432.1 | unavailable |
| AV post-HMX requant ticks | 1,199.0 | 1,199.2 | unavailable |
| Attention pipeline-wait ticks | 1,074.0 | 1,075.4 | unavailable |
| HMX commands/block | 176 | 176 | unavailable |
| U8xS8 tile pairs/block | 49,408 | 49,408 | unavailable |
| Weight DMA descriptors/block | 512 | 512 | unavailable |
| Weight DDR read bytes/block | 25,444,352 | 25,444,352 | unavailable |
| Intermediate DDR read/write bytes | 0 / 0 | 0 / 0 | unavailable |
| Spill/fill count | 0 | 0 | unavailable |
| Requested/acquired VTCM bytes | 8,388,608 / 8,388,608 | 8,388,608 / 8,388,608 | unavailable |
| Peak live VTCM bytes | 5,306,080 | 5,306,080 | unavailable |
| FastRPC execution units / HMX owners | 1 / 1 | 1 / 1 | unavailable |
| QNN dependency | none | none | unavailable |
| Ledger unattributed ticks | 0.0 | 0.0 | unavailable |

## Correctness and decision

| Gate | Result |
|---|---:|
| Accepted two-stage formulation vs archived AV | 0 mismatches, 0 LSB |
| One-stage HMX affine candidate vs accepted AV | 19,713 mismatches, 5 LSB |
| Mandatory byte-exact gate | FAIL |
| Build and device execution | not run after mandatory gate failure |
| Local speed gate | unavailable |
| Local adoption eligibility | NO |
| Selected Baseline changed | NO |

Evidence is retained at
`D:/llm_exp/results/qwen3-block-htp/exp0076/20260830T113701Z_cb74da2_formal`.
The control profiling provenance is EXP-0075 formal evidence at
`D:/llm_exp/results/qwen3-block-htp/exp0075/20260830T112231Z_8e3fe8561e5b_formal`.
