# EXP0247 — Dense HMX R3 single-layer result

Implemented the requested naive dense online matrix multiply; no butterfly/FWHT. All seven linear projections retain native HMX W4 reads. Weights are exactly C64 layer0; other recipes and baselines remain frozen.

**Outcome: component arithmetic/layout checks pass; the frozen whole-layer numerical gate fails. Five short rounds and ten formal timing rounds were not started.**

## Software quality, from the independently completed EXP0246

| Variant | PPL |
|---|---:|
| F | 26.8074 |
| C64 | 27.9047 |
| A8 | 37.5051 |
| recal_L0 | 37.5051 |
| recal_ALL | 37.7765 |
| R3_L0 | 33.2076 |
| R3_ALL | 32.3915 |

Fresh256 documents/4096targets, four balanced English/Chinese wiki/news cells. ALL R3 lowers PPL13.63% versus original A8 and14.25% versus matched recalibration; paired95% ratios[0.8262,0.9042] and[0.8206,0.8969]. It remains16.08% above matched C64 W4A16 and20.83% above F16. Software results are not device PPL.

## Hardware verification

On PJZ110 HTP, batch all24 Q/K heads into a dense matrix:1536 live rows in M64 prefill and24 live rows (32 padded) in M1 decode. Explicit128x128 sign weights and FP16 normalization; one HMX FP16 GEMM. RoPE-to-FP16 preparation, packing, normalization metadata, dense multiplication, repacking and a QKV completion barrier are in the runtime. Gate/up and HMX projection scratch are reused after their prior lifetimes end.

Both Android/Hexagon builds passed. Independent parity and identity layout, Float64 dense oracle (declared rounded FP16 scale), exact U8 packing, prefix K native cache reconstruction and all-seven W4 packing checks passed. Maximum dense error is0.589 local FP16 ULP. U8 encoding of the captured HMX result is exact against independent scalar quantization.

Against an ideal Float64-accumulated dense multiply rounded to FP16, prefill Q/K has143 changed codes out of196608 (0.07273%), each only1 code. Complete layer output amplifies this discrepancy:

| Step | Maximum U8-code difference | Dequantized cosine | Frozen <=2 codes and >=0.999 |
|---|---:|---:|---|
| prefill64 | 7 | 0.989692 | fail |
| decode 1 | 3 | 0.983452 | fail |
| decode 2 | 0 | 1.000000 | pass |
| decode 3 | 5 | 0.974874 | fail |
| decode 4 | 0 | 1.000000 | pass |
| decode 5 | 3 | 0.984049 | fail |
| decode 6 | 3 | 0.981652 | fail |
| decode 7 | 4 | 0.979159 | fail |
| decode 8 | 0 | 1.000000 | pass |

A separate audit first executes HMX then substitutes the scalar dense result, preserving the HMX state transition. Its complete output is byte-exact to scalar-only across all9 steps. Candidate output is also byte-exact across the two builds. This excludes the tested HMX state-transition hypothesis. Raw pre-rotation carriers are identical across candidate, scalar and identity. Native cache packing and identity layout are exact. Evidence supports propagation of small Q/K rounding differences through the current integer attention and later A8 chain; the precise first amplifying boundary is not yet localized.

Preserved initial checker failure: np.spacing at a negative FP16 binade edge used only the smaller adjacent spacing. The independent checker now uses both neighboring representable values; the frozen one-ULP criterion was not changed. That tooling correction does not eliminate the real whole-layer gate failure.

## Performance and limits

7 audit runs/63 layer RPCs only. Each requested and received8MiB VTCM; planned peak6682752bytes. Audits explicitly export tensors to DDR. Their Host-wall times are not comparable formal performance evidence, and zero legacy mismatch fields are not independent success claims. Five short/ten formal rounds, the10% speed gate and E2E token/s are N/A because the numerical prerequisite failed. No claim that dense R3 is fast or slow follows from these audits.

Device EOS passes through native A8 computations and is transformed online into cache. It is not the imported unquantized warm-prefix tensor used by the software proxy. DSP PPL and end-to-end text generation remain unmeasured. No full-model deployment or baseline promotion.

## Next discussion

Keep R3 as a promising quality direction. First align a software reference with actual HMX arithmetic and trace QK score, integer Softmax/AV, O and residual/MLP boundaries to locate the amplification. Then assess whether the software PPL gain survives device arithmetic; alternatively test a more precise dense accumulation implementation under a separately frozen protocol. Do not silently raise the2-code/0.999 gate, infer hardware PPL from software, or optimize the pipeline before this mismatch is understood.

![Software PPL](/mnt/d/llm_exp/results/qwen3-block-htp/exp0246/figures/final_ppl.png)

![Activation distributions](/mnt/d/llm_exp/results/qwen3-block-htp/exp0246/figures/key_distributions.png)
