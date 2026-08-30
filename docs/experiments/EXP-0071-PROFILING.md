# EXP-0071 — Profiling closure

## Three-variant repeat-ten overview

F16F16 and W4F16 reuse the retained EXP-0038 evidence. W4U8 uses the
gate-passing EXP-0071 candidate. Positive speed means W4U8 is faster than
W4F16.

| Module | W16A16 | W4A16 | W4A8 EXP-0071 | A8 vs A16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +79.2% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 18.7 us (0.9%) | +128.5% |
| QKV + Q/K Norm/RoPE | 397.1 us (16.0%) | 439.2 us (19.7%) | 421.3 us (20.4%) | +4.2% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 243.2 us (11.8%) | -42.2% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 170.1 us (8.2%) | +2.3% |
| Post-attn residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 42.4 us (2.1%) | -2.8% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 707.8 us (34.3%) | +37.4% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 358.2 us (17.3%) | -8.6% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 21.6 us (1.0%) | -76.8% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 78.2 us (3.8%) | +1.8% |
| **Complete block Host wall** | **2488.3 us** | **2229.7 us** | **2065.6 us** | **+7.9%** |

## Formal A/B result

| Metric | Repeat | EXP-0070 control | EXP-0071 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|---:|
| Host wall ns/block | 1 | 2,556,511.0 | 2,499,427.0 | -2.233% | -2.155% |
| Input RMSNorm ticks/block | 1 | 1,554.0 | 365.0 | -76.512% | -76.268% |
| Host wall ns/block | 10 | 2,134,432.3 | 2,065,567.7 | -3.226% | -3.169% |
| Input RMSNorm ticks/block | 10 | 1,543.7 | 358.8 | -76.757% | -76.742% |

The candidate completes sixteen tasks per block and uses the same per-row
reduction, scalar square root, gamma, requantization, and native HMX activation
store as the control. Main and worker aggregate work are overlapping diagnostic
counters; only the Input RMSNorm ledger interval is additive.

Final output hash is `69f22eeb035e5ec5`; native Input RMSNorm hash is
`7255c2406108617c`; QK, probability, and AV hashes remain
`32aa949912e365be`, `94f2e218f06f9627`, and `f853658f52032bde`.
Requested/acquired VTCM is 8,388,608 bytes, peak plan is 5,306,080 bytes,
intermediate DDR read/write and spill/fill are zero, HMX commands remain 176,
and tile pairs remain 49,408. The formal gate passes. Full evidence is retained
at `D:/llm_exp/results/qwen3-block-htp/exp0071/20260830T101057Z_14526c4fb1a8_formal`.
