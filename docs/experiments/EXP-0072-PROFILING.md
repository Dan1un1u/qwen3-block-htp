# EXP-0072 — Profiling closure

## Three-variant repeat-ten overview

F16F16 and W4F16 reuse the retained EXP-0038 evidence. W4U8 uses the
gate-passing EXP-0072 candidate. Positive speed means W4U8 is faster than
W4F16.

| Module | W16A16 | W4A16 | W4A8 EXP-0072 | A8 vs A16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +79.4% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 19.0 us (0.9%) | +124.4% |
| QKV + Q/K Norm/RoPE | 397.1 us (16.0%) | 439.2 us (19.7%) | 424.7 us (20.6%) | +3.4% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 243.5 us (11.8%) | -42.2% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 170.4 us (8.3%) | +2.0% |
| Post-attn residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 33.0 us (1.6%) | +24.7% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 712.7 us (34.6%) | +36.5% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 357.5 us (17.3%) | -8.4% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 17.4 us (0.8%) | -71.2% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 78.4 us (3.8%) | +1.6% |
| **Complete block Host wall** | **2488.3 us** | **2229.7 us** | **2060.8 us** | **+8.2%** |

## Formal A/B result

| Metric | Repeat | pool4 control | pool6 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|---:|
| Host wall ns/block | 1 | 2,479,948.0 | 2,471,198.0 | -0.353% | -0.530% |
| Combined residual ticks/block | 1 | 1,238.0 | 970.0 | -21.648% | -20.723% |
| Host wall ns/block | 10 | 2,073,885.4 | 2,060,849.0 | -0.629% | -0.344% |
| Combined residual ticks/block | 10 | 1,232.6 | 967.6 | -21.499% | -21.393% |

Both modes execute sixteen identical tasks per residual boundary. The candidate
changes only the claiming domain from main plus three to main plus five
workers. Final output hash is `69f22eeb035e5ec5`; Input RMSNorm, QK,
probability, and AV hashes remain `7255c2406108617c`, `32aa949912e365be`,
`94f2e218f06f9627`, and `f853658f52032bde`. Requested/acquired VTCM is
8,388,608 bytes, peak plan is 5,306,080 bytes, intermediate DDR read/write and
spill/fill are zero, HMX commands remain 176, and tile pairs remain 49,408.
The formal gate passes. Full evidence is retained at
`D:/llm_exp/results/qwen3-block-htp/exp0072/20260830T102828Z_6a7be24f93a5_formal`.
