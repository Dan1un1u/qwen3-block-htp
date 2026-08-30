# EXP-0070 — Profiling closure

## Three-variant repeat-ten overview

F16F16 and W4F16 reuse the selected EXP-0038 evidence. W4U8 uses the
gate-passing EXP-0070 candidate. Positive speed means W4U8 is faster than
W4F16.

| Module | W16A16 | W4A16 | W4A8 EXP-0070 | A8 vs A16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +81.5% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 80.3 us (3.8%) | -46.8% |
| QKV + Q/K Norm/RoPE | 397.1 us (16.0%) | 439.2 us (19.7%) | 422.1 us (19.9%) | +4.0% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 243.7 us (11.5%) | -42.3% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 170.3 us (8.0%) | +2.1% |
| Post-attn residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 42.5 us (2.0%) | -3.0% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 704.1 us (33.1%) | +38.1% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 356.8 us (16.8%) | -8.3% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 21.7 us (1.0%) | -77.0% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 78.6 us (3.7%) | +1.3% |
| **Complete block Host wall** | **2488.3 us** | **2229.7 us** | **2124.2 us** | **+5.0%** |

## Formal A/B result

| Metric | Repeat | EXP-0068 control | EXP-0070 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|---:|
| Host wall ns/block | 1 | 2,585,000.0 | 2,557,969.0 | -1.046% | -1.025% |
| Attention ticks/block | 1 | 4,898.0 | 4,670.0 | -4.655% | -4.512% |
| QKV + Attention ticks/block | 1 | 12,992.0 | 12,849.0 | -1.101% | -1.535% |
| Host wall ns/block | 10 | 2,139,989.5 | 2,124,156.3 | -0.740% | -0.457% |
| Attention ticks/block | 10 | 4,937.0 | 4,679.9 | -5.208% | -5.208% |
| QKV + Attention ticks/block | 10 | 13,051.6 | 12,788.4 | -2.017% | -1.864% |

At repeat ten, QK HMX work falls 44.068%, AV HMX work falls 40.787%, and
Attention pipeline wait falls 50.923%. HMX tile pairs remain 49,408, while
Attention commands fall from 32 to 16 and block commands from 192 to 176.
All DMA traffic, tasks, context counts, VTCM residency and non-Attention
stages remain invariant.

Final output hash is `69f22eeb035e5ec5`; QK, probability, and AV hashes are
`32aa949912e365be`, `94f2e218f06f9627`, and `f853658f52032bde`.
Requested/acquired VTCM is 8,388,608 bytes, peak plan is 5,306,080 bytes,
and intermediate DDR read/write and spill/fill are all zero. The formal gate
passes. Full evidence is retained at
`D:/llm_exp/results/qwen3-block-htp/exp0070/20260830T094952Z_ce84f2ecb593_formal`.
