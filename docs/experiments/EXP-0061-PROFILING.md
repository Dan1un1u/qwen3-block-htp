# EXP-0061 — Complete profiling report

The direct control is EXP-0060. Both paths use the exact HVX word-tree
centered-square reduction. The candidate batches the two heads and sixteen
rows in each chunk into one 32-lane `qhmath_hvx_rsqrt_vf` invocation, retains
the corresponding 4 KiB of row data in a per-worker VTCM overlay, and leaves
gamma, RoPE, U8 requantization, projections, Attention, residuals, and MLP
unchanged.

| Repeat | Metric | EXP-0060 control | EXP-0061 candidate | Delta | Paired delta |
|---:|---|---:|---:|---:|---:|
| 1 | Host wall ns/block | 2,844,115.0 | 2,828,282.0 | -0.557% | -0.316% |
| 1 | Q/K Norm+RoPE ticks/block | 28,395.0 | 27,124.0 | -4.476% | -4.613% |
| 1 | QKV ledger ticks/block | 11,131.0 | 10,699.0 | -3.881% | -3.981% |
| 10 | Host wall ns/block | 2,426,703.1 | 2,409,963.6 | -0.690% | -1.111% |
| 10 | Q/K Norm+RoPE ticks/block | 28,108.7 | 26,897.2 | -4.310% | -4.253% |
| 10 | QKV ledger ticks/block | 10,980.1 | 10,677.4 | -2.757% | -2.712% |

The seven-round interleaved speed gate passes at repeat one and repeat ten.
Final output hash is `69f22eeb035e5ec5`; QK, probability, and AV hashes are
`32aa949912e365be`, `94f2e218f06f9627`, and `f853658f52032bde` for both
control and candidate. Peak planned VTCM remains 5,306,080 bytes inside the
mandatory 8 MiB request. Intermediate DDR read/write and spill/fill are zero.

Formal evidence:
`D:/llm_exp/results/qwen3-block-htp/exp0061/20260830T062409Z_21006b8d9ab7_formal`.
