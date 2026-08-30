# EXP-0062 — Complete profiling report

The direct control is EXP-0061. The candidate converts the 64 common RoPE rows
once into a read-only 64 KiB SF32 VTCM overlay instead of repeating FP16-to-SF32
conversion in twelve paired Q/K tasks. Reciprocal-square-root, normalization,
rotation, U8 encoding, projections, Attention, residuals, and MLP are fixed.

| Repeat | Metric | EXP-0061 control | EXP-0062 candidate | Delta | Paired delta |
|---:|---|---:|---:|---:|---:|
| 1 | Host wall ns/block | 2,813,646.0 | 2,800,313.0 | -0.474% | -0.474% |
| 1 | Q/K Norm+RoPE ticks/block | 27,013.0 | 26,885.0 | -0.474% | -0.374% |
| 1 | QKV ledger ticks/block | 10,692.0 | 10,650.0 | -0.393% | -0.384% |
| 10 | Host wall ns/block | 2,416,947.9 | 2,406,359.4 | -0.438% | -0.413% |
| 10 | Q/K Norm+RoPE ticks/block | 26,881.3 | 26,823.1 | -0.217% | -0.445% |
| 10 | QKV ledger ticks/block | 10,669.9 | 10,654.0 | -0.149% | -0.163% |

Final output hash is `69f22eeb035e5ec5`; QK, probability, and AV hashes are
`32aa949912e365be`, `94f2e218f06f9627`, and `f853658f52032bde` for both
variants. Peak planned VTCM is 5,306,080 bytes inside the exact 8 MiB request.
Intermediate DDR and spill/fill remain zero. All repeat-one and repeat-ten
ordinary and paired gates pass.

Formal evidence:
`D:/llm_exp/results/qwen3-block-htp/exp0062/20260830T063618Z_c14de10d0059_formal`.
