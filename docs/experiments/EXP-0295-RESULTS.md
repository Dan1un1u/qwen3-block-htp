# EXP-0295 — two-size Qwen migration

Both Qwen sizes now support the L32-0051-style chunked long-prefill native-layout/concurrent-GQA method with unchanged model-specific arithmetic. Correctness and performance gates are separate.

| Model | Sequence | Prefill token/s | Decode token/s | M64 parity |
|---|---|---:|---:|---|
| Qwen3-0.6B | 536+46 | 2486.77 | 24.84 | FAIL |
| Qwen3-0.6B | 741+3 | 2410.92 | 21.11 | FAIL |
| Qwen3-1.7B | 536+46 | 1693.52 | 19.95 | FAIL |
| Qwen3-1.7B | 741+3 | 1677.19 | 17.51 | FAIL |

Each size: six candidate boundary/long-shape audits, one sequential control audit, five short rounds and ten formal rotated six-arm repeat10 rounds. Read each size RESULTS.md/MODULES.md for controls, timing boundary, CIs and limitations. No numerical-tolerance relaxation, model-quality claim or baseline promotion.
