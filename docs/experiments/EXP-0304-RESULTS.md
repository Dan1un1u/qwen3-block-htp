# SP2 / uniform INT16 Down supplementary comparison

Only the Down LUT and associated middle reconstruction scale change. Same binary, frozen W4/other boundaries, FP32 residual, no rotation. Fixed token fixtures, not dataset quality evaluation.

| Down input | Prefill token/s | Decode token/s | Prefill Host ms | Decode Host ms |
|---|---:|---:|---:|---:|
| sp2 | 2998.3426 | 93.2698 | 21.345126 | 450.306534 |
| int16 | 2993.3312 | 93.3147 | 21.380862 | 450.089926 |

| Phase | INT16/SP2 Host wall | Overhead | Paired95% CI | Within10% |
|---|---:|---:|---|---|
| prefill | 1.001674 | +0.1674% | [0.998536, 1.005070] | True |
| decode | 0.999519 | -0.0481% | [0.995031, 1.004536] | True |

SP2 and INT16 have identical paired HMX command/tile counts, weight DMA bytes/descriptors and VTCM peak. All timed selected tokens/codes match each arm own reference. Numerical and physical checks are separate from model quality; no PPL or baseline promotion.
