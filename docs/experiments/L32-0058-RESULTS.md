# SP2 / uniform INT16 Down supplementary comparison

Model: 3B; fixed prompt/decode: 741+3. Five short paired repeat10 rounds and ten formal AB/BA repeat10 rounds.

Only the Down LUT and associated middle reconstruction scale change. Same binary, frozen W4/other boundaries, FP32 residual, no rotation. Fixed token fixtures, not dataset quality evaluation.

| Down input | Prefill token/s | Decode token/s | Prefill Host ms | Decode Host ms |
|---|---:|---:|---:|---:|
| sp2 | 1176.2148 | 23.4397 | 629.986976 | 127.987801 |
| int16 | 1176.8026 | 23.4427 | 629.672315 | 127.971423 |

| Phase | INT16/SP2 Host wall | Overhead | Paired95% CI | Within10% |
|---|---:|---:|---|---|
| prefill | 0.999501 | -0.0499% | [0.997857, 1.000969] | True |
| decode | 0.999872 | -0.0128% | [0.996729, 1.002743] | True |

SP2 and INT16 have identical paired HMX command/tile counts, weight DMA bytes/descriptors and VTCM peak. All timed selected tokens/codes match each arm own reference. Numerical and physical checks are separate from model quality; no PPL or baseline promotion.

Independent INT16 reference initially differed only in an exact-cancellation zero sign. Archived V79 disassembly plus the instruction simulator prove the expected negative zero. Recomputed CPU reference evaluates every zero-result epilogue through that simulator; all420 layer hashes match, without relaxing bit-exact checks or changing hardware. Both arms also pass840 exact KV snapshots each. See reference-recovery.md; rejected libnative diagnostic and original failed reference attempts remain.
