# L32-0015 paused for numerical-gate discussion (lock retained)

Read docs/experiments/L32-0015.md then docs/LLAMA32_R3_R4_SP2_CHECKPOINT.md.
R3/R4 isolated components pass. R4-SP2 full layer0 actual-arithmetic tail exact.
Prefill ideal gate passes. Decode has one1LSB change among2048 and cosine0.992509,
below0.999; reference squared norm is only67 LSB^2. No gate weakening or speed claim.
Initial DSP address crash fixed with32KiB dense-operand alignment. All attempts retained.
No expansion/formal profiling until discussed. R3 fullblock remains pending.
Baseline L32-0012, other recipes,Qwen,frozen rotation branch unchanged.
