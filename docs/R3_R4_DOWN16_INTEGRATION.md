# Qwen1.7B R3+R4 on uniform INT16 Down

EXP0332 admits existing production denseR3 mode1 together with recent R4 mode4 OPT12 (HVX FP32) or mode5 OPT10 (HMX FP16). Uniform INT16 Down mode8 and FP32 residual2 are required. Qwen0.6/Llama and scalar/identity/refined R3 combination paths remain outside this capability.

R3 is after Q/K norm/RoPE and before attention. Its Gate/Up/HMX projection scratch becomes dead once quantized Q/K and cache preparation are complete; attention and O finish before MLP repopulates Gate/Up and R4 reuses phase-dead arenas. R4 follows completed SwiGLU producers and precedes two-byte nativeW4 Down. HMX is owned by one existing worker and all existing joins remain; no new VTCM/DDR arena or threads.

Use matching frozen R3 Q/K qparams, score config and once-rotated prefix K. R4 Down weights/scale and FP16 SwiGLU LUT come from the recent full6144 original-derived package. Non-QK metadata/weights must agree before composition. Projection rounding2 retains the EXP0326 half-scale-aware load-time correction. R4-only rounding0 is a separate historical-arithmetic anchor; comparisons with rounding2 require explicit labels.

Example options: QBH_SP2=8 QBH_FP32_RESIDUAL=2 QBH_DENSE_R3=1 QBH_R3_OPT=2 QBH_PROJECTION_ROUNDING=2 QBH_DENSE_R4=5 QBH_R4_OPT=10. R3/R4 flags must match the selected package; merely toggling R3 on an OFF package is invalid.

This admits a combination rather than changing either rotation kernel. Independent component/conditional downstream checks and current actual full-model quality establish the combination; historical independent passes alone do not establish it. Dense R4 FP16 and butterfly R4 FP32 contracts remain separate. No idealFP32 R3 bit equality or default/baseline promotion is implied.
