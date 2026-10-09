# Qwen1.7B R3+R4 on uniform INT16 Down

EXP0332 admits existing production denseR3 mode1 together with recent R4 mode4 OPT12 (HVX FP32) or mode5 OPT10 (HMX FP16). Uniform INT16 Down mode8 and FP32 residual2 are required. Qwen0.6/Llama and scalar/identity/refined R3 combination paths remain outside this capability.

R3 is after Q/K norm/RoPE and before attention. Its Gate/Up/HMX projection scratch becomes dead once quantized Q/K and cache preparation are complete; attention and O finish before MLP repopulates Gate/Up and R4 reuses phase-dead arenas. R4 follows completed SwiGLU producers and precedes two-byte nativeW4 Down. HMX is owned by one existing worker and all existing joins remain; no new VTCM/DDR arena or threads.

Use matching frozen R3 Q/K qparams, score config and once-rotated prefix K. R4 Down weights/scale and FP16 SwiGLU LUT come from the recent full6144 original-derived package. Non-QK metadata/weights must agree before composition. Projection rounding2 retains the EXP0326 half-scale-aware load-time correction. R4-only rounding0 is a separate historical-arithmetic anchor; comparisons with rounding2 require explicit labels.

Example options: QBH_SP2=8 QBH_FP32_RESIDUAL=2 QBH_DENSE_R3=1 QBH_R3_OPT=2 QBH_PROJECTION_ROUNDING=2 QBH_DENSE_R4=5 QBH_R4_OPT=10. R3/R4 flags must match the selected package; merely toggling R3 on an OFF package is invalid.

This admits a combination rather than changing either rotation kernel. Independent component/conditional downstream checks and current actual full-model quality establish the combination; historical independent passes alone do not establish it. Dense R4 FP16 and butterfly R4 FP32 contracts remain separate. No idealFP32 R3 bit equality or default/baseline promotion is implied.

## Package selection and retained switches

| R3 | R4 | Compatible source package |
|---|---|---|
| off | off | Original EXP0308 uniform INT16 Down; original prefix |
| on | off | EXP0324 matched Q/K grids, score metadata, transformed prefix |
| off | dense or butterfly | EXP0329 rotated Down and FP16 SwiGLU LUT; original prefix |
| on | dense or butterfly | EXP0332 composed `on` package; transformed prefix |

The host exposes all switches. EXP0332's five-arm driver covers R4-only and the combination, and deliberately does not offer a no-R4 arm using a rotated package. Original and R3-only packages are retained separately. Long-context combination and other models are not validated by this context64 experiment.

Completed component/1/3/28-layer validation and full-model text/PPL: R3+denseR4 41.43340654 versus denseR4-only rounding2 49.84815219; R3+butterflyR4 41.82856257 versus butterflyR4-only rounding2 48.53350996. Scope is the same 8192 WT2 test targets,64-token warmup,43-target resets with tail retained. Historical BF16 teacher is27.93590920. This is a relative quality improvement and retains a substantial teacher gap.

Normal generation's R3 audit write count now includes all per-layer captures. Dense R4's quality histogram now uses dead Gate storage, with a live-range guard; its nominal Down alias was overwritten by final norm/head bias. The rejected vocabulary-count attempt remains in evidence. Histogram repair leaves every audited generation tensor and layer output unchanged. No rotation arithmetic, VTCM size or timed projection pipeline was changed.

See EXP0332 results and immutable raw evidence. `--build-tag` selects the sealed runtime for fixed or quality runs; quality defaults to the repaired `quality` build. Rerunning requires a newly authorized experiment/evidence namespace; existing records are not overwritten.
