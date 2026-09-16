# EXP-0284: A8 / SP2 / Uniform INT16 Down

Only the SwiGLU-to-Down reconstruction grid and its corresponding middle scale change. Native per-channel W4, FP32 residual, all non-Down boundaries and no-rotation settings remain frozen. SP2 and INT16 use the same fused LUT, two-byte physical carrier, native packed-W4 matrix path and pipeline. No new DSP kernel or online conversion. The uniform grid uses ties-to-even and symmetric [-32767,32767] codes; the existing SP2 grid/tie rule is unchanged. Both retain the same nominal frozen clipping alpha, with FP32-stored scales. Ordinary A8 is the original frozen quantizer, not newly recalibrated.

Each trajectory: one warmup per arm, repeat1 auxiliary, five short and ten rotated formal repeat10 rounds. Primary fixed-token and supplementary greedy are separate estimates; no optional stopping. Complete M64+15 Host wall includes embedding, all layers, final norm, head, greedy, FastRPC, excludes cold loading/ADB/external tokenizer. Headline and module times are arithmetic means of the ten round means. Paired bootstrap20000 with fixed seed.

| Trajectory | Down input | Prefill token/s | Decode token/s | Prefill Host ms | 15-decode Host ms |
|---|---|---:|---:|---:|---:|
| fixed | A8 | 1857.8547 | 47.2170 | 34.448335 | 317.681955 |
| fixed | SP2 | 1846.7884 | 47.1178 | 34.654755 | 318.351099 |
| fixed | INT16 | 1841.7867 | 47.1440 | 34.748866 | 318.174210 |
| greedy | A8 | 1861.3444 | 47.3031 | 34.383751 | 317.103799 |
| greedy | SP2 | 1848.2329 | 47.3235 | 34.627671 | 316.967186 |
| greedy | INT16 | 1839.9065 | 47.1959 | 34.784377 | 317.824304 |

| Trajectory / phase | Comparison | Host overhead | Paired 95% CI of wall ratio |
|---|---|---:|---|
| fixed / prefill | SP2_over_A8 | +0.599% | [1.003499, 1.008670] |
| fixed / prefill | INT16_over_A8 | +0.872% | [1.005923, 1.012037] |
| fixed / prefill | INT16_over_SP2 | +0.272% | [1.000742, 1.005142] |
| fixed / decode | SP2_over_A8 | +0.211% | [0.999781, 1.003867] |
| fixed / decode | INT16_over_A8 | +0.155% | [0.999727, 1.003260] |
| fixed / decode | INT16_over_SP2 | -0.056% | [0.997479, 1.001421] |
| greedy / prefill | SP2_over_A8 | +0.709% | [1.002606, 1.011325] |
| greedy / prefill | INT16_over_A8 | +1.165% | [1.008062, 1.015545] |
| greedy / prefill | INT16_over_SP2 | +0.453% | [0.999448, 1.010314] |
| greedy / decode | SP2_over_A8 | -0.043% | [0.994061, 1.003396] |
| greedy / decode | INT16_over_A8 | +0.227% | [0.999808, 1.005151] |
| greedy / decode | INT16_over_SP2 | +0.270% | [0.998247, 1.008696] |

All 14400 short/formal token-boundary profiles verified. Same weight DDR bytes, HMX command counts and VTCM peak across arms. SP2 and INT16 HMX tile work is equal; relative to A8, extra prefill tile pairs 344064, extra decode pairs0. This does not reveal HMX internal W4 implementation.

Numerical implementation correctness does not establish model quality. No PPL or quality acceptance. This comparison measures efficient support for the software-prescribed SP2 representation; it does not establish SP2 superiority over uniform INT16.

The reused internal flag remains SP2 mode8 for both SP2 and INT16; manifests, LUTs, scales and arm labels distinguish the two quantization contracts. Failed tooling attempts, if any, remain in the evidence directory; see closure notes.


## Verification and retained attempts

All signed16 code values and endpoint co-packed/separate HMX component checks pass. Across all28 layers the conservative signed24 raw magnitude bound is1,819,170 and signed32 reconstructed magnitude bound is461,373,440. All28 fused LUTs exhaustively checked over65536 Gate/Up pairs. Selected layer14 and consecutive3 gates pass. Complete independent CPU INT16 references match all896 layer-output hashes across fixed and greedy trajectories, plus65536 final-hidden FP32 values, final norm and full-vocabulary head selection. Original A8/SP2 controls reproduce sealed EXP0282 A5-fair-a02 results. No native or build implementation changed from the floating-baseline refresh.

Preserved a01 failures identified an experiment-local reference mismatch, not a native numerical failure. AV multiplier1 uses output_zero_point directly inside saturation. Historical reference clipped around128 first, then translated, creating two layer4 values20 instead of0. EXP0284 wrapper fixes that branch; SDK-emulator exhaustive65537 accumulator integers and standalone131072 FP32 outputs confirm the repair. Historical helper and evidence stay untouched. Corrected complete references are frontend-reference-a03; a02 was interrupted solely to expand the CPU matrix cache16→256. The250 common intermediate tensors remain exact. Initial standalone diagnostic package omitted mandatory physical KV-reference files; a02 repairs packaging and preserves a01. No timed runs were discarded or repeated.

Measured binary source eaea89c81dd0e5f3424dfdea9d1e2b47347f8d72; corrected CPU reference source4e2b50745353808a2c5385af149e3e1a058ce9e7. Native source trees unchanged. Device released on closure. No PPL, no baseline promotion.
