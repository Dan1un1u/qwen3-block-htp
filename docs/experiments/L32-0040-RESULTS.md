# L32-0040: A8 / SP2 / Uniform INT16 Down

Only the SwiGLU-to-Down reconstruction grid and its corresponding middle scale change. Native per-channel W4, FP32 residual, all non-Down boundaries and no-rotation settings remain frozen. SP2 and INT16 use the same fused LUT, two-byte physical carrier, native packed-W4 matrix path and pipeline. No new DSP kernel or online conversion. The uniform grid uses ties-to-even and symmetric [-32767,32767] codes; the existing SP2 grid/tie rule is unchanged. Both retain the same nominal frozen clipping alpha, with FP32-stored scales. Ordinary A8 is the original frozen quantizer, not newly recalibrated.

Each trajectory: one warmup per arm, repeat1 auxiliary, five short and ten rotated formal repeat10 rounds. Primary fixed-token and supplementary greedy are separate estimates; no optional stopping. Complete M64+15 Host wall includes embedding, all layers, final norm, head, greedy, FastRPC, excludes cold loading/ADB/external tokenizer. Headline and module times are arithmetic means of the ten round means. Paired bootstrap20000 with fixed seed.

| Trajectory | Down input | Prefill token/s | Decode token/s | Prefill Host ms | 15-decode Host ms |
|---|---|---:|---:|---:|---:|
| fixed | A8 | 2373.1055 | 45.7049 | 26.968881 | 328.192547 |
| fixed | SP2 | 2335.2361 | 45.7404 | 27.406222 | 327.937891 |
| fixed | INT16 | 2318.2070 | 45.6310 | 27.607543 | 328.724105 |
| greedy | A8 | 2359.3882 | 45.6114 | 27.125676 | 328.865338 |
| greedy | SP2 | 2330.5597 | 45.6325 | 27.461214 | 328.712978 |
| greedy | INT16 | 2316.7224 | 45.6136 | 27.625234 | 328.849623 |

| Trajectory / phase | Comparison | Host overhead | Paired 95% CI of wall ratio |
|---|---|---:|---|
| fixed / prefill | SP2_over_A8 | +1.622% | [1.013524, 1.019196] |
| fixed / prefill | INT16_over_A8 | +2.368% | [1.019475, 1.028208] |
| fixed / prefill | INT16_over_SP2 | +0.735% | [1.003109, 1.012004] |
| fixed / decode | SP2_over_A8 | -0.078% | [0.997507, 1.000954] |
| fixed / decode | INT16_over_A8 | +0.162% | [0.998993, 1.004629] |
| fixed / decode | INT16_over_SP2 | +0.240% | [0.999063, 1.006137] |
| greedy / prefill | SP2_over_A8 | +1.237% | [1.007962, 1.016607] |
| greedy / prefill | INT16_over_A8 | +1.842% | [1.015602, 1.021933] |
| greedy / prefill | INT16_over_SP2 | +0.597% | [1.001101, 1.010899] |
| greedy / decode | SP2_over_A8 | -0.046% | [0.995422, 1.003541] |
| greedy / decode | INT16_over_A8 | -0.005% | [0.996977, 1.002105] |
| greedy / decode | INT16_over_SP2 | +0.042% | [0.996432, 1.003943] |

All 14400 short/formal token-boundary profiles verified. Same weight DDR bytes, HMX command counts and VTCM peak across arms. SP2 and INT16 HMX tile work is equal; relative to A8, extra prefill tile pairs 262144, extra decode pairs0. This does not reveal HMX internal W4 implementation.

Numerical implementation correctness does not establish model quality. No PPL or quality acceptance. This comparison measures efficient support for the software-prescribed SP2 representation; it does not establish SP2 superiority over uniform INT16.

The reused internal flag remains SP2 mode8 for both SP2 and INT16; manifests, LUTs, scales and arm labels distinguish the two quantization contracts. Failed tooling attempts, if any, remain in the evidence directory; see closure notes.

## Exact implementation evidence

All16 per-layer LUTs passed independent scalar checks for65536 Gate/Up pairs each. Complete signed16 carrier coverage and endpoint/row-co-pack/separate-pass device probes passed exact. Maximum all-input raw signed24 bound2,394,960; signed32 merged bound599,392,256. Independently referenced selected layer7,chain3,chain16 prefill/decode passed exactly. Full16 fixed and greedy selected tokens/codes matched independent CPU trajectories; an extra untimed boundary audit matched all65,536 exported FP32 last-row hidden values across both trajectories exactly. No failed hardware or numerical attempt. `full_hidden_gate.json` records this stronger evidence.
