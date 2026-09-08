# EXP-0250 integer attention attribution

conditional full-model software PPL with independently validated integer attention, not DSP PPL

F16 PPL 29.486850; C64 W4A16 PPL 29.665686.

| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |
|---|---:|---:|---:|
| carrier | 49.943143 | 39.089126 | 0.7827 [0.7260, 0.8410] |
| unbounded | 51.267663 | 39.027082 | 0.7612 [0.7062, 0.8171] |
| wide_float | 50.981746 | 39.027082 | 0.7655 [0.7119, 0.8211] |
| score | 7658.548439 | 6371.563253 | 0.8320 [0.7135, 0.9675] |
| wide_exact | 52.388373 | 38.961148 | 0.7437 [0.6879, 0.8019] |
| wide_sole | 61.559435 | 41.873633 | 0.6802 [0.6231, 0.7408] |
| sole | 11271.881495 | 8272.142744 | 0.7339 [0.6253, 0.8594] |

## Conditional increments (positive means worse)

| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |
|---|---:|---:|
| carrier->unbounded | +0.026175 [-0.033456, +0.086925] | -0.001588 [-0.050881, +0.047116] |
| unbounded->wide_float | -0.005593 [-0.061322, +0.050168] | +0.000000 [+0.000000, +0.000000] |
| wide_float->score | +5.012110 [+4.790003, +5.243184] | +5.095344 [+4.895198, +5.310212] |
| wide_float->wide_exact | +0.027217 [-0.023298, +0.077400] | -0.001691 [-0.052078, +0.049737] |
| wide_exact->wide_sole | +0.161318 [+0.085106, +0.239865] | +0.072091 [+0.009421, +0.133284] |
| sole->wide_sole | -5.210063 [-5.459861, -4.971044] | -5.285993 [-5.513429, -5.071217] |

## Cells

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 27.585201 | 25.533726 | 24.277671 | 44.209560 |
| C64 | 26.172385 | 26.665188 | 24.564196 | 45.178011 |
| OFF_carrier | 45.649527 | 50.761401 | 43.775160 | 61.334646 |
| OFF_unbounded | 45.629718 | 50.839385 | 40.792887 | 73.003145 |
| OFF_wide_float | 43.805266 | 51.661614 | 43.148485 | 69.182933 |
| OFF_score | 8591.846913 | 7706.145430 | 9154.058774 | 5676.078938 |
| OFF_wide_exact | 48.006744 | 52.767784 | 42.405789 | 70.120239 |
| OFF_wide_sole | 56.343213 | 60.019377 | 52.825382 | 80.390133 |
| OFF_sole | 12794.080756 | 12658.036268 | 12890.882260 | 7732.636442 |
| R3_carrier | 35.995964 | 39.650972 | 30.532191 | 53.574512 |
| R3_unbounded | 31.839948 | 37.745618 | 33.422417 | 57.754734 |
| R3_wide_float | 31.839948 | 37.745618 | 33.422417 | 57.754734 |
| R3_score | 7031.444279 | 6061.482156 | 6888.962965 | 5613.151641 |
| R3_wide_exact | 31.699427 | 41.513411 | 30.875196 | 56.712324 |
| R3_wide_sole | 35.046686 | 41.018993 | 32.436521 | 65.932013 |
| R3_sole | 8881.586586 | 7739.527014 | 9104.718719 | 7481.704864 |

## Evidence and limits

All eight parent controls (F/C64 and OFF/R3 carrier/score/sole) reproduce EXP0249 per-token NLL and top1 exactly. Independent int64/scalar and GPU reference agree, including retained EXP0042/EXP0248 QK/probability/AV captures, random masked prefill/decode, all56 layer configurations,384 old division cases and72 wide-difference safety checks.
Fresh128 documents,32 per English/Chinese wiki/news cell,2048 targets. Historical/calibration documents, text hashes and token32-grams excluded. Independent reconstruction, repeat, causal, CE and immutable weights/prefix checks pass. All16 arms frozen before inference. This lightweight diagnostic panel does not replace deployment acceptance. Pointwise paired document bootstrap10000seed250; no multiplicity correction.
Conditional comparisons: carrier->unbounded combines score rounding and power-of-two gain approximation; unbounded->wide_float retains first raw U8 saturation; wide_float->score adds the second U8 saturation; wide_float->wide_exact adds capped integer exponents; wide_exact->wide_sole adds reciprocal approximation. These effects are conditional and nonadditive. sole->wide_sole tests the complete proposed repair against the old integer attention.
R3 vs OFF includes frozen Q/K calibration differences. Linear/norm/R3 software arithmetic, raw offline prefix, signed K clipping, V recentering, AV conversion and existing probability scale discrepancy remain unchanged. This is not full-device PPL. No new calibration, weights, hardware, deployment or baseline promotion.

E2E token/s: N/A (no device/full-model timing).

## Interpretation

The proposed repair is effective on this independent panel: R3 full integer attention PPL 8272.14 -> 41.87, OFF 11271.88 -> 61.56. The second score saturation is the dominant measured failure. R3 with frozen Q/K recalibration improves the repaired path relative to OFF. However repaired R3 is still 42.0% above F16 PPL; the exact-normalization reference is 32.1% above F16. Both fail the unchanged accuracy reference. SOLE adds a significant conditional penalty in both groups, while integer-exponent pointwise intervals include zero loss. The repair succeeds numerically; A8 quality acceptance remains unmet.

Masked row maximum is now computed on raw native U8 scores. The difference is explicitly signed16 before multiplication and exponent coding, avoiding the second U8 saturation. The bound255*18+4=4594 fits int16. This changes downstream score handling only; native W4 linear semantics and first HMX U8 score output are retained.
A simple same-range example: raw scores[160,180], multiplier5. The old second conversion maps both to255, making their difference zero. The repaired path retains a difference100 in F3 units before exponent coding. Moving row-max subtraction later cannot reconstruct that difference once both codes have saturated.
Full-model software results must be interpreted against the paired controls below. They do not demonstrate a new device kernel or recover precision already lost in the first raw U8 conversion. The latter is isolated separately by unbounded versus wide_float.

| Complete integer path | Old PPL | Repaired PPL | Repaired / old (95% CI) |
|---|---:|---:|---:|
| OFF | 11271.881495 | 61.559435 | 0.005461 [0.004254, 0.006936] |
| R3 | 8272.142744 | 41.873633 | 0.005062 [0.004032, 0.006275] |

First-stage raw-U8 versus unbounded per-token NLL/top1 exactness on the final panel: {"OFF": false, "R3": true}. Equality on this finite panel does not prove saturation is impossible on other inputs or longer contexts.

Eight exposed development controls exactly reproduce the parent experiment. All16 independent arms were specified before inference, so every arm is reported. No range or candidate tuning used the final panel. Better conditional PPL after rounding or exponent encoding can reflect interacting errors, and is not proof that those approximations are error-free.

## Same-input Softmax precision on retained exposed development traces

| Group | Standard Softmax + U8: mean row L1 / mass | Integer exponent + exact division | Integer exponent + SOLE |
|---|---:|---:|---:|
| OFF | 0.025145 / 0.994473 | 0.079822 / 0.993357 | 0.134819 / 0.970531 |
| R3 | 0.025093 / 0.994935 | 0.078506 / 0.994452 | 0.135280 / 0.970655 |

Each entry is an equal-layer average over all28 retained EXP0249 carrier-trajectory prefill traces for four exposed development documents. Within each group the source scores and masks are identical. L1 is relative to standard Softmax of the repaired, first-stage-U8 scores. Codes are divided by255 for this local diagnostic; the slightly clipped frozen AV probability scales remain unchanged in model PPL. These traces are explanatory and are not additional independent final samples.

## Next direction and limits

Prioritize checking a device realization of raw U8 masked maximum and widened HVX score differences while preserving native HMX W4 and the existing memory/RPC contract. Keep the existing single-layer Host-wall >10% slowdown stop. Before full-model deployment, re-evaluate exponent and reciprocal precision with a small fixed candidate set on development inputs; exact division here is a diagnostic upper reference for normalization, not a demonstrated fast DSP implementation. Do not equate software dense R3 with the expensive EXP0248 device refinement or silently inherit its timing.
The quality reference remains <=5% overall PPL increase and <=10% in every cell relative to F16, with pointwise uncertainty reported. A successful numerical repair does not mean A8 quality acceptance. Remaining carrier, V/AV and nonlinear errors are still bundled outside these conditional comparisons. Other recipes and weights remain frozen. No new experiment, hardware deployment or baseline promotion is implied by this proposal.

Artifacts: raw_score_repair.png/.pdf, summary.json, local_wide_precision.json, independent_integrity_checks.json, FULL_PROFILE.md. E2E token/s N/A.

Closure recovery: an expected aggregate parent-audit file was missing, not a hash mismatch. Full pinned-ledger verification of545 historical files and the second independent integrity run passed; initial failure is preserved in integrity_attempt1.log. See RECOVERY.md.
