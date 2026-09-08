# EXP-0253 complete floating attention core

On the same fresh128 documents /2048 targets, fixed per-channel W4 and U8 Q/K/V caches, full floating internal attention with dense R3 has PPL 33.043536, F16 26.407342, and C64 W4A16 27.331961. Its PPL/F16 ratio is 1.251301 (95% CI [1.176173,1.334539]); original overall5percent/every-cell10percent point gate False, confidence gate False.

Relative to R3+NR64, floating-core PPL change is -7.910% (paired ratio95%CI [0.871631,0.971993]). Relative to historical R3+standard-Softmax/probability-A8, change is +3.594% (paired ratio95%CI [0.988284764642254, 1.0863920911831333]). These are paired full-model conditional software results, with frozen offline EOS prefix and no device inference. Rotation comparisons also include the frozen historical Q/K recalibration; they are not rotation-only causality tests.

The new path fully removes score/log2 exponent/probability quantization and signed-K/V/integer-AV internals. It retains actual U8 Q/K/V codes and original output context QDQ once per invocation. Float16 decoded inputs -> FP32 accurate QK, exponential Softmax, FP32 AV -> FP16 -> frozen context QDQ. To satisfy unchanged Float64 numerical tolerance at large logits, FP32 QK preserves TwoSum summation residuals before common maximum subtraction. Original failed test and repair rationale are retained; no PPL had been observed when repaired. This is a reference for the mathematical attention operation, not a native FlashAttention kernel, guarantee of HMX rounding, theoretical upper bound, or speed measurement.

All56 qparam configurations /280 numerical cases passed independent Float64 dense and chunked online references. All six historical development controls reproduce per-token scores exactly. Fresh data independently retokenized and checked against all previous panels/calibration/training. All16 phase-arm scores passed repeat/causality/CE/cache/prefix/weights and output-QDQ invocation checks. Independent NLL/PPL reconstruction and unchanged quality-gate calculation passed. Other recipes, weight/model packages and native binaries unchanged.

# EXP-0253 floating attention diagnostic

conditional full-model software PPL comparing standard FP32 attention core with retained U8 cache and output boundaries; not DSP PPL or a native FlashAttention kernel

F16 PPL 26.407342; C64 W4A16 PPL 27.331961.

| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |
|---|---:|---:|---:|
| legacy | 39.024535 | 31.897286 | 0.8174 [0.7723, 0.8628] |
| wide_nr64 | 46.391480 | 35.881765 | 0.7735 [0.7234, 0.8278] |
| float_core | 37.560899 | 33.043536 | 0.8797 [0.8302, 0.9340] |

## Conditional increments (positive means worse)

| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |
|---|---:|---:|
| legacy->wide_nr64 | +0.172925 [+0.112534, +0.237112] | +0.117708 [+0.062524, +0.172823] |
| legacy->float_core | -0.038227 [-0.085829, +0.010072] | +0.035305 [-0.011784, +0.082862] |
| wide_nr64->float_core | -0.211152 [-0.274338, -0.147875] | -0.082403 [-0.137389, -0.028407] |

## Cells

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 18.391632 | 27.160144 | 27.618548 | 35.248899 |
| C64 | 18.680383 | 29.876803 | 28.602674 | 34.958811 |
| OFF_legacy | 25.964434 | 43.991499 | 43.027016 | 47.191319 |
| OFF_wide_nr64 | 31.105248 | 55.197256 | 49.463986 | 54.539687 |
| OFF_float_core | 24.647319 | 41.337728 | 44.083475 | 44.315108 |
| R3_legacy | 21.254773 | 36.507453 | 33.131826 | 40.265356 |
| R3_wide_nr64 | 23.853628 | 41.559647 | 36.102170 | 46.316481 |
| R3_float_core | 22.062428 | 37.752488 | 34.156562 | 41.905693 |

Eight prespecified arms. All six historical development controls reproduce exactly (EXP0249 legacy; EXP0252 F/C64/NR64). Fresh128documents/2048targets; frozen weights, parameters, prefix, independent data audit. Numerical, repeat, causal, CE, immutable weight/prefix checks. No final-data selection. C64 denotes the fixed per-channel W4A16 weight control, not group64. The floating core decodes identical U8 cache semantics and removes only attention-internal quantization; output FP16 cast and context QDQ remain.

No hardware execution in this experiment. E2E tokens/s: N/A (not measured).

## Unchanged quality gates versus F16

| Arm | PPL / F16 (95% CI) | Point pass | Confidence pass |
|---|---:|---|---|
| C64 | 1.035014 [0.999105, 1.073381] | False | False |
| OFF_legacy | 1.477791 [1.376173, 1.590997] | False | False |
| OFF_wide_nr64 | 1.756764 [1.616405, 1.916808] | False | False |
| OFF_float_core | 1.422366 [1.319591, 1.535714] | False | False |
| R3_legacy | 1.207895 [1.131400, 1.294128] | False | False |
| R3_wide_nr64 | 1.358780 [1.264109, 1.460804] | False | False |
| R3_float_core | 1.251301 [1.176173, 1.334539] | False | False |

Float-core QK uses FP32 TwoSum residual reduction before common maximum subtraction to satisfy the unchanged Float64 numerical reference gate; see ARITHMETIC_REPAIR.md. Its arithmetic is a standard attention diagnostic reference, not the measured precision or speed of a native FlashAttention implementation. Retained U8 input codes and one output context QDQ are checked on every invocation.

E2E tokens/s: N/A — no new device run.
