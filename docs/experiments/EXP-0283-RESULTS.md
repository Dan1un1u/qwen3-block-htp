# EXP-0283: refreshed W16A16 paper baseline

Model: Qwen3-1.7B. Complete warm M64 prefill plus 15 continuous decode passes; includes embedding, all layers, final norm, FP16 head, greedy and FastRPC. Excludes cold loading, host tokenizer and transport. Same binary, model package and prompt for both arms.

| Configuration | Prefill token/s | Decode token/s | Prefill Host ms | Total 15-decode Host ms |
|---|---:|---:|---:|---:|
| Current unmodified mathematics / original schedule, OPT0 | 795.235 | 9.067 | 80.479359 | 1654.287674 |
| Exact conversion + head prefetch + consumed-row final norm, OPT3 | 797.802 | 12.500 | 80.220424 | 1200.041005 |

Paired candidate/control wall ratio: prefill 0.99568296, bootstrap95 CI [0.994535565969829, 0.9968526043067385]; decode 0.72389658, CI [0.7229581586150178, 0.7249402462698407]. Median of ten round means, fixed balanced AB/BA; five short rounds passed before ten formal rounds; repeat10 primary, one repeat1 auxiliary pair. All runs preserved.

## Correctness and physical scope

Full-model audit: 912 hidden/Norm/cache files byte-identical across original, conversion-only and final candidates, all 16 token IDs and selected FP16 logit codes exact, every layer output hash exact. Component audit checks finite binary16 patterns and probability rounding midpoints plus actual softmax outputs. Existing single-layer / consecutive-layer checks precede full-model candidate execution. Timed runs disable audit exports and retain zero intermediate and output DDR. No PPL rerun or new model-quality claim.

VTCM grant remains 8MiB; peak 6875904 bytes, identical across arms. Weight bytes, HMX commands and FP16 tile-pair work are equal for every matched step. Head prefetch changes from zero to 593 per pass, using already allocated alternating buffers. One matrix owner and one RPC per full pass/token. See attribution.json and MODULE_TABLES.md.

## Changes and limits

The EXP0260 exact A16 conversion/HVX copy path is enabled for FP16 stored weights. Scalar expf, ordered FP32 reduction/division and both FP16 probability roundings are preserved. FP16 head stages the next weight group while the current HMX/argmax completes. Only the final consumed prompt row is normalized for the head. Backbone M64 carrier computation has not been deeply redesigned; intrinsic FP16 weight bandwidth remains a real cost. No floating approximation change, quantization, rotation, or changes to other recipes.

Implementation selector: QBH_F16F16_OPT=0 original, 2 exact decode only, 3 final candidate. Existing W4F16 flags keep their prior meanings. OPT3 is an eligible measured implementation; this report does not promote a Selected Baseline.

Source build: 81e16c4ca4d9f2929c982dc4378f00dd436d8846; package manifest 0f8a359b559f252cb13329f57d4cabd56f17ca9bfb64bc5643f4d6014795070f. Full build/runtime seals, commands and hashes are retained. Historical W16 measurements are non-paired context, not the optimization denominator.

## Retained issues

Initial invocation attempted a non-executable build script; corrected by invoking bash. The first formal orchestration attempt used system Python without NumPy; resumed with the existing scientific Python environment before any formal samples. All failed logs retained. Add any later attributable tooling repairs before closure.
