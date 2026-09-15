# L32-0039: refreshed W16A16 paper baseline

Model: Llama-3.2-1B-Instruct. Complete warm M64 prefill plus 15 continuous decode passes; includes embedding, all layers, final norm, FP16 head, greedy and FastRPC. Excludes cold loading, host tokenizer and transport. Same binary, model package and prompt for both arms.

| Configuration | Prefill token/s | Decode token/s | Prefill Host ms | Total 15-decode Host ms |
|---|---:|---:|---:|---:|
| Current unmodified mathematics / original schedule, OPT0 | 964.478 | 12.956 | 66.357109 | 1157.735586 |
| Exact conversion + head prefetch + consumed-row final norm, OPT3 | 974.175 | 17.472 | 65.696620 | 858.501927 |

Paired candidate/control wall ratio: prefill 0.99159146, bootstrap95 CI [0.9899611969433127, 0.992177764140824]; decode 0.74165230, CI [0.7408542488603129, 0.742329158231732]. Median of ten round means, fixed balanced AB/BA; five short rounds passed before ten formal rounds; repeat10 primary, one repeat1 auxiliary pair. All runs preserved.

## Correctness and physical scope

Full-model audit: 528 hidden/Norm/cache files byte-identical across original, conversion-only and final candidates, all 16 token IDs and selected FP16 logit codes exact, every layer output hash exact. Component audit checks finite binary16 patterns and probability rounding midpoints plus actual softmax outputs. Existing single-layer / consecutive-layer checks precede full-model candidate execution. Timed runs disable audit exports and retain zero intermediate and output DDR. No PPL rerun or new model-quality claim.

VTCM grant remains 8MiB; peak 7842560 bytes, identical across arms. Weight bytes, HMX commands and FP16 tile-pair work are equal for every matched step. Head prefetch changes from zero to 500 per pass, using already allocated alternating buffers. One matrix owner and one RPC per full pass/token. See attribution.json and MODULE_TABLES.md.

## Changes and limits

The EXP0260 exact A16 conversion/HVX copy path is enabled for FP16 stored weights. Scalar expf, ordered FP32 reduction/division and both FP16 probability roundings are preserved. FP16 head stages the next weight group while the current HMX/argmax completes. Only the final consumed prompt row is normalized for the head. Backbone M64 carrier computation has not been deeply redesigned; intrinsic FP16 weight bandwidth remains a real cost. No floating approximation change, quantization, rotation, or changes to other recipes.

Implementation selector: QBH_F16F16_OPT=0 original, 2 exact decode only, 3 final candidate. Existing W4F16 flags keep their prior meanings. OPT3 is an eligible measured implementation; this report does not promote a Selected Baseline.

Source build: 6dd974ff2b1920c84e1bd86aa9663a098e849cf9; package manifest c5df4b14a06af3274063a228f1fe9fb5480064b0386d0af5c90e24476ce0d5d9. Full build/runtime seals, commands and hashes are retained. Historical W16 measurements are non-paired context, not the optimization denominator.

## Retained issues

No failed hardware or numerical attempt in this Llama refresh. The scientific Python environment and existing build scripts were reused. Six independently referenced selected-layer cases and a three-layer continuous replay passed before the full-model candidate. All attempts and exact outputs are retained.
