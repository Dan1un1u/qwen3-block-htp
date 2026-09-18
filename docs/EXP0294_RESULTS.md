# EXP0294 complete: Qwen3-0.6B ordinary A8

Same native binary, frozen W4 weights/scales, FP32 residual mode2 and all non-Down boundaries. No rotation, new calibration, quality acceptance or baseline promotion. Only Down input representation and required one-pass/two-plane execution differ. Existing equivalent Gate/Up streams enabled on both.

| Recipe | Prefill TPS | Decode TPS | Prefill ms | Decode42 ms |
|---|---:|---:|---:|---:|
| SP2 | 3022.363201 | 92.918015 | 21.175483 | 452.011377 |
| A8 | 3074.858385 | 93.465647 | 20.813967 | 449.362963 |

Five short + ten balanced paired formal repeat10 rounds; repeat1 auxiliary. Complete embedding/all28/finalNorm/head/greedy/FastRPC wall, excluding cold load/tokenizer. Fixed same SP2 input trajectory; not named-dataset performance.

Independent selected0/14/27 and chain3 exact; full fixed and free-greedy A8, plus SP2 control, exact hidden/norm/KV/IDs/codes after reference correction. Prior reference failed at one layer24 K code: SDK rsqrt is a three-step approximation, not mathematically rounded rsqrt. Vendor ISA simulator supplies that operation only from reference-derived inputs. Hardware mathematics unchanged; no threshold relaxation. See reference_recovery.md and sdk-reference-provenance.json.
