# Qwen3-1.7B matched integer and scale-fusion ablations

M64 prefill +42 fixed-input decode. Native per-channel W4, uniform INT16 Down, no rotation, FP32 residual and RMSNorm, unchanged KV and high precision embedding. Same production layout, scheduling flags, workers and token trace within the model. Host wall includes embedding, all layers, final norm, LM head, greedy and FastRPC; excludes loading, tokenizer and ADB. No model-quality/PPL claim or baseline promotion.

F: vector floating Softmax/SwiGLU with fused necessary QDQ. I: existing integer nonlinear producers and standalone AV conversion. M: AV common multiplier and zero correction folded into O, where numerically eligible. F/I attributes the combined nonlinear implementation and associated boundary handling, not QDQ alone; I/M attributes only AV-to-O. Other fusion/layout evidence remains historical and is not relabeled as measured INT16.

Five short rounds and ten formal rounds, repeat10 each, warmup before measurement. Formal rounds alternate or rotate complete arms. All samples retained;95% intervals bootstrap the ten paired round means (20,000 draws). No independence assumption across the ten repeats inside a round. Decode latency below is the complete42-token phase.

| Configuration | Prefill ms | Prefill token/s | Decode42 ms | Decode token/s |
|---|---:|---:|---:|---:|
| F | 52.352471 | 1222.48 | 897.085064 | 46.82 |
| I | 34.834264 | 1837.27 | 889.610297 | 47.21 |

| Comparison | Phase | Wall reduction | Optimized/control wall ratio,95%CI |
|---|---|---:|---|
| F->I | prefill | 33.4620% | 0.665380 [0.663989, 0.666446] |
| F->I | decode | 0.8332% | 0.991668 [0.988743, 0.993893] |

All 8,600 formal invocations pass the hardware status,8MiB/no intermediate DDR/no spill and exact additive ledger checks. Peak VTCM 8,365,824bytes. Single-layer,three-layer and full-model audits use independent own-contract references. Each full arm checks 1,204 layer outputs plus43 head token/code pairs. The FP component exhaustively checks all 1,835,008 Gate/Up input code pairs. Own-contract correctness does not assert F/I or I/M bit equality.

Qwen unconditional AV folding is rejected before timing: 1,310/6,078,464 candidate AV values exceed the original U8 output range, including6 in the first layer on identical inputs. Saturation cannot commute through O. Candidate package/reference are preserved as negative evidence, not a valid speedup arm. No tolerance or saturation rule was relaxed.

Qwen starts from the verified909 recovered EXP0284 payloads. The32 retired old chain arrays are explicitly excluded according to the prior recovery record. New own references are recomputed. Only the expected-token capacity metadata was extended from16 to64 in control-v2; the16 historical fixed inputs are continued by repeating the final frozen token. Arithmetic payloads are unchanged.

## Scope of performance evidence

These are end-to-end comparisons with the same production overlap. Component counters below are worker or substage observations; they must not be stacked as additional Host-wall percentages. In particular, eliminated AV conversion is not guaranteed to reduce the full critical path by its entire duration.

| Phase | Arm | AV RQ µs | Softmax worker µs | SwiGLU worker µs | O stage µs |
|---|---|---:|---:|---:|---:|
| prefill | F | 1651.132 | 11147.477 | 0.000 | 2082.197 |
| prefill | I | 1649.068 | 8717.161 | 0.000 | 2085.917 |
| decode | F | 148.148 | 320.661 | 2148.802 | 1368.705 |
| decode | I | 147.825 | 358.372 | 541.677 | 1368.913 |

Per-module additive tables: MODULES.md and modules.json. Raw profiles, commands, build seals, independent references and failed attempts remain beside this report.

The legacy SP2 mode8 flag selects the shared byte-carrier execution path; this experiment uses the verified uniform INT16 LUT/scale package. It is not a SP2-grid measurement. Complete formal profile comparisons confirm equal matrix commands/tile work, weight traffic, DMA descriptors, VTCM peak and producer-release counts. See counter-invariants.json and protocol-invariants.json.
