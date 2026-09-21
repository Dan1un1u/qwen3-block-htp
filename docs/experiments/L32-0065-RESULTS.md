# Llama3.2-1B matched integer and scale-fusion ablations

M64 prefill +42 fixed-input decode. Native per-channel W4, uniform INT16 Down, no rotation, FP32 residual and RMSNorm, unchanged KV and high precision embedding. Same production layout, scheduling flags, workers and token trace within the model. Host wall includes embedding, all layers, final norm, LM head, greedy and FastRPC; excludes loading, tokenizer and ADB. No model-quality/PPL claim or baseline promotion.

F: vector floating Softmax/SwiGLU with fused necessary QDQ. I: existing integer nonlinear producers and standalone AV conversion. M: AV common multiplier and zero correction folded into O, where numerically eligible. F/I attributes the combined nonlinear implementation and associated boundary handling, not QDQ alone; I/M attributes only AV-to-O. Other fusion/layout evidence remains historical and is not relabeled as measured INT16.

Five short rounds and ten formal rounds, repeat10 each, warmup before measurement. Formal rounds alternate or rotate complete arms. All samples retained;95% intervals bootstrap the ten paired round means (20,000 draws). No independence assumption across the ten repeats inside a round. Decode latency below is the complete42-token phase.

| Configuration | Prefill ms | Prefill token/s | Decode42 ms | Decode token/s |
|---|---:|---:|---:|---:|
| F | 41.204356 | 1553.23 | 913.127081 | 46.00 |
| I | 27.599084 | 2318.92 | 910.341852 | 46.14 |
| M | 27.641559 | 2315.35 | 906.631522 | 46.33 |

| Comparison | Phase | Wall reduction | Optimized/control wall ratio,95%CI |
|---|---|---:|---|
| F->I | prefill | 33.0190% | 0.669810 [0.669110, 0.670471] |
| I->M | prefill | -0.1539% | 1.001539 [0.999025, 1.004324] |
| F->M | prefill | 32.9159% | 0.670841 [0.669192, 0.672521] |
| F->I | decode | 0.3050% | 0.996950 [0.995522, 0.998377] |
| I->M | decode | 0.4076% | 0.995924 [0.993435, 0.998477] |
| F->M | decode | 0.7114% | 0.992886 [0.990916, 0.995158] |

All 12,900 formal invocations pass the hardware status,8MiB/no intermediate DDR/no spill and exact additive ledger checks. Peak VTCM 8,098,272bytes. Single-layer,three-layer and full-model audits use independent own-contract references. Each full arm checks 688 layer outputs plus43 head token/code pairs. The FP component exhaustively checks all 1,048,576 Gate/Up input code pairs. Own-contract correctness does not assert F/I or I/M bit equality.

The generic Llama scan fallback formerly honored live-row AV conversion only under the3B compile guard. This experiment uses the common audited4-row HVX granule for1B too and separates AV conversion timing from matrix timing. Padding-poison tests pass. The fair control removes most of the previously reported AV-fold speedup: prefill has no confirmed benefit; decode improves only about0.4%. Do not reuse the older~4% decode claim as this matched result.

Llama I/M reuse independently computed L32-0062 references under verified unchanged payloads. On the checked trajectories the fold bypasses no AV saturation. Floating scale reassociation can still change rounding and downstream quantized decisions; bitwise I/M equivalence and PPL acceptability are not claimed.

## Scope of performance evidence

These are end-to-end comparisons with the same production overlap. Component counters below are worker or substage observations; they must not be stacked as additional Host-wall percentages. In particular, eliminated AV conversion is not guaranteed to reduce the full critical path by its entire duration.

| Phase | Arm | AV RQ µs | Softmax worker µs | SwiGLU worker µs | O stage µs |
|---|---|---:|---:|---:|---:|
| prefill | F | 920.182 | 12302.795 | 0.000 | 1591.852 |
| prefill | I | 925.803 | 8570.534 | 0.000 | 1589.858 |
| prefill | M | 5.121 | 8531.442 | 0.000 | 1587.478 |
| decode | F | 106.340 | 332.988 | 1789.612 | 907.795 |
| decode | I | 109.616 | 343.482 | 485.475 | 904.898 |
| decode | M | 36.594 | 343.762 | 486.712 | 903.356 |

Per-module additive tables: MODULES.md and modules.json. Raw profiles, commands, build seals, independent references and failed attempts remain beside this report.

The legacy SP2 mode8 flag selects the shared byte-carrier execution path; this experiment uses the verified uniform INT16 LUT/scale package. It is not a SP2-grid measurement. Complete formal profile comparisons confirm equal matrix commands/tile work, weight traffic, DMA descriptors, VTCM peak and producer-release counts. See counter-invariants.json and protocol-invariants.json.
