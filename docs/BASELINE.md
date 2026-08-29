# Selected Baseline — W4F16-EXP0038-NORMS

The user promoted this Project Variant on 2026-08-29. `PROJECT_STATUS.yaml` is
the authoritative pointer; this document records the complete runnable and
evidence identity.

## Identity

- Execution Unit: real Qwen3-1.7B layer-14 complete middle block, `M=64`.
- Numerical variant: W4F16 with signed symmetric per-output-channel W4 and
  FP16 activations on FP16 HMX.
- Source branch: `codex/exp-0038-crouton-native-boundaries`.
- Implementation commit: `db791a430631eaba8f90bf3d0e8914c4998a15f5`.
- Source head including the formal evidence validator:
  `24917512fc35adfd22f97b5123473bb6a7518fd0`.
- Formal evidence:
  `D:\llm_exp\results\qwen3-block-htp\exp0038\20260828T193721Z_db791a430631`.
- Retained binaries:
  `D:\llm_exp\models\qwen3-block-htp\exp0038\artifacts\db791a430631`.

## Runtime selection

The Baseline uses three W4F16 HVX workers, 32-tile regions,
`adaptive_down96_gate4_dma8_cross_prefetch`, `combined_hvx` Attention packing,
the four-context `gqa_qkv_overlap` Attention pipeline, and the four-context
`crouton_native_batch8` MLP with 64-vector chunks. Common operators use
`hvx_fp16`, residual mode is `hvx_fused_post_norm`, and the selected EXP-0038
boundary mode is exactly `norms`.

`qkv`, `av_to_o`, and `all` are not Baseline modes. The QKV and AV-to-O
candidates reduced their local boundaries but failed their cross-variant
complete-block speed gates.

## Formal result

The output hash is `f18b9abbe1487231`, byte-exact with the paired W4F16
control and independent implementation reference. The plan requests exactly
8,388,608 VTCM bytes, reports a peak plan of 8,171,008 bytes, uses one FastRPC
Execution Unit and one HMX owner, and records zero intermediate DDR, zero
spill/fill, and no QNN dependency.

Median Host wall is 2,661,771 ns per block at repeat one and 2,229,682.3 ns per
block at repeat ten. Against the same-run F16F16 `norms` comparator, W4F16 is
7.50% faster at repeat one and 10.39% faster at repeat ten. Against the inherited
W4F16 control without direct Norm stores, it is 0.61% and 0.58% faster. An
additional eleven-round paired W4F16 check measured median improvements of
0.48% and 1.05% at repeat one and repeat ten.

## Repeat-ten profiling

The primary steady-state Host result is 2,229.7 microseconds per block. The
exclusive DSP ledger contains 41,376.1 qtimer ticks, or approximately 2,155.0
microseconds at 19.2 MHz. The remaining approximately 74.7 microseconds is the
per-block Host/FastRPC envelope. At repeat one the Host result is 2,661.8
microseconds while the DSP ledger is approximately 2,230.1 microseconds; most
of the repeat-one versus repeat-ten difference is therefore invocation-envelope
amortization rather than a radically different DSP pipeline.

| Exclusive ledger stage | Median ticks/block | Approx. us | Ledger share |
| --- | ---: | ---: | ---: |
| Gate + Up + Crouton-native SwiGLU | 18,675.6 | 972.7 | 45.14% |
| Q/K/V projections | 8,432.3 | 439.2 | 20.38% |
| Down projection | 6,285.0 | 327.3 | 15.19% |
| O projection | 3,338.5 | 173.9 | 8.07% |
| GQA Attention pipeline | 2,700.5 | 140.7 | 6.53% |
| Input RMSNorm direct Crouton | 819.9 | 42.7 | 1.98% |
| Post-Attention residual/direct Norm handoff | 791.5 | 41.2 | 1.91% |
| Final residual | 95.4 | 5.0 | 0.23% |
| Runtime, metadata, I/O and remaining ledger | 238.0 | 12.4 | 0.58% |

Projection stages account for approximately 88.8% of the steady-state DSP
ledger. Gate/Up plus Down alone account for approximately 60.3%. Against the
same-run F16F16 `norms` comparator, W4F16 is 10.59% slower in QKV but 13.70%
faster in O, 12.88% faster in Gate/Up, and 28.82% faster in Down. Attention and
the shared common operators are essentially numerical-variant neutral.

Engine work counters are overlapping diagnostics and must not be added to the
ledger. Per block they report about 25.18 MB of W4 weight DDR reads, 208 HMX
commands, 98,816 FP16 HMX tile pairs, 41,467.9 weight-DMA ticks, 24,863.8 W4
expansion interval ticks, 51,057.7 aggregate expansion-work ticks, 4,240.9 HMX
compute ticks and 27,173.5 projection HMX-wait ticks. Gate/Up alone reports
25,238.7 weight-DMA ticks, 11,769.0 expansion ticks, 12,615.2 HMX-wait ticks and
48 HMX commands. These counters show that compressed-weight delivery,
expansion/readiness and command completion dominate the projection critical
path; raw HMX arithmetic is not the sole limiting resource.
