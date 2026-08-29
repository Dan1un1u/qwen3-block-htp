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
