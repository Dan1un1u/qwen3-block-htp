# Matched A8 / uniform INT16 Down comparison

Model llama l32-0068; full 16 layers, fixed64+42, same binary and shared optimizations, FP32 residual, no rotation. Five short and ten formal alternating paired repeat10. No quality or Selected baseline promotion.

| Phase | A8 token/s | INT16 token/s | INT16/A8 wall | 95% paired CI |
|---|---:|---:|---:|---|
| prefill | 2399.0435 | 2351.5047 | 1.020216 | [1.0145496729672252, 1.0254243812132295] |
| decode | 46.3016 | 46.2212 | 1.001741 | [0.9966563938923672, 1.006151195345091] |

Only Down input qparam/LUT and required byte-MAC mode differ. Each arm verified against its own independent mathematical arithmetic reference; cross-recipe outputs need not match. Performance uses full Host wall, excludes cold loading and external tokenization. Historical unrelated rounds are not treated as a paired speedup.

## Focus modules
| Phase | Counter | A8 us/layer | INT16 us/layer | Ratio |
|---|---|---:|---:|---:|
| prefill | down_ticks | 163.6742 | 190.4563 | 1.1636307117320517 |
| prefill | gate_up_ticks | 340.7351 | 340.5498 | 0.999456120025416 |
| prefill | u8_attention_av_requant_ticks | 0.3310 | 0.3313 | 1.0009834775767115 |
| prefill | o_projection_ticks | 75.6446 | 75.6817 | 1.0004901448186443 |
| prefill | w4u8_gate_up_swiglu_worker_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_ready_wait_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_join_wait_ticks | 26.9146 | 26.8701 | 0.9983502960154327 |
| decode | down_ticks | 161.8463 | 162.3956 | 1.00339442699493 |
| decode | gate_up_ticks | 296.2461 | 296.6249 | 1.0012784967403732 |
| decode | u8_attention_av_requant_ticks | 2.2376 | 2.2382 | 1.0002535475001497 |
| decode | o_projection_ticks | 55.8713 | 55.9313 | 1.0010747936169864 |
| decode | w4u8_gate_up_swiglu_worker_ticks | 30.6698 | 31.1957 | 1.017144841546817 |
| decode | w4u8_gate_up_swiglu_ready_wait_ticks | 253.8869 | 253.7224 | 0.9993520874803277 |
| decode | w4u8_gate_up_swiglu_join_wait_ticks | 3.4657 | 3.5434 | 1.022429532422689 |

Per-layer values above are full-model totals divided by executed layer count, not standalone operator timings. Worker/wait counters may overlap and are not added to module Host-wall attribution.

## Evidence
formal-summary.json, modules.json, payload-fairness.json, postflight.json, immutable per-arm protocols/records and oracle summaries.


## Historical comparability audit

The historical Table A cells were not contemporaneous paired results. All new pairs use one binary, identical fixed prompts/decode inputs, weights and non-Down metadata, KV settings and shared optimizations. Ordinary Down uses the original affine U8 quantizer; uniform INT16 uses its audited symmetric signed16 LUT. Necessary Down quantizer and byte-MAC mode changes are intentional; identical clipping endpoints or cross-recipe outputs are not claimed.

The old A8 row was L32-0057; the old INT16 row was L32-0065 M. New arms both use the L32-0067 production fused mask0 path (no ablation scratch), explicit HVX lane handling that avoids vector-array stack materialization, valid-row AV handling and AV-to-O scale folding. The new A8 independent full16 reference observed zero clipped values under the original AV saturation contract. Both recipes pass own arithmetic; no new DSP algorithm was introduced in this experiment. Cross-run speed differences are not separately attributed to these changes without an old/new paired experiment.

Workbook A six scoped rows updated across the two experiments. H/I/J/K append-only new paired provenance and modules. Other models/recipes and B/C/D/E/F/G are unchanged. Fixed-length hardware measurements, not actual dataset-quality evaluations.
