# Matched A8 / uniform INT16 Down comparison

Model qwen 0.6B; full 28 layers, fixed64+42, same binary and shared optimizations, FP32 residual, no rotation. Five short and ten formal alternating paired repeat10. No quality or Selected baseline promotion.

| Phase | A8 token/s | INT16 token/s | INT16/A8 wall | 95% paired CI |
|---|---:|---:|---:|---|
| prefill | 3074.9857 | 3024.7046 | 1.016623 | [1.0141288615930535, 1.0185800932400018] |
| decode | 94.9080 | 94.3276 | 1.006153 | [1.0031854142588612, 1.0095694799967885] |

Only Down input qparam/LUT and required byte-MAC mode differ. Each arm verified against its own independent mathematical arithmetic reference; cross-recipe outputs need not match. Performance uses full Host wall, excludes cold loading and external tokenization. Historical unrelated rounds are not treated as a paired speedup.

## Focus modules
| Phase | Counter | A8 us/layer | INT16 us/layer | Ratio |
|---|---|---:|---:|---:|
| prefill | down_ticks | 46.3130 | 57.1749 | 1.2345327006147115 |
| prefill | gate_up_ticks | 95.9930 | 96.9838 | 1.0103220915780429 |
| prefill | u8_attention_av_requant_ticks | 61.2711 | 61.2858 | 1.0002389239617662 |
| prefill | o_projection_ticks | 43.8016 | 43.5062 | 0.9932570966530943 |
| prefill | w4u8_gate_up_swiglu_worker_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_ready_wait_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_join_wait_ticks | 26.5590 | 27.8597 | 1.0489741661028678 |
| decode | down_ticks | 38.1420 | 40.0925 | 1.0511363643620815 |
| decode | gate_up_ticks | 63.8973 | 63.7562 | 0.9977916828415041 |
| decode | u8_attention_av_requant_ticks | 5.3645 | 5.3659 | 1.0002487472443173 |
| decode | o_projection_ticks | 27.4847 | 27.3686 | 0.9957774956913026 |
| decode | w4u8_gate_up_swiglu_worker_ticks | 9.6877 | 9.9518 | 1.0272617345774877 |
| decode | w4u8_gate_up_swiglu_ready_wait_ticks | 47.2830 | 47.1287 | 0.996737466817654 |
| decode | w4u8_gate_up_swiglu_join_wait_ticks | 3.4235 | 3.4979 | 1.0217324822175375 |

Per-layer values above are full-model totals divided by executed layer count, not standalone operator timings. Worker/wait counters may overlap and are not added to module Host-wall attribution.

## Evidence
formal-summary.json, modules.json, payload-fairness.json, postflight.json, immutable per-arm protocols/records and oracle summaries.


# Matched A8 / uniform INT16 Down comparison

Model qwen 1.7B; full 28 layers, fixed64+42, same binary and shared optimizations, FP32 residual, no rotation. Five short and ten formal alternating paired repeat10. No quality or Selected baseline promotion.

| Phase | A8 token/s | INT16 token/s | INT16/A8 wall | 95% paired CI |
|---|---:|---:|---:|---|
| prefill | 1860.8790 | 1843.1446 | 1.009622 | [1.0067109308369806, 1.011909708047124] |
| decode | 47.6464 | 47.5358 | 1.002325 | [0.9976139814975333, 1.0057504861969488] |

Only Down input qparam/LUT and required byte-MAC mode differ. Each arm verified against its own independent mathematical arithmetic reference; cross-recipe outputs need not match. Performance uses full Host wall, excludes cold loading and external tokenization. Historical unrelated rounds are not treated as a paired speedup.

## Focus modules
| Phase | Counter | A8 us/layer | INT16 us/layer | Ratio |
|---|---|---:|---:|---:|
| prefill | down_ticks | 128.6375 | 138.1714 | 1.0741147124957524 |
| prefill | gate_up_ticks | 255.3512 | 254.8626 | 0.9980862755999156 |
| prefill | u8_attention_av_requant_ticks | 58.8888 | 58.9298 | 1.0006971247316292 |
| prefill | o_projection_ticks | 74.8140 | 75.1792 | 1.0048816484924674 |
| prefill | w4u8_gate_up_swiglu_worker_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_ready_wait_ticks | 0.0000 | 0.0000 | None |
| prefill | w4u8_gate_up_swiglu_join_wait_ticks | 14.0195 | 14.3123 | 1.0208879536359874 |
| decode | down_ticks | 125.2197 | 127.1356 | 1.0152996279142326 |
| decode | gate_up_ticks | 229.0130 | 228.8803 | 0.999420531693109 |
| decode | u8_attention_av_requant_ticks | 5.3496 | 5.3481 | 0.9997359022729126 |
| decode | o_projection_ticks | 49.1011 | 49.1309 | 1.0006068843886997 |
| decode | w4u8_gate_up_swiglu_worker_ticks | 19.5662 | 20.3613 | 1.0406399547077019 |
| decode | w4u8_gate_up_swiglu_ready_wait_ticks | 195.8542 | 194.9550 | 0.9954089759760573 |
| decode | w4u8_gate_up_swiglu_join_wait_ticks | 3.2932 | 3.3710 | 1.023625680440603 |

Per-layer values above are full-model totals divided by executed layer count, not standalone operator timings. Worker/wait counters may overlap and are not added to module Host-wall attribution.

## Evidence
formal-summary.json, modules.json, payload-fairness.json, postflight.json, immutable per-arm protocols/records and oracle summaries.


## Historical comparability audit

The historical Table A cells were not contemporaneous paired results. All new pairs use one binary, identical fixed prompts/decode inputs, weights and non-Down metadata, KV settings and shared optimizations. Ordinary Down uses the original affine U8 quantizer; uniform INT16 uses its audited symmetric signed16 LUT. Necessary Down quantizer and byte-MAC mode changes are intentional; identical clipping endpoints or cross-recipe outputs are not claimed.

Historical ordinary A8 already used U8_PREFILL_OPT3, matching INT16 mode8 Gate/Up publication and stream scheduling. No missing shared prefill optimization was found in the two Qwen paths. Original AV requantization is retained equally, respecting the prior saturation rejection. Qwen1.7 historical INT16 EXP0307 used a different decode token trace from A-table ordinary A8; both new arms use the original A fixed trace. Prefill inputs were already the same, so this trace correction is not claimed as an explanation for prefill gains. Prefix KV is enabled equally for1.7 and disabled equally for0.6. Retired0.6 payload recovered from the device matches all1016 original file hashes; no historical hash was replaced.

Workbook A six scoped rows updated across the two experiments. H/I/J/K append-only new paired provenance and modules. Other models/recipes and B/C/D/E/F/G are unchanged. Fixed-length hardware measurements, not actual dataset-quality evaluations.
