# EXP-0220 complete profiling comparison

Same frozen ABI108 binary, independent original versus R2-only token sequences. One M64 plus15 feedback steps per session; five short then ten alternating formal rounds. Frozen other recipes are historical nonpaired columns; no activation-only attribution with different W4 values. Quality scoring disabled.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 R2-only EXP-0220 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 376.3 (0.60%) | 247.4 (0.63%) | +52.12% |
| Input RMSNorm | 489.7 (0.61%) | 492.1 (0.78%) | 554.0 (1.40%) | -11.17% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11716.4 (18.61%) | 7052.7 (17.82%) | +66.13% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3962.8 (6.30%) | 3214.5 (8.12%) | +23.28% |
| O projection | 5757.7 (7.14%) | 5019.1 (7.97%) | 1256.4 (3.17%) | +299.48% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.9 (0.75%) | 654.0 (1.65%) | -27.54% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22381.2 (35.56%) | 14442.7 (36.49%) | +54.97% |
| Down | 13447.9 (16.67%) | 8559.6 (13.60%) | 3428.5 (8.66%) | +149.66% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.85% |
| KV carrier conversion | 174.0 (0.22%) | 172.0 (0.27%) | 203.2 (0.51%) | -15.35% |
| KV append DMA | 343.6 (0.43%) | 338.8 (0.54%) | 463.9 (1.17%) | -26.96% |
| Block orchestration | 16.1 (0.02%) | 19.0 (0.03%) | 34.6 (0.09%) | -44.95% |
| Layer bookkeeping | 23.9 (0.03%) | 23.9 (0.04%) | 23.2 (0.06%) | +3.37% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.4 (0.01%) | 22.5 (0.06%) | -62.73% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 95.1 (0.15%) | 105.6 (0.27%) | -9.96% |
| Embedding | 68.1 (0.08%) | 67.6 (0.11%) | 62.4 (0.16%) | +8.26% |
| Final model RMSNorm | 49.7 (0.06%) | 48.2 (0.08%) | 3.7 (0.01%) | +1194.41% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6735.4 (10.70%) | 5284.7 (13.35%) | +27.45% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2334.1 (3.71%) | 2466.6 (6.23%) | -5.37% |
| 完整 Host wall | 80692.2 (100.00%) | 62947.1 (100.00%) | 39575.9 (100.00%) | +59.05% |


All numeric fields are retained below. Additive timing ledger fields are mutually exclusive; engine-work, waits and diagnostic counters overlap and must not be summed. Host-DSP boundary is computed for each record as Host wall minus DSP invocation. Relative changes below are candidate/control minus one, so positive timing changes mean slower. Zero control denominators are N/A, not omitted evidence.

## prefill

| Field | R1 original | R1 R2-only | R1 change | R10 original median | R10 R2-only median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.000000 | 12.000000 | +33.3333% | 12.500000 | 12.500000 | +0.0000% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 76375.000000 | 75919.000000 | -0.5971% | 75867.500000 | 75910.000000 | +0.0560% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 124.000000 | 119.000000 | -4.0323% | 122.000000 | 122.000000 | +0.0000% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76554.000000 | 76094.000000 | -0.6009% | 76043.500000 | 76085.000000 | +0.0546% |
| attention_unattributed_ticks | 55.000000 | 56.000000 | +1.8182% | 56.000000 | 56.500000 | +0.8929% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 367.000000 | 367.000000 | +0.0000% | 367.500000 | 365.500000 | -0.5442% |
| boundary_ddr_read_bytes | 1423616.000000 | 1423616.000000 | +0.0000% | 1423616.000000 | 1423616.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 233.000000 | 233.000000 | +0.0000% | 233.000000 | 233.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 164389.000000 | 164605.000000 | +0.1314% | 164108.000000 | 164344.000000 | +0.1438% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9712.000000 | 9722.000000 | +0.1030% | 9734.500000 | 9781.500000 | +0.4828% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2700.000000 | 2684.000000 | -0.5926% | 2691.500000 | 2687.500000 | -0.1486% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 429347.000000 | 428365.000000 | -0.2287% | 429572.000000 | 429705.000000 | +0.0310% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1303.000000 | 1314.000000 | +0.8442% | 1296.500000 | 1298.000000 | +0.1157% |
| generation_final_norm_ticks | 920.000000 | 921.000000 | +0.1087% | 921.000000 | 925.500000 | +0.4886% |
| generation_lm_head_argmax_ticks | 6570.000000 | 6837.000000 | +4.0639% | 6577.500000 | 6834.500000 | +3.9073% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128964.000000 | 129298.000000 | +0.2590% | 128820.500000 | 129320.500000 | +0.3881% |
| generation_lm_head_expand_ticks | 111628.000000 | 111726.000000 | +0.0878% | 111482.000000 | 111744.500000 | +0.2355% |
| generation_lm_head_hmx_tail_wait_ticks | 2499.000000 | 2401.000000 | -3.9216% | 2417.000000 | 2333.500000 | -3.4547% |
| generation_lm_head_hmx_ticks | 127958.000000 | 128283.000000 | +0.2540% | 127813.500000 | 128301.500000 | +0.3818% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 237.000000 | 239.000000 | +0.8439% | 236.500000 | 239.000000 | +1.0571% |
| generation_lm_head_scale_init_ticks | 572.000000 | 553.000000 | -3.3217% | 570.500000 | 565.500000 | -0.8764% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129884.000000 | 130219.000000 | +0.2579% | 129741.000000 | 130240.500000 | +0.3850% |
| generation_lm_head_weight_dma_ticks | 128088.000000 | 128405.000000 | +0.2475% | 127944.500000 | 128416.000000 | +0.3685% |
| generation_lm_head_weight_dma_wait_ticks | 2737.000000 | 2903.000000 | +6.0650% | 2856.000000 | 2879.000000 | +0.8053% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 131817.000000 | 133308.000000 | +1.1311% | 130604.500000 | 129121.000000 | -1.1359% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2313.176750 | 2341.614333 | +1.2294% | 2331.276000 | 2334.062583 | +0.1195% |
| host_us | 62936.458000 | 62880.781000 | -0.0885% | 62953.151000 | 62947.057000 | -0.0097% |
| host_wall_ns | 62936458.000000 | 62880781.000000 | -0.0885% | 62953151.000000 | 62947057.000000 | -0.0097% |
| input_norm_ticks | 9482.000000 | 9442.000000 | -0.4219% | 9448.000000 | 9448.500000 | +0.0053% |
| input_stage_ticks | 10.000000 | 10.000000 | +0.0000% | 10.500000 | 9.500000 | -9.5238% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1163967.000000 | 1162352.000000 | -0.1387% | 1163837.000000 | 1163631.500000 | -0.0177% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 452.000000 | 432.000000 | -4.4248% | 457.000000 | 459.500000 | +0.5470% |
| ledger_named_ticks | 1163967.000000 | 1162352.000000 | -0.1387% | 1163837.000000 | 1163631.500000 | -0.0177% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7182.000000 | 7112.000000 | -0.9747% | 7165.500000 | 7216.500000 | +0.7117% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96698.000000 | 96346.000000 | -0.3640% | 96430.000000 | 96366.500000 | -0.0659% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 20.000000 | 22.000000 | +10.0000% | 23.000000 | 23.000000 | +0.0000% |
| post_attention_residual_ticks | 9086.000000 | 9090.000000 | +0.0440% | 9080.500000 | 9075.000000 | -0.0606% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682533.000000 | 680637.000000 | -0.2778% | 682602.000000 | 682648.000000 | +0.0067% |
| projection_pack_ticks | 3158.000000 | 3175.000000 | +0.5383% | 3165.500000 | 3169.000000 | +0.1106% |
| projection_unpack_ticks | 10969.000000 | 10895.000000 | -0.6746% | 10782.000000 | 10769.000000 | -0.1206% |
| qk_norm_rope_ticks | 26.000000 | 27.000000 | +3.8462% | 21.000000 | 22.500000 | +7.1429% |
| qkv_projection_ticks | 224714.000000 | 224584.000000 | -0.0579% | 224840.500000 | 224932.000000 | +0.0407% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 923.000000 | 881.000000 | -4.5504% | 907.500000 | 915.000000 | +0.8264% |
| runtime_teardown_ticks | 913.000000 | 838.000000 | -8.2147% | 895.500000 | 911.000000 | +1.7309% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6447.000000 | 6457.000000 | +0.1551% | 6464.500000 | 6505.000000 | +0.6265% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3295.000000 | 3291.000000 | -0.1214% | 3295.500000 | 3303.000000 | +0.2276% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 166.000000 | 160.000000 | -3.6145% | 163.000000 | 161.000000 | -1.2270% |
| total_ticks | 1163044.000000 | 1161471.000000 | -0.1352% | 1162926.000000 | 1162712.500000 | -0.0184% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 249276.000000 | 249664.000000 | +0.1557% | 249576.000000 | 249655.000000 | +0.0317% |
| w4f16_cross_prefetch_wait_ticks | 1127.000000 | 1383.000000 | +22.7152% | 1129.000000 | 1142.000000 | +1.1515% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75229.000000 | 75847.000000 | +0.8215% | 75477.500000 | 75753.000000 | +0.3650% |
| w4f16_expand_ticks | 615146.000000 | 615172.000000 | +0.0042% | 615275.000000 | 615298.000000 | +0.0037% |
| w4f16_expand_work_ticks | 1692989.000000 | 1694865.000000 | +0.1108% | 1693671.000000 | 1694889.000000 | +0.0719% |
| w4f16_gate_up_expand_pool_wait_ticks | 30575.000000 | 29849.000000 | -2.3745% | 30474.500000 | 30657.500000 | +0.6005% |
| w4f16_gate_up_expand_ticks | 252564.000000 | 252065.000000 | -0.1976% | 253031.500000 | 253162.000000 | +0.0516% |
| w4f16_gate_up_expand_work_ticks | 158678.000000 | 159179.000000 | +0.3157% | 159043.000000 | 159084.000000 | +0.0258% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7264.000000 | 7409.000000 | +1.9961% | 7221.000000 | 7434.500000 | +2.9567% |
| w4f16_gate_up_hmx_wait_ticks | 278294.000000 | 277300.000000 | -0.3572% | 278510.500000 | 278726.000000 | +0.0774% |
| w4f16_gate_up_stream_join_wait_ticks | 695.000000 | 872.000000 | +25.4676% | 707.000000 | 716.000000 | +1.2730% |
| w4f16_gate_up_stream_ready_wait_ticks | 1177.000000 | 1241.000000 | +5.4376% | 1332.500000 | 1284.500000 | -3.6023% |
| w4f16_gate_up_stream_work_ticks | 134371.000000 | 134431.000000 | +0.0447% | 134370.500000 | 134397.000000 | +0.0197% |
| w4f16_gate_up_weight_dma_ticks | 559124.000000 | 556547.000000 | -0.4609% | 559661.500000 | 559595.500000 | -0.0118% |
| w4f16_hmx_tail_wait_ticks | 19071.000000 | 18527.000000 | -2.8525% | 19130.000000 | 19311.000000 | +0.9462% |
| w4f16_prefetch_wait_ticks | 29063.000000 | 28220.000000 | -2.9006% | 29201.000000 | 29111.000000 | -0.3082% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1138809.000000 | 1135736.000000 | -0.2698% | 1138513.500000 | 1138608.500000 | +0.0083% |


## decode

| Field | R1 original | R1 R2-only | R1 change | R10 original median | R10 R2-only median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.666667 | 9.533333 | -1.3793% | 8.566667 | 9.400000 | +9.7276% |
| attention_av_hmx_ticks | 8188.800000 | 7953.866667 | -2.8690% | 7991.266667 | 7999.400000 | +0.1018% |
| attention_av_pack_ticks | 3045.666667 | 3030.266667 | -0.5056% | 3034.700000 | 3036.833333 | +0.0703% |
| attention_av_unpack_ticks | 4005.133333 | 4015.266667 | +0.2530% | 4009.933333 | 4008.066667 | -0.0466% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8280.733333 | 8031.133333 | -3.0142% | 8075.566667 | 8052.633333 | -0.2840% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6381.000000 | 6374.800000 | -0.0972% | 6380.566667 | 6374.766667 | -0.0909% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132649.133333 | 132824.933333 | +0.1325% | 132765.233333 | 132809.166667 | +0.0331% |
| attention_ticks | 644354.466667 | 643976.066667 | -0.0587% | 644082.966667 | 644058.366667 | -0.0038% |
| attention_unattributed_ticks | 481804.000000 | 481745.800000 | -0.0121% | 481801.233333 | 481775.366667 | -0.0054% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 280.133333 | 280.866667 | +0.2618% | 283.800000 | 283.666667 | -0.0470% |
| boundary_ddr_read_bytes | 1165568.000000 | 1165568.000000 | +0.0000% | 1165568.000000 | 1165568.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 170.000000 | 170.000000 | +0.0000% | 170.000000 | 170.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 164380.733333 | 164127.000000 | -0.1544% | 164398.600000 | 164285.700000 | -0.0687% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29890.066667 | 29881.200000 | -0.0297% | 29905.966667 | 29908.700000 | +0.0091% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2671.200000 | 2670.600000 | -0.0225% | 2671.366667 | 2671.166667 | -0.0075% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 428758.533333 | 427950.600000 | -0.1884% | 428414.266667 | 428345.366667 | -0.0161% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 38.866667 | 34.933333 | -10.1201% | 36.733333 | 34.700000 | -5.5354% |
| generation_final_norm_ticks | 41.066667 | 40.733333 | -0.8117% | 40.033333 | 39.733333 | -0.7494% |
| generation_lm_head_argmax_ticks | 6595.733333 | 6574.000000 | -0.3295% | 6596.733333 | 6571.700000 | -0.3795% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128895.400000 | 129016.066667 | +0.0936% | 128926.500000 | 128883.600000 | -0.0333% |
| generation_lm_head_expand_ticks | 111839.933333 | 111918.600000 | +0.0703% | 111867.633333 | 111860.066667 | -0.0068% |
| generation_lm_head_hmx_tail_wait_ticks | 2289.000000 | 2310.733333 | +0.9495% | 2288.500000 | 2286.200000 | -0.1005% |
| generation_lm_head_hmx_ticks | 127962.000000 | 128088.066667 | +0.0985% | 127999.500000 | 127956.366667 | -0.0337% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 211.066667 | 209.066667 | -0.9476% | 211.133333 | 210.533333 | -0.2842% |
| generation_lm_head_scale_init_ticks | 569.000000 | 572.200000 | +0.5624% | 566.833333 | 565.166667 | -0.2940% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128936.466667 | 129056.800000 | +0.0933% | 128966.900000 | 128923.366667 | -0.0338% |
| generation_lm_head_weight_dma_ticks | 128069.400000 | 128191.400000 | +0.0953% | 128101.666667 | 128059.466667 | -0.0329% |
| generation_lm_head_weight_dma_wait_ticks | 2856.466667 | 2899.733333 | +1.5147% | 2856.433333 | 2860.833333 | +0.1540% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 123713.933333 | 124755.800000 | +0.8422% | 123851.866667 | 123729.133333 | -0.0991% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1998.770678 | 2017.808978 | +0.9525% | 2091.883525 | 1873.901050 | -10.4204% |
| host_us | 92696.555400 | 92613.656200 | -0.0894% | 92763.274133 | 92523.928833 | -0.2580% |
| host_wall_ns | 92696555.400000 | 92613656.200000 | -0.0894% | 92763274.133333 | 92523928.833333 | -0.2580% |
| input_norm_ticks | 9361.733333 | 9354.400000 | -0.0783% | 9362.233333 | 9361.633333 | -0.0064% |
| input_stage_ticks | 6.933333 | 8.866667 | +27.8846% | 7.733333 | 8.200000 | +6.0345% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741397.466667 | 1739440.266667 | -0.1124% | 1740912.233333 | 1740084.833333 | -0.0475% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 323.933333 | 319.733333 | -1.2966% | 322.066667 | 322.266667 | +0.0621% |
| ledger_named_ticks | 1741397.466667 | 1739440.266667 | -0.1124% | 1740912.233333 | 1740084.833333 | -0.0475% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 5675.333333 | 5619.133333 | -0.9903% | 5618.433333 | 5702.133333 | +1.4897% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96264.066667 | 96090.733333 | -0.1801% | 96201.433333 | 96092.233333 | -0.1135% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 18.666667 | 17.533333 | -6.0714% | 17.266667 | 17.533333 | +1.5444% |
| post_attention_residual_ticks | 9021.733333 | 9013.800000 | -0.0879% | 9019.666667 | 9017.366667 | -0.0255% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682071.733333 | 680496.333333 | -0.2310% | 681740.200000 | 681177.500000 | -0.0825% |
| projection_pack_ticks | 422.333333 | 417.866667 | -1.0576% | 422.433333 | 418.700000 | -0.8838% |
| projection_unpack_ticks | 10835.800000 | 10830.466667 | -0.0492% | 10808.433333 | 10798.333333 | -0.0934% |
| qk_norm_rope_ticks | 17.800000 | 18.133333 | +1.8727% | 17.466667 | 17.566667 | +0.5725% |
| qkv_projection_ticks | 224478.400000 | 224090.400000 | -0.1728% | 224414.033333 | 224279.033333 | -0.0602% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 618.533333 | 629.733333 | +1.8107% | 619.533333 | 621.900000 | +0.3820% |
| runtime_teardown_ticks | 576.000000 | 580.933333 | +0.8565% | 576.933333 | 577.533333 | +0.1040% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4254.066667 | 4239.666667 | -0.3385% | 4240.500000 | 4240.233333 | -0.0063% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21311.133333 | 21309.600000 | -0.0072% | 21323.766667 | 21327.933333 | +0.0195% |
| scan_cache_stage_ticks | 13793.200000 | 13717.066667 | -0.5520% | 13782.400000 | 13731.633333 | -0.3683% |
| scan_dynamic_attention_ticks | 644303.266667 | 643926.266667 | -0.0585% | 644033.633333 | 644007.500000 | -0.0041% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 39.066667 | 41.200000 | +5.4608% | 38.166667 | 36.766667 | -3.6681% |
| total_ticks | 1740778.933333 | 1738810.533333 | -0.1131% | 1740294.400000 | 1739466.233333 | -0.0476% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 833399.600000 | 833020.666667 | -0.0455% | 833007.200000 | 832995.600000 | -0.0014% |
| w4f16_cross_prefetch_wait_ticks | 1141.800000 | 1196.466667 | +4.7878% | 1173.833333 | 1176.366667 | +0.2158% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 76671.400000 | 76867.400000 | +0.2556% | 76810.900000 | 76869.900000 | +0.0768% |
| w4f16_expand_ticks | 614616.800000 | 614087.666667 | -0.0861% | 614591.333333 | 614457.300000 | -0.0218% |
| w4f16_expand_work_ticks | 1694007.800000 | 1694459.000000 | +0.0266% | 1694020.166667 | 1694145.433333 | +0.0074% |
| w4f16_gate_up_expand_pool_wait_ticks | 31420.066667 | 31197.600000 | -0.7080% | 31277.633333 | 31369.700000 | +0.2944% |
| w4f16_gate_up_expand_ticks | 252796.533333 | 252230.066667 | -0.2241% | 252506.733333 | 252534.900000 | +0.0112% |
| w4f16_gate_up_expand_work_ticks | 158103.533333 | 158005.333333 | -0.0621% | 158064.933333 | 158076.633333 | +0.0074% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7303.466667 | 7340.600000 | +0.5084% | 7227.000000 | 7256.466667 | +0.4077% |
| w4f16_gate_up_hmx_wait_ticks | 278259.200000 | 277490.866667 | -0.2761% | 277849.000000 | 277848.500000 | -0.0002% |
| w4f16_gate_up_stream_join_wait_ticks | 671.133333 | 696.600000 | +3.7946% | 675.333333 | 690.866667 | +2.3001% |
| w4f16_gate_up_stream_ready_wait_ticks | 1311.866667 | 1259.533333 | -3.9892% | 1330.133333 | 1320.933333 | -0.6917% |
| w4f16_gate_up_stream_work_ticks | 134361.866667 | 134398.466667 | +0.0272% | 134381.266667 | 134377.033333 | -0.0032% |
| w4f16_gate_up_weight_dma_ticks | 558201.333333 | 556596.266667 | -0.2875% | 557603.600000 | 557485.566667 | -0.0212% |
| w4f16_hmx_tail_wait_ticks | 19363.533333 | 19131.866667 | -1.1964% | 19151.500000 | 19132.433333 | -0.0996% |
| w4f16_prefetch_wait_ticks | 29250.800000 | 28542.600000 | -2.4211% | 28924.000000 | 28880.100000 | -0.1518% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1133970.866667 | 1131717.000000 | -0.1988% | 1133614.833333 | 1132924.900000 | -0.0609% |



## Identity, correctness and direct throughput

## Completed profiling

Original/R2 each pass one warmup, five alternating short rounds and ten alternating formal rounds. All 320 formal invocation ledgers and 8,960 layer ledgers close exactly with zero unattributed ticks. Every selected token agrees with its independently generated software sequence, including the candidate's first EOS. This fixed 16-step benchmark deliberately continues after EOS; R2 speed is a diagnostic of full-model execution, not usable generated-text throughput. Host wall covers the complete model token-in/token-out pass, excluding cold staging and separate WSL tokenizer/detokenizer work (retained in execution JSON).

Measured script/source state: 1bebf5dcc47e5304dfe93b69cd9f4187fae3dfca on codex/exp-0220-gamma-fold-attribution-r2-only. Frozen runtime source d981072513d06ed61731c14743c76ac6bc81617f. Full numeric original/R2 repeat-one and repeat-ten tables and signed changes are in full_profiling_report.md. Other recipe columns below are frozen EXP-0218 nonpaired references. R2 quality failed; this is not a promoted Selected Baseline or an activation-only comparison.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 R2-only EXP-0220 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 376.3 (0.60%) | 247.4 (0.63%) | +52.12% |
| Input RMSNorm | 489.7 (0.61%) | 492.1 (0.78%) | 554.0 (1.40%) | -11.17% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11716.4 (18.61%) | 7052.7 (17.82%) | +66.13% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3962.8 (6.30%) | 3214.5 (8.12%) | +23.28% |
| O projection | 5757.7 (7.14%) | 5019.1 (7.97%) | 1256.4 (3.17%) | +299.48% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.9 (0.75%) | 654.0 (1.65%) | -27.54% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22381.2 (35.56%) | 14442.7 (36.49%) | +54.97% |
| Down | 13447.9 (16.67%) | 8559.6 (13.60%) | 3428.5 (8.66%) | +149.66% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.85% |
| KV carrier conversion | 174.0 (0.22%) | 172.0 (0.27%) | 203.2 (0.51%) | -15.35% |
| KV append DMA | 343.6 (0.43%) | 338.8 (0.54%) | 463.9 (1.17%) | -26.96% |
| Block orchestration | 16.1 (0.02%) | 19.0 (0.03%) | 34.6 (0.09%) | -44.95% |
| Layer bookkeeping | 23.9 (0.03%) | 23.9 (0.04%) | 23.2 (0.06%) | +3.37% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.4 (0.01%) | 22.5 (0.06%) | -62.73% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 95.1 (0.15%) | 105.6 (0.27%) | -9.96% |
| Embedding | 68.1 (0.08%) | 67.6 (0.11%) | 62.4 (0.16%) | +8.26% |
| Final model RMSNorm | 49.7 (0.06%) | 48.2 (0.08%) | 3.7 (0.01%) | +1194.41% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6735.4 (10.70%) | 5284.7 (13.35%) | +27.45% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2334.1 (3.71%) | 2466.6 (6.23%) | -5.37% |
| 完整 Host wall | 80692.2 (100.00%) | 62947.1 (100.00%) | 39575.9 (100.00%) | +59.05% |


Direct E2E medians and paired speed changes:

```json
{
  "times": {
    "original": {
      "prefill_tokens": 64,
      "prefill_host_us": 62953.151,
      "prefill_tokens_per_second": 1016.6290167111731,
      "decode_tokens": 15,
      "decode_total_host_us": 1391449.112,
      "decode_tokens_per_second": 10.780128335731764
    },
    "r2_only": {
      "prefill_tokens": 64,
      "prefill_host_us": 62947.057,
      "prefill_tokens_per_second": 1016.7274381072335,
      "decode_tokens": 15,
      "decode_total_host_us": 1387858.9324999999,
      "decode_tokens_per_second": 10.808014884466655
    }
  },
  "paired_speed_percent": {
    "prefill": -0.011636250434343687,
    "decode": 0.29325381936629036
  }
}
```

Sub-percent timing changes are reported without a performance-improvement claim. No baseline is promoted.
