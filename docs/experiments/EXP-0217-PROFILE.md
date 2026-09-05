# EXP-0217 complete profiling closure

F16F16 owns tokenizer/prepared token IDs, FP16 embedding, 28 transformer layers, final RMSNorm, FP16 stored LM head and greedy feedback. Ten independent M64+15 sessions. No equivalent earlier full-generation F16F16 control exists; control/candidate deltas are N/A, speed is report-only. R1 is the first formal session; R10 is the median of all ten sessions. Each RPC repeat_count=1.

{
  "source_commit": "4cf5512ac483cc440fa2ff02926b06a0e4f093b5",
  "binary_build_commit": "c3c1e708c27528c627245677c9207b1e6479b69b",
  "scope": "M64 prompt + 15 continuous decode passes; 16 selected tokens",
  "same_scope_control": null,
  "rounds": 10
}

Token IDs match the independent Transformers FP16 reference in all 160 comparisons. This does not prove full-logit equality or general quality. Generation profile hidden/cache error fields with zero compared elements are placeholders, not independent tensor comparisons. All-layer cache lengths and timed DDR/residency are checked. The unchanged F16 transformer retains its previously approved numerical contract. Dedicated transformer replay regression is archived separately.

All 160 invocation and 4480 per-layer exclusive ledgers close exactly. Final norm is subtracted from the inclusive LM-head field for additive reporting. Engine timings may overlap. Individual row medians need not sum to Host median.

Three-recipe table: F16F16 EXP-0217 current full-generation measurement; W4F16 selected EXP-0166 and W4U8 EXP-0216 are non-paired historical references. Historical W4U8 decode uses 192 steps and is not an equivalent 15-step ranking.

| 模块 | F16F16 EXP-0217 | W4F16 EXP-0166 | W4U8 EXP-0216 | W4U8 相对 W4F16 增速 |
|---|---|---|---|---|
| I/O、metadata | 98.5 (0.12%) | 366.8 (0.58%) | 238.8 (0.61%) | +53.64% |
| Input RMSNorm | 490.3 (0.60%) | 493.9 (0.78%) | 559.2 (1.42%) | -11.67% |
| QKV＋Q/K Norm-RoPE | 11542.9 (14.22%) | 11772.6 (18.69%) | 7033.0 (17.90%) | +67.39% |
| QK–Softmax–AV | 3981.5 (4.90%) | 3974.7 (6.31%) | 3233.3 (8.23%) | +22.93% |
| O projection | 5805.4 (7.15%) | 5044.9 (8.01%) | 1240.3 (3.16%) | +306.76% |
| Post-attention residual＋RMSNorm | 473.7 (0.58%) | 471.8 (0.75%) | 658.4 (1.68%) | -28.34% |
| Gate/Up＋SwiGLU | 29858.5 (36.78%) | 22459.5 (35.66%) | 14296.8 (36.38%) | +57.09% |
| Down | 13637.2 (16.80%) | 8516.2 (13.52%) | 3339.1 (8.50%) | +155.04% |
| Final residual | 140.8 (0.17%) | 140.3 (0.22%) | 184.7 (0.47%) | -24.05% |
| KV carrier conversion | 175.8 (0.22%) | 148.0 (0.24%) | 205.4 (0.52%) | -27.96% |
| KV append DMA | 346.2 (0.43%) | 338.9 (0.54%) | 455.9 (1.16%) | -25.66% |
| Block orchestration | 16.7 (0.02%) | 16.8 (0.03%) | 34.1 (0.09%) | -50.84% |
| Layer bookkeeping | 23.6 (0.03%) | 22.1 (0.04%) | 21.1 (0.05%) | +4.43% |
| Stage-boundary bookkeeping | 7.7 (0.01%) | 4.4 (0.01%) | 27.0 (0.07%) | -83.77% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 83.5 (0.10%) | 99.9 (0.16%) | 109.8 (0.28%) | -9.06% |
| Embedding | 68.7 (0.08%) | 77.5 (0.12%) | 68.5 (0.17%) | +13.11% |
| Final model RMSNorm | 48.8 (0.06%) | 48.3 (0.08%) | 4.7 (0.01%) | +925.41% |
| LM head＋greedy，不含 final norm | 11915.7 (14.68%) | 6702.0 (10.64%) | 5223.5 (13.29%) | +28.30% |
| Host–DSP 边界 | 2577.9 (3.18%) | 2288.5 (3.63%) | 2311.4 (5.88%) | -0.99% |
| 完整 Host wall | 81182.8 (100.00%) | 62974.7 (100.00%) | 39296.5 (100.00%) | +60.26% |


## prefill complete exported diagnostics

Every numeric exported field is retained. *_ticks are qtimer ticks (19.2 ticks/us), *_us are microseconds; other fields preserve raw units. Nested layer ledgers are verified in every raw record. No same-scope F16 control exists.

| Field | R1 | R10 median | Control/delta |
|---|---|---|---|
| activation_ticks | 9.000000 | 10.000000 | N/A: new integration scope |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_gqa_pipeline_ticks | 76324.000000 | 76245.000000 | N/A: new integration scope |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_setup_ticks | 123.000000 | 119.000000 | N/A: new integration scope |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_ticks | 76523.000000 | 76445.000000 | N/A: new integration scope |
| attention_unattributed_ticks | 76.000000 | 82.500000 | N/A: new integration scope |
| block_invocation_count | 28.000000 | 28.000000 | N/A: new integration scope |
| block_orchestration_ticks | 310.000000 | 320.000000 | N/A: new integration scope |
| boundary_ddr_read_bytes | 1423616.000000 | 1423616.000000 | N/A: new integration scope |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| boundary_dma_descriptor_count | 233.000000 | 233.000000 | N/A: new integration scope |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | N/A: new integration scope |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | N/A: new integration scope |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_min_cosine | 1.000000 | 1.000000 | N/A: new integration scope |
| cache_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: new integration scope |
| down_ticks | 260463.000000 | 261834.500000 | N/A: new integration scope |
| dsp_status | 3.000000 | 3.000000 | N/A: new integration scope |
| experiment | 217.000000 | 217.000000 | N/A: new integration scope |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: new integration scope |
| f16_cache_native_append_update_ticks | 9915.000000 | 9963.500000 | N/A: new integration scope |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | N/A: new integration scope |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | N/A: new integration scope |
| final_residual_ticks | 2705.000000 | 2703.000000 | N/A: new integration scope |
| first_position | 0.000000 | 0.000000 | N/A: new integration scope |
| gate_up_ticks | 571491.000000 | 573274.000000 | N/A: new integration scope |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | N/A: new integration scope |
| generation_embedding_ticks | 1297.000000 | 1318.500000 | N/A: new integration scope |
| generation_final_norm_ticks | 939.000000 | 937.000000 | N/A: new integration scope |
| generation_lm_head_argmax_ticks | 6657.000000 | 7239.500000 | N/A: new integration scope |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | N/A: new integration scope |
| generation_lm_head_command_count | 594.000000 | 594.000000 | N/A: new integration scope |
| generation_lm_head_ddr_read_bytes | 622329856.000000 | 622329856.000000 | N/A: new integration scope |
| generation_lm_head_exclusive_ticks | 227639.000000 | 228781.500000 | N/A: new integration scope |
| generation_lm_head_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_hmx_ticks | 22067.000000 | 21352.000000 | N/A: new integration scope |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | N/A: new integration scope |
| generation_lm_head_prefetch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_dma_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_init_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_resident_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_ticks | 228578.000000 | 229722.000000 | N/A: new integration scope |
| generation_lm_head_weight_dma_ticks | 198514.000000 | 199828.000000 | N/A: new integration scope |
| generation_lm_head_weight_dma_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_step | 0.000000 | 0.000000 | N/A: new integration scope |
| hmx_command_count | 6418.000000 | 6418.000000 | N/A: new integration scope |
| hmx_compute_ticks | 140336.000000 | 139980.000000 | N/A: new integration scope |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | N/A: new integration scope |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: new integration scope |
| host_boundary_us | 2578.854500 | 2577.917083 | N/A: new integration scope |
| host_us | 81022.292000 | 81182.838500 | N/A: new integration scope |
| host_wall_ns | 81022292.000000 | 81182838.500000 | N/A: new integration scope |
| input_norm_ticks | 9406.000000 | 9413.000000 | N/A: new integration scope |
| input_stage_ticks | 7.000000 | 5.000000 | N/A: new integration scope |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: new integration scope |
| invocation_ticks | 1506114.000000 | 1512482.500000 | N/A: new integration scope |
| kv_cache_k_format | 4.000000 | 4.000000 | N/A: new integration scope |
| kv_cache_v_format | 5.000000 | 5.000000 | N/A: new integration scope |
| layer_bookkeeping_ticks | 487.000000 | 453.000000 | N/A: new integration scope |
| ledger_named_ticks | 1506114.000000 | 1512482.500000 | N/A: new integration scope |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| logical_m | 64.000000 | 64.000000 | N/A: new integration scope |
| metadata_stage_ticks | 1883.000000 | 1887.500000 | N/A: new integration scope |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: new integration scope |
| numerical_status | 1.000000 | 1.000000 | N/A: new integration scope |
| o_projection_ticks | 111434.000000 | 111464.000000 | N/A: new integration scope |
| output_cosine | 1.000000 | 1.000000 | N/A: new integration scope |
| output_fp16_atol | 0.062500 | 0.062500 | N/A: new integration scope |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | N/A: new integration scope |
| output_fp16_rtol | 0.002000 | 0.002000 | N/A: new integration scope |
| output_max_abs | 0.000000 | 0.000000 | N/A: new integration scope |
| output_max_lsb | 0.000000 | 0.000000 | N/A: new integration scope |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: new integration scope |
| output_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: new integration scope |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: new integration scope |
| output_nrmse | 0.000000 | 0.000000 | N/A: new integration scope |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| post_attention_norm_ticks | 27.000000 | 26.500000 | N/A: new integration scope |
| post_attention_residual_ticks | 9052.000000 | 9068.000000 | N/A: new integration scope |
| prepared_session_run_index | 1.000000 | 1.000000 | N/A: new integration scope |
| projection_failure_index | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_result | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_step | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_hmx_wait_ticks | 337396.000000 | 329920.500000 | N/A: new integration scope |
| projection_pack_ticks | 3198.000000 | 3142.500000 | N/A: new integration scope |
| projection_unpack_ticks | 103126.000000 | 104063.000000 | N/A: new integration scope |
| qk_norm_rope_ticks | 27.000000 | 27.000000 | N/A: new integration scope |
| qkv_projection_ticks | 220792.000000 | 221596.000000 | N/A: new integration scope |
| repeat_count | 1.000000 | 1.000000 | N/A: new integration scope |
| runtime_setup_ticks | 795.000000 | 808.500000 | N/A: new integration scope |
| runtime_teardown_ticks | 760.000000 | 806.500000 | N/A: new integration scope |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_cache_append_ticks | 6608.000000 | 6648.000000 | N/A: new integration scope |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | N/A: new integration scope |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | N/A: new integration scope |
| scan_cache_pack_ticks | 3330.000000 | 3375.500000 | N/A: new integration scope |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_logical_m_observed | 64.000000 | 64.000000 | N/A: new integration scope |
| scan_padded_kv_length | 64.000000 | 64.000000 | N/A: new integration scope |
| scan_total_kv_length | 64.000000 | 64.000000 | N/A: new integration scope |
| stage_boundary_ticks | 130.000000 | 147.500000 | N/A: new integration scope |
| total_ticks | 1505319.000000 | 1511686.500000 | N/A: new integration scope |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| valid_length | 64.000000 | 64.000000 | N/A: new integration scope |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | N/A: new integration scope |
| vtcm_peak_plan_bytes | 6875904.000000 | 6875904.000000 | N/A: new integration scope |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | N/A: new integration scope |
| w4f16_cross_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_cross_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_hmx_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_join_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_weight_dma_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | N/A: new integration scope |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| weight_ddr_read_bytes | 3440906240.000000 | 3440906240.000000 | N/A: new integration scope |
| weight_dma_descriptor_count | 5522.000000 | 5522.000000 | N/A: new integration scope |
| weight_dma_ticks | 1283672.000000 | 1289640.000000 | N/A: new integration scope |


## decode complete exported diagnostics

Every numeric exported field is retained. *_ticks are qtimer ticks (19.2 ticks/us), *_us are microseconds; other fields preserve raw units. Nested layer ledgers are verified in every raw record. No same-scope F16 control exists.

| Field | R1 | R10 median | Control/delta |
|---|---|---|---|
| activation_ticks | 5.800000 | 6.600000 | N/A: new integration scope |
| attention_av_hmx_ticks | 8050.400000 | 7893.866667 | N/A: new integration scope |
| attention_av_pack_ticks | 3057.533333 | 3044.700000 | N/A: new integration scope |
| attention_av_unpack_ticks | 3975.666667 | 3980.733333 | N/A: new integration scope |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_qk_hmx_ticks | 8050.200000 | 7850.500000 | N/A: new integration scope |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_qk_unpack_ticks | 6386.600000 | 6397.933333 | N/A: new integration scope |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| attention_softmax_ticks | 134181.933333 | 134188.333333 | N/A: new integration scope |
| attention_ticks | 642668.200000 | 642307.033333 | N/A: new integration scope |
| attention_unattributed_ticks | 478965.866667 | 478915.466667 | N/A: new integration scope |
| block_invocation_count | 28.000000 | 28.000000 | N/A: new integration scope |
| block_orchestration_ticks | 248.800000 | 248.433333 | N/A: new integration scope |
| boundary_ddr_read_bytes | 1165568.000000 | 1165568.000000 | N/A: new integration scope |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| boundary_dma_descriptor_count | 170.000000 | 170.000000 | N/A: new integration scope |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | N/A: new integration scope |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | N/A: new integration scope |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_min_cosine | 1.000000 | 1.000000 | N/A: new integration scope |
| cache_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: new integration scope |
| down_ticks | 261714.200000 | 263231.500000 | N/A: new integration scope |
| dsp_status | 3.000000 | 3.000000 | N/A: new integration scope |
| experiment | 217.000000 | 217.000000 | N/A: new integration scope |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: new integration scope |
| f16_cache_native_append_update_ticks | 16776.533333 | 16751.200000 | N/A: new integration scope |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | N/A: new integration scope |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: new integration scope |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| final_residual_ticks | 2676.066667 | 2675.766667 | N/A: new integration scope |
| first_position | 71.000000 | 71.000000 | N/A: new integration scope |
| gate_up_ticks | 573025.200000 | 575367.100000 | N/A: new integration scope |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | N/A: new integration scope |
| generation_embedding_ticks | 43.400000 | 43.833333 | N/A: new integration scope |
| generation_final_norm_ticks | 57.866667 | 57.300000 | N/A: new integration scope |
| generation_lm_head_argmax_ticks | 7035.133333 | 7311.433333 | N/A: new integration scope |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | N/A: new integration scope |
| generation_lm_head_command_count | 594.000000 | 594.000000 | N/A: new integration scope |
| generation_lm_head_ddr_read_bytes | 622329856.000000 | 622329856.000000 | N/A: new integration scope |
| generation_lm_head_exclusive_ticks | 228431.000000 | 229585.233333 | N/A: new integration scope |
| generation_lm_head_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_hmx_ticks | 21756.066667 | 21360.766667 | N/A: new integration scope |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | N/A: new integration scope |
| generation_lm_head_prefetch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_dma_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_init_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_scale_resident_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_lm_head_ticks | 228488.866667 | 229640.000000 | N/A: new integration scope |
| generation_lm_head_weight_dma_ticks | 199316.400000 | 200602.000000 | N/A: new integration scope |
| generation_lm_head_weight_dma_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| generation_step | 8.000000 | 8.000000 | N/A: new integration scope |
| hmx_command_count | 5970.000000 | 5970.000000 | N/A: new integration scope |
| hmx_compute_ticks | 135572.800000 | 135152.433333 | N/A: new integration scope |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | N/A: new integration scope |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: new integration scope |
| host_boundary_us | 2102.267261 | 2333.824686 | N/A: new integration scope |
| host_us | 110297.312400 | 110760.529633 | N/A: new integration scope |
| host_wall_ns | 110297312.400000 | 110760529.633333 | N/A: new integration scope |
| input_norm_ticks | 9325.733333 | 9327.100000 | N/A: new integration scope |
| input_stage_ticks | 4.733333 | 5.500000 | N/A: new integration scope |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: new integration scope |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: new integration scope |
| invocation_ticks | 2077344.866667 | 2081823.633333 | N/A: new integration scope |
| kv_cache_k_format | 4.000000 | 4.000000 | N/A: new integration scope |
| kv_cache_v_format | 5.000000 | 5.000000 | N/A: new integration scope |
| layer_bookkeeping_ticks | 317.466667 | 318.466667 | N/A: new integration scope |
| ledger_named_ticks | 2077344.866667 | 2081823.633333 | N/A: new integration scope |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| logical_m | 1.000000 | 1.000000 | N/A: new integration scope |
| metadata_stage_ticks | 2135.733333 | 2034.633333 | N/A: new integration scope |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: new integration scope |
| numerical_status | 1.000000 | 1.000000 | N/A: new integration scope |
| o_projection_ticks | 112137.600000 | 112207.300000 | N/A: new integration scope |
| output_cosine | 1.000000 | 1.000000 | N/A: new integration scope |
| output_fp16_atol | 0.062500 | 0.062500 | N/A: new integration scope |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | N/A: new integration scope |
| output_fp16_rtol | 0.002000 | 0.002000 | N/A: new integration scope |
| output_max_abs | 0.000000 | 0.000000 | N/A: new integration scope |
| output_max_lsb | 0.000000 | 0.000000 | N/A: new integration scope |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: new integration scope |
| output_mismatches | 0.000000 | 0.000000 | N/A: new integration scope |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: new integration scope |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: new integration scope |
| output_nrmse | 0.000000 | 0.000000 | N/A: new integration scope |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| post_attention_norm_ticks | 16.933333 | 16.800000 | N/A: new integration scope |
| post_attention_residual_ticks | 9002.333333 | 9004.000000 | N/A: new integration scope |
| prepared_session_run_index | 9.000000 | 9.000000 | N/A: new integration scope |
| projection_failure_index | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_result | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_failure_step | 0.000000 | 0.000000 | N/A: new integration scope |
| projection_hmx_wait_ticks | 333990.200000 | 333686.900000 | N/A: new integration scope |
| projection_pack_ticks | 393.066667 | 393.366667 | N/A: new integration scope |
| projection_unpack_ticks | 100820.800000 | 101730.466667 | N/A: new integration scope |
| qk_norm_rope_ticks | 18.600000 | 18.733333 | N/A: new integration scope |
| qkv_projection_ticks | 222062.600000 | 222296.000000 | N/A: new integration scope |
| repeat_count | 1.000000 | 1.000000 | N/A: new integration scope |
| runtime_setup_ticks | 511.466667 | 510.133333 | N/A: new integration scope |
| runtime_teardown_ticks | 479.466667 | 477.500000 | N/A: new integration scope |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | N/A: new integration scope |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | N/A: new integration scope |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| scan_cache_append_ticks | 4346.200000 | 4342.600000 | N/A: new integration scope |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | N/A: new integration scope |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | N/A: new integration scope |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | N/A: new integration scope |
| scan_cache_pack_ticks | 8084.133333 | 8085.200000 | N/A: new integration scope |
| scan_cache_stage_ticks | 13856.466667 | 13820.800000 | N/A: new integration scope |
| scan_dynamic_attention_ticks | 642609.333333 | 642247.866667 | N/A: new integration scope |
| scan_logical_m_observed | 1.000000 | 1.000000 | N/A: new integration scope |
| scan_padded_kv_length | 96.000000 | 96.000000 | N/A: new integration scope |
| scan_total_kv_length | 72.000000 | 72.000000 | N/A: new integration scope |
| stage_boundary_ticks | 31.333333 | 31.100000 | N/A: new integration scope |
| total_ticks | 2076833.400000 | 2081310.600000 | N/A: new integration scope |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: new integration scope |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: new integration scope |
| valid_length | 72.000000 | 72.000000 | N/A: new integration scope |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | N/A: new integration scope |
| vtcm_peak_plan_bytes | 6875904.000000 | 6875904.000000 | N/A: new integration scope |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | N/A: new integration scope |
| w4f16_cross_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_cross_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_expand_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_expand_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_hmx_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_join_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_stream_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_gate_up_weight_dma_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_hmx_tail_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4f16_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | N/A: new integration scope |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | N/A: new integration scope |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: new integration scope |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: new integration scope |
| weight_ddr_read_bytes | 3440906240.000000 | 3440906240.000000 | N/A: new integration scope |
| weight_dma_descriptor_count | 5522.000000 | 5522.000000 | N/A: new integration scope |
| weight_dma_ticks | 1288925.066667 | 1294107.333333 | N/A: new integration scope |


## Direct E2E throughput

| Mode | Tokens | Complete Host us | tok/s |
|---|---|---|---|
| prefill | 64 | 81182.838 | 788.343955 |
| decode | 15 | 1661407.944 | 9.028487 |


## Retained artifacts and regression closure

{
  "experiment": "EXP-0217",
  "sessions": 10,
  "selected_token_checks": 160,
  "independent_selected_token_mismatches": 0,
  "invocation_ledgers_checked": 160,
  "layer_ledgers_checked": 4480,
  "full_logit_quality_assessment": "not_run",
  "text": "低比特量化能够减少大模型推理的内存带宽，因为量化后",
  "prefill": {
    "tokens": 64,
    "complete_host_us": 81182.8385,
    "tokens_per_second": 788.3439552313757
  },
  "decode": {
    "tokens": 15,
    "complete_host_us": 1661407.9444999998,
    "tokens_per_second": 9.028486982776675
  },
  "final_source_commit": "bcea4e3b309a1f0e64422bc9dec189d78a9cc4d2",
  "artifact_directory": "/mnt/d/llm_exp/models/qwen3-block-htp/exp0217/artifacts/bcea4e3b309a1f0e64422bc9dec189d78a9cc4d2",
  "stable_device_selected_logit_bits": true,
  "transformer_regression": {
    "steps": 9,
    "all_pass": true,
    "output_min_cosine": 0.999997239,
    "cache_min_cosine": 0.999990536,
    "cache_prefix_mismatches": 0,
    "source_commit": "bcea4e3b309a1f0e64422bc9dec189d78a9cc4d2",
    "log_sha256": "45616f1dd31efa1012e2e2038b94ffb48aacd92a545e7411ab2439310b655d01",
    "note": "Legacy transformer replay JSON experiment label 163 is inherited; binary ABI107 EXP217 verified by generation profiles and retained hashes."
  },
  "w4f16_regression": {
    "all_runs_exact": true,
    "generated_tokens": 16,
    "decoded_text": "低比特量化（Low-Bitwidth Quantization）在大模型推理中"
  },
  "general_quality_evaluation": "proposal_only_not_run",
  "evidence_validity": "valid",
  "local_gate": "pass",
  "selected_baseline_changed": false
}

Device SHA256 matched all three runtime binaries and the six model-boundary/manifest artifacts. The FP16 head and embedding are each 622329856 bytes. Full local package manifest audit verified 1079 files. All 160 selected token IDs match the independent FP16 reference; selected FP16 logit bits are stable across device sessions, without claiming independent full-logit equality. Existing transformer replay passed nine steps; W4F16 generation passed its 16-token reference. No baseline promotion was performed.
