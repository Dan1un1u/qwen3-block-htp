# EXP0327 complete profiling comparison

Full28 Qwen3-1.7B, W4A8 uniformINT16 Down, FP32 residual2, rounding2. Control R3 OPT1 old long QKV; candidate R3 OPT2 + all-chunk nativeW4; OFF uses same shared nativeW4 optimization with frozen OFF QK/prefix. Same native binary c163b3e, source closure may include documentation only. Fixed tokens,64 or2048 prefill and64 continuous decode; no greedy feedback divergence in timed inputs.5shortrepeat1,10formalrepeat10 rotated arm order. Early auxiliary short rounds0/1 may overlap OFF-package SHA verification and are not performance evidence; all formal samples start after verification. One fullmodel RPC/pass; Host includes staging, excludes coldload/ADB/tokenizer. No teacher-forcing PPL rerun.

Own deployed arithmetic preserved; exact1/3/full28 hashes and short184capture checks in correctness.json. Audit outputs,cacheguards and poison tests are separate from warm timing. Independent conditional component evidence inherited EXP0326; no ideal fullmodel equality claim.

All counters below retained; overlapping work/wait/engine counters are NOT additive. Stage ledger is exclusive and all per-invocation and per-layer unattributed fractions <=0.1%. Ticks at19.2MHz; totals per full prefill or64decode, VTCM maxima. No intermediateDDR or spill, oneHMX owner. Historical F16F16/W4F16 long measurements N/A; not extrapolated.

## short, 64+64, prefill

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|3458.489583|3423.854167|3466.197917|-1.001%|-1.222%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|block_orchestration_ticks|36.718750|37.135417|36.927083|+1.135%|+0.564%|
|boundary_ddr_read_bytes|5107072.000000|5107072.000000|5107072.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|289.000000|289.000000|289.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|917504.000000|917504.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|3303.281250|3310.468750|0.000000|+0.218%|N/A zero denominator|
|dense_r3_total_matmul_ticks|522.239583|524.531250|0.000000|+0.439%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|8325.000000|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|8079.687500|371.302083|0.000000|-95.404%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|3801.458333|3787.760417|3835.937500|-0.360%|-1.256%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|3.020833|2.760417|2.864583|-8.621%|-3.636%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|6948.125000|6937.291667|6935.729167|-0.156%|+0.023%|
|generation_embedding_ddr_read_bytes|262400.000000|262400.000000|262400.000000|+0.000%|+0.000%|
|generation_embedding_ticks|64.375000|69.739583|65.520833|+8.333%|+6.439%|
|generation_final_norm_ticks|17.864583|17.395833|15.052083|-2.624%|+15.571%|
|generation_lm_head_argmax_ticks|596.093750|595.937500|598.229167|-0.026%|-0.383%|
|generation_lm_head_command_count|594.000000|594.000000|594.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|156797952.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_expand_ticks|3693.020833|3675.156250|3673.645833|-0.484%|+0.041%|
|generation_lm_head_hmx_tail_wait_ticks|150.729167|159.635417|180.677083|+5.909%|-11.646%|
|generation_lm_head_hmx_ticks|4543.281250|4538.489583|4546.197917|-0.105%|-0.170%|
|generation_lm_head_prefetch_count|593.000000|593.000000|593.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|22.343750|22.239583|22.395833|-0.466%|-0.698%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|1215488.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|5204.687500|5198.333333|5205.677083|-0.122%|-0.141%|
|generation_lm_head_weight_dma_ticks|5145.416667|5141.979167|5149.895833|-0.067%|-0.154%|
|generation_lm_head_weight_dma_wait_ticks|299.062500|299.739583|297.187500|+0.226%|+0.859%|
|hmx_command_count|1910.000000|1910.000000|1882.000000|+0.000%|+1.488%|
|hmx_compute_ticks|12951.875000|12960.833333|12870.729167|+0.069%|+0.700%|
|hmx_fp16_tile_pair_count|21504.000000|21504.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|2031360.000000|2031360.000000|2031360.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|2439.530500|2271.822083|2409.115000|-6.875%|-5.699%|
|host_wall_ns|43684.584000|36023.802000|36275.365000|-17.537%|-0.693%|
|input_norm_ticks|2063.593750|2067.968750|2094.895833|+0.212%|-1.285%|
|input_stage_ticks|0.364583|0.364583|0.312500|+0.000%|+16.667%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|41174.531250|33766.927083|33868.385417|-17.991%|-0.300%|
|layer_bookkeeping_ticks|21.875000|21.666667|22.708333|-0.952%|-4.587%|
|ledger_named_ticks|41174.531250|33766.927083|33868.385417|-17.991%|-0.300%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|254.375000|256.093750|254.062500|+0.676%|+0.800%|
|o_projection_ticks|2092.187500|2078.177083|2103.750000|-0.670%|-1.216%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|1.510417|1.406250|1.354167|-6.897%|+3.846%|
|post_attention_residual_ticks|2078.854167|2076.614583|2081.093750|-0.108%|-0.215%|
|prefix_group_patch_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|projection_hmx_wait_ticks|1809.062500|1802.812500|1838.802083|-0.345%|-1.957%|
|projection_pack_ticks|4.375000|4.270833|4.322917|-2.381%|-1.205%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|0.520833|0.468750|0.468750|-10.000%|+0.000%|
|qkv_projection_ticks|14407.083333|7053.125000|7067.500000|-51.044%|-0.203%|
|repeat_count|1.000000|1.000000|1.000000|+0.000%|+0.000%|
|runtime_setup_ticks|54.322917|56.302083|54.218750|+3.643%|+3.842%|
|runtime_teardown_ticks|54.687500|52.656250|52.135417|-3.714%|+0.999%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0.000000|0|0|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|458.385417|451.770833|450.833333|-1.443%|+0.208%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|448.000000|448.000000|448.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|135.677083|136.875000|141.354167|+0.883%|-3.169%|
|scan_cache_stage_ticks|0.000000|0.000000|0.000000|0|0|
|scan_dynamic_attention_ticks|0.000000|0.000000|0.000000|0|0|
|stage_boundary_ticks|7.812500|8.072917|7.708333|+3.333%|+4.730%|
|total_ticks|41113.645833|33706.406250|33814.479167|-18.016%|-0.320%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|629.739583|625.312500|627.135417|-0.703%|-0.291%|
|u8_attention_av_requant_ticks|1650.208333|1647.500000|1647.291667|-0.164%|+0.013%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_pipeline_wait_ticks|1985.625000|1920.104167|1969.375000|-3.300%|-2.502%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|592.083333|590.729167|585.000000|-0.229%|+0.979%|
|u8_attention_qk_norm_rope_ticks|11906.197917|12535.312500|24495.312500|+5.284%|-48.826%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|8697.864583|8696.614583|8689.687500|-0.014%|+0.080%|
|u8_attention_v_pack_ticks|4924.218750|4877.656250|4945.937500|-0.946%|-1.381%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|840.000000|840.000000|840.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|196.000000|196.000000|196.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|704643072.000000|704643072.000000|704643072.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|399.895833|400.104167|399.427083|+0.052%|+0.170%|
|w4u8_gate_up_swiglu_publish_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|112.000000|112.000000|112.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|2121.979167|2130.000000|2134.479167|+0.378%|-0.210%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|672.000000|672.000000|672.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|522.187500|564.635417|565.520833|+8.129%|-0.157%|
|w4u8_qkv_ring_hmx_dispatch_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1387.968750|1423.333333|1416.458333|+2.548%|+0.485%|
|w4u8_qkv_ring_pipeline_ticks|2455.156250|2804.322917|7029.166667|+14.222%|-60.104%|
|w4u8_qkv_ring_pool_wait_ticks|4.114583|323.177083|4533.906250|+7754.430%|-92.872%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|5.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|21.093750|11.562500|12.708333|-45.185%|-9.016%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|13500.729167|13510.104167|13541.875000|+0.069%|-0.235%|
|w4u8_qkvo_prefetch_wait_ticks|2123.385417|2131.041667|2135.468750|+0.361%|-0.207%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|866032640.000000|866032640.000000|866032640.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|2275.000000|2275.000000|2275.000000|+0.000%|+0.000%|
|weight_dma_ticks|18035.104167|18046.927083|18047.812500|+0.066%|-0.005%|
## short, 64+64, decode

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|154788.437500|154790.729167|155005.833333|+0.001%|-0.139%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|block_orchestration_ticks|1947.864583|1950.312500|1952.552083|+0.126%|-0.115%|
|boundary_ddr_read_bytes|310337536.000000|310337536.000000|310337536.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|14464.000000|14464.000000|14464.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|58720256.000000|58720256.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|4796.822917|4790.677083|0.000000|-0.128%|N/A zero denominator|
|dense_r3_total_matmul_ticks|5031.354167|4598.750000|0.000000|-8.598%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|72322.552083|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|12617.343750|1979.739583|0.000000|-84.309%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|223804.218750|223904.218750|223584.479167|+0.045%|+0.143%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|120.104167|120.364583|120.729167|+0.217%|-0.302%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|398902.083333|399209.375000|399214.322917|+0.077%|-0.001%|
|generation_embedding_ddr_read_bytes|278528.000000|278528.000000|278528.000000|+0.000%|+0.000%|
|generation_embedding_ticks|369.895833|366.979167|370.416667|-0.789%|-0.928%|
|generation_final_norm_ticks|924.531250|923.906250|914.114583|-0.068%|+1.071%|
|generation_lm_head_argmax_ticks|29771.718750|29760.781250|29844.166667|-0.037%|-0.279%|
|generation_lm_head_command_count|9536.000000|9536.000000|9536.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|10035068928.000000|10035068928.000000|10035068928.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|9408.000000|9408.000000|9408.000000|+0.000%|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_hmx_tail_wait_ticks|955.468750|910.000000|1149.583333|-4.759%|-20.841%|
|generation_lm_head_hmx_ticks|171510.833333|171984.687500|171925.572917|+0.276%|+0.034%|
|generation_lm_head_prefetch_count|9472.000000|9472.000000|9472.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|1387.343750|1386.302083|1368.645833|-0.075%|+1.290%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|77791232.000000|77791232.000000|77791232.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|205476.041667|205972.239583|205984.947917|+0.241%|-0.006%|
|generation_lm_head_weight_dma_ticks|172125.312500|172687.708333|172318.437500|+0.327%|+0.214%|
|generation_lm_head_weight_dma_wait_ticks|167801.458333|168367.604167|167978.333333|+0.337%|+0.232%|
|hmx_command_count|93760.000000|93760.000000|91968.000000|+0.000%|+1.949%|
|hmx_compute_ticks|584870.416667|583323.645833|587796.770833|-0.264%|-0.761%|
|hmx_fp16_tile_pair_count|28672.000000|28672.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|108331008.000000|108331008.000000|108331008.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|55073.176167|51250.834583|53889.217083|-6.940%|-4.896%|
|host_wall_ns|1392069.322000|1378509.895000|1369439.165000|-0.974%|+0.662%|
|input_norm_ticks|23410.520833|23408.281250|23411.979167|-0.010%|-0.016%|
|input_stage_ticks|17.239583|16.510417|17.083333|-4.230%|-3.354%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1336418.645833|1326092.864583|1315549.947917|-0.773%|+0.801%|
|layer_bookkeeping_ticks|1057.187500|1055.468750|1069.270833|-0.163%|-1.291%|
|ledger_named_ticks|1336418.645833|1326092.864583|1315549.947917|-0.773%|+0.801%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|14912.343750|14844.427083|14782.083333|-0.455%|+0.422%|
|o_projection_ticks|87132.968750|87117.135417|87092.656250|-0.018%|+0.028%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|64.114583|64.218750|64.010417|+0.162%|+0.325%|
|post_attention_residual_ticks|23786.979167|23775.520833|23678.385417|-0.048%|+0.410%|
|prefix_group_patch_count|0.000000|0.000000|0.000000|0|0|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0.000000|0|0|
|projection_hmx_wait_ticks|52745.468750|52882.291667|53365.885417|+0.259%|-0.906%|
|projection_pack_ticks|267.135417|278.072917|277.864583|+4.094%|+0.075%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|21.458333|22.604167|20.989583|+5.340%|+7.692%|
|qkv_projection_ticks|180279.947917|169852.239583|158918.020833|-5.784%|+6.880%|
|repeat_count|64.000000|64.000000|64.000000|+0.000%|+0.000%|
|runtime_setup_ticks|2435.416667|2431.666667|2440.625000|-0.154%|-0.367%|
|runtime_teardown_ticks|2297.135417|2299.322917|2295.260417|+0.095%|+0.177%|
|scan_attention_overlay_capacity_bytes|176160768.000000|176160768.000000|176160768.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|18087936.000000|18087936.000000|18087936.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|14447.343750|14354.062500|14069.531250|-0.646%|+2.022%|
|scan_cache_ddr_read_bytes|354156544.000000|354156544.000000|354156544.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|919.322917|917.552083|1408.958333|-0.193%|-34.877%|
|scan_cache_stage_ticks|26064.218750|26059.947917|25981.406250|-0.016%|+0.302%|
|scan_dynamic_attention_ticks|154381.562500|154363.802083|154612.031250|-0.012%|-0.161%|
|stage_boundary_ticks|112.343750|113.020833|112.135417|+0.603%|+0.790%|
|total_ticks|1333985.833333|1323663.750000|1313104.375000|-0.774%|+0.804%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|55017.395833|55112.968750|55119.166667|+0.174%|-0.011%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|19881.093750|19860.885417|19839.322917|-0.102%|+0.109%|
|u8_attention_pipeline_wait_ticks|98979.062500|99035.989583|99045.364583|+0.058%|-0.009%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|39616.927083|39664.791667|39800.989583|+0.121%|-0.342%|
|u8_attention_qk_norm_rope_ticks|22450.572917|83707.135417|132546.875000|+272.851%|-36.847%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|43225.781250|43257.083333|43332.031250|+0.072%|-0.173%|
|u8_attention_v_pack_ticks|37618.750000|37665.052083|37615.572917|+0.123%|+0.132%|
|u8_cache_full_prefix_pack_count|28672.000000|28672.000000|28672.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|63296.000000|63296.000000|63296.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|12608.000000|12608.000000|12608.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|55054434304.000000|55054434304.000000|55054434304.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|7168.000000|0|-100.000%|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_decode_swiglu_vector_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|5977.083333|5997.656250|5972.708333|+0.344%|+0.418%|
|w4u8_gate_up_swiglu_publish_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|337569.687500|338729.583333|337926.041667|+0.344%|+0.238%|
|w4u8_gate_up_swiglu_worker_ticks|37042.708333|36484.218750|36444.791667|-1.508%|+0.108%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|14336.000000|14336.000000|14336.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|33267.656250|33329.947917|33326.302083|+0.187%|+0.011%|
|w4u8_o_gate_prefetch_start_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|8810.364583|8867.500000|8962.395833|+0.649%|-1.059%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|135051.770833|135525.937500|135801.718750|+0.351%|-0.203%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|43008.000000|43008.000000|43008.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|33523.906250|33463.385417|36943.593750|-0.181%|-9.420%|
|w4u8_qkv_ring_hmx_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|88527.291667|92006.093750|89417.708333|+3.930%|+2.895%|
|w4u8_qkv_ring_pipeline_ticks|155759.947917|156424.270833|156941.250000|+0.427%|-0.329%|
|w4u8_qkv_ring_pool_wait_ticks|244.635417|244.270833|245.520833|-0.149%|-0.509%|
|w4u8_qkv_ring_prep_worker_count|320.000000|320.000000|320.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|990.885417|818.541667|1019.739583|-17.393%|-19.730%|
|w4u8_qkv_ring_slot_count|128.000000|128.000000|128.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|796347.708333|797120.885417|797761.093750|+0.097%|-0.080%|
|w4u8_qkvo_prefetch_wait_ticks|135121.718750|135597.500000|135870.156250|+0.352%|-0.201%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|55426088960.000000|55426088960.000000|55426088960.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|117120.000000|117120.000000|117120.000000|+0.000%|+0.000%|
|weight_dma_ticks|989143.593750|989501.093750|989024.322917|+0.036%|+0.048%|
## short, 2048+64, prefill

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|399161.562500|399258.593750|399298.541667|+0.024%|-0.010%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|block_orchestration_ticks|989.322917|987.812500|981.979167|-0.153%|+0.594%|
|boundary_ddr_read_bytes|163426304.000000|163426304.000000|163426304.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|9248.000000|9248.000000|9248.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|29360128.000000|29360128.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|105677.291667|105676.875000|0.000000|-0.000%|N/A zero denominator|
|dense_r3_total_matmul_ticks|16728.020833|16729.270833|0.000000|+0.007%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|265870.833333|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|258255.364583|11753.125000|0.000000|-95.449%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|121372.187500|121990.208333|121795.000000|+0.509%|+0.160%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|62.500000|62.031250|61.822917|-0.750%|+0.337%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|222684.843750|222726.145833|222454.531250|+0.019%|+0.122%|
|generation_embedding_ddr_read_bytes|8396800.000000|8396800.000000|8396800.000000|+0.000%|+0.000%|
|generation_embedding_ticks|1872.187500|1894.791667|1882.656250|+1.207%|+0.645%|
|generation_final_norm_ticks|15.468750|15.520833|15.104167|+0.337%|+2.759%|
|generation_lm_head_argmax_ticks|602.031250|601.614583|598.385417|-0.069%|+0.540%|
|generation_lm_head_command_count|594.000000|594.000000|594.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|156797952.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_expand_ticks|3678.802083|3678.854167|3681.354167|+0.001%|-0.068%|
|generation_lm_head_hmx_tail_wait_ticks|142.864583|149.479167|155.833333|+4.630%|-4.078%|
|generation_lm_head_hmx_ticks|4525.937500|4544.062500|4544.270833|+0.400%|-0.005%|
|generation_lm_head_prefetch_count|593.000000|593.000000|593.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|21.770833|21.875000|22.864583|+0.478%|-4.328%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|1215488.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|5188.177083|5207.708333|5204.531250|+0.376%|+0.061%|
|generation_lm_head_weight_dma_ticks|5134.427083|5152.604167|5147.760417|+0.354%|+0.094%|
|generation_lm_head_weight_dma_wait_ticks|299.635417|301.145833|300.416667|+0.504%|+0.243%|
|hmx_command_count|65274.000000|42706.000000|41810.000000|-34.574%|+2.143%|
|hmx_compute_ticks|373997.656250|337247.395833|334780.312500|-9.826%|+0.737%|
|hmx_fp16_tile_pair_count|688128.000000|688128.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|59138816.000000|59138816.000000|59138816.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|31871.559500|37174.473917|35978.436083|+16.638%|+3.324%|
|host_wall_ns|1511009.742000|1242295.776000|1241530.314000|-17.784%|+0.062%|
|input_norm_ticks|66690.104167|66710.729167|66768.906250|+0.031%|-0.087%|
|input_stage_ticks|10.520833|10.312500|11.145833|-1.980%|-7.477%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1479766.979167|1205121.302083|1205628.281250|-18.560%|-0.042%|
|layer_bookkeeping_ticks|551.875000|550.572917|553.750000|-0.236%|-0.574%|
|ledger_named_ticks|1479766.979167|1205121.302083|1205628.281250|-18.560%|-0.042%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|8107.968750|8131.562500|8127.968750|+0.291%|+0.044%|
|o_projection_ticks|67535.833333|67083.906250|67151.145833|-0.669%|-0.100%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|33.125000|32.812500|32.604167|-0.943%|+0.639%|
|post_attention_residual_ticks|66545.156250|66561.250000|66557.083333|+0.024%|+0.006%|
|prefix_group_patch_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|projection_hmx_wait_ticks|60885.260417|58189.531250|58403.854167|-4.428%|-0.367%|
|projection_pack_ticks|115.885417|116.041667|112.552083|+0.135%|+3.100%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|11.562500|10.520833|10.208333|-9.009%|+3.061%|
|qkv_projection_ticks|498861.666667|224071.145833|224625.781250|-55.084%|-0.247%|
|repeat_count|32.000000|32.000000|32.000000|+0.000%|+0.000%|
|runtime_setup_ticks|1239.895833|1240.885417|1237.447917|+0.080%|+0.278%|
|runtime_teardown_ticks|1161.041667|1160.885417|1159.843750|-0.013%|+0.090%|
|scan_attention_overlay_capacity_bytes|85327872.000000|85327872.000000|85327872.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|51707904.000000|51707904.000000|51707904.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|14873.281250|14877.031250|14572.291667|+0.025%|+2.091%|
|scan_cache_ddr_read_bytes|1934098432.000000|1934098432.000000|1934098432.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|117440512.000000|117440512.000000|117440512.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|28224.000000|28224.000000|28224.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|2473.750000|2473.697917|2739.270833|-0.002%|-9.695%|
|scan_cache_stage_ticks|45456.875000|45567.604167|45535.208333|+0.244%|+0.071%|
|scan_dynamic_attention_ticks|395500.989583|395622.656250|395639.531250|+0.031%|-0.004%|
|stage_boundary_ticks|59.322917|60.260417|60.885417|+1.580%|-1.027%|
|total_ticks|1478527.083333|1203879.687500|1204390.833333|-18.576%|-0.042%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|71564.531250|71706.145833|71592.968750|+0.198%|+0.158%|
|u8_attention_av_requant_ticks|1649.010417|1646.458333|1647.083333|-0.155%|-0.038%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|264618.541667|264610.104167|264459.114583|-0.003%|+0.057%|
|u8_attention_pipeline_wait_ticks|16779.739583|16842.656250|17182.031250|+0.375%|-1.975%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|37524.531250|37658.593750|37587.604167|+0.357%|+0.189%|
|u8_attention_qk_norm_rope_ticks|380622.604167|400042.135417|783013.020833|+5.102%|-48.910%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|710988.593750|711100.677083|710767.135417|+0.016%|+0.047%|
|u8_attention_v_pack_ticks|108750.572917|108750.416667|108776.927083|-0.000%|-0.024%|
|u8_cache_full_prefix_pack_count|13888.000000|13888.000000|13888.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|6834.322917|0.000000|0.000000|-100.000%|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|21672.000000|26880.000000|26880.000000|+24.031%|+0.000%|
|w4u8_decode_direct_n_projection_count|3668.000000|6272.000000|6272.000000|+70.992%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|18907922432.000000|22548578304.000000|22548578304.000000|+19.255%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|5376.000000|5376.000000|5376.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|13482.239583|13488.333333|13570.208333|+0.045%|-0.603%|
|w4u8_gate_up_swiglu_publish_count|5376.000000|5376.000000|5376.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|3584.000000|3584.000000|3584.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|27944.000000|5376.000000|5376.000000|-80.762%|+0.000%|
|w4u8_qkv_ring_dispatch_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|92699.270833|67770.937500|67657.135417|-26.892%|+0.168%|
|w4u8_qkv_ring_expand_task_count|111104.000000|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_expand_ticks|191199.062500|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_expand_worker_count|93.000000|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_head_publish_count|21504.000000|21504.000000|21504.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|13106.770833|17342.916667|17625.833333|+32.320%|-1.605%|
|w4u8_qkv_ring_hmx_dispatch_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|66881.614583|45394.583333|45336.510417|-32.127%|+0.128%|
|w4u8_qkv_ring_pipeline_ticks|117028.281250|88726.458333|223482.916667|-24.184%|-60.298%|
|w4u8_qkv_ring_pool_wait_ticks|128.750000|10206.718750|144771.562500|+7827.549%|-92.950%|
|w4u8_qkv_ring_prep_worker_count|67.000000|160.000000|160.000000|+138.806%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4695.625000|345.052083|461.041667|-92.652%|-25.158%|
|w4u8_qkv_ring_slot_count|126.000000|64.000000|64.000000|-49.206%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|468652.968750|432189.531250|431901.979167|-7.780%|+0.067%|
|w4u8_qkvo_prefetch_wait_ticks|92860.625000|67805.989583|67691.250000|-26.981%|+0.170%|
|w4u8_qkvo_weight_expand_ticks|191199.062500|0.000000|0.000000|-100.000%|0|
|weight_ddr_read_bytes|22852180992.000000|22852180992.000000|22852180992.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|99491.000000|54355.000000|54355.000000|-45.367%|+0.000%|
|weight_dma_ticks|447197.395833|416042.447917|415870.468750|-6.967%|+0.041%|
## short, 2048+64, decode

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|516509.895833|516466.250000|516328.697917|-0.008%|+0.027%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|block_orchestration_ticks|1936.458333|1938.437500|1939.010417|+0.102%|-0.030%|
|boundary_ddr_read_bytes|310337536.000000|310337536.000000|310337536.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|14464.000000|14464.000000|14464.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|58720256.000000|58720256.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|4781.614583|4785.312500|0.000000|+0.077%|N/A zero denominator|
|dense_r3_total_matmul_ticks|5137.864583|4662.968750|0.000000|-9.243%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|71665.416667|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|12658.802083|1979.479167|0.000000|-84.363%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|224211.562500|224268.750000|223844.322917|+0.026%|+0.190%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|120.104167|120.416667|121.250000|+0.260%|-0.687%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|399209.739583|399585.781250|399350.781250|+0.094%|+0.059%|
|generation_embedding_ddr_read_bytes|278528.000000|278528.000000|278528.000000|+0.000%|+0.000%|
|generation_embedding_ticks|367.604167|369.739583|368.645833|+0.581%|+0.297%|
|generation_final_norm_ticks|946.979167|943.854167|929.010417|-0.330%|+1.598%|
|generation_lm_head_argmax_ticks|29646.406250|29644.166667|29601.875000|-0.008%|+0.143%|
|generation_lm_head_command_count|9536.000000|9536.000000|9536.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|10035068928.000000|10035068928.000000|10035068928.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|9408.000000|9408.000000|9408.000000|+0.000%|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_hmx_tail_wait_ticks|1030.729167|990.052083|1025.052083|-3.946%|-3.414%|
|generation_lm_head_hmx_ticks|171472.552083|172347.552083|171997.500000|+0.510%|+0.204%|
|generation_lm_head_prefetch_count|9472.000000|9472.000000|9472.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|1331.406250|1373.958333|1381.666667|+3.196%|-0.558%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|77791232.000000|77791232.000000|77791232.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|205287.916667|206178.802083|205815.260417|+0.434%|+0.177%|
|generation_lm_head_weight_dma_ticks|172036.927083|172876.093750|172713.177083|+0.488%|+0.094%|
|generation_lm_head_weight_dma_wait_ticks|167710.989583|168527.968750|168416.979167|+0.487%|+0.066%|
|hmx_command_count|93760.000000|93760.000000|91968.000000|+0.000%|+1.949%|
|hmx_compute_ticks|649019.010417|645234.531250|646105.156250|-0.583%|-0.135%|
|hmx_fp16_tile_pair_count|28672.000000|28672.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|122552320.000000|122552320.000000|122552320.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|33550.834583|32177.081917|32187.920583|-4.095%|-0.034%|
|host_wall_ns|1730092.288000|1721032.134000|1710528.378000|-0.524%|+0.614%|
|input_norm_ticks|23304.687500|23313.802083|23357.135417|+0.039%|-0.186%|
|input_stage_ticks|17.656250|17.239583|16.770833|-2.360%|+2.795%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1698153.750000|1688855.052083|1677162.135417|-0.548%|+0.697%|
|layer_bookkeeping_ticks|1054.531250|1053.333333|1068.697917|-0.114%|-1.438%|
|ledger_named_ticks|1698153.750000|1688855.052083|1677162.135417|-0.548%|+0.697%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|15101.927083|15035.156250|15151.979167|-0.442%|-0.771%|
|o_projection_ticks|87471.770833|87965.104167|87721.927083|+0.564%|+0.277%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|62.812500|64.947917|63.645833|+3.400%|+2.046%|
|post_attention_residual_ticks|23704.895833|23693.125000|23640.520833|-0.050%|+0.223%|
|prefix_group_patch_count|0.000000|0.000000|0.000000|0|0|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0.000000|0|0|
|projection_hmx_wait_ticks|52783.802083|53389.687500|52613.385417|+1.148%|+1.475%|
|projection_pack_ticks|258.645833|270.625000|273.541667|+4.631%|-1.066%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|21.145833|21.406250|21.666667|+1.232%|-1.202%|
|qkv_projection_ticks|180538.437500|169897.656250|159189.427083|-5.894%|+6.727%|
|repeat_count|64.000000|64.000000|64.000000|+0.000%|+0.000%|
|runtime_setup_ticks|2433.958333|2440.104167|2432.656250|+0.253%|+0.306%|
|runtime_teardown_ticks|2294.947917|2295.416667|2291.718750|+0.020%|+0.161%|
|scan_attention_overlay_capacity_bytes|176160768.000000|176160768.000000|176160768.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|163315712.000000|163315712.000000|163315712.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|13550.312500|13580.937500|13236.302083|+0.226%|+2.604%|
|scan_cache_ddr_read_bytes|7635468288.000000|7635468288.000000|7635468288.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|917.760417|923.593750|1418.125000|+0.636%|-34.872%|
|scan_cache_stage_ticks|147760.416667|147865.364583|147616.510417|+0.071%|+0.169%|
|scan_dynamic_attention_ticks|516098.437500|516040.833333|515935.625000|-0.011%|+0.020%|
|stage_boundary_ticks|111.770833|111.145833|112.916667|-0.559%|-1.568%|
|total_ticks|1695718.072917|1686417.187500|1674724.687500|-0.548%|+0.698%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|55151.510417|55175.729167|55223.906250|+0.044%|-0.087%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|207960.208333|207970.729167|207885.052083|+0.005%|+0.041%|
|u8_attention_pipeline_wait_ticks|11776.250000|11300.364583|11533.802083|-4.041%|-2.024%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|94608.541667|94506.406250|94348.437500|-0.108%|+0.167%|
|u8_attention_qk_norm_rope_ticks|22578.697917|82889.583333|133133.802083|+267.114%|-37.740%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|284888.958333|284628.906250|284789.583333|-0.091%|-0.056%|
|u8_attention_v_pack_ticks|397803.854167|397794.479167|397827.656250|-0.002%|-0.008%|
|u8_cache_full_prefix_pack_count|28672.000000|28672.000000|28672.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|63296.000000|63296.000000|63296.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|12608.000000|12608.000000|12608.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|55054434304.000000|55054434304.000000|55054434304.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|7168.000000|0|-100.000%|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_decode_swiglu_vector_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|5894.947917|5923.697917|5948.593750|+0.488%|-0.419%|
|w4u8_gate_up_swiglu_publish_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|339367.187500|339471.406250|339284.270833|+0.031%|+0.055%|
|w4u8_gate_up_swiglu_worker_ticks|36137.343750|36230.677083|36388.437500|+0.258%|-0.434%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|14336.000000|14336.000000|14336.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|33347.708333|33331.614583|33410.572917|-0.048%|-0.236%|
|w4u8_o_gate_prefetch_start_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|8960.885417|8948.437500|9085.364583|-0.139%|-1.507%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|135082.500000|135301.979167|135670.625000|+0.162%|-0.272%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|43008.000000|43008.000000|43008.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|33531.250000|33473.020833|36766.770833|-0.174%|-8.958%|
|w4u8_qkv_ring_hmx_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|89950.156250|91766.770833|90290.572917|+2.020%|+1.635%|
|w4u8_qkv_ring_pipeline_ticks|155777.343750|156317.916667|157211.406250|+0.347%|-0.568%|
|w4u8_qkv_ring_pool_wait_ticks|244.218750|245.208333|246.041667|+0.405%|-0.339%|
|w4u8_qkv_ring_prep_worker_count|320.000000|320.000000|320.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|1073.020833|732.656250|954.583333|-31.720%|-23.249%|
|w4u8_qkv_ring_slot_count|128.000000|128.000000|128.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|796813.645833|798013.489583|798186.406250|+0.151%|-0.022%|
|w4u8_qkvo_prefetch_wait_ticks|135153.437500|135372.708333|135740.937500|+0.162%|-0.271%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|55426088960.000000|55426088960.000000|55426088960.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|117120.000000|117120.000000|117120.000000|+0.000%|+0.000%|
|weight_dma_ticks|990084.479167|991043.229167|991335.208333|+0.097%|-0.029%|
## formal, 64+64, prefill

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|3460.539063|3429.776042|3476.640625|-0.889%|-1.348%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|block_orchestration_ticks|37.815104|38.289063|38.377604|+1.253%|-0.231%|
|boundary_ddr_read_bytes|5107072.000000|5107072.000000|5107072.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|289.000000|289.000000|289.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|917504.000000|917504.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|3304.765625|3302.989583|0.000000|-0.054%|N/A zero denominator|
|dense_r3_total_matmul_ticks|524.403646|523.911458|0.000000|-0.094%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|8320.619792|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|8077.630208|371.458333|0.000000|-95.401%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|3843.690104|3852.614583|3845.057292|+0.232%|+0.197%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|3.062500|3.164062|3.119792|+3.316%|+1.419%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|7101.924479|7126.221354|7108.932292|+0.342%|+0.243%|
|generation_embedding_ddr_read_bytes|262400.000000|262400.000000|262400.000000|+0.000%|+0.000%|
|generation_embedding_ticks|67.638021|68.257813|68.085938|+0.916%|+0.252%|
|generation_final_norm_ticks|17.138021|17.216146|14.242188|+0.456%|+20.881%|
|generation_lm_head_argmax_ticks|596.033854|596.138021|598.234375|+0.017%|-0.350%|
|generation_lm_head_command_count|594.000000|594.000000|594.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|156797952.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_expand_ticks|3698.893229|3699.679688|3699.763021|+0.021%|-0.002%|
|generation_lm_head_hmx_tail_wait_ticks|163.929688|173.015625|164.328125|+5.543%|+5.287%|
|generation_lm_head_hmx_ticks|4570.153646|4576.369792|4572.736979|+0.136%|+0.079%|
|generation_lm_head_prefetch_count|593.000000|593.000000|593.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|21.731771|22.341146|21.888021|+2.804%|+2.070%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|1215488.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|5227.846354|5234.778646|5231.054688|+0.133%|+0.071%|
|generation_lm_head_weight_dma_ticks|5171.755208|5178.187500|5176.895833|+0.124%|+0.025%|
|generation_lm_head_weight_dma_wait_ticks|300.645833|301.617188|301.815104|+0.323%|-0.066%|
|hmx_command_count|1910.000000|1910.000000|1882.000000|+0.000%|+1.488%|
|hmx_compute_ticks|13001.447917|13059.197917|12896.122396|+0.444%|+1.265%|
|hmx_fp16_tile_pair_count|21504.000000|21504.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|2031360.000000|2031360.000000|2031360.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|792.187425|802.252629|830.512904|+1.271%|-3.403%|
|host_wall_ns|42148.283800|34746.614650|34849.749850|-17.561%|-0.296%|
|input_norm_ticks|2066.690104|2065.986979|2096.333333|-0.034%|-1.448%|
|input_stage_ticks|0.330729|0.356771|0.338542|+7.874%|+5.385%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|41340.208333|33933.460938|34025.966146|-17.917%|-0.272%|
|layer_bookkeeping_ticks|24.643229|24.505208|26.526042|-0.560%|-7.618%|
|ledger_named_ticks|41340.208333|33933.460938|34025.966146|-17.917%|-0.272%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|256.057292|255.283854|257.026042|-0.302%|-0.678%|
|o_projection_ticks|2107.867188|2098.419271|2105.768229|-0.448%|-0.349%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|1.536458|1.580729|1.492188|+2.881%|+5.934%|
|post_attention_residual_ticks|2078.304688|2079.145833|2084.140625|+0.040%|-0.240%|
|prefix_group_patch_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|projection_hmx_wait_ticks|1743.296875|1755.867188|1750.166667|+0.721%|+0.326%|
|projection_pack_ticks|4.341146|4.359375|4.401042|+0.420%|-0.947%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|0.567708|0.578125|0.596354|+1.835%|-3.057%|
|qkv_projection_ticks|14432.164062|7028.067708|7045.924479|-51.303%|-0.253%|
|repeat_count|1.000000|1.000000|1.000000|+0.000%|+0.000%|
|runtime_setup_ticks|66.721354|66.791667|68.075521|+0.105%|-1.886%|
|runtime_teardown_ticks|50.911458|51.208333|51.710938|+0.583%|-0.972%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0.000000|0|0|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|376.156250|378.070312|371.895833|+0.509%|+1.660%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|448.000000|448.000000|448.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|127.687500|127.593750|135.695313|-0.073%|-5.970%|
|scan_cache_stage_ticks|0.000000|0.000000|0.000000|0|0|
|scan_dynamic_attention_ticks|0.000000|0.000000|0.000000|0|0|
|stage_boundary_ticks|7.510417|7.382812|7.572917|-1.699%|-2.510%|
|total_ticks|41275.125000|33866.421875|33957.609375|-17.950%|-0.269%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|629.684896|633.148438|633.942708|+0.550%|-0.125%|
|u8_attention_av_requant_ticks|1648.606771|1649.565104|1648.739583|+0.058%|+0.050%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_pipeline_wait_ticks|1988.869792|1956.049479|1998.390625|-1.650%|-2.119%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|593.713542|595.677083|589.841146|+0.331%|+0.989%|
|u8_attention_qk_norm_rope_ticks|11906.682292|12517.333333|24512.864583|+5.129%|-48.936%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|8701.143229|8709.302083|8697.286458|+0.094%|+0.138%|
|u8_attention_v_pack_ticks|4922.523438|4866.101562|4945.835938|-1.146%|-1.612%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|840.000000|840.000000|840.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|196.000000|196.000000|196.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|704643072.000000|704643072.000000|704643072.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|399.119792|399.695312|400.789062|+0.144%|-0.273%|
|w4u8_gate_up_swiglu_publish_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|112.000000|112.000000|112.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|168.000000|168.000000|168.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|2164.692708|2167.523438|2167.697917|+0.131%|-0.008%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|672.000000|672.000000|672.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|520.914062|538.617188|547.429688|+3.398%|-1.610%|
|w4u8_qkv_ring_hmx_dispatch_count|28.000000|28.000000|28.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1447.559896|1472.841146|1472.015625|+1.746%|+0.056%|
|w4u8_qkv_ring_pipeline_ticks|2485.955729|2790.127604|7007.473958|+12.236%|-60.184%|
|w4u8_qkv_ring_pool_wait_ticks|3.960938|281.052083|4490.294271|+6995.595%|-93.741%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|5.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|13.908854|8.468750|10.856771|-39.113%|-21.996%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|13717.742188|13746.789062|13736.372396|+0.212%|+0.076%|
|w4u8_qkvo_prefetch_wait_ticks|2165.817708|2168.559896|2168.768229|+0.127%|-0.010%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|866032640.000000|866032640.000000|866032640.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|2275.000000|2275.000000|2275.000000|+0.000%|+0.000%|
|weight_dma_ticks|18367.140625|18402.697917|18392.828125|+0.194%|+0.054%|
## formal, 64+64, decode

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|155369.817708|155483.013021|155317.179688|+0.073%|+0.107%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|block_orchestration_ticks|1955.205729|1954.549479|1954.835938|-0.034%|-0.015%|
|boundary_ddr_read_bytes|310337536.000000|310337536.000000|310337536.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|14464.000000|14464.000000|14464.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|58720256.000000|58720256.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|4789.640625|4784.252604|0.000000|-0.112%|N/A zero denominator|
|dense_r3_total_matmul_ticks|5131.992188|4630.393229|0.000000|-9.774%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|71281.369792|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|12626.122396|1974.679688|0.000000|-84.360%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|227769.544271|227660.846354|227521.721354|-0.048%|+0.061%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|120.476562|120.507812|120.630208|+0.026%|-0.101%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|409379.703125|410115.583333|409266.101563|+0.180%|+0.208%|
|generation_embedding_ddr_read_bytes|278528.000000|278528.000000|278528.000000|+0.000%|+0.000%|
|generation_embedding_ticks|366.919271|367.372396|367.544271|+0.123%|-0.047%|
|generation_final_norm_ticks|925.398438|927.315104|913.562500|+0.207%|+1.505%|
|generation_lm_head_argmax_ticks|29760.437500|29760.018229|29836.080729|-0.001%|-0.255%|
|generation_lm_head_command_count|9536.000000|9536.000000|9536.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|10035068928.000000|10035068928.000000|10035068928.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|9408.000000|9408.000000|9408.000000|+0.000%|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_hmx_tail_wait_ticks|875.281250|880.721354|1008.437500|+0.622%|-12.665%|
|generation_lm_head_hmx_ticks|175303.307292|175041.942708|175196.898438|-0.149%|-0.088%|
|generation_lm_head_prefetch_count|9472.000000|9472.000000|9472.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|1386.677083|1429.867188|1401.755208|+3.115%|+2.005%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|77791232.000000|77791232.000000|77791232.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|209321.484375|209101.809896|209281.838542|-0.105%|-0.086%|
|generation_lm_head_weight_dma_ticks|175919.463542|175667.708333|175792.166667|-0.143%|-0.071%|
|generation_lm_head_weight_dma_wait_ticks|171607.510417|171371.164062|171478.127604|-0.138%|-0.062%|
|hmx_command_count|93760.000000|93760.000000|91968.000000|+0.000%|+1.949%|
|hmx_compute_ticks|583051.328125|582319.760417|582944.791667|-0.125%|-0.107%|
|hmx_fp16_tile_pair_count|28672.000000|28672.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|108331008.000000|108331008.000000|108331008.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|37282.007383|36806.254129|36916.942817|-1.276%|-0.300%|
|host_wall_ns|1396369.616050|1387503.181950|1375827.475950|-0.635%|+0.849%|
|input_norm_ticks|23410.846354|23409.119792|23409.947917|-0.007%|-0.004%|
|input_stage_ticks|16.669271|16.867188|16.476563|+1.187%|+2.371%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1359105.911458|1350025.450521|1338039.468750|-0.668%|+0.896%|
|layer_bookkeeping_ticks|1057.130208|1057.458333|1069.296875|+0.031%|-1.107%|
|ledger_named_ticks|1359105.911458|1350025.450521|1338039.468750|-0.668%|+0.896%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|15347.179688|15350.000000|15395.020833|+0.018%|-0.292%|
|o_projection_ticks|87850.000000|88051.880208|87873.052083|+0.230%|+0.204%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|63.789062|63.820312|64.013021|+0.049%|-0.301%|
|post_attention_residual_ticks|23785.843750|23789.250000|23694.041667|+0.014%|+0.402%|
|prefix_group_patch_count|0.000000|0.000000|0.000000|0|0|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0.000000|0|0|
|projection_hmx_wait_ticks|50488.674479|50793.151042|50647.171875|+0.603%|+0.288%|
|projection_pack_ticks|266.625000|278.153646|277.890625|+4.324%|+0.095%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|21.640625|21.466146|21.106771|-0.806%|+1.703%|
|qkv_projection_ticks|183390.182292|173131.453125|162046.567708|-5.594%|+6.841%|
|repeat_count|64.000000|64.000000|64.000000|+0.000%|+0.000%|
|runtime_setup_ticks|2435.835938|2433.252604|2438.346354|-0.106%|-0.209%|
|runtime_teardown_ticks|2293.041667|2294.252604|2292.677083|+0.053%|+0.069%|
|scan_attention_overlay_capacity_bytes|176160768.000000|176160768.000000|176160768.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|18087936.000000|18087936.000000|18087936.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|14404.859375|14448.877604|13969.039063|+0.306%|+3.435%|
|scan_cache_ddr_read_bytes|354156544.000000|354156544.000000|354156544.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|917.041667|922.117188|1411.117188|+0.553%|-34.653%|
|scan_cache_stage_ticks|26214.229167|26276.291667|26169.231771|+0.237%|+0.409%|
|scan_dynamic_attention_ticks|154954.304688|155053.382813|154918.554688|+0.064%|+0.087%|
|stage_boundary_ticks|111.700521|111.687500|111.901042|-0.012%|-0.191%|
|total_ticks|1356672.539062|1347587.361979|1335596.687500|-0.670%|+0.898%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|55360.479167|55358.583333|55269.075521|-0.003%|+0.162%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|19896.341146|19902.726562|19860.437500|+0.032%|+0.213%|
|u8_attention_pipeline_wait_ticks|99785.015625|99736.052083|99780.273438|-0.049%|-0.044%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|39916.013021|39977.268229|39907.088542|+0.153%|+0.176%|
|u8_attention_qk_norm_rope_ticks|22548.125000|82734.695312|129790.721354|+266.925%|-36.255%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|43287.278646|43277.421875|43358.041667|-0.023%|-0.186%|
|u8_attention_v_pack_ticks|37752.385417|37698.968750|37702.203125|-0.141%|-0.009%|
|u8_cache_full_prefix_pack_count|28672.000000|28672.000000|28672.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|63296.000000|63296.000000|63296.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|12608.000000|12608.000000|12608.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|55054434304.000000|55054434304.000000|55054434304.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|7168.000000|0|-100.000%|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_decode_swiglu_vector_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|6003.156250|5990.002604|5973.882812|-0.219%|+0.270%|
|w4u8_gate_up_swiglu_publish_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|348156.000000|349969.390625|348288.802083|+0.521%|+0.483%|
|w4u8_gate_up_swiglu_worker_ticks|36329.325521|35760.098958|36206.458333|-1.567%|-1.233%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|14336.000000|14336.000000|14336.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|34065.002604|34072.632813|34151.604167|+0.022%|-0.231%|
|w4u8_o_gate_prefetch_start_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|9590.726563|9591.544271|9768.044271|+0.009%|-1.807%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|138219.830729|138864.666667|139166.130208|+0.467%|-0.217%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|43008.000000|43008.000000|43008.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|33283.296875|33413.437500|35636.385417|+0.391%|-6.238%|
|w4u8_qkv_ring_hmx_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|92946.692708|95602.843750|94403.718750|+2.858%|+1.270%|
|w4u8_qkv_ring_pipeline_ticks|158701.263021|159570.419271|160065.776042|+0.548%|-0.309%|
|w4u8_qkv_ring_pool_wait_ticks|245.361979|246.585938|245.687500|+0.499%|+0.366%|
|w4u8_qkv_ring_prep_worker_count|320.000000|320.000000|320.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|828.044271|649.330729|708.289063|-21.583%|-8.324%|
|w4u8_qkv_ring_slot_count|128.000000|128.000000|128.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|812443.606771|814313.669271|813847.130208|+0.230%|+0.057%|
|w4u8_qkvo_prefetch_wait_ticks|138290.934896|138933.627604|139234.135417|+0.465%|-0.216%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|55426088960.000000|55426088960.000000|55426088960.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|117120.000000|117120.000000|117120.000000|+0.000%|+0.000%|
|weight_dma_ticks|1013538.718750|1014450.218750|1014276.604167|+0.090%|+0.017%|
## formal, 2048+64, prefill

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|403631.307292|399809.744792|399947.434896|-0.947%|-0.034%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|block_orchestration_ticks|997.033854|988.333333|984.802083|-0.873%|+0.359%|
|boundary_ddr_read_bytes|163426304.000000|163426304.000000|163426304.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|9248.000000|9248.000000|9248.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|29360128.000000|29360128.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|105696.716146|105677.184896|0.000000|-0.018%|N/A zero denominator|
|dense_r3_total_matmul_ticks|16793.557292|16741.526042|0.000000|-0.310%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|266373.130208|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|258283.205729|11757.151042|0.000000|-95.448%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|141139.901042|123963.320312|124069.992188|-12.170%|-0.086%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|63.921875|64.033854|64.151042|+0.175%|-0.183%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|265825.328125|230821.627604|230719.309896|-13.168%|+0.044%|
|generation_embedding_ddr_read_bytes|8396800.000000|8396800.000000|8396800.000000|+0.000%|+0.000%|
|generation_embedding_ticks|2061.867188|1936.723958|1929.643229|-6.069%|+0.367%|
|generation_final_norm_ticks|15.692708|15.455729|15.382813|-1.510%|+0.474%|
|generation_lm_head_argmax_ticks|603.848958|602.148438|600.343750|-0.282%|+0.301%|
|generation_lm_head_command_count|594.000000|594.000000|594.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|156797952.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_expand_ticks|4337.403646|3943.437500|3935.174479|-9.083%|+0.210%|
|generation_lm_head_hmx_tail_wait_ticks|786.783854|389.049479|372.411458|-50.552%|+4.468%|
|generation_lm_head_hmx_ticks|5894.247396|5051.291667|5019.468750|-14.301%|+0.634%|
|generation_lm_head_prefetch_count|593.000000|593.000000|593.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|31.973958|26.044271|25.783854|-18.545%|+1.010%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|1215488.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|6575.934896|5722.195312|5687.986979|-12.983%|+0.601%|
|generation_lm_head_weight_dma_ticks|6508.549479|5661.687500|5628.135417|-13.012%|+0.596%|
|generation_lm_head_weight_dma_wait_ticks|361.822917|328.421875|326.169271|-9.231%|+0.691%|
|hmx_command_count|65274.000000|42706.000000|41810.000000|-34.574%|+2.143%|
|hmx_compute_ticks|402433.182292|339354.458333|337256.630208|-15.674%|+0.622%|
|hmx_fp16_tile_pair_count|688128.000000|688128.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|59138816.000000|59138816.000000|59138816.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|22568.134775|21489.134912|22238.224021|-4.781%|-3.368%|
|host_wall_ns|1593129.447350|1240280.783950|1241159.903100|-22.148%|-0.071%|
|input_norm_ticks|66757.057292|66700.346354|66779.486979|-0.085%|-0.119%|
|input_stage_ticks|10.981771|11.018229|10.940104|+0.332%|+0.714%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1570959.716146|1219135.221354|1218802.028646|-22.396%|+0.027%|
|layer_bookkeeping_ticks|554.609375|554.692708|561.223958|+0.015%|-1.164%|
|ledger_named_ticks|1570959.716146|1219135.221354|1218802.028646|-22.396%|+0.027%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|9797.947917|8923.455729|8918.164062|-8.925%|+0.059%|
|o_projection_ticks|70798.747396|67266.395833|67401.046875|-4.989%|-0.200%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|32.549479|32.476562|32.518229|-0.224%|-0.128%|
|post_attention_residual_ticks|66621.221354|66556.197917|66570.028646|-0.098%|-0.021%|
|prefix_group_patch_count|224.000000|224.000000|224.000000|+0.000%|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|projection_hmx_wait_ticks|55058.335938|55629.997396|55819.940104|+1.038%|-0.340%|
|projection_pack_ticks|115.692708|116.242188|111.117188|+0.475%|+4.612%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|10.989583|10.445313|10.221354|-4.953%|+2.191%|
|qkv_projection_ticks|514250.411458|225287.906250|224976.921875|-56.191%|+0.138%|
|repeat_count|32.000000|32.000000|32.000000|+0.000%|+0.000%|
|runtime_setup_ticks|1251.632812|1251.010417|1255.557292|-0.050%|-0.362%|
|runtime_teardown_ticks|1163.656250|1156.989583|1159.520833|-0.573%|-0.218%|
|scan_attention_overlay_capacity_bytes|85327872.000000|85327872.000000|85327872.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|51707904.000000|51707904.000000|51707904.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|17158.505208|15543.140625|15290.153646|-9.414%|+1.655%|
|scan_cache_ddr_read_bytes|1934098432.000000|1934098432.000000|1934098432.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|117440512.000000|117440512.000000|117440512.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|28224.000000|28224.000000|28224.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|2464.726562|2465.820312|2736.640625|+0.044%|-9.896%|
|scan_cache_stage_ticks|56024.330729|47169.364583|47100.007812|-15.806%|+0.147%|
|scan_dynamic_attention_ticks|399962.794271|396174.463542|396267.429688|-0.947%|-0.023%|
|stage_boundary_ticks|60.851562|60.598958|60.528646|-0.415%|+0.116%|
|total_ticks|1569708.083333|1217884.854167|1217548.817708|-22.413%|+0.028%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|73987.583333|71814.627604|71745.880208|-2.937%|+0.096%|
|u8_attention_av_requant_ticks|1649.026042|1649.598958|1649.000000|+0.035%|+0.036%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|264736.622396|264663.252604|264497.622396|-0.028%|+0.063%|
|u8_attention_pipeline_wait_ticks|13377.286458|16101.536458|16892.645833|+20.365%|-4.683%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|38448.369792|37685.408854|37726.445312|-1.984%|-0.109%|
|u8_attention_qk_norm_rope_ticks|380772.721354|400529.445312|782548.138021|+5.189%|-48.817%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|710792.893229|711112.916667|710791.747396|+0.045%|+0.045%|
|u8_attention_v_pack_ticks|109213.270833|108769.851562|108816.632813|-0.406%|-0.043%|
|u8_cache_full_prefix_pack_count|13888.000000|13888.000000|13888.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|6832.591146|0.000000|0.000000|-100.000%|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|21672.000000|26880.000000|26880.000000|+24.031%|+0.000%|
|w4u8_decode_direct_n_projection_count|3668.000000|6272.000000|6272.000000|+70.992%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|18907922432.000000|22548578304.000000|22548578304.000000|+19.255%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|5376.000000|5376.000000|5376.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|13548.877604|13498.744792|13562.656250|-0.370%|-0.471%|
|w4u8_gate_up_swiglu_publish_count|5376.000000|5376.000000|5376.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|3584.000000|3584.000000|3584.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|0|0|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|27944.000000|5376.000000|5376.000000|-80.762%|+0.000%|
|w4u8_qkv_ring_dispatch_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|110661.927083|70376.656250|70370.671875|-36.404%|+0.009%|
|w4u8_qkv_ring_expand_task_count|111104.000000|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_expand_ticks|191220.067708|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_expand_worker_count|93.000000|0.000000|0.000000|-100.000%|0|
|w4u8_qkv_ring_head_publish_count|21504.000000|21504.000000|21504.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|12391.812500|17342.164062|17641.609375|+39.949%|-1.697%|
|w4u8_qkv_ring_hmx_dispatch_count|896.000000|896.000000|896.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|73797.669271|47915.572917|48117.929688|-35.072%|-0.421%|
|w4u8_qkv_ring_pipeline_ticks|132244.135417|89882.526042|223833.638021|-32.033%|-59.844%|
|w4u8_qkv_ring_pool_wait_ticks|129.234375|8752.085938|142484.031250|+6672.258%|-93.857%|
|w4u8_qkv_ring_prep_worker_count|67.000000|160.000000|160.000000|+138.806%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|2735.557292|281.750000|400.179688|-89.700%|-29.594%|
|w4u8_qkv_ring_slot_count|126.000000|64.000000|64.000000|-49.206%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|541550.828125|443723.096354|443776.802083|-18.064%|-0.012%|
|w4u8_qkvo_prefetch_wait_ticks|110821.158854|70411.132812|70404.744792|-36.464%|+0.009%|
|w4u8_qkvo_weight_expand_ticks|191220.067708|0.000000|0.000000|-100.000%|0|
|weight_ddr_read_bytes|22852180992.000000|22852180992.000000|22852180992.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|99491.000000|54355.000000|54355.000000|-45.367%|+0.000%|
|weight_dma_ticks|539096.421875|433573.296875|433610.343750|-19.574%|-0.009%|
## formal, 2048+64, decode

|Counter|R3 prior|R3 candidate|OFF optimized|Candidate/prior change|Candidate/OFF change|
|---|---:|---:|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|attention_setup_ticks|0.000000|0.000000|0.000000|0|0|
|attention_softmax_ticks|0.000000|0.000000|0.000000|0|0|
|attention_ticks|518402.067708|517780.960938|517852.096354|-0.120%|-0.014%|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|block_invocation_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|block_orchestration_ticks|1938.080729|1940.104167|1939.695313|+0.104%|+0.021%|
|boundary_ddr_read_bytes|310337536.000000|310337536.000000|310337536.000000|+0.000%|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|boundary_dma_descriptor_count|14464.000000|14464.000000|14464.000000|+0.000%|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|0|0|
|cache_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|cache_tensor_count|0.000000|0.000000|0.000000|0|0|
|dense_r3_constant_read_bytes|58720256.000000|58720256.000000|0.000000|+0.000%|N/A zero denominator|
|dense_r3_total_finish_ticks|4785.765625|4781.252604|0.000000|-0.094%|N/A zero denominator|
|dense_r3_total_matmul_ticks|5151.617188|4658.072917|0.000000|-9.580%|N/A zero denominator|
|dense_r3_total_parallel_work_ticks|0.000000|70204.304688|0.000000|N/A zero denominator|N/A zero denominator|
|dense_r3_total_prepare_ticks|12647.489583|1982.203125|0.000000|-84.327%|N/A zero denominator|
|dense_r4_audit_bytes|0.000000|0.000000|0.000000|0|0|
|dense_r4_finish_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_layout_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_matmul_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_consume_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_publish_count|0.000000|0.000000|0.000000|0|0|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0.000000|0|0|
|dense_r4_prepare_ticks|0.000000|0.000000|0.000000|0|0|
|down_ticks|229527.346354|228580.013021|228723.289063|-0.413%|-0.063%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|final_residual_ticks|120.348958|120.640625|120.692708|+0.242%|-0.043%|
|fp16_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0.000000|0|0|
|fp_softmax_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_softmax_q_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_compute_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_dq_ticks|0.000000|0.000000|0.000000|0|0|
|fp_swiglu_q_ticks|0.000000|0.000000|0.000000|0|0|
|gate_up_ticks|413964.119792|411992.726562|412850.661458|-0.476%|-0.208%|
|generation_embedding_ddr_read_bytes|278528.000000|278528.000000|278528.000000|+0.000%|+0.000%|
|generation_embedding_ticks|371.760417|370.682292|371.091146|-0.290%|-0.110%|
|generation_final_norm_ticks|954.153646|951.335938|931.359375|-0.295%|+2.145%|
|generation_lm_head_argmax_ticks|29640.346354|29640.708333|29598.169271|+0.001%|+0.144%|
|generation_lm_head_command_count|9536.000000|9536.000000|9536.000000|+0.000%|+0.000%|
|generation_lm_head_ddr_read_bytes|10035068928.000000|10035068928.000000|10035068928.000000|+0.000%|+0.000%|
|generation_lm_head_direct_slot_join_count|9408.000000|9408.000000|9408.000000|+0.000%|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_hmx_tail_wait_ticks|1018.307292|957.609375|993.096354|-5.961%|-3.573%|
|generation_lm_head_hmx_ticks|176608.838542|176014.242188|176677.432292|-0.337%|-0.375%|
|generation_lm_head_prefetch_count|9472.000000|9472.000000|9472.000000|+0.000%|+0.000%|
|generation_lm_head_scale_dma_ticks|1423.479167|1420.708333|1416.578125|-0.195%|+0.292%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|0|0|
|generation_lm_head_scale_resident_bytes|77791232.000000|77791232.000000|77791232.000000|+0.000%|+0.000%|
|generation_lm_head_ticks|210597.364583|209958.481771|210577.361979|-0.303%|-0.294%|
|generation_lm_head_weight_dma_ticks|177308.851563|176555.908854|177204.960938|-0.425%|-0.366%|
|generation_lm_head_weight_dma_wait_ticks|172961.354167|172263.244792|172867.799479|-0.404%|-0.350%|
|hmx_command_count|93760.000000|93760.000000|91968.000000|+0.000%|+1.949%|
|hmx_compute_ticks|646610.127604|645072.364583|643726.591146|-0.238%|+0.209%|
|hmx_fp16_tile_pair_count|28672.000000|28672.000000|0.000000|+0.000%|N/A zero denominator|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|hmx_u8s8_tile_pair_count|122552320.000000|122552320.000000|122552320.000000|+0.000%|+0.000%|
|host_dsp_boundary_ns|39872.483113|40142.213046|39971.988683|+0.676%|+0.426%|
|host_wall_ns|1771479.791050|1757594.874450|1746914.788900|-0.784%|+0.611%|
|input_norm_ticks|23309.729167|23310.013021|23355.947917|+0.001%|-0.197%|
|input_stage_ticks|16.799479|16.625000|16.666667|-1.039%|-0.250%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|0|0|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|0|0|
|invocation_ticks|1732005.817708|1717179.835938|1708262.234375|-0.856%|+0.522%|
|layer_bookkeeping_ticks|1054.239583|1054.583333|1068.664062|+0.033%|-1.318%|
|ledger_named_ticks|1732005.817708|1717179.835938|1708262.234375|-0.856%|+0.522%|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|0|0|
|metadata_stage_ticks|15882.315104|15830.695312|15805.885417|-0.325%|+0.157%|
|o_projection_ticks|88823.479167|88652.565104|88866.632813|-0.192%|-0.241%|
|output_nonfinite_count|0.000000|0.000000|0.000000|0|0|
|output_stage_ticks|0.000000|0.000000|0.000000|0|0|
|post_attention_norm_ticks|63.960938|63.648438|63.734375|-0.489%|-0.135%|
|post_attention_residual_ticks|23713.221354|23705.791667|23636.208333|-0.031%|+0.294%|
|prefix_group_patch_count|0.000000|0.000000|0.000000|0|0|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0.000000|0|0|
|projection_hmx_wait_ticks|50538.377604|50102.848958|50618.026042|-0.862%|-1.018%|
|projection_pack_ticks|258.494792|269.815104|272.833333|+4.379%|-1.106%|
|projection_unpack_ticks|0.000000|0.000000|0.000000|0|0|
|qk_norm_rope_ticks|21.757812|21.622396|21.143229|-0.622%|+2.266%|
|qkv_projection_ticks|184887.651042|173838.471354|163034.382813|-5.976%|+6.627%|
|repeat_count|64.000000|64.000000|64.000000|+0.000%|+0.000%|
|runtime_setup_ticks|2430.723958|2430.143229|2435.164062|-0.024%|-0.206%|
|runtime_teardown_ticks|2292.760417|2294.713542|2295.130208|+0.085%|-0.018%|
|scan_attention_overlay_capacity_bytes|176160768.000000|176160768.000000|176160768.000000|+0.000%|+0.000%|
|scan_attention_overlay_required_bytes|163315712.000000|163315712.000000|163315712.000000|+0.000%|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|0|0|
|scan_cache_append_ticks|14126.664063|13988.174479|13586.591146|-0.980%|+2.956%|
|scan_cache_ddr_read_bytes|7635468288.000000|7635468288.000000|7635468288.000000|+0.000%|+0.000%|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|3670016.000000|+0.000%|+0.000%|
|scan_cache_dma_descriptor_count|57344.000000|57344.000000|57344.000000|+0.000%|+0.000%|
|scan_cache_pack_ticks|917.252604|921.195312|1416.078125|+0.430%|-34.947%|
|scan_cache_stage_ticks|152020.630208|151168.453125|151226.513021|-0.561%|-0.038%|
|scan_dynamic_attention_ticks|517994.135417|517352.609375|517467.619792|-0.124%|-0.022%|
|stage_boundary_ticks|111.445312|111.713542|111.807292|+0.241%|-0.084%|
|total_ticks|1729565.424479|1714751.893229|1705826.622396|-0.856%|+0.523%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|0|0|
|u8_attention_av_hmx_ticks|56322.541667|56304.750000|56141.226563|-0.032%|+0.291%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_k_pack_ticks|208187.054688|208271.929688|208231.552083|+0.041%|+0.019%|
|u8_attention_pipeline_wait_ticks|10089.380208|9882.078125|10020.427083|-2.055%|-1.381%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|0|0|
|u8_attention_qk_hmx_ticks|94498.062500|94478.520833|94299.161458|-0.021%|+0.190%|
|u8_attention_qk_norm_rope_ticks|22592.669271|81601.580729|130209.817708|+261.186%|-37.331%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|0|0|
|u8_attention_softmax_ticks|284708.213542|284757.307292|285098.476562|+0.017%|-0.120%|
|u8_attention_v_pack_ticks|397823.575521|397815.096354|397835.580729|-0.002%|-0.005%|
|u8_cache_full_prefix_pack_count|28672.000000|28672.000000|28672.000000|+0.000%|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|0|0|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|0|0|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|7013600.000000|+0.000%|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|+0.000%|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|0|0|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_direct_n_hmx_command_count|63296.000000|63296.000000|63296.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_projection_count|12608.000000|12608.000000|12608.000000|+0.000%|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|55054434304.000000|55054434304.000000|55054434304.000000|+0.000%|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|7168.000000|0|-100.000%|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|14336.000000|0|-100.000%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|0|0|
|w4u8_decode_swiglu_row4_call_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_decode_swiglu_vector_count|344064.000000|344064.000000|344064.000000|+0.000%|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_gate_up_swiglu_consume_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|5930.971354|5943.299479|5951.867188|+0.208%|-0.144%|
|w4u8_gate_up_swiglu_publish_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|353199.911458|351876.070313|352450.875000|-0.375%|-0.163%|
|w4u8_gate_up_swiglu_worker_ticks|35626.742188|35337.229167|35647.286458|-0.813%|-0.870%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_down_hmx_command_count|14336.000000|14336.000000|14336.000000|+0.000%|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_o_batch_count|7168.000000|7168.000000|7168.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|34532.690104|34297.333333|34259.895833|-0.682%|+0.109%|
|w4u8_o_gate_prefetch_start_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|10132.200521|9891.848958|9938.395833|-2.372%|-0.468%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_batch_count|10752.000000|10752.000000|10752.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|139506.760417|139413.289063|139948.044271|-0.067%|-0.382%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0.000000|0|0|
|w4u8_qkv_ring_head_publish_count|43008.000000|43008.000000|43008.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|33366.640625|33473.348958|35752.348958|+0.320%|-6.374%|
|w4u8_qkv_ring_hmx_dispatch_count|1792.000000|1792.000000|1792.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|94264.585938|95758.611979|95317.716146|+1.585%|+0.463%|
|w4u8_qkv_ring_pipeline_ticks|160149.942708|160231.809896|161052.869792|+0.051%|-0.510%|
|w4u8_qkv_ring_pool_wait_ticks|245.023438|245.906250|246.330729|+0.360%|-0.172%|
|w4u8_qkv_ring_prep_worker_count|320.000000|320.000000|320.000000|+0.000%|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|847.416667|664.856771|750.000000|-21.543%|-11.352%|
|w4u8_qkv_ring_slot_count|128.000000|128.000000|128.000000|+0.000%|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|820164.468750|817638.799479|819712.005208|-0.308%|-0.253%|
|w4u8_qkvo_prefetch_wait_ticks|139578.609375|139483.007812|140015.044271|-0.068%|-0.380%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0.000000|0|0|
|weight_ddr_read_bytes|55426088960.000000|55426088960.000000|55426088960.000000|+0.000%|+0.000%|
|weight_dma_descriptor_count|117120.000000|117120.000000|117120.000000|+0.000%|+0.000%|
|weight_dma_ticks|1023724.013021|1020627.789062|1022734.187500|-0.302%|-0.206%|

## E2E formal results

Complete Host-wall paired statistics in summary.json:
```json
{
  "short": {
    "64": {
      "control_to_candidate": {
        "prefill_ms": {
          "tokens": 64,
          "control_ms": 43.688698,
          "candidate_ms": 36.027552,
          "control_tps": 1464.9097576677611,
          "candidate_tps": 1776.418225695712,
          "wall_ratio": 0.8245834197530146,
          "ci95": [
            0.7853566088077115,
            0.836075536485042
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1392.351144,
          "candidate_ms": 1378.793386,
          "control_tps": 45.96541632173213,
          "candidate_tps": 46.41739701527695,
          "wall_ratio": 0.9911950368909489,
          "ci95": [
            0.9867856495257779,
            0.9979281564405754
          ],
          "within10percent": true
        }
      },
      "off_to_candidate": {
        "prefill_ms": {
          "tokens": 64,
          "control_ms": 36.278802,
          "candidate_ms": 36.027552,
          "control_tps": 1764.1155846325908,
          "candidate_tps": 1776.418225695712,
          "wall_ratio": 0.9930744681150167,
          "ci95": [
            0.9450262109929675,
            1.0052964657828316
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1369.723541,
          "candidate_ms": 1378.793386,
          "control_tps": 46.72475728443364,
          "candidate_tps": 46.41739701527695,
          "wall_ratio": 1.0080061017218072,
          "ci95": [
            0.9992683093780687,
            1.0307975728324759
          ],
          "within10percent": true
        }
      }
    },
    "2048": {
      "control_to_candidate": {
        "prefill_ms": {
          "tokens": 2048,
          "control_ms": 1511.23865,
          "candidate_ms": 1242.542337,
          "control_tps": 1355.179739480591,
          "candidate_tps": 1648.2335764467396,
          "wall_ratio": 0.8217382522608193,
          "ci95": [
            0.8187010312646203,
            0.8250317758103201
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1730.426457,
          "candidate_ms": 1721.400464,
          "control_tps": 36.98510256885191,
          "candidate_tps": 37.17903029448701,
          "wall_ratio": 0.993637908583528,
          "ci95": [
            0.993042651295806,
            0.996513358210832
          ],
          "within10percent": true
        }
      },
      "off_to_candidate": {
        "prefill_ms": {
          "tokens": 2048,
          "control_ms": 1241.753647,
          "candidate_ms": 1242.542337,
          "control_tps": 1649.2804389565042,
          "candidate_tps": 1648.2335764467396,
          "wall_ratio": 0.9995461584108943,
          "ci95": [
            0.9980134393694563,
            1.009080734352778
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1710.865405,
          "candidate_ms": 1721.400464,
          "control_tps": 37.40796898047044,
          "candidate_tps": 37.17903029448701,
          "wall_ratio": 1.0063718488825966,
          "ci95": [
            1.0057807042980098,
            1.0071710218057983
          ],
          "within10percent": true
        }
      }
    }
  },
  "formal": {
    "64": {
      "control_to_candidate": {
        "prefill_ms": {
          "tokens": 64,
          "control_ms": 42.15840615,
          "candidate_ms": 34.75870834999999,
          "control_tps": 1518.0839563119964,
          "candidate_tps": 1841.265197646506,
          "wall_ratio": 0.8258807596118133,
          "ci95": [
            0.8229469463442629,
            0.8266973054338985
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1396.76862455,
          "candidate_ms": 1387.9512203,
          "control_tps": 45.82004411834424,
          "candidate_tps": 46.11113061031544,
          "wall_ratio": 0.9927926644652421,
          "ci95": [
            0.9916865282727119,
            0.9971894278327875
          ],
          "within10percent": true
        }
      },
      "off_to_candidate": {
        "prefill_ms": {
          "tokens": 64,
          "control_ms": 34.85999975,
          "candidate_ms": 34.75870834999999,
          "control_tps": 1835.915102093482,
          "candidate_tps": 1841.265197646506,
          "wall_ratio": 0.9973820775445701,
          "ci95": [
            0.9960879482194409,
            0.9996393656459344
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1376.17641825,
          "candidate_ms": 1387.9512203,
          "control_tps": 46.50566537202033,
          "candidate_tps": 46.11113061031544,
          "wall_ratio": 1.008871074065091,
          "ci95": [
            1.007507414257741,
            1.0103239583975694
          ],
          "within10percent": true
        }
      }
    },
    "2048": {
      "control_to_candidate": {
        "prefill_ms": {
          "tokens": 2048,
          "control_ms": 1593.55561615,
          "candidate_ms": 1240.8471301,
          "control_tps": 1285.176356096017,
          "candidate_tps": 1650.485342086379,
          "wall_ratio": 0.7785902383227449,
          "ci95": [
            0.7767466323725591,
            0.7799156965557621
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1772.0331501,
          "candidate_ms": 1758.10272725,
          "control_tps": 36.11670582821113,
          "candidate_tps": 36.40287851672235,
          "wall_ratio": 0.9923069155869465,
          "ci95": [
            0.9903793824717975,
            0.9929085046568592
          ],
          "within10percent": true
        }
      },
      "off_to_candidate": {
        "prefill_ms": {
          "tokens": 2048,
          "control_ms": 1241.63417375,
          "candidate_ms": 1240.8471301,
          "control_tps": 1649.4391369839664,
          "candidate_tps": 1650.485342086379,
          "wall_ratio": 0.9992423668817405,
          "ci95": [
            0.9985048955059247,
            1.0007275159507123
          ],
          "within10percent": true
        },
        "decode_ms": {
          "tokens": 64,
          "control_ms": 1747.4355678000002,
          "candidate_ms": 1758.10272725,
          "control_tps": 36.62509861841442,
          "candidate_tps": 36.40287851672235,
          "wall_ratio": 1.006124471705073,
          "ci95": [
            1.0053011885763792,
            1.006642603381341
          ],
          "within10percent": true
        }
      }
    }
  }
}
```

## Provenance and failures

Frozen package SHA/remote file verification: remote-package-verification.json. Exact runtime file hashes: runtime-native-full.json. Initial second-chunk OPT2 rejection and documentation-dirty preflight interruption retained under RECOVERY_NOTES.md. No measurement resampling or numerical gate relaxation. No baseline promotion.

# EXP0327 stable module overview

F16F16 and W4F16 were not measured at this experiment scope; N/A is not zero. Current W4U8 uses R3+Down16+FP32 residual. Counters in microseconds and fraction of complete Host wall. Overlapping engine counters are excluded from this additive overview.

## 64-token prefill

|Module|F16F16|W4F16|W4U8 candidate us (%Host)|W4F16/W4U8 gain|
|---|---|---|---:|---|
|I/O and metadata|N/A|N/A|255.641 (0.735%)|N/A|
|Input RMSNorm|N/A|N/A|2065.987 (5.944%)|N/A|
|QKV + Q/K Norm/RoPE + R3|N/A|N/A|7028.646 (20.221%)|N/A|
|QK-Softmax-AV|N/A|N/A|3429.776 (9.867%)|N/A|
|O projection|N/A|N/A|2098.419 (6.037%)|N/A|
|Post-attention residual + RMSNorm|N/A|N/A|2080.727 (5.986%)|N/A|
|Gate/Up + SwiGLU|N/A|N/A|7126.221 (20.502%)|N/A|
|Down|N/A|N/A|3852.615 (11.084%)|N/A|
|Final residual|N/A|N/A|3.164 (0.009%)|N/A|
|KV carrier conversion|N/A|N/A|127.594 (0.367%)|N/A|
|KV append DMA|N/A|N/A|378.070 (1.088%)|N/A|
|Block orchestration|N/A|N/A|38.289 (0.110%)|N/A|
|Layer bookkeeping|N/A|N/A|24.505 (0.071%)|N/A|
|Stage bookkeeping|N/A|N/A|7.383 (0.021%)|N/A|
|DSP unattributed|N/A|N/A|0.000 (0.000%)|N/A|
|Runtime setup/teardown|N/A|N/A|118.000 (0.339%)|N/A|
|Embedding|N/A|N/A|68.258 (0.196%)|N/A|
|Final model RMSNorm|N/A|N/A|17.216 (0.050%)|N/A|
|LM head + greedy|N/A|N/A|5234.779 (15.060%)|N/A|
|Host-DSP boundary|N/A|N/A|802.253 (2.308%)|N/A|
|Complete Host wall, including input staging|N/A|N/A|34758.708 (100%)|N/A|

Per-field medians need not sum exactly. Input staging is included in complete Host wall; separately available in raw long_complete records.

## 2048-token prefill

|Module|F16F16|W4F16|W4U8 candidate us (%Host)|W4F16/W4U8 gain|
|---|---|---|---:|---|
|I/O and metadata|N/A|N/A|8934.474 (0.720%)|N/A|
|Input RMSNorm|N/A|N/A|66700.346 (5.375%)|N/A|
|QKV + Q/K Norm/RoPE + R3|N/A|N/A|225298.352 (18.157%)|N/A|
|QK-Softmax-AV|N/A|N/A|399809.745 (32.221%)|N/A|
|O projection|N/A|N/A|67266.396 (5.421%)|N/A|
|Post-attention residual + RMSNorm|N/A|N/A|66588.674 (5.366%)|N/A|
|Gate/Up + SwiGLU|N/A|N/A|230821.628 (18.602%)|N/A|
|Down|N/A|N/A|123963.320 (9.990%)|N/A|
|Final residual|N/A|N/A|64.034 (0.005%)|N/A|
|KV carrier conversion|N/A|N/A|2465.820 (0.199%)|N/A|
|KV append DMA|N/A|N/A|15543.141 (1.253%)|N/A|
|Block orchestration|N/A|N/A|988.333 (0.080%)|N/A|
|Layer bookkeeping|N/A|N/A|554.693 (0.045%)|N/A|
|Stage bookkeeping|N/A|N/A|60.599 (0.005%)|N/A|
|DSP unattributed|N/A|N/A|0.000 (0.000%)|N/A|
|Runtime setup/teardown|N/A|N/A|2408.000 (0.194%)|N/A|
|Embedding|N/A|N/A|1936.724 (0.156%)|N/A|
|Final model RMSNorm|N/A|N/A|15.456 (0.001%)|N/A|
|LM head + greedy|N/A|N/A|5722.195 (0.461%)|N/A|
|Host-DSP boundary|N/A|N/A|21489.135 (1.732%)|N/A|
|Complete Host wall, including input staging|N/A|N/A|1240847.130 (100%)|N/A|

Per-field medians need not sum exactly. Input staging is included in complete Host wall; separately available in raw long_complete records.

