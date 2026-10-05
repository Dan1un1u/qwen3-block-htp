# EXP-0326 complete paired performance record

Source e3859c3; Qwen3-1.7B full28 W4A8 uniformINT16Down/FP32 residual/denseR3. Same archived runtime, frozen package and fixed M64+15 tokens. Control QBH_PROJECTION_ROUNDING=0 versus candidate2.5shortrepeat1 and10formalrepeat10 alternating pairs. Host wall includes token staging and FastRPC, excludes cold load and tokenization. Arithmetic intentionally differs between arms; within-arm outputs reproduce exactly.

F16F16 and W4F16 were not measured in this accuracy experiment; their module cells are N/A, not zeros, and no cross-recipe speedup is inferred. W4U8 current-candidate results below.

Numerical evidence: single/three/full28 conditional integration4/12/112 layersteps exact; R3 component bounds unchanged. Control112 hashes and184captures exactly reproduce EXP0324. Timing audits disabled: raw zero mismatch fields are not numerical proof. No hidden-tensor injection is used in independent PPL simulation; conditional integration explicitly injects checked R3 QK codes only.

## short E2E

|Scope|Control ms|Candidate ms|Control token/s|Candidate token/s|Wall delta|Ratio95%CI|
|---|---:|---:|---:|---:|---:|---|
|prefill_ns (64 tokens)|43.892188|43.752552|1458.118|1462.772|-0.318%|[0.9821194427768094, 1.0049252930691213]|
|decode_ns (15 tokens)|335.739478|337.336510|44.677|44.466|+0.476%|[0.9837233224095809, 1.0098799373244989]|

### short prefill counters

Ticks converted to microseconds (19.2MHz); ns converted to microseconds. Byte/counter fields retain their units. Engine/work/wait/lifetime counters overlap and must NOT be summed. Only the declared named stage ledger is additive. VTCM fields use per-run maxima; work/traffic use per-complete-prefill or per15-token-decode totals. Host–DSP excludes separately reported input staging.

|Field|Control|Candidate|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_setup_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_softmax_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_ticks|3462.552083|3458.854167|-0.107%|
|attention_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|block_invocation_count|28.000000|28.000000|+0.000%|
|block_orchestration_ticks|38.125000|36.979167|-3.005%|
|boundary_ddr_read_bytes|5107072.000000|5107072.000000|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|boundary_dma_descriptor_count|289.000000|289.000000|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|cache_tensor_count|0.000000|0.000000|0 (both measured zero)|
|dense_r3_constant_read_bytes|917504.000000|917504.000000|+0.000%|
|dense_r3_total_finish_ticks|3304.322917|3304.375000|+0.002%|
|dense_r3_total_matmul_ticks|524.843750|524.687500|-0.030%|
|dense_r3_total_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r3_total_prepare_ticks|8075.312500|8068.802083|-0.081%|
|dense_r4_audit_bytes|0.000000|0.000000|0 (both measured zero)|
|dense_r4_finish_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_layout_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_matmul_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_consume_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_publish_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prepare_ticks|0.000000|0.000000|0 (both measured zero)|
|down_ticks|3815.468750|3830.520833|+0.395%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|final_residual_ticks|2.760417|2.708333|-1.887%|
|fp16_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_q_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_q_ticks|0.000000|0.000000|0 (both measured zero)|
|gate_up_ticks|6951.614583|6935.208333|-0.236%|
|generation_embedding_ddr_read_bytes|262400.000000|262400.000000|+0.000%|
|generation_embedding_ticks|68.281250|64.479167|-5.568%|
|generation_final_norm_ticks|20.104167|17.916667|-10.881%|
|generation_lm_head_argmax_ticks|598.958333|598.020833|-0.157%|
|generation_lm_head_command_count|594.000000|594.000000|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_expand_ticks|3696.927083|3683.541667|-0.362%|
|generation_lm_head_hmx_tail_wait_ticks|145.989583|144.531250|-0.999%|
|generation_lm_head_hmx_ticks|4547.604167|4523.229167|-0.536%|
|generation_lm_head_prefetch_count|593.000000|593.000000|+0.000%|
|generation_lm_head_scale_dma_ticks|22.864583|22.239583|-2.733%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|+0.000%|
|generation_lm_head_ticks|5212.968750|5184.166667|-0.553%|
|generation_lm_head_weight_dma_ticks|5152.656250|5126.093750|-0.516%|
|generation_lm_head_weight_dma_wait_ticks|298.802083|302.500000|+1.238%|
|hmx_command_count|1910.000000|1910.000000|+0.000%|
|hmx_compute_ticks|12895.937500|12953.072917|+0.443%|
|hmx_fp16_tile_pair_count|21504.000000|21504.000000|+0.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|hmx_u8s8_tile_pair_count|2031360.000000|2031360.000000|+0.000%|
|host_dsp_boundary_ns|2589.479167|2587.082750|-0.093%|
|host_wall_ns|43887.500000|43748.489000|-0.317%|
|input_norm_ticks|2150.937500|2065.989583|-3.949%|
|input_stage_ticks|0.364583|0.364583|+0.000%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_dma_descriptor_count|0.000000|0.000000|0 (both measured zero)|
|intermediate_spill_fill_count|0.000000|0.000000|0 (both measured zero)|
|invocation_ticks|41353.645833|41184.322917|-0.409%|
|layer_bookkeeping_ticks|21.979167|22.447917|+2.133%|
|ledger_named_ticks|41353.645833|41184.322917|-0.409%|
|ledger_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|metadata_stage_ticks|257.552083|259.635417|+0.809%|
|o_projection_ticks|2106.458333|2125.468750|+0.902%|
|output_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|output_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|post_attention_norm_ticks|1.354167|1.510417|+11.538%|
|post_attention_residual_ticks|2085.520833|2077.187500|-0.400%|
|prefix_group_patch_count|224.000000|224.000000|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|+0.000%|
|projection_hmx_wait_ticks|1821.145833|1836.250000|+0.829%|
|projection_pack_ticks|4.687500|4.166667|-11.111%|
|projection_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|qk_norm_rope_ticks|0.572917|0.520833|-9.091%|
|qkv_projection_ticks|14399.479167|14396.614583|-0.020%|
|repeat_count|1.000000|1.000000|+0.000%|
|runtime_setup_ticks|61.770833|55.729167|-9.781%|
|runtime_teardown_ticks|52.291667|54.270833|+3.785%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_ticks|476.302083|479.062500|+0.580%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|+0.000%|
|scan_cache_dma_descriptor_count|448.000000|448.000000|+0.000%|
|scan_cache_pack_ticks|135.364583|134.270833|-0.808%|
|scan_cache_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|scan_dynamic_attention_ticks|0.000000|0.000000|0 (both measured zero)|
|stage_boundary_ticks|7.760417|7.864583|+1.342%|
|total_ticks|41299.479167|41130.885417|-0.408%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_attention_av_hmx_ticks|629.427083|628.750000|-0.108%|
|u8_attention_av_requant_ticks|1649.687500|1650.937500|+0.076%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_k_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_pipeline_wait_ticks|1947.864583|2004.739583|+2.920%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_qk_hmx_ticks|589.375000|600.520833|+1.891%|
|u8_attention_qk_norm_rope_ticks|11904.895833|11898.385417|-0.055%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_softmax_ticks|8716.562500|8716.041667|-0.006%|
|u8_attention_v_pack_ticks|4937.916667|4909.270833|-0.580%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_common_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_direct_n_hmx_command_count|840.000000|840.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|196.000000|196.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|704643072.000000|704643072.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_consume_count|168.000000|168.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|400.364583|398.177083|-0.546%|
|w4u8_gate_up_swiglu_publish_count|168.000000|168.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_down_hmx_command_count|224.000000|224.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_batch_count|112.000000|112.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_batch_count|168.000000|168.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|28.000000|28.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|2124.270833|2125.729167|+0.069%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_head_publish_count|672.000000|672.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|523.541667|525.520833|+0.378%|
|w4u8_qkv_ring_hmx_dispatch_count|28.000000|28.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1413.906250|1409.479167|-0.313%|
|w4u8_qkv_ring_pipeline_ticks|2444.166667|2455.885417|+0.479%|
|w4u8_qkv_ring_pool_wait_ticks|3.750000|3.750000|+0.000%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|15.781250|19.427083|+23.102%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|13619.791667|13550.208333|-0.511%|
|w4u8_qkvo_prefetch_wait_ticks|2125.833333|2126.666667|+0.039%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|weight_ddr_read_bytes|866032640.000000|866032640.000000|+0.000%|
|weight_dma_descriptor_count|2275.000000|2275.000000|+0.000%|
|weight_dma_ticks|18120.989583|18023.125000|-0.540%|

### short decode counters

Ticks converted to microseconds (19.2MHz); ns converted to microseconds. Byte/counter fields retain their units. Engine/work/wait/lifetime counters overlap and must NOT be summed. Only the declared named stage ledger is additive. VTCM fields use per-run maxima; work/traffic use per-complete-prefill or per15-token-decode totals. Host–DSP excludes separately reported input staging.

|Field|Control|Candidate|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_setup_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_softmax_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_ticks|35482.239583|35456.354167|-0.073%|
|attention_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|block_invocation_count|420.000000|420.000000|+0.000%|
|block_orchestration_ticks|474.479167|473.281250|-0.252%|
|boundary_ddr_read_bytes|72735360.000000|72735360.000000|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|boundary_dma_descriptor_count|3390.000000|3390.000000|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|cache_tensor_count|0.000000|0.000000|0 (both measured zero)|
|dense_r3_constant_read_bytes|13762560.000000|13762560.000000|+0.000%|
|dense_r3_total_finish_ticks|1124.479167|1120.052083|-0.394%|
|dense_r3_total_matmul_ticks|1203.697917|1200.312500|-0.281%|
|dense_r3_total_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r3_total_prepare_ticks|2946.770833|2950.885417|+0.140%|
|dense_r4_audit_bytes|0.000000|0.000000|0 (both measured zero)|
|dense_r4_finish_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_layout_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_matmul_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_consume_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_publish_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prepare_ticks|0.000000|0.000000|0 (both measured zero)|
|down_ticks|52277.552083|52234.062500|-0.083%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|final_residual_ticks|28.281250|28.072917|-0.737%|
|fp16_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_q_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_q_ticks|0.000000|0.000000|0 (both measured zero)|
|gate_up_ticks|93297.864583|93181.614583|-0.125%|
|generation_embedding_ddr_read_bytes|65280.000000|65280.000000|+0.000%|
|generation_embedding_ticks|89.843750|85.208333|-5.159%|
|generation_final_norm_ticks|217.083333|224.479167|+3.407%|
|generation_lm_head_argmax_ticks|6983.125000|6980.208333|-0.042%|
|generation_lm_head_command_count|2235.000000|2235.000000|+0.000%|
|generation_lm_head_ddr_read_bytes|2351969280.000000|2351969280.000000|+0.000%|
|generation_lm_head_direct_slot_join_count|2205.000000|2205.000000|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_hmx_tail_wait_ticks|228.645833|249.062500|+8.929%|
|generation_lm_head_hmx_ticks|40185.729167|40228.645833|+0.107%|
|generation_lm_head_prefetch_count|2220.000000|2220.000000|+0.000%|
|generation_lm_head_scale_dma_ticks|327.239583|318.489583|-2.674%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_scale_resident_bytes|18232320.000000|18232320.000000|+0.000%|
|generation_lm_head_ticks|48174.635417|48226.458333|+0.108%|
|generation_lm_head_weight_dma_ticks|40338.541667|40375.625000|+0.092%|
|generation_lm_head_weight_dma_wait_ticks|39324.895833|39364.843750|+0.102%|
|hmx_command_count|21975.000000|21975.000000|+0.000%|
|hmx_compute_ticks|136201.927083|136387.083333|+0.136%|
|hmx_fp16_tile_pair_count|6720.000000|6720.000000|+0.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|hmx_u8s8_tile_pair_count|25363200.000000|25363200.000000|+0.000%|
|host_dsp_boundary_ns|23424.217333|25037.395750|+6.887%|
|host_wall_ns|335680.259000|337271.927000|+0.474%|
|input_norm_ticks|5515.937500|5517.291667|+0.025%|
|input_stage_ticks|4.218750|4.114583|-2.469%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_dma_descriptor_count|0.000000|0.000000|0 (both measured zero)|
|intermediate_spill_fill_count|0.000000|0.000000|0 (both measured zero)|
|invocation_ticks|312124.479167|312234.531250|+0.035%|
|layer_bookkeeping_ticks|248.593750|248.541667|-0.021%|
|ledger_named_ticks|312124.479167|312234.531250|+0.035%|
|ledger_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|metadata_stage_ticks|3539.322917|3525.000000|-0.405%|
|o_projection_ticks|20306.093750|20407.083333|+0.497%|
|output_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|output_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|post_attention_norm_ticks|14.427083|14.322917|-0.722%|
|post_attention_residual_ticks|5488.385417|5593.437500|+1.914%|
|prefix_group_patch_count|0.000000|0.000000|0 (both measured zero)|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0 (both measured zero)|
|projection_hmx_wait_ticks|12157.708333|12175.416667|+0.146%|
|projection_pack_ticks|62.656250|62.604167|-0.083%|
|projection_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|qk_norm_rope_ticks|5.052083|5.052083|+0.000%|
|qkv_projection_ticks|42212.239583|42268.437500|+0.133%|
|repeat_count|15.000000|15.000000|+0.000%|
|runtime_setup_ticks|574.947917|574.479167|-0.082%|
|runtime_teardown_ticks|544.062500|542.187500|-0.345%|
|scan_attention_overlay_capacity_bytes|41287680.000000|41287680.000000|+0.000%|
|scan_attention_overlay_required_bytes|3870720.000000|3870720.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_ticks|3477.552083|3512.604167|+1.008%|
|scan_cache_ddr_read_bytes|61931520.000000|61931520.000000|+0.000%|
|scan_cache_ddr_write_bytes|860160.000000|860160.000000|+0.000%|
|scan_cache_dma_descriptor_count|13440.000000|13440.000000|+0.000%|
|scan_cache_pack_ticks|214.427083|216.875000|+1.142%|
|scan_cache_stage_ticks|5550.000000|5535.312500|-0.265%|
|scan_dynamic_attention_ticks|35387.968750|35364.895833|-0.065%|
|stage_boundary_ticks|26.145833|25.833333|-1.195%|
|total_ticks|311542.500000|311661.875000|+0.038%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_attention_av_hmx_ticks|12572.083333|12563.802083|-0.066%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_k_pack_ticks|4311.250000|4322.291667|+0.256%|
|u8_attention_pipeline_wait_ticks|23215.260417|23049.114583|-0.716%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_qk_hmx_ticks|8848.854167|8856.770833|+0.089%|
|u8_attention_qk_norm_rope_ticks|5263.281250|5271.145833|+0.149%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_softmax_ticks|9652.187500|9637.708333|-0.150%|
|u8_attention_v_pack_ticks|7530.416667|7525.468750|-0.066%|
|u8_cache_full_prefix_pack_count|6720.000000|6720.000000|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_common_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_direct_n_hmx_command_count|14835.000000|14835.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|2955.000000|2955.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|12903383040.000000|12903383040.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_row4_call_count|80640.000000|80640.000000|+0.000%|
|w4u8_decode_swiglu_vector_count|80640.000000|80640.000000|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_consume_count|2520.000000|2520.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|1407.760417|1402.031250|-0.407%|
|w4u8_gate_up_swiglu_publish_count|2520.000000|2520.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|79009.583333|78879.531250|-0.165%|
|w4u8_gate_up_swiglu_worker_ticks|8731.979167|8884.791667|+1.750%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_down_hmx_command_count|3360.000000|3360.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_batch_count|1680.000000|1680.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|420.000000|420.000000|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|7793.593750|7784.583333|-0.116%|
|w4u8_o_gate_prefetch_start_count|420.000000|420.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|2136.041667|2020.156250|-5.425%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_batch_count|2520.000000|2520.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|420.000000|420.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|31576.458333|31743.020833|+0.527%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_head_publish_count|10080.000000|10080.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|7823.541667|7869.635417|+0.589%|
|w4u8_qkv_ring_hmx_dispatch_count|420.000000|420.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|20952.708333|20943.177083|-0.045%|
|w4u8_qkv_ring_pipeline_ticks|36436.406250|36497.708333|+0.168%|
|w4u8_qkv_ring_pool_wait_ticks|54.843750|54.322917|-0.950%|
|w4u8_qkv_ring_prep_worker_count|75.000000|75.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|259.166667|286.406250|+10.510%|
|w4u8_qkv_ring_slot_count|30.000000|30.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|186063.541667|186191.406250|+0.069%|
|w4u8_qkvo_prefetch_wait_ticks|31592.395833|31761.458333|+0.535%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|weight_ddr_read_bytes|12990489600.000000|12990489600.000000|+0.000%|
|weight_dma_descriptor_count|27450.000000|27450.000000|+0.000%|
|weight_dma_ticks|231440.677083|231467.239583|+0.011%|
## formal E2E

|Scope|Control ms|Candidate ms|Control token/s|Candidate token/s|Wall delta|Ratio95%CI|
|---|---:|---:|---:|---:|---:|---|
|prefill_ns (64 tokens)|42.336263|42.293648|1511.706|1513.230|-0.101%|[0.9954936391650501, 1.0020980937414647]|
|decode_ns (15 tokens)|326.221894|326.216596|45.981|45.982|-0.002%|[0.9987430160924126, 1.0014320751507908]|

### formal prefill counters

Ticks converted to microseconds (19.2MHz); ns converted to microseconds. Byte/counter fields retain their units. Engine/work/wait/lifetime counters overlap and must NOT be summed. Only the declared named stage ledger is additive. VTCM fields use per-run maxima; work/traffic use per-complete-prefill or per15-token-decode totals. Host–DSP excludes separately reported input staging.

|Field|Control|Candidate|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_setup_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_softmax_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_ticks|3465.609375|3459.697917|-0.171%|
|attention_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|block_invocation_count|28.000000|28.000000|+0.000%|
|block_orchestration_ticks|38.377604|38.671875|+0.767%|
|boundary_ddr_read_bytes|5107072.000000|5107072.000000|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|boundary_dma_descriptor_count|289.000000|289.000000|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|cache_tensor_count|0.000000|0.000000|0 (both measured zero)|
|dense_r3_constant_read_bytes|917504.000000|917504.000000|+0.000%|
|dense_r3_total_finish_ticks|3304.549479|3304.648438|+0.003%|
|dense_r3_total_matmul_ticks|524.283854|523.911458|-0.071%|
|dense_r3_total_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r3_total_prepare_ticks|8075.828125|8077.119792|+0.016%|
|dense_r4_audit_bytes|0.000000|0.000000|0 (both measured zero)|
|dense_r4_finish_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_layout_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_matmul_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_consume_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_publish_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prepare_ticks|0.000000|0.000000|0 (both measured zero)|
|down_ticks|3813.325521|3808.309896|-0.132%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|final_residual_ticks|2.947917|2.945312|-0.088%|
|fp16_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_q_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_q_ticks|0.000000|0.000000|0 (both measured zero)|
|gate_up_ticks|6984.903646|6978.697917|-0.089%|
|generation_embedding_ddr_read_bytes|262400.000000|262400.000000|+0.000%|
|generation_embedding_ticks|65.328125|65.835938|+0.777%|
|generation_final_norm_ticks|19.255208|17.429688|-9.481%|
|generation_lm_head_argmax_ticks|599.135417|597.549479|-0.265%|
|generation_lm_head_command_count|594.000000|594.000000|+0.000%|
|generation_lm_head_ddr_read_bytes|156797952.000000|156797952.000000|+0.000%|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_expand_ticks|3686.255208|3684.539062|-0.047%|
|generation_lm_head_hmx_tail_wait_ticks|162.450521|156.908854|-3.411%|
|generation_lm_head_hmx_ticks|4557.955729|4547.653646|-0.226%|
|generation_lm_head_prefetch_count|593.000000|593.000000|+0.000%|
|generation_lm_head_scale_dma_ticks|21.664062|21.575521|-0.409%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_scale_resident_bytes|1215488.000000|1215488.000000|+0.000%|
|generation_lm_head_ticks|5221.984375|5208.630208|-0.256%|
|generation_lm_head_weight_dma_ticks|5163.531250|5151.661458|-0.230%|
|generation_lm_head_weight_dma_wait_ticks|301.401042|300.294271|-0.367%|
|hmx_command_count|1910.000000|1910.000000|+0.000%|
|hmx_compute_ticks|12914.783854|12908.752604|-0.047%|
|hmx_fp16_tile_pair_count|21504.000000|21504.000000|+0.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|hmx_u8s8_tile_pair_count|2031360.000000|2031360.000000|+0.000%|
|host_dsp_boundary_ns|1051.955858|1107.424475|+5.273%|
|host_wall_ns|42324.513150|42281.440100|-0.102%|
|input_norm_ticks|2154.898438|2066.122396|-4.120%|
|input_stage_ticks|0.333333|0.330729|-0.781%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_dma_descriptor_count|0.000000|0.000000|0 (both measured zero)|
|intermediate_spill_fill_count|0.000000|0.000000|0 (both measured zero)|
|invocation_ticks|41275.919271|41162.606771|-0.275%|
|layer_bookkeeping_ticks|24.080729|24.335938|+1.060%|
|ledger_named_ticks|41275.919271|41162.606771|-0.275%|
|ledger_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|metadata_stage_ticks|270.562500|268.171875|-0.884%|
|o_projection_ticks|2102.888021|2103.088542|+0.010%|
|output_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|output_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|post_attention_norm_ticks|1.406250|1.533854|+9.074%|
|post_attention_residual_ticks|2086.744792|2078.585938|-0.391%|
|prefix_group_patch_count|224.000000|224.000000|+0.000%|
|prefix_seed_metadata_read_bytes|57344.000000|57344.000000|+0.000%|
|projection_hmx_wait_ticks|1788.432292|1787.804688|-0.035%|
|projection_pack_ticks|4.307292|4.432292|+2.902%|
|projection_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|qk_norm_rope_ticks|0.630208|0.622396|-1.240%|
|qkv_projection_ticks|14397.630208|14400.187500|+0.018%|
|repeat_count|1.000000|1.000000|+0.000%|
|runtime_setup_ticks|66.026042|66.820312|+1.203%|
|runtime_teardown_ticks|51.580729|52.080729|+0.969%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_ticks|395.057292|393.601562|-0.368%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|scan_cache_ddr_write_bytes|3670016.000000|3670016.000000|+0.000%|
|scan_cache_dma_descriptor_count|448.000000|448.000000|+0.000%|
|scan_cache_pack_ticks|127.815104|128.325521|+0.399%|
|scan_cache_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|scan_dynamic_attention_ticks|0.000000|0.000000|0 (both measured zero)|
|stage_boundary_ticks|7.221354|7.377604|+2.164%|
|total_ticks|41209.812500|41094.270833|-0.280%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_attention_av_hmx_ticks|630.114583|631.338542|+0.194%|
|u8_attention_av_requant_ticks|1648.403646|1648.382812|-0.001%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_k_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_pipeline_wait_ticks|1974.088542|1983.007812|+0.452%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_qk_hmx_ticks|590.861979|591.164062|+0.051%|
|u8_attention_qk_norm_rope_ticks|11904.760417|11905.653646|+0.008%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_softmax_ticks|8707.906250|8699.208333|-0.100%|
|u8_attention_v_pack_ticks|4940.815104|4920.330729|-0.415%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_common_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_direct_n_hmx_command_count|840.000000|840.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|196.000000|196.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|704643072.000000|704643072.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_consume_count|168.000000|168.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|400.953125|400.656250|-0.074%|
|w4u8_gate_up_swiglu_publish_count|168.000000|168.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_down_hmx_command_count|224.000000|224.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_batch_count|112.000000|112.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_batch_count|168.000000|168.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|28.000000|28.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|2130.351562|2129.971354|-0.018%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_head_publish_count|672.000000|672.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|523.703125|525.195312|+0.285%|
|w4u8_qkv_ring_hmx_dispatch_count|28.000000|28.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1410.304688|1398.989583|-0.802%|
|w4u8_qkv_ring_pipeline_ticks|2453.052083|2454.104167|+0.043%|
|w4u8_qkv_ring_pool_wait_ticks|3.812500|3.848958|+0.956%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|16.369792|15.981771|-2.370%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|13550.830729|13543.614583|-0.053%|
|w4u8_qkvo_prefetch_wait_ticks|2131.343750|2130.937500|-0.019%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|weight_ddr_read_bytes|866032640.000000|866032640.000000|+0.000%|
|weight_dma_descriptor_count|2275.000000|2275.000000|+0.000%|
|weight_dma_ticks|18140.770833|18118.296875|-0.124%|

### formal decode counters

Ticks converted to microseconds (19.2MHz); ns converted to microseconds. Byte/counter fields retain their units. Engine/work/wait/lifetime counters overlap and must NOT be summed. Only the declared named stage ledger is additive. VTCM fields use per-run maxima; work/traffic use per-complete-prefill or per15-token-decode totals. Host–DSP excludes separately reported input staging.

|Field|Control|Candidate|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_av_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_hmx_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_pack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_qk_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_setup_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_softmax_ticks|0.000000|0.000000|0 (both measured zero)|
|attention_ticks|35505.192708|35508.015625|+0.008%|
|attention_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|block_invocation_count|420.000000|420.000000|+0.000%|
|block_orchestration_ticks|473.507813|474.541667|+0.218%|
|boundary_ddr_read_bytes|72735360.000000|72735360.000000|+0.000%|
|boundary_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|boundary_dma_descriptor_count|3390.000000|3390.000000|+0.000%|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0 (both measured zero)|
|cache_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|cache_tensor_count|0.000000|0.000000|0 (both measured zero)|
|dense_r3_constant_read_bytes|13762560.000000|13762560.000000|+0.000%|
|dense_r3_total_finish_ticks|1122.174479|1121.398438|-0.069%|
|dense_r3_total_matmul_ticks|1197.942708|1187.151042|-0.901%|
|dense_r3_total_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r3_total_prepare_ticks|2948.177083|2950.500000|+0.079%|
|dense_r4_audit_bytes|0.000000|0.000000|0 (both measured zero)|
|dense_r4_finish_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_layout_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_matmul_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_parallel_work_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_consume_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_join_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_publish_count|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|0 (both measured zero)|
|dense_r4_prepare_ticks|0.000000|0.000000|0 (both measured zero)|
|down_ticks|52581.979167|52599.434896|+0.033%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|final_residual_ticks|28.179688|28.244792|+0.231%|
|fp16_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp16_post_residual_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_softmax_q_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_dq_ticks|0.000000|0.000000|0 (both measured zero)|
|fp_swiglu_q_ticks|0.000000|0.000000|0 (both measured zero)|
|gate_up_ticks|94168.041667|93956.473958|-0.225%|
|generation_embedding_ddr_read_bytes|65280.000000|65280.000000|+0.000%|
|generation_embedding_ticks|88.505208|89.242188|+0.833%|
|generation_final_norm_ticks|216.911458|228.059896|+5.140%|
|generation_lm_head_argmax_ticks|6978.510417|6979.596354|+0.016%|
|generation_lm_head_command_count|2235.000000|2235.000000|+0.000%|
|generation_lm_head_ddr_read_bytes|2351969280.000000|2351969280.000000|+0.000%|
|generation_lm_head_direct_slot_join_count|2205.000000|2205.000000|+0.000%|
|generation_lm_head_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_hmx_tail_wait_ticks|224.760417|218.177083|-2.929%|
|generation_lm_head_hmx_ticks|40441.325521|40440.190104|-0.003%|
|generation_lm_head_prefetch_count|2220.000000|2220.000000|+0.000%|
|generation_lm_head_scale_dma_ticks|325.981771|325.890625|-0.028%|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0 (both measured zero)|
|generation_lm_head_scale_resident_bytes|18232320.000000|18232320.000000|+0.000%|
|generation_lm_head_ticks|48406.440104|48418.276042|+0.024%|
|generation_lm_head_weight_dma_ticks|40599.356771|40587.687500|-0.029%|
|generation_lm_head_weight_dma_wait_ticks|39587.593750|39576.755208|-0.027%|
|hmx_command_count|21975.000000|21975.000000|+0.000%|
|hmx_compute_ticks|135700.835938|135829.875000|+0.095%|
|hmx_fp16_tile_pair_count|6720.000000|6720.000000|+0.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|hmx_u8s8_tile_pair_count|25363200.000000|25363200.000000|+0.000%|
|host_dsp_boundary_ns|11870.901371|12070.759829|+1.684%|
|host_wall_ns|326115.820850|326099.947500|-0.005%|
|input_norm_ticks|5516.372396|5516.955729|+0.011%|
|input_stage_ticks|3.966146|3.828125|-3.480%|
|intermediate_ddr_read_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|intermediate_dma_descriptor_count|0.000000|0.000000|0 (both measured zero)|
|intermediate_spill_fill_count|0.000000|0.000000|0 (both measured zero)|
|invocation_ticks|314239.148438|314224.083333|-0.005%|
|layer_bookkeeping_ticks|249.000000|249.213542|+0.086%|
|ledger_named_ticks|314239.148438|314224.083333|-0.005%|
|ledger_unattributed_ticks|0.000000|0.000000|0 (both measured zero)|
|metadata_stage_ticks|3904.114583|3891.854167|-0.314%|
|o_projection_ticks|20465.981771|20440.096354|-0.126%|
|output_nonfinite_count|0.000000|0.000000|0 (both measured zero)|
|output_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|post_attention_norm_ticks|14.640625|14.549479|-0.623%|
|post_attention_residual_ticks|5491.916667|5599.148438|+1.953%|
|prefix_group_patch_count|0.000000|0.000000|0 (both measured zero)|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|0 (both measured zero)|
|projection_hmx_wait_ticks|12025.429688|12050.820312|+0.211%|
|projection_pack_ticks|62.411458|62.377604|-0.054%|
|projection_unpack_ticks|0.000000|0.000000|0 (both measured zero)|
|qk_norm_rope_ticks|5.067708|5.164063|+1.901%|
|qkv_projection_ticks|42461.848958|42469.507813|+0.018%|
|repeat_count|15.000000|15.000000|+0.000%|
|runtime_setup_ticks|575.794271|576.791667|+0.173%|
|runtime_teardown_ticks|539.940104|539.934896|-0.001%|
|scan_attention_overlay_capacity_bytes|41287680.000000|41287680.000000|+0.000%|
|scan_attention_overlay_required_bytes|3870720.000000|3870720.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|scan_cache_append_ticks|3536.338542|3511.567708|-0.700%|
|scan_cache_ddr_read_bytes|61931520.000000|61931520.000000|+0.000%|
|scan_cache_ddr_write_bytes|860160.000000|860160.000000|+0.000%|
|scan_cache_dma_descriptor_count|13440.000000|13440.000000|+0.000%|
|scan_cache_pack_ticks|215.466146|215.690104|+0.104%|
|scan_cache_stage_ticks|5548.783854|5563.388021|+0.263%|
|scan_dynamic_attention_ticks|35411.820312|35414.476562|+0.008%|
|stage_boundary_ticks|26.260417|26.236979|-0.089%|
|total_ticks|313663.151042|313646.562500|-0.005%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_attention_av_hmx_ticks|12577.500000|12586.109375|+0.068%|
|u8_attention_av_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_k_pack_ticks|4322.807292|4323.385417|+0.013%|
|u8_attention_pipeline_wait_ticks|23193.510417|23189.119792|-0.019%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0 (both measured zero)|
|u8_attention_qk_hmx_ticks|8861.963542|8861.304688|-0.007%|
|u8_attention_qk_norm_rope_ticks|5267.268229|5258.848958|-0.160%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_attention_softmax_ticks|9648.625000|9643.533854|-0.053%|
|u8_attention_v_pack_ticks|7531.544271|7527.242188|-0.057%|
|u8_cache_full_prefix_pack_count|6720.000000|6720.000000|+0.000%|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_append_update_ticks|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_seal_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0 (both measured zero)|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0 (both measured zero)|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|7013600.000000|7013600.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_av_requant_vector_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_common_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_direct_n_hmx_command_count|14835.000000|14835.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|2955.000000|2955.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|12903383040.000000|12903383040.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_decode_swiglu_row4_call_count|80640.000000|80640.000000|+0.000%|
|w4u8_decode_swiglu_vector_count|80640.000000|80640.000000|+0.000%|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_gate_up_swiglu_consume_count|2520.000000|2520.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|1401.057292|1400.070312|-0.070%|
|w4u8_gate_up_swiglu_publish_count|2520.000000|2520.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|79855.593750|79714.489583|-0.177%|
|w4u8_gate_up_swiglu_worker_ticks|8615.820312|8595.622396|-0.234%|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_down_hmx_command_count|3360.000000|3360.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_o_batch_count|1680.000000|1680.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|420.000000|420.000000|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|7846.927083|7838.062500|-0.113%|
|w4u8_o_gate_prefetch_start_count|420.000000|420.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|2188.893229|2072.658854|-5.310%|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_batch_count|2520.000000|2520.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|420.000000|420.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|31862.130208|31883.247396|+0.066%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|0 (both measured zero)|
|w4u8_qkv_ring_head_publish_count|10080.000000|10080.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|7825.080729|7846.166667|+0.269%|
|w4u8_qkv_ring_hmx_dispatch_count|420.000000|420.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|21277.434896|21038.578125|-1.123%|
|w4u8_qkv_ring_pipeline_ticks|36695.281250|36704.716146|+0.026%|
|w4u8_qkv_ring_pool_wait_ticks|54.812500|54.882812|+0.128%|
|w4u8_qkv_ring_prep_worker_count|75.000000|75.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|234.653646|240.908854|+2.666%|
|w4u8_qkv_ring_slot_count|30.000000|30.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|187530.125000|187484.098958|-0.025%|
|w4u8_qkvo_prefetch_wait_ticks|31878.750000|31899.802083|+0.066%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|0 (both measured zero)|
|weight_ddr_read_bytes|12990489600.000000|12990489600.000000|+0.000%|
|weight_dma_descriptor_count|27450.000000|27450.000000|+0.000%|
|weight_dma_ticks|233691.492188|233501.289062|-0.081%|

## Physical and provenance

All measured invocations request/acquire8MiB VTCM and report zero intermediate DDR read/write/spill. One full-model FastRPC perpass; matrix ownership is unchanged from the verified runtime. Fresh host executable changes only load-time bias preparation; both shared libraries and the unused llama executable are SHA-identical to EXP0324. Exact hashes in binary-regression.json and runtime-full28.json. Raw records retain all metadata and overlapping counters. No baseline promotion or general model-quality acceptance.
