# EXP-0309 Qwen3-0.6B complete paired profiling

Full model fixed64+42, ordinary W4A8 versus HVX R4. FP32 residual, R3/SP2/INT16 Down off. Timings exclude audit dumping, cold load and external tokenizer. Each timed head agrees with its own independent reference.

Formal: ten alternating pairs, repeat10 per arm. Auxiliary: one warmup repeat1 per arm, descriptive only. Module means reconcile to Host wall; counter tables show medians of per-round phase means. Decode is per step, not all42. Raw engine/worker/wait counters overlap and MUST NOT be summed as an additive ledger. R4 prepare/core/finish are subsets of Activation.

Evidence directory: /mnt/d/llm_exp/results/qwen3-block-htp/exp0309/0.6B; sealed package hash and flags are in per-run protocol.json. Measured source is in the protocol-referenced binary seal. No quality or speed acceptance claim.

## auxiliary prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|218.177083|224.166667|0.97181%|0.84332%|
|Input RMSNorm|1018.385417|1039.583333|4.53612%|3.91092%|
|QKV＋RoPE|6688.958333|6678.906250|29.79415%|25.12609%|
|QK–Softmax–AV|3321.510417|3364.166667|14.79477%|12.65602%|
|O projection|1240.520833|1238.645833|5.52556%|4.65979%|
|Post-attention residual＋RMSNorm|1011.666667|1020.416667|4.50620%|3.83881%|
|Gate/Up＋SwiGLU|2682.447917|6445.520833|11.94824%|24.24809%|
|Down|1300.781250|1278.437500|5.79398%|4.80949%|
|Final residual|2.812500|2.916667|0.01253%|0.01097%|
|KV carrier conversion|134.687500|133.958333|0.59993%|0.50395%|
|KV append DMA|258.072917|256.927083|1.14952%|0.96656%|
|Block orchestration|38.541667|36.406250|0.17167%|0.13696%|
|Layer bookkeeping|22.031250|22.500000|0.09813%|0.08465%|
|Stage-boundary bookkeeping|33.645833|33.333333|0.14987%|0.12540%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|109.114583|107.187500|0.48602%|0.40324%|
|Embedding|45.468750|44.843750|0.20253%|0.16870%|
|Final model RMSNorm|7.447917|7.447917|0.03317%|0.02802%|
|LM head＋greedy（不含 final norm）|1990.208333|1971.822917|8.86484%|7.41801%|
|Host_boundary_us|2326.093833|2674.374500|10.36096%|10.06101%|
|Host_wall_us|22450.573000|26581.562000|100.00000%|100.00000%|

## auxiliary prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|22450573.000000|26581562.000000|18.40037%|
|output_mismatches|raw|1.000000|1.000000|0.00000%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.000000|0.000000|N/A: zero control|
|output_nrmse|raw|0.000000|0.000000|N/A: zero control|
|output_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|output_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|output_max_required_rtol_after_atol|raw|0.000000|0.000000|N/A: zero control|
|output_fp16_atol|raw|0.062500|0.062500|0.00000%|
|output_fp16_rtol|raw|0.002000|0.002000|0.00000%|
|output_fp16_max_composed_nrmse|raw|0.003000|0.003000|0.00000%|
|output_max_lsb|raw|0.000000|0.000000|N/A: zero control|
|cache_prefix_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_structure_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_min_cosine|raw|1.000000|1.000000|0.00000%|
|cache_max_mixed_tolerance_violation_fraction|raw|0.000000|0.000000|N/A: zero control|
|cache_max_nrmse|raw|0.000000|0.000000|N/A: zero control|
|cache_compared_elements|raw|0.000000|0.000000|N/A: zero control|
|cache_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|cache_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|cache_tensor_count|raw|0.000000|0.000000|N/A: zero control|
|cache_composed_cosine_diagnostic_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_legacy_mixed_bound_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_fp16_min_cosine|raw|0.999990|0.999990|0.00000%|
|cache_fp16_max_violation_fraction|raw|0.010000|0.010000|0.00000%|
|fp16_norm_contexts|raw|4.000000|4.000000|0.00000%|
|fp16_norm_rows_per_task|raw|4.000000|4.000000|0.00000%|
|fp16_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_input_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|1.000000|1.000000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|14.000000|14.000000|0.00000%|
|kv_cache_v_format|raw|12.000000|12.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|56.000000|56.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_vector_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|2.000000|2.000000|0.00000%|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|5.000000|5.000000|0.00000%|
|w4u8_qkv_ring_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_head_publish_count|raw|672.000000|672.000000|0.00000%|
|w4u8_qkv_ring_pipeline_ticks|us|6652.135417|6641.875000|-0.15424%|
|w4u8_qkv_ring_dma_wait_ticks|us|1164.947917|1167.864583|0.25037%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|32.708333|32.031250|-2.07006%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|726.093750|758.281250|4.43297%|
|w4u8_qkv_ring_hmx_compute_ticks|us|328.020833|313.125000|-4.54112%|
|w4u8_qkv_ring_pool_wait_ticks|us|5144.791667|5132.343750|-0.24195%|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|758.697917|1973.750000|160.14965%|
|w4u8_decode_swiglu_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_norm_rope_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_pair_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_pair_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_rows_processed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_temp_carrier_skipped_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_padding_poison_pair_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_residual_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_vector_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|64.000000|64.000000|0.00000%|
|scan_total_kv_length|raw|64.000000|64.000000|0.00000%|
|scan_padded_kv_length|raw|64.000000|64.000000|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_attention_overlay_required_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_cache_dma_descriptor_count|raw|448.000000|448.000000|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_write_bytes|bytes|5906432.000000|5906432.000000|0.00000%|
|scan_cache_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|scan_cache_append_ticks|us|258.072917|256.927083|-0.44400%|
|scan_cache_pack_ticks|us|134.687500|133.958333|-0.54138%|
|block_orchestration_ticks|us|38.541667|36.406250|-5.54054%|
|layer_bookkeeping_ticks|us|22.031250|22.500000|2.12766%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|20065.520833|23849.375000|18.85749%|
|invocation_ticks|us|20124.479167|23907.187500|18.79655%|
|runtime_setup_ticks|us|58.958333|57.812500|-1.94346%|
|runtime_teardown_ticks|us|50.156250|49.375000|-1.55763%|
|stage_boundary_ticks|us|33.645833|33.333333|-0.92879%|
|ledger_named_ticks|us|20124.479167|23907.187500|18.79655%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.312500|0.416667|33.33333%|
|metadata_stage_ticks|us|217.864583|223.750000|2.70141%|
|input_norm_ticks|us|1018.385417|1039.583333|2.08152%|
|qkv_projection_ticks|us|6688.437500|6678.281250|-0.15185%|
|qk_norm_rope_ticks|us|0.520833|0.625000|20.00000%|
|attention_ticks|us|3321.510417|3364.166667|1.28424%|
|o_projection_ticks|us|1240.520833|1238.645833|-0.15115%|
|post_attention_residual_ticks|us|1010.000000|1018.802083|0.87149%|
|post_attention_norm_ticks|us|1.666667|1.614583|-3.12500%|
|gate_up_ticks|us|2682.447917|3950.260417|47.26327%|
|activation_ticks|us|0.000000|2495.260417|N/A: zero control|
|down_ticks|us|1300.781250|1278.437500|-1.71772%|
|final_residual_ticks|us|2.812500|2.916667|3.70370%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|45.468750|44.843750|-1.37457%|
|generation_final_norm_ticks|us|7.447917|7.447917|0.00000%|
|generation_lm_head_ticks|us|1997.656250|1979.270833|-0.92035%|
|generation_lm_head_weight_dma_ticks|us|1422.864583|1427.291667|0.31114%|
|generation_lm_head_scale_dma_ticks|us|19.375000|19.218750|-0.80645%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|1478.177083|1455.416667|-1.53976%|
|generation_lm_head_argmax_ticks|us|468.177083|471.458333|0.70086%|
|generation_lm_head_weight_dma_wait_ticks|us|1348.541667|1359.531250|0.81492%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|71.041667|45.000000|-36.65689%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|131328.000000|131328.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|79006720.000000|79006720.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|653.000000|653.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|297992192.000000|297992192.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|595984384.000000|595984384.000000|0.00000%|
|weight_dma_ticks|us|5952.395833|5971.250000|0.31675%|
|hmx_compute_ticks|us|5125.000000|5177.812500|1.03049%|
|projection_pack_ticks|us|4.010417|3.958333|-1.29870%|
|projection_hmx_wait_ticks|us|1263.489583|1278.020833|1.15009%|
|projection_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_setup_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_softmax_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_gqa_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_norm_rope_ticks|us|24298.854167|24312.864583|0.05766%|
|u8_attention_k_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_v_pack_ticks|us|4066.406250|4007.343750|-1.45245%|
|u8_cache_native_append_update_ticks|us|390.937500|389.322917|-0.41300%|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|4014080.000000|4014080.000000|0.00000%|
|u8_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_tail_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_sealed_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|1.000000|1.000000|0.00000%|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|917504.000000|917504.000000|0.00000%|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|1.000000|1.000000|0.00000%|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|827904.000000|827904.000000|0.00000%|
|u8_cache_k_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_correction_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_hvx_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|580.625000|614.010417|5.74991%|
|wide_score_mode|raw|4.000000|4.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|fp32_residual|raw|2.000000|2.000000|0.00000%|
|dense_r4_mode|raw|0.000000|4.000000|N/A: zero control|
|dense_r4_optimization|raw|6.000000|6.000000|0.00000%|
|dense_r4_calls|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_rows|raw|0.000000|1792.000000|N/A: zero control|
|dense_r4_pipeline_batches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_pipeline_hvx_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_dispatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_prepare_tiles|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_finish_groups|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_publish_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_consume_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prepare_ticks|us|0.000000|286.510417|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|1361.458333|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|843.541667|N/A: zero control|
|dense_r4_audit_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_mode|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_optimization|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_audit|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt_calls|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_conversion_audit_mismatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_heads|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_constant_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_direct_slot_join_count|raw|147.000000|147.000000|0.00000%|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|0.000000|0.000000|N/A: zero control|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|8713.489583|8730.208333|0.19187%|
|u8_attention_av_hmx_ticks|us|611.770833|625.677083|2.27311%|
|u8_attention_av_requant_ticks|us|1713.697917|1716.875000|0.18539%|
|u8_attention_pipeline_wait_ticks|us|1952.656250|2209.739583|13.16583%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|1166.041667|1169.114583|0.26353%|
|w4u8_qkvo_hmx_lifetime_ticks|us|4998.958333|5015.000000|0.32090%|
|w4f16_gate_up_weight_dma_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_join_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_gate_up_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_down_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_activation_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_expanded_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|vtcm_requested_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_acquired_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_peak_plan_bytes|bytes|5744384.000000|6006528.000000|4.56348%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1101.000000|1101.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|589184.000000|589184.000000|0.00000%|
|weight_dma_descriptor_count|raw|1158.000000|1158.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|301962240.000000|301962240.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4861312.000000|4861312.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|2326.093833|2674.374500|14.97277%|
|Host_wall_us|us|22450.573000|26581.562000|18.40037%|
|generation_lm_head_exclusive_ticks|us|1990.208333|1971.822917|-0.92379%|

## auxiliary decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|212.767857|212.218502|1.97326%|1.92287%|
|Input RMSNorm|186.855159|185.802331|1.73294%|1.68352%|
|QKV＋RoPE|1475.813492|1484.769345|13.68706%|13.45318%|
|QK–Softmax–AV|1917.430556|1915.564236|17.78272%|17.35653%|
|O projection|788.168403|775.280258|7.30967%|7.02465%|
|Post-attention residual＋RMSNorm|191.250000|192.046131|1.77370%|1.74009%|
|Gate/Up＋SwiGLU|1758.159722|2174.889633|16.30561%|19.70622%|
|Down|1079.933036|1052.306548|10.01557%|9.53473%|
|Final residual|2.439236|2.486359|0.02262%|0.02253%|
|KV carrier conversion|164.840030|164.469246|1.52877%|1.49022%|
|KV append DMA|88.539187|88.308532|0.82113%|0.80015%|
|Block orchestration|29.985119|29.212550|0.27809%|0.26469%|
|Layer bookkeeping|16.192956|16.176835|0.15018%|0.14657%|
|Stage-boundary bookkeeping|1.777034|1.801835|0.01648%|0.01633%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|72.077133|72.194940|0.66846%|0.65414%|
|Embedding|1.532738|1.542659|0.01421%|0.01398%|
|Final model RMSNorm|7.663690|7.502480|0.07107%|0.06798%|
|LM head＋greedy（不含 final norm）|1979.533730|1962.198661|18.35868%|17.77907%|
|Host_boundary_us|807.587970|697.793776|7.48977%|6.32256%|
|Host_wall_us|10782.547048|11036.564857|100.00000%|100.00000%|

## auxiliary decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|10782547.047619|11036564.857143|2.35582%|
|output_mismatches|raw|0.357143|0.238095|-33.33333%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.642857|0.761905|18.51852%|
|output_nrmse|raw|0.000000|0.000000|N/A: zero control|
|output_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|output_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|output_max_required_rtol_after_atol|raw|0.000000|0.000000|N/A: zero control|
|output_fp16_atol|raw|0.062500|0.062500|0.00000%|
|output_fp16_rtol|raw|0.002000|0.002000|0.00000%|
|output_fp16_max_composed_nrmse|raw|0.003000|0.003000|0.00000%|
|output_max_lsb|raw|0.000000|0.000000|N/A: zero control|
|cache_prefix_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_structure_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_min_cosine|raw|1.000000|1.000000|0.00000%|
|cache_max_mixed_tolerance_violation_fraction|raw|0.000000|0.000000|N/A: zero control|
|cache_max_nrmse|raw|0.000000|0.000000|N/A: zero control|
|cache_compared_elements|raw|0.000000|0.000000|N/A: zero control|
|cache_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|cache_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|cache_tensor_count|raw|0.000000|0.000000|N/A: zero control|
|cache_composed_cosine_diagnostic_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_legacy_mixed_bound_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_fp16_min_cosine|raw|0.999990|0.999990|0.00000%|
|cache_fp16_max_violation_fraction|raw|0.010000|0.010000|0.00000%|
|fp16_norm_contexts|raw|4.000000|4.000000|0.00000%|
|fp16_norm_rows_per_task|raw|4.000000|4.000000|0.00000%|
|fp16_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_input_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|22.500000|22.500000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|14.000000|14.000000|0.00000%|
|kv_cache_v_format|raw|12.000000|12.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|56.000000|56.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_av_requant_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_av_requant_vector_count|raw|1792.000000|1792.000000|0.00000%|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|2.000000|2.000000|0.00000%|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|5.000000|5.000000|0.00000%|
|w4u8_qkv_ring_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_head_publish_count|raw|672.000000|672.000000|0.00000%|
|w4u8_qkv_ring_pipeline_ticks|us|1445.342262|1454.040179|0.60179%|
|w4u8_qkv_ring_dma_wait_ticks|us|1118.462302|1140.243056|1.94738%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|48.206845|34.439484|-28.55893%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|678.462302|721.536458|6.34879%|
|w4u8_qkv_ring_hmx_compute_ticks|us|320.839534|311.765873|-2.82810%|
|w4u8_qkv_ring_pool_wait_ticks|us|3.985615|3.846726|-3.48475%|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|70.927579|74.237351|4.66641%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|271.883681|275.978423|1.50606%|
|w4u8_gate_up_swiglu_publish_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|291.875000|1146.913442|292.94679%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|1282.162698|696.778274|-45.65602%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|96.566220|316.302083|227.54941%|
|w4u8_decode_swiglu_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_norm_rope_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_q_pair_row4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_k_pair_row4_call_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_qk_rows_processed|raw|2688.000000|2688.000000|0.00000%|
|w4u8_decode_k_temp_carrier_skipped_count|raw|224.000000|224.000000|0.00000%|
|w4u8_qk_padding_poison_pair_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_residual_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|2688.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|2688.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|1507328.000000|1730560.000000|14.80978%|
|scan_attention_overlay_required_bytes|bytes|83675.428571|83675.428571|0.00000%|
|scan_cache_dma_descriptor_count|raw|1202.666667|1202.666667|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4559445.333333|4559445.333333|0.00000%|
|scan_cache_ddr_write_bytes|bytes|77312.000000|77312.000000|0.00000%|
|scan_cache_stage_ticks|us|534.723462|532.795139|-0.36062%|
|scan_cache_append_ticks|us|88.539187|88.308532|-0.26051%|
|scan_cache_pack_ticks|us|164.840030|164.469246|-0.22494%|
|block_orchestration_ticks|us|29.985119|29.212550|-2.57651%|
|layer_bookkeeping_ticks|us|16.192956|16.176835|-0.09956%|
|scan_dynamic_attention_ticks|us|1911.734871|1909.877232|-0.09717%|
|total_ticks|us|9936.763393|10300.700645|3.66253%|
|invocation_ticks|us|9974.959077|10338.771081|3.64725%|
|runtime_setup_ticks|us|38.195685|38.070437|-0.32791%|
|runtime_teardown_ticks|us|33.881448|34.124504|0.71737%|
|stage_boundary_ticks|us|1.777034|1.801835|1.39567%|
|ledger_named_ticks|us|9974.959077|10338.771081|3.64725%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.390625|0.374504|-4.12698%|
|metadata_stage_ticks|us|212.377232|211.843998|-0.25108%|
|input_norm_ticks|us|186.855159|185.802331|-0.56345%|
|qkv_projection_ticks|us|1475.507192|1484.423363|0.60428%|
|qk_norm_rope_ticks|us|0.306300|0.345982|12.95547%|
|attention_ticks|us|1917.430556|1915.564236|-0.09733%|
|o_projection_ticks|us|788.168403|775.280258|-1.63520%|
|post_attention_residual_ticks|us|190.204613|190.999504|0.41791%|
|post_attention_norm_ticks|us|1.045387|1.046627|0.11862%|
|gate_up_ticks|us|1758.159722|2018.015873|14.78001%|
|activation_ticks|us|0.000000|156.873760|N/A: zero control|
|down_ticks|us|1079.933036|1052.306548|-2.55817%|
|final_residual_ticks|us|2.439236|2.486359|1.93188%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|1.532738|1.542659|0.64725%|
|generation_final_norm_ticks|us|7.663690|7.502480|-2.10356%|
|generation_lm_head_ticks|us|1987.197421|1969.701141|-0.88045%|
|generation_lm_head_weight_dma_ticks|us|1406.237599|1415.504712|0.65900%|
|generation_lm_head_scale_dma_ticks|us|19.438244|19.476687|0.19777%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|1473.576389|1457.478919|-1.09241%|
|generation_lm_head_argmax_ticks|us|466.418651|465.052083|-0.29299%|
|generation_lm_head_weight_dma_wait_ticks|us|1331.204117|1346.558780|1.15344%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|81.499256|56.250000|-30.98096%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|2304.000000|2304.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|79006720.000000|79006720.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|653.000000|653.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|297992192.000000|297992192.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|595984384.000000|595984384.000000|0.00000%|
|weight_dma_ticks|us|5746.145833|5840.949901|1.64987%|
|hmx_compute_ticks|us|4250.752728|4024.052579|-5.33318%|
|projection_pack_ticks|us|4.236111|4.218750|-0.40984%|
|projection_hmx_wait_ticks|us|644.451885|559.073661|-13.24819%|
|projection_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_setup_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_softmax_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_gqa_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_unattributed_ticks|us|867.154018|865.509673|-0.18963%|
|u8_attention_qk_norm_rope_ticks|us|2030.432788|2004.556052|-1.27444%|
|u8_attention_k_pack_ticks|us|49.701141|49.671379|-0.05988%|
|u8_attention_v_pack_ticks|us|117.580605|117.074653|-0.43030%|
|u8_cache_native_append_update_ticks|us|251.801835|251.214038|-0.23344%|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_native_incremental_append_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_tail_append_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_segment_seal_count|raw|0.666667|0.666667|0.00000%|
|u8_cache_segment_sealed_bytes|bytes|45056.000000|45056.000000|0.00000%|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|224.000000|224.000000|0.00000%|
|u8_cache_v_vtcm_tail_publish_count|raw|53.333333|53.333333|0.00000%|
|u8_cache_v_vtcm_tail_seal_count|raw|5.333333|5.333333|0.00000%|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|336.000000|336.000000|0.00000%|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|420522.666667|420522.666667|0.00000%|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_seal_count|raw|4.666667|4.666667|0.00000%|
|u8_cache_k_vtcm_tail_cached_head_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_native_load_bytes|bytes|783701.333333|783701.333333|0.00000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|bytes|10285.333333|10285.333333|0.00000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|39.663938|39.377480|-0.72221%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|25088.000000|25088.000000|0.00000%|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|146.075149|146.221478|0.10017%|
|wide_score_mode|raw|4.000000|4.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|fp32_residual|raw|2.000000|2.000000|0.00000%|
|dense_r4_mode|raw|0.000000|4.000000|N/A: zero control|
|dense_r4_optimization|raw|6.000000|6.000000|0.00000%|
|dense_r4_calls|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_rows|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_pipeline_batches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_pipeline_hvx_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_dispatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_prepare_tiles|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_finish_groups|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_publish_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_consume_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prepare_ticks|us|0.000000|25.658482|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|80.280258|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|48.655754|N/A: zero control|
|dense_r4_audit_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_mode|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_optimization|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_audit|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt_calls|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_conversion_audit_mismatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_heads|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_constant_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_direct_slot_join_count|raw|147.000000|147.000000|0.00000%|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|0.000000|0.000000|N/A: zero control|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|354.987599|355.876736|0.25047%|
|u8_attention_av_hmx_ticks|us|168.391617|168.665675|0.16275%|
|u8_attention_av_requant_ticks|us|149.025298|147.566964|-0.97858%|
|u8_attention_pipeline_wait_ticks|us|64.515129|64.977679|0.71696%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|1119.440724|1141.221478|1.94568%|
|w4u8_qkvo_hmx_lifetime_ticks|us|4256.649306|4229.626736|-0.63483%|
|w4f16_gate_up_weight_dma_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_join_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_gate_up_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_down_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_activation_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_expanded_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|vtcm_requested_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_acquired_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_peak_plan_bytes|bytes|5744384.000000|6006528.000000|4.56348%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1101.000000|1101.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|593621.333333|593621.333333|0.00000%|
|weight_dma_descriptor_count|raw|1158.000000|1158.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|301962240.000000|301962240.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4732288.000000|4732288.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|807.587970|697.793776|-13.59532%|
|Host_wall_us|us|10782.547048|11036.564857|2.35582%|
|generation_lm_head_exclusive_ticks|us|1979.533730|1962.198661|-0.87571%|

## formal prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|227.728125|228.764583|1.08728%|0.92595%|
|Input RMSNorm|1024.977083|1039.475000|4.89370%|4.20740%|
|QKV＋RoPE|6639.671875|6637.846875|31.70074%|26.86751%|
|QK–Softmax–AV|3341.709896|3330.252083|15.95481%|13.47961%|
|O projection|1237.557292|1234.732292|5.90865%|4.99773%|
|Post-attention residual＋RMSNorm|1015.095833|1020.646875|4.84652%|4.13119%|
|Gate/Up＋SwiGLU|2675.266146|6421.882812|12.77291%|25.99336%|
|Down|1294.852604|1303.307813|6.18220%|5.27530%|
|Final residual|3.286979|3.257292|0.01569%|0.01318%|
|KV carrier conversion|135.896354|135.378646|0.64883%|0.54796%|
|KV append DMA|246.195833|246.276563|1.17545%|0.99683%|
|Block orchestration|39.100521|37.651042|0.18668%|0.15240%|
|Layer bookkeeping|23.973438|23.916146|0.11446%|0.09680%|
|Stage-boundary bookkeeping|33.503125|32.741667|0.15996%|0.13253%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|120.194792|116.928125|0.57386%|0.47328%|
|Embedding|42.508333|42.543229|0.20295%|0.17220%|
|Final model RMSNorm|7.569271|7.503125|0.03614%|0.03037%|
|LM head＋greedy（不含 final norm）|1977.532812|1981.419792|9.44162%|8.02004%|
|Host_boundary_us|858.225017|861.330162|4.09755%|3.48634%|
|Host_wall_us|20944.845330|24705.854120|100.00000%|100.00000%|

## formal prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|20954083.350000|24706078.100000|17.90579%|
|output_mismatches|raw|1.000000|1.000000|0.00000%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.000000|0.000000|N/A: zero control|
|output_nrmse|raw|0.000000|0.000000|N/A: zero control|
|output_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|output_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|output_max_required_rtol_after_atol|raw|0.000000|0.000000|N/A: zero control|
|output_fp16_atol|raw|0.062500|0.062500|0.00000%|
|output_fp16_rtol|raw|0.002000|0.002000|0.00000%|
|output_fp16_max_composed_nrmse|raw|0.003000|0.003000|0.00000%|
|output_max_lsb|raw|0.000000|0.000000|N/A: zero control|
|cache_prefix_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_structure_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_min_cosine|raw|1.000000|1.000000|0.00000%|
|cache_max_mixed_tolerance_violation_fraction|raw|0.000000|0.000000|N/A: zero control|
|cache_max_nrmse|raw|0.000000|0.000000|N/A: zero control|
|cache_compared_elements|raw|0.000000|0.000000|N/A: zero control|
|cache_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|cache_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|cache_tensor_count|raw|0.000000|0.000000|N/A: zero control|
|cache_composed_cosine_diagnostic_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_legacy_mixed_bound_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_fp16_min_cosine|raw|0.999990|0.999990|0.00000%|
|cache_fp16_max_violation_fraction|raw|0.010000|0.010000|0.00000%|
|fp16_norm_contexts|raw|4.000000|4.000000|0.00000%|
|fp16_norm_rows_per_task|raw|4.000000|4.000000|0.00000%|
|fp16_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_input_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|1.000000|1.000000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|14.000000|14.000000|0.00000%|
|kv_cache_v_format|raw|12.000000|12.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|56.000000|56.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_vector_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|2.000000|2.000000|0.00000%|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|5.000000|5.000000|0.00000%|
|w4u8_qkv_ring_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_head_publish_count|raw|672.000000|672.000000|0.00000%|
|w4u8_qkv_ring_pipeline_ticks|us|6603.091146|6601.289062|-0.02729%|
|w4u8_qkv_ring_dma_wait_ticks|us|1142.122396|1142.057292|-0.00570%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|26.609375|27.132812|1.96712%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|733.617188|737.065104|0.46999%|
|w4u8_qkv_ring_hmx_compute_ticks|us|299.098958|298.812500|-0.09577%|
|w4u8_qkv_ring_pool_wait_ticks|us|5124.411458|5122.781250|-0.03181%|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|765.085938|1968.294271|157.26447%|
|w4u8_decode_swiglu_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_norm_rope_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_pair_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_pair_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_rows_processed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_temp_carrier_skipped_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_padding_poison_pair_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_residual_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_vector_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|64.000000|64.000000|0.00000%|
|scan_total_kv_length|raw|64.000000|64.000000|0.00000%|
|scan_padded_kv_length|raw|64.000000|64.000000|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_attention_overlay_required_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_cache_dma_descriptor_count|raw|448.000000|448.000000|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_write_bytes|bytes|5906432.000000|5906432.000000|0.00000%|
|scan_cache_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|scan_cache_append_ticks|us|246.205729|246.296875|0.03702%|
|scan_cache_pack_ticks|us|135.867188|135.518229|-0.25684%|
|block_orchestration_ticks|us|38.841146|37.552083|-3.31881%|
|layer_bookkeeping_ticks|us|23.856771|23.898438|0.17465%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|20005.570312|23776.039062|18.84709%|
|invocation_ticks|us|20071.885417|23843.885417|18.79245%|
|runtime_setup_ticks|us|68.580729|67.721354|-1.25309%|
|runtime_teardown_ticks|us|49.231771|49.468750|0.48135%|
|stage_boundary_ticks|us|33.276042|32.635417|-1.92518%|
|ledger_named_ticks|us|20071.885417|23843.885417|18.79245%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.486979|0.460938|-5.34759%|
|metadata_stage_ticks|us|227.286458|228.606771|0.58090%|
|input_norm_ticks|us|1024.940104|1039.578125|1.42818%|
|qkv_projection_ticks|us|6640.315104|6638.289062|-0.03051%|
|qk_norm_rope_ticks|us|0.630208|0.619792|-1.65289%|
|attention_ticks|us|3341.497396|3330.567708|-0.32709%|
|o_projection_ticks|us|1237.911458|1234.843750|-0.24781%|
|post_attention_residual_ticks|us|1013.505208|1019.070312|0.54909%|
|post_attention_norm_ticks|us|1.583333|1.539062|-2.79605%|
|gate_up_ticks|us|2681.414062|3930.020833|46.56524%|
|activation_ticks|us|0.000000|2494.445312|N/A: zero control|
|down_ticks|us|1292.653646|1299.315104|0.51533%|
|final_residual_ticks|us|3.281250|3.268229|-0.39683%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|42.338542|42.585938|0.58433%|
|generation_final_norm_ticks|us|7.531250|7.510417|-0.27663%|
|generation_lm_head_ticks|us|1986.393229|1983.924479|-0.12428%|
|generation_lm_head_weight_dma_ticks|us|1427.322917|1425.776042|-0.10838%|
|generation_lm_head_scale_dma_ticks|us|19.747396|19.682292|-0.32968%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|1466.757812|1460.067708|-0.45612%|
|generation_lm_head_argmax_ticks|us|468.122396|471.960938|0.81999%|
|generation_lm_head_weight_dma_wait_ticks|us|1354.539062|1356.992188|0.18110%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|58.989583|47.151042|-20.06887%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|131328.000000|131328.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|79006720.000000|79006720.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|653.000000|653.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|297992192.000000|297992192.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|595984384.000000|595984384.000000|0.00000%|
|weight_dma_ticks|us|5933.651042|5935.067708|0.02388%|
|hmx_compute_ticks|us|5058.533854|5177.041667|2.34273%|
|projection_pack_ticks|us|4.213542|4.151042|-1.48331%|
|projection_hmx_wait_ticks|us|1255.492188|1291.419271|2.86159%|
|projection_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_setup_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_softmax_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_gqa_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_norm_rope_ticks|us|24317.778646|24307.937500|-0.04047%|
|u8_attention_k_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_v_pack_ticks|us|4054.651042|4013.510417|-1.01465%|
|u8_cache_native_append_update_ticks|us|380.302083|379.690104|-0.16092%|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|4014080.000000|4014080.000000|0.00000%|
|u8_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_tail_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_sealed_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|1.000000|1.000000|0.00000%|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|917504.000000|917504.000000|0.00000%|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|1.000000|1.000000|0.00000%|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|827904.000000|827904.000000|0.00000%|
|u8_cache_k_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_correction_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_hvx_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|601.325521|597.536458|-0.63012%|
|wide_score_mode|raw|4.000000|4.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|fp32_residual|raw|2.000000|2.000000|0.00000%|
|dense_r4_mode|raw|0.000000|4.000000|N/A: zero control|
|dense_r4_optimization|raw|6.000000|6.000000|0.00000%|
|dense_r4_calls|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_rows|raw|0.000000|1792.000000|N/A: zero control|
|dense_r4_pipeline_batches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_pipeline_hvx_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_dispatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_prepare_tiles|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_finish_groups|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_publish_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_consume_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prepare_ticks|us|0.000000|285.627604|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|1363.330729|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|841.778646|N/A: zero control|
|dense_r4_audit_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_mode|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_optimization|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_audit|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt_calls|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_conversion_audit_mismatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_heads|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_constant_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_direct_slot_join_count|raw|147.000000|147.000000|0.00000%|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|0.000000|0.000000|N/A: zero control|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|8716.539062|8720.315104|0.04332%|
|u8_attention_av_hmx_ticks|us|625.033854|625.437500|0.06458%|
|u8_attention_av_requant_ticks|us|1716.791667|1717.039062|0.01441%|
|u8_attention_pipeline_wait_ticks|us|2037.338542|2032.289062|-0.24785%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|1143.161458|1143.080729|-0.00706%|
|w4u8_qkvo_hmx_lifetime_ticks|us|4944.101562|4989.997396|0.92829%|
|w4f16_gate_up_weight_dma_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_join_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_gate_up_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_down_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_activation_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_expanded_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|vtcm_requested_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_acquired_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_peak_plan_bytes|bytes|5744384.000000|6006528.000000|4.56348%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1101.000000|1101.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|589184.000000|589184.000000|0.00000%|
|weight_dma_descriptor_count|raw|1158.000000|1158.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|301962240.000000|301962240.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4861312.000000|4861312.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|869.799496|862.820300|-0.80239%|
|Host_wall_us|us|20954.083350|24706.078100|17.90579%|
|generation_lm_head_exclusive_ticks|us|1978.911458|1976.528646|-0.12041%|

## formal decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|224.818886|224.766592|2.10770%|2.03763%|
|Input RMSNorm|186.994829|185.777691|1.75310%|1.68418%|
|QKV＋RoPE|1486.889187|1486.572359|13.93976%|13.47661%|
|QK–Softmax–AV|1919.598115|1919.267150|17.99646%|17.39923%|
|O projection|775.271230|774.267212|7.26826%|7.01916%|
|Post-attention residual＋RMSNorm|191.345747|191.997148|1.79389%|1.74056%|
|Gate/Up＋SwiGLU|1770.212661|2185.274182|16.59595%|19.81073%|
|Down|1063.539844|1066.644147|9.97081%|9.66972%|
|Final residual|2.414869|2.480618|0.02264%|0.02249%|
|KV carrier conversion|164.509623|164.311496|1.54230%|1.48958%|
|KV append DMA|88.329216|88.375980|0.82810%|0.80118%|
|Block orchestration|29.767175|29.182639|0.27907%|0.26456%|
|Layer bookkeeping|16.267870|16.257490|0.15251%|0.14738%|
|Stage-boundary bookkeeping|1.751166|1.787388|0.01642%|0.01620%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|72.119370|72.065749|0.67613%|0.65332%|
|Embedding|1.587835|1.608259|0.01489%|0.01458%|
|Final model RMSNorm|7.661037|7.511520|0.07182%|0.06810%|
|LM head＋greedy（不含 final norm）|1973.058842|1970.545412|18.49766%|17.86409%|
|Host_boundary_us|690.396306|642.068744|6.47255%|5.82071%|
|Host_wall_us|10666.533806|11030.761775|100.00000%|100.00000%|

## formal decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|10681894.210714|11024037.984524|3.20303%|
|output_mismatches|raw|0.357143|0.238095|-33.33333%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.642857|0.761905|18.51852%|
|output_nrmse|raw|0.000000|0.000000|N/A: zero control|
|output_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|output_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|output_max_required_rtol_after_atol|raw|0.000000|0.000000|N/A: zero control|
|output_fp16_atol|raw|0.062500|0.062500|0.00000%|
|output_fp16_rtol|raw|0.002000|0.002000|0.00000%|
|output_fp16_max_composed_nrmse|raw|0.003000|0.003000|0.00000%|
|output_max_lsb|raw|0.000000|0.000000|N/A: zero control|
|cache_prefix_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_structure_mismatches|raw|0.000000|0.000000|N/A: zero control|
|cache_min_cosine|raw|1.000000|1.000000|0.00000%|
|cache_max_mixed_tolerance_violation_fraction|raw|0.000000|0.000000|N/A: zero control|
|cache_max_nrmse|raw|0.000000|0.000000|N/A: zero control|
|cache_compared_elements|raw|0.000000|0.000000|N/A: zero control|
|cache_mixed_tolerance_violations|raw|0.000000|0.000000|N/A: zero control|
|cache_nonfinite_count|raw|0.000000|0.000000|N/A: zero control|
|cache_tensor_count|raw|0.000000|0.000000|N/A: zero control|
|cache_composed_cosine_diagnostic_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_legacy_mixed_bound_failure_count|raw|0.000000|0.000000|N/A: zero control|
|cache_fp16_min_cosine|raw|0.999990|0.999990|0.00000%|
|cache_fp16_max_violation_fraction|raw|0.010000|0.010000|0.00000%|
|fp16_norm_contexts|raw|4.000000|4.000000|0.00000%|
|fp16_norm_rows_per_task|raw|4.000000|4.000000|0.00000%|
|fp16_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_input_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|fp16_post_residual_norm_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|22.500000|22.500000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|14.000000|14.000000|0.00000%|
|kv_cache_v_format|raw|12.000000|12.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|56.000000|56.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_av_requant_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_av_requant_vector_count|raw|1792.000000|1792.000000|0.00000%|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|2.000000|2.000000|0.00000%|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|5.000000|5.000000|0.00000%|
|w4u8_qkv_ring_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|28.000000|28.000000|0.00000%|
|w4u8_qkv_ring_head_publish_count|raw|672.000000|672.000000|0.00000%|
|w4u8_qkv_ring_pipeline_ticks|us|1456.649368|1456.137835|-0.03512%|
|w4u8_qkv_ring_dma_wait_ticks|us|1142.257812|1142.873822|0.05393%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|37.211868|35.043279|-5.82768%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|722.297743|725.335503|0.42057%|
|w4u8_qkv_ring_hmx_compute_ticks|us|313.870102|312.747830|-0.35756%|
|w4u8_qkv_ring_pool_wait_ticks|us|3.916791|3.925347|0.21846%|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|76.149988|75.732763|-0.54790%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|277.227617|277.450087|0.08025%|
|w4u8_gate_up_swiglu_publish_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|84.000000|84.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|276.623264|1138.065538|311.41353%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|1304.391245|711.797433|-45.43068%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|95.345486|317.433470|232.92973%|
|w4u8_decode_swiglu_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qk_norm_rope_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_q_pair_row4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_k_pair_row4_call_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_qk_rows_processed|raw|2688.000000|2688.000000|0.00000%|
|w4u8_decode_k_temp_carrier_skipped_count|raw|224.000000|224.000000|0.00000%|
|w4u8_qk_padding_poison_pair_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_q_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_k_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_residual_active_contexts|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_main_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_worker_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_hvx_tile4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|2688.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|2688.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|1507328.000000|1730560.000000|14.80978%|
|scan_attention_overlay_required_bytes|bytes|83675.428571|83675.428571|0.00000%|
|scan_cache_dma_descriptor_count|raw|1202.666667|1202.666667|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4559445.333333|4559445.333333|0.00000%|
|scan_cache_ddr_write_bytes|bytes|77312.000000|77312.000000|0.00000%|
|scan_cache_stage_ticks|us|537.178075|537.353237|0.03261%|
|scan_cache_append_ticks|us|88.310392|88.379154|0.07786%|
|scan_cache_pack_ticks|us|164.511037|164.311942|-0.12102%|
|block_orchestration_ticks|us|29.771019|29.182664|-1.97627%|
|layer_bookkeeping_ticks|us|16.258309|16.255828|-0.01525%|
|scan_dynamic_attention_ticks|us|1913.708953|1913.153894|-0.02900%|
|total_ticks|us|9944.517795|10345.477431|4.03197%|
|invocation_ticks|us|9982.513641|10383.580357|4.01769%|
|runtime_setup_ticks|us|38.055804|37.960875|-0.24944%|
|runtime_teardown_ticks|us|34.046689|34.077381|0.09015%|
|stage_boundary_ticks|us|1.753472|1.784474|1.76803%|
|ledger_named_ticks|us|9982.513641|10383.580357|4.01769%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.384673|0.381324|-0.87041%|
|metadata_stage_ticks|us|224.569630|224.368862|-0.08940%|
|input_norm_ticks|us|186.977803|185.741195|-0.66137%|
|qkv_projection_ticks|us|1486.905382|1486.408978|-0.03339%|
|qk_norm_rope_ticks|us|0.334139|0.333457|-0.20412%|
|attention_ticks|us|1919.393973|1918.839844|-0.02887%|
|o_projection_ticks|us|777.565476|772.401600|-0.66411%|
|post_attention_residual_ticks|us|190.278398|190.925161|0.33990%|
|post_attention_norm_ticks|us|1.062624|1.065786|0.29758%|
|gate_up_ticks|us|1768.552207|2029.354477|14.74665%|
|activation_ticks|us|0.000000|156.788132|N/A: zero control|
|down_ticks|us|1061.334697|1065.619606|0.40373%|
|final_residual_ticks|us|2.419395|2.478609|2.44746%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|1.579303|1.616009|2.32421%|
|generation_final_norm_ticks|us|7.662264|7.512773|-1.95101%|
|generation_lm_head_ticks|us|1979.958209|1969.533172|-0.52653%|
|generation_lm_head_weight_dma_ticks|us|1421.391431|1421.532242|0.00991%|
|generation_lm_head_scale_dma_ticks|us|19.640191|19.623946|-0.08271%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|1467.560950|1457.741195|-0.66912%|
|generation_lm_head_argmax_ticks|us|464.857763|464.472470|-0.08288%|
|generation_lm_head_weight_dma_wait_ticks|us|1351.069878|1355.440724|0.32351%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|60.908358|48.784846|-19.90451%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|2304.000000|2304.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|79006720.000000|79006720.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|653.000000|653.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|112.000000|112.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|297992192.000000|297992192.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|595984384.000000|595984384.000000|0.00000%|
|weight_dma_ticks|us|5864.012959|5874.098710|0.17199%|
|hmx_compute_ticks|us|4062.615203|4035.974826|-0.65574%|
|projection_pack_ticks|us|4.181238|4.196305|0.36035%|
|projection_hmx_wait_ticks|us|560.784350|565.436074|0.82950%|
|projection_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_cross_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_setup_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_qk_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_softmax_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_hmx_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_av_unpack_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_gqa_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|attention_unattributed_ticks|us|868.051959|870.158792|0.24271%|
|u8_attention_qk_norm_rope_ticks|us|2024.190476|2022.770957|-0.07013%|
|u8_attention_k_pack_ticks|us|49.731461|49.753038|0.04339%|
|u8_attention_v_pack_ticks|us|117.446367|117.162512|-0.24169%|
|u8_cache_native_append_update_ticks|us|251.224268|251.133991|-0.03594%|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_native_incremental_append_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_tail_append_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_segment_seal_count|raw|0.666667|0.666667|0.00000%|
|u8_cache_segment_sealed_bytes|bytes|45056.000000|45056.000000|0.00000%|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|224.000000|224.000000|0.00000%|
|u8_cache_v_vtcm_tail_publish_count|raw|53.333333|53.333333|0.00000%|
|u8_cache_v_vtcm_tail_seal_count|raw|5.333333|5.333333|0.00000%|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|336.000000|336.000000|0.00000%|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|420522.666667|420522.666667|0.00000%|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_seal_count|raw|4.666667|4.666667|0.00000%|
|u8_cache_k_vtcm_tail_cached_head_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|28.000000|28.000000|0.00000%|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_native_load_bytes|bytes|783701.333333|783701.333333|0.00000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|bytes|10285.333333|10285.333333|0.00000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|39.517051|39.376922|-0.35460%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|25088.000000|25088.000000|0.00000%|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|146.459015|145.354167|-0.75437%|
|wide_score_mode|raw|4.000000|4.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|fp32_residual|raw|2.000000|2.000000|0.00000%|
|dense_r4_mode|raw|0.000000|4.000000|N/A: zero control|
|dense_r4_optimization|raw|6.000000|6.000000|0.00000%|
|dense_r4_calls|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_rows|raw|0.000000|28.000000|N/A: zero control|
|dense_r4_pipeline_batches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_pipeline_hvx_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_join_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_dispatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_prepare_tiles|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_parallel_finish_groups|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_publish_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prefill_consume_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r4_prepare_ticks|us|0.000000|25.596354|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|80.292039|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|48.694382|N/A: zero control|
|dense_r4_audit_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_mode|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_optimization|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_audit|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_opt_calls|raw|0.000000|0.000000|N/A: zero control|
|w4f16_decode_conversion_audit_mismatches|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_heads|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_constant_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|dense_r3_total_parallel_work_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_direct_slot_join_count|raw|147.000000|147.000000|0.00000%|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|0.000000|0.000000|N/A: zero control|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|355.320747|354.584635|-0.20717%|
|u8_attention_av_hmx_ticks|us|169.044395|168.589162|-0.26930%|
|u8_attention_av_requant_ticks|us|148.196677|146.760665|-0.96899%|
|u8_attention_pipeline_wait_ticks|us|65.133371|66.598214|2.24899%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|1143.234623|1143.844246|0.05332%|
|w4u8_qkvo_hmx_lifetime_ticks|us|4235.577877|4247.949529|0.29209%|
|w4f16_gate_up_weight_dma_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_expand_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_hmx_tail_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4f16_gate_up_stream_join_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_gate_up_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_down_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_activation_work_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_mlp_expanded_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|vtcm_requested_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_acquired_bytes|bytes|8388608.000000|8388608.000000|0.00000%|
|vtcm_peak_plan_bytes|bytes|5744384.000000|6006528.000000|4.56348%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1101.000000|1101.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|593621.333333|593621.333333|0.00000%|
|weight_dma_descriptor_count|raw|1158.000000|1158.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|301962240.000000|301962240.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4732288.000000|4732288.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|694.118238|644.377266|-7.16607%|
|Host_wall_us|us|10681.894211|11024.037985|3.20303%|
|generation_lm_head_exclusive_ticks|us|1972.304067|1962.014385|-0.52171%|

## Validation
8686 profiled invocations reconciled exactly. Layer hashes, padding, final hidden, KV payloads, final norm and bounds are in the model evidence. Failed candidate attempts in RECOVERY.md are excluded from all tables. No baseline/default promotion.
