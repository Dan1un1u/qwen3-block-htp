# L32-0072 Llama3.2-3B complete paired profiling

Full model fixed64+42, ordinary W4A8 versus HVX R4. FP32 residual, R3/SP2/INT16 Down off. Timings exclude audit dumping, cold load and external tokenizer. Each timed head agrees with its own independent reference.

Formal: ten alternating pairs, repeat10 per arm. Auxiliary: one warmup repeat1 per arm, descriptive only. Module means reconcile to Host wall; counter tables show medians of per-round phase means. Decode is per step, not all42. Raw engine/worker/wait counters overlap and MUST NOT be summed as an additive ledger. R4 prepare/core/finish are subsets of Activation.

Evidence directory: /mnt/d/llm_exp/results/llama32-htp/l32-0072/3B; sealed package hash and flags are in per-run protocol.json. Measured source is in the protocol-referenced binary seal. No quality or speed acceptance claim.

## auxiliary prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|243.697917|237.187500|0.45136%|0.38163%|
|Input RMSNorm|3724.843750|3717.343750|6.89894%|5.98107%|
|QKV＋RoPE|9237.552083|9253.385417|17.10925%|14.88836%|
|QK–Softmax–AV|6004.479167|6011.666667|11.12114%|9.67255%|
|O projection|3134.270833|3123.020833|5.80511%|5.02483%|
|Post-attention residual＋RMSNorm|3676.875000|3732.187500|6.81009%|6.00495%|
|Gate/Up＋SwiGLU|14377.291667|21871.822917|26.62878%|35.19096%|
|Down|6527.968750|7049.010417|12.09072%|11.34160%|
|Final residual|3.854167|4.322917|0.00714%|0.00696%|
|KV carrier conversion|81.250000|81.197917|0.15049%|0.13064%|
|KV append DMA|244.479167|243.750000|0.45281%|0.39218%|
|Block orchestration|37.187500|37.864583|0.06888%|0.06092%|
|Layer bookkeeping|23.697917|22.968750|0.04389%|0.03696%|
|Stage-boundary bookkeeping|6.979167|7.656250|0.01293%|0.01232%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|80.885417|80.104167|0.14981%|0.12888%|
|Embedding|70.364583|71.406250|0.13033%|0.11489%|
|Final model RMSNorm|87.656250|85.937500|0.16235%|0.13827%|
|LM head＋greedy（不含 final norm）|4142.760417|4025.520833|7.67298%|6.47692%|
|Host_boundary_us|2285.468250|2495.467833|4.23301%|4.01512%|
|Host_wall_us|53991.562000|62151.822000|100.00000%|100.00000%|

## auxiliary prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|53991562.000000|62151822.000000|15.11395%|
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
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|1.000000|1.000000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|1.000000|1.000000|0.00000%|
|kv_cache_v_format|raw|1.000000|1.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|168.000000|168.000000|0.00000%|
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
|w4u8_decode_qk_norm_rope_rows|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_batch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_head_publish_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dma_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|551.510417|1009.166667|82.98234%|
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
|scan_cache_ddr_write_bytes|bytes|3670016.000000|3670016.000000|0.00000%|
|scan_cache_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|scan_cache_append_ticks|us|244.479167|243.750000|-0.29825%|
|scan_cache_pack_ticks|us|81.250000|81.197917|-0.06410%|
|block_orchestration_ticks|us|37.187500|37.864583|1.82073%|
|layer_bookkeeping_ticks|us|23.697917|22.968750|-3.07692%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|51665.833333|59616.041667|15.38775%|
|invocation_ticks|us|51706.093750|59656.354167|15.37587%|
|runtime_setup_ticks|us|40.260417|40.312500|0.12937%|
|runtime_teardown_ticks|us|40.625000|39.791667|-2.05128%|
|stage_boundary_ticks|us|6.979167|7.656250|9.70149%|
|ledger_named_ticks|us|51706.093750|59656.354167|15.37587%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.364583|0.312500|-14.28571%|
|metadata_stage_ticks|us|243.333333|236.875000|-2.65411%|
|input_norm_ticks|us|3724.843750|3717.343750|-0.20135%|
|qkv_projection_ticks|us|3600.885417|3607.604167|0.18659%|
|qk_norm_rope_ticks|us|5636.666667|5645.781250|0.16170%|
|attention_ticks|us|6004.479167|6011.666667|0.11970%|
|o_projection_ticks|us|3134.270833|3123.020833|-0.35894%|
|post_attention_residual_ticks|us|3675.833333|3731.197917|1.50618%|
|post_attention_norm_ticks|us|1.041667|0.989583|-5.00000%|
|gate_up_ticks|us|14377.291667|14497.708333|0.83755%|
|activation_ticks|us|0.000000|7374.114583|N/A: zero control|
|down_ticks|us|6527.968750|7049.010417|7.98168%|
|final_residual_ticks|us|3.854167|4.322917|12.16216%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|70.364583|71.406250|1.48038%|
|generation_final_norm_ticks|us|87.656250|85.937500|-1.96078%|
|generation_lm_head_ticks|us|4230.416667|4111.458333|-2.81198%|
|generation_lm_head_weight_dma_ticks|us|3714.270833|3597.395833|-3.14665%|
|generation_lm_head_scale_dma_ticks|us|19.218750|19.218750|0.00000%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|3687.291667|3568.802083|-3.21346%|
|generation_lm_head_argmax_ticks|us|393.541667|392.083333|-0.37057%|
|generation_lm_head_weight_dma_wait_ticks|us|3642.708333|3522.708333|-3.29425%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|9.479167|10.729167|13.18681%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|126.000000|126.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4008.000000|4008.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|125.000000|125.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1026048.000000|1026048.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|393472.000000|393472.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|198027264.000000|198027264.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|1218.000000|1218.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|336.000000|336.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|1606287360.000000|1606287360.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|3212574720.000000|3212574720.000000|0.00000%|
|weight_dma_ticks|us|33111.041667|32892.864583|-0.65893%|
|hmx_compute_ticks|us|13899.791667|13810.416667|-0.64300%|
|projection_pack_ticks|us|13.697917|13.906250|1.52091%|
|projection_hmx_wait_ticks|us|1213.333333|1468.020833|20.99073%|
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
|u8_attention_qk_norm_rope_ticks|us|1.093750|1.927083|76.19048%|
|u8_attention_k_pack_ticks|us|1915.416667|1920.052083|0.24201%|
|u8_attention_v_pack_ticks|us|3315.208333|3319.843750|0.13982%|
|u8_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_attention_qk_hmx_ticks|us|649.791667|644.218750|-0.85765%|
|wide_score_mode|raw|8.000000|8.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|llama_fp32_residual|raw|1.000000|1.000000|0.00000%|
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
|dense_r4_prepare_ticks|us|0.000000|661.458333|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|4577.656250|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|2131.770833|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|124.000000|124.000000|0.00000%|
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
|u8_attention_qk_requant_ticks|us|742.343750|747.708333|0.72265%|
|u8_attention_softmax_ticks|us|15064.375000|15099.635417|0.23406%|
|u8_attention_av_hmx_ticks|us|661.979167|661.041667|-0.14162%|
|u8_attention_av_requant_ticks|us|8.854167|9.270833|4.70588%|
|u8_attention_pipeline_wait_ticks|us|936.927083|912.916667|-2.56268%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_hmx_lifetime_ticks|us|25707.916667|25328.958333|-1.47409%|
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
|vtcm_peak_plan_bytes|bytes|8360416.000000|8360416.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1666.000000|1666.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|3148032.000000|3148032.000000|0.00000%|
|weight_dma_descriptor_count|raw|2311.000000|2311.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|1613512704.000000|1613512704.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|5352832.000000|5352832.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_error|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_ticks|us|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_ticks|us|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|2285.468250|2495.467833|9.18847%|
|Host_wall_us|us|53991.562000|62151.822000|15.11395%|
|generation_lm_head_exclusive_ticks|us|4142.760417|4025.520833|-2.82999%|

## auxiliary decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|280.562996|251.541419|0.64475%|0.56890%|
|Input RMSNorm|2315.540675|2315.572917|5.32126%|5.23700%|
|QKV＋RoPE|4004.722222|4016.834077|9.20310%|9.08464%|
|QK–Softmax–AV|6418.328373|6420.014881|14.74972%|14.51978%|
|O projection|2949.409722|2958.371776|6.77793%|6.69078%|
|Post-attention residual＋RMSNorm|2312.341270|2315.982143|5.31390%|5.23792%|
|Gate/Up＋SwiGLU|13140.166171|13506.440972|30.19692%|30.54674%|
|Down|6478.375496|7044.977679|14.88771%|15.93322%|
|Final residual|2.094494|2.105655|0.00481%|0.00476%|
|KV carrier conversion|12.787698|12.555804|0.02939%|0.02840%|
|KV append DMA|183.829365|179.068700|0.42245%|0.40499%|
|Block orchestration|32.003968|31.180556|0.07355%|0.07052%|
|Layer bookkeeping|16.448413|16.566220|0.03780%|0.03747%|
|Stage-boundary bookkeeping|1.851438|1.800595|0.00425%|0.00407%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|51.328125|51.664187|0.11796%|0.11685%|
|Embedding|2.580605|2.434276|0.00593%|0.00551%|
|Final model RMSNorm|84.428323|84.192708|0.19402%|0.19041%|
|LM head＋greedy（不含 final norm）|4139.285714|4023.726438|9.51234%|9.10023%|
|Host_boundary_us|1088.839192|980.618879|2.50222%|2.21781%|
|Host_wall_us|43514.924262|44215.649881|100.00000%|100.00000%|

## auxiliary decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|43514924.261905|44215649.880952|1.61031%|
|output_mismatches|raw|0.785714|0.952381|21.21212%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.214286|0.047619|-77.77778%|
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
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|22.500000|22.500000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|1.000000|1.000000|0.00000%|
|kv_cache_v_format|raw|1.000000|1.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_av_requant_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_av_requant_vector_count|raw|2688.000000|2688.000000|0.00000%|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_batch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_head_publish_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dma_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|2.240823|2.655010|18.48367%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|2324.476687|2328.544147|0.17498%|
|w4u8_gate_up_swiglu_publish_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|715.926339|3075.400546|329.56941%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|12363.696677|9875.799851|-20.12260%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|88.102679|304.662698|245.80413%|
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
|w4u8_decode_softmax_hvx_tile4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|7168.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|7168.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|3295232.000000|3295232.000000|0.00000%|
|scan_attention_overlay_required_bytes|bytes|61537.523810|61537.523810|0.00000%|
|scan_cache_dma_descriptor_count|raw|896.000000|896.000000|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4902912.000000|4902912.000000|0.00000%|
|scan_cache_ddr_write_bytes|bytes|57344.000000|57344.000000|0.00000%|
|scan_cache_stage_ticks|us|474.786706|475.119048|0.07000%|
|scan_cache_append_ticks|us|183.829365|179.068700|-2.58972%|
|scan_cache_pack_ticks|us|12.787698|12.555804|-1.81342%|
|block_orchestration_ticks|us|32.003968|31.180556|-2.57285%|
|layer_bookkeeping_ticks|us|16.448413|16.566220|0.71622%|
|scan_dynamic_attention_ticks|us|6413.768601|6415.401786|0.02546%|
|total_ticks|us|42399.455605|43207.878224|1.90668%|
|invocation_ticks|us|42426.085069|43235.031002|1.90672%|
|runtime_setup_ticks|us|26.629464|27.152778|1.96517%|
|runtime_teardown_ticks|us|24.698661|24.511409|-0.75815%|
|stage_boundary_ticks|us|1.851438|1.800595|-2.74615%|
|ledger_named_ticks|us|42426.085069|43235.031002|1.90672%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.319940|0.293899|-8.13953%|
|metadata_stage_ticks|us|280.243056|251.247520|-10.34657%|
|input_norm_ticks|us|2315.540675|2315.572917|0.00139%|
|qkv_projection_ticks|us|3621.457093|3632.807540|0.31342%|
|qk_norm_rope_ticks|us|383.265129|384.026538|0.19866%|
|attention_ticks|us|6418.328373|6420.014881|0.02628%|
|o_projection_ticks|us|2949.409722|2958.371776|0.30386%|
|post_attention_residual_ticks|us|2311.522817|2315.178571|0.15815%|
|post_attention_norm_ticks|us|0.818452|0.803571|-1.81818%|
|gate_up_ticks|us|13140.166171|13039.372520|-0.76707%|
|activation_ticks|us|0.000000|467.068452|N/A: zero control|
|down_ticks|us|6478.375496|7044.977679|8.74605%|
|final_residual_ticks|us|2.094494|2.105655|0.53286%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|2.580605|2.434276|-5.67035%|
|generation_final_norm_ticks|us|84.428323|84.192708|-0.27907%|
|generation_lm_head_ticks|us|4223.714038|4107.919147|-2.74154%|
|generation_lm_head_weight_dma_ticks|us|3714.900794|3598.401538|-3.13600%|
|generation_lm_head_scale_dma_ticks|us|18.701637|19.501488|4.27690%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|3691.765873|3573.062996|-3.21534%|
|generation_lm_head_argmax_ticks|us|390.713046|390.695685|-0.00444%|
|generation_lm_head_weight_dma_wait_ticks|us|3645.580357|3527.361111|-3.24281%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|10.607639|10.524554|-0.78326%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|126.000000|126.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4008.000000|4008.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|125.000000|125.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1026048.000000|1026048.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|6400.000000|6400.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|198027264.000000|198027264.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|1218.000000|1218.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|336.000000|336.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|1606287360.000000|1606287360.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|3212574720.000000|3212574720.000000|0.00000%|
|weight_dma_ticks|us|33297.118056|33157.921627|-0.41804%|
|hmx_compute_ticks|us|12568.511905|12606.026786|0.29848%|
|projection_pack_ticks|us|12.729415|12.750496|0.16561%|
|projection_hmx_wait_ticks|us|983.142361|1233.151042|25.42955%|
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
|attention_unattributed_ticks|us|792.901786|793.151042|0.03144%|
|u8_attention_qk_norm_rope_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_k_pack_ticks|us|2131.491815|2131.984127|0.02310%|
|u8_attention_v_pack_ticks|us|1650.096726|1649.749504|-0.02104%|
|u8_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_full_prefix_pack_count|raw|448.000000|448.000000|0.00000%|
|u8_cache_segment_tail_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_sealed_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_attention_qk_hmx_ticks|us|627.806300|628.397817|0.09422%|
|wide_score_mode|raw|8.000000|8.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|llama_fp32_residual|raw|1.000000|1.000000|0.00000%|
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
|dense_r4_prepare_ticks|us|0.000000|56.200397|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|279.603175|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|129.265873|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|124.000000|124.000000|0.00000%|
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
|u8_attention_softmax_ticks|us|555.998264|555.533234|-0.08364%|
|u8_attention_av_hmx_ticks|us|648.754960|649.899554|0.17643%|
|u8_attention_av_requant_ticks|us|11.278522|11.299603|0.18692%|
|u8_attention_pipeline_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_hmx_lifetime_ticks|us|25566.820437|25244.475446|-1.26079%|
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
|vtcm_peak_plan_bytes|bytes|8360416.000000|8360416.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1666.000000|1666.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|3143082.666667|3143082.666667|0.00000%|
|weight_dma_descriptor_count|raw|2311.000000|2311.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|1613512704.000000|1613512704.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4965760.000000|4965760.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_error|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_ticks|us|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_ticks|us|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|1088.839192|980.618879|-9.93905%|
|Host_wall_us|us|43514.924262|44215.649881|1.61031%|
|generation_lm_head_exclusive_ticks|us|4139.285714|4023.726438|-2.79177%|

## formal prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|241.442708|243.346354|0.47162%|0.40977%|
|Input RMSNorm|3705.080208|3696.493750|7.23730%|6.22445%|
|QKV＋RoPE|9102.469271|9116.186458|17.78026%|15.35057%|
|QK–Softmax–AV|6022.598437|6014.350521|11.76421%|10.12745%|
|O projection|3120.893750|3122.006771|6.09618%|5.25709%|
|Post-attention residual＋RMSNorm|3676.207292|3734.142188|7.18090%|6.28785%|
|Gate/Up＋SwiGLU|13704.885937|21332.405729|26.77037%|35.92122%|
|Down|6270.565625|6780.807292|12.24858%|11.41807%|
|Final residual|3.681250|3.990625|0.00719%|0.00672%|
|KV carrier conversion|81.142708|81.670833|0.15850%|0.13752%|
|KV append DMA|236.818750|236.158854|0.46259%|0.39766%|
|Block orchestration|40.155208|40.578646|0.07844%|0.06833%|
|Layer bookkeeping|24.140104|24.283854|0.04715%|0.04089%|
|Stage-boundary bookkeeping|7.510417|7.763021|0.01467%|0.01307%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|93.623958|95.645833|0.18288%|0.16106%|
|Embedding|75.296354|75.614063|0.14708%|0.12733%|
|Final model RMSNorm|85.564583|85.641667|0.16714%|0.14421%|
|LM head＋greedy（不含 final norm）|3865.068229|3837.086979|7.54981%|6.46120%|
|Host_boundary_us|837.092708|858.468182|1.63513%|1.44556%|
|Host_wall_us|51194.237500|59386.641620|100.00000%|100.00000%|

## formal prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|51185630.250000|59356112.000000|15.96245%|
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
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|1.000000|1.000000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|1.000000|1.000000|0.00000%|
|kv_cache_v_format|raw|1.000000|1.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|168.000000|168.000000|0.00000%|
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
|w4u8_decode_qk_norm_rope_rows|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_batch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_head_publish_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dma_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|544.932292|1023.179688|87.76272%|
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
|scan_cache_ddr_write_bytes|bytes|3670016.000000|3670016.000000|0.00000%|
|scan_cache_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|scan_cache_append_ticks|us|236.841146|235.859375|-0.41453%|
|scan_cache_pack_ticks|us|81.083333|81.692708|0.75154%|
|block_orchestration_ticks|us|40.359375|40.372396|0.03226%|
|layer_bookkeeping_ticks|us|23.914062|23.986979|0.30491%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|50304.682292|58492.013021|16.27548%|
|invocation_ticks|us|50359.171875|58550.606771|16.26602%|
|runtime_setup_ticks|us|55.138021|54.877604|-0.47230%|
|runtime_teardown_ticks|us|39.140625|39.955729|2.08250%|
|stage_boundary_ticks|us|7.598958|7.851562|3.32419%|
|ledger_named_ticks|us|50359.171875|58550.606771|16.26602%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.468750|0.463542|-1.11111%|
|metadata_stage_ticks|us|240.343750|245.294271|2.05977%|
|input_norm_ticks|us|3704.986979|3696.151042|-0.23849%|
|qkv_projection_ticks|us|3467.296875|3467.156250|-0.00406%|
|qk_norm_rope_ticks|us|5634.411458|5649.007812|0.25906%|
|attention_ticks|us|6022.729167|6014.940104|-0.12933%|
|o_projection_ticks|us|3116.841146|3124.794271|0.25517%|
|post_attention_residual_ticks|us|3674.877604|3732.710938|1.57375%|
|post_attention_norm_ticks|us|1.166667|1.210938|3.79464%|
|gate_up_ticks|us|13702.526042|13958.502604|1.86810%|
|activation_ticks|us|0.000000|7371.955729|N/A: zero control|
|down_ticks|us|6262.179688|6780.497396|8.27695%|
|final_residual_ticks|us|3.695312|3.976562|7.61099%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|75.302083|75.388021|0.11412%|
|generation_final_norm_ticks|us|85.557292|85.705729|0.17349%|
|generation_lm_head_ticks|us|3949.854167|3920.557292|-0.74172%|
|generation_lm_head_weight_dma_ticks|us|3430.432292|3404.343750|-0.76050%|
|generation_lm_head_scale_dma_ticks|us|19.414062|19.239583|-0.89873%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|3409.278646|3381.625000|-0.81113%|
|generation_lm_head_argmax_ticks|us|393.546875|392.205729|-0.34078%|
|generation_lm_head_weight_dma_wait_ticks|us|3358.156250|3332.437500|-0.76586%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|14.820312|13.804688|-6.85293%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|126.000000|126.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4008.000000|4008.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|125.000000|125.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1026048.000000|1026048.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|393472.000000|393472.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|198027264.000000|198027264.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|1218.000000|1218.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|336.000000|336.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|1606287360.000000|1606287360.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|3212574720.000000|3212574720.000000|0.00000%|
|weight_dma_ticks|us|31667.239583|31611.148438|-0.17713%|
|hmx_compute_ticks|us|14440.309896|14516.343750|0.52654%|
|projection_pack_ticks|us|14.117188|14.190104|0.51651%|
|projection_hmx_wait_ticks|us|1281.320312|1594.908854|24.47386%|
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
|u8_attention_qk_norm_rope_ticks|us|1.481771|1.479167|-0.17575%|
|u8_attention_k_pack_ticks|us|1918.778646|1918.411458|-0.01914%|
|u8_attention_v_pack_ticks|us|3322.372396|3310.447917|-0.35891%|
|u8_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_attention_qk_hmx_ticks|us|656.755208|656.270833|-0.07375%|
|wide_score_mode|raw|8.000000|8.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|llama_fp32_residual|raw|1.000000|1.000000|0.00000%|
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
|dense_r4_prepare_ticks|us|0.000000|659.203125|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|4577.854167|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|2131.296875|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|124.000000|124.000000|0.00000%|
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
|u8_attention_qk_requant_ticks|us|746.802083|747.906250|0.14785%|
|u8_attention_softmax_ticks|us|15077.203125|15077.924479|0.00478%|
|u8_attention_av_hmx_ticks|us|675.942708|671.484375|-0.65957%|
|u8_attention_av_requant_ticks|us|8.598958|8.721354|1.42338%|
|u8_attention_pipeline_wait_ticks|us|958.906250|949.841146|-0.94536%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_hmx_lifetime_ticks|us|24692.231771|24449.398438|-0.98344%|
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
|vtcm_peak_plan_bytes|bytes|8360416.000000|8360416.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1666.000000|1666.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|3148032.000000|3148032.000000|0.00000%|
|weight_dma_descriptor_count|raw|2311.000000|2311.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|1613512704.000000|1613512704.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|5352832.000000|5352832.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_error|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_ticks|us|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_ticks|us|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|839.372383|864.268096|2.96599%|
|Host_wall_us|us|51185.630250|59356.112000|15.96245%|
|generation_lm_head_exclusive_ticks|us|3864.106771|3835.088542|-0.75097%|

## formal decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|240.827195|242.827765|0.57943%|0.57059%|
|Input RMSNorm|2315.667981|2315.489807|5.57146%|5.44090%|
|QKV＋RoPE|3848.907478|3853.850682|9.26042%|9.05571%|
|QK–Softmax–AV|6422.002827|6419.260082|15.45125%|15.08386%|
|O projection|2873.553906|2879.360392|6.91373%|6.76587%|
|Post-attention residual＋RMSNorm|2312.597123|2316.276290|5.56408%|5.44274%|
|Gate/Up＋SwiGLU|12444.999256|12905.913182|29.94249%|30.32608%|
|Down|6213.467671|6738.267250|14.94951%|15.83346%|
|Final residual|2.178212|2.194643|0.00524%|0.00516%|
|KV carrier conversion|12.516667|12.491865|0.03011%|0.02935%|
|KV append DMA|164.305432|164.402480|0.39532%|0.38631%|
|Block orchestration|32.086644|31.410702|0.07720%|0.07381%|
|Layer bookkeeping|16.554973|16.594296|0.03983%|0.03899%|
|Stage-boundary bookkeeping|1.808420|1.812686|0.00435%|0.00426%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|51.049182|51.068403|0.12282%|0.12000%|
|Embedding|2.309908|2.302914|0.00556%|0.00541%|
|Final model RMSNorm|83.363083|83.443204|0.20057%|0.19607%|
|LM head＋greedy（不含 final norm）|3851.235838|3834.153100|9.26602%|9.00942%|
|Host_boundary_us|673.577242|686.025426|1.62062%|1.61201%|
|Host_wall_us|41563.009037|42557.145168|100.00000%|100.00000%|

## formal decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|41564640.857143|42548558.958333|2.36720%|
|output_mismatches|raw|0.785714|0.952381|21.21212%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.214286|0.047619|-77.77778%|
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
|repeat_count|raw|1.000000|1.000000|0.00000%|
|prepared_session_run_index|raw|22.500000|22.500000|0.00000%|
|numerical_audit_enabled|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_result|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_index|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_n_tile|raw|0.000000|0.000000|N/A: zero control|
|projection_failure_step|raw|0.000000|0.000000|N/A: zero control|
|kv_cache_k_format|raw|1.000000|1.000000|0.00000%|
|kv_cache_v_format|raw|1.000000|1.000000|0.00000%|
|w4u8_prefill_cache_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_delta_reconstruction_mode|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_softmax_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_lm_head_group_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_o_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_n_tiles_observed|raw|16.000000|16.000000|0.00000%|
|w4u8_o_batch_count|raw|168.000000|168.000000|0.00000%|
|w4u8_decode_av_requant_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_av_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_av_requant_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_av_requant_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_av_requant_vector_count|raw|2688.000000|2688.000000|0.00000%|
|w4u8_av_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_common_op_rows|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_common_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_op_rows_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_input_norm_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_post_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_final_residual_direct_row4_call_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_common_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_qk_norm_rope_rows|raw|64.000000|64.000000|0.00000%|
|w4u8_decode_qk_padding_poison|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_projection_mode|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_mask|raw|63.000000|63.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_continuous|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_gate_prefetch|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|raw|16.000000|16.000000|0.00000%|
|w4u8_decode_direct_n_q_batch_n_tiles|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_direct_n_down_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|w4u8_decode_direct_n_down_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_decode_direct_n_o_single_dma|raw|1.000000|1.000000|0.00000%|
|w4u8_qkv_ring_slot_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_prep_worker_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_batch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_task_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_dispatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_head_publish_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pipeline_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_dma_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_compute_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_pool_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|2.270089|2.253224|-0.74293%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|2324.792101|2328.457589|0.15767%|
|w4u8_gate_up_swiglu_publish_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|224.000000|224.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|834.580915|3422.892733|310.13312%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|11546.573165|8909.479167|-22.83876%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|89.687128|303.382688|238.26782%|
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
|w4u8_decode_softmax_hvx_tile4_call_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_swiglu_rows_observed|raw|4.000000|4.000000|0.00000%|
|w4u8_decode_swiglu_row4_call_count|raw|7168.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|7168.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|3295232.000000|3295232.000000|0.00000%|
|scan_attention_overlay_required_bytes|bytes|61537.523810|61537.523810|0.00000%|
|scan_cache_dma_descriptor_count|raw|896.000000|896.000000|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4902912.000000|4902912.000000|0.00000%|
|scan_cache_ddr_write_bytes|bytes|57344.000000|57344.000000|0.00000%|
|scan_cache_stage_ticks|us|472.599082|472.334945|-0.05589%|
|scan_cache_append_ticks|us|164.192026|164.420759|0.13931%|
|scan_cache_pack_ticks|us|12.503906|12.493242|-0.08529%|
|block_orchestration_ticks|us|32.065414|31.403956|-2.06284%|
|layer_bookkeeping_ticks|us|16.553137|16.578249|0.15170%|
|scan_dynamic_attention_ticks|us|6411.061694|6411.264695|0.00317%|
|total_ticks|us|40873.415551|41863.942398|2.42340%|
|invocation_ticks|us|40899.958891|41890.510975|2.42189%|
|runtime_setup_ticks|us|26.543341|26.549727|0.02406%|
|runtime_teardown_ticks|us|24.506200|24.514819|0.03517%|
|stage_boundary_ticks|us|1.810392|1.809462|-0.05137%|
|ledger_named_ticks|us|40899.958891|41890.510975|2.42189%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.300843|0.296813|-1.33965%|
|metadata_stage_ticks|us|241.165985|246.423115|2.17988%|
|input_norm_ticks|us|2315.641803|2315.490761|-0.00652%|
|qkv_projection_ticks|us|3467.054688|3471.639323|0.13223%|
|qk_norm_rope_ticks|us|383.553881|384.184524|0.16442%|
|attention_ticks|us|6415.661830|6415.857081|0.00304%|
|o_projection_ticks|us|2874.766431|2881.612041|0.23813%|
|post_attention_residual_ticks|us|2311.781622|2315.464100|0.15929%|
|post_attention_norm_ticks|us|0.821863|0.822111|0.03018%|
|gate_up_ticks|us|12448.181362|12435.134673|-0.10481%|
|activation_ticks|us|0.000000|467.112537|N/A: zero control|
|down_ticks|us|6213.673239|6736.573723|8.41532%|
|final_residual_ticks|us|2.182106|2.194196|0.55409%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|2.303943|2.307230|0.14263%|
|generation_final_norm_ticks|us|83.357081|83.417783|0.07282%|
|generation_lm_head_ticks|us|3944.663504|3917.193390|-0.69639%|
|generation_lm_head_weight_dma_ticks|us|3432.507626|3406.602617|-0.75470%|
|generation_lm_head_scale_dma_ticks|us|19.414435|19.086372|-1.68979%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|3413.319940|3387.131386|-0.76725%|
|generation_lm_head_argmax_ticks|us|390.689236|390.738219|0.01254%|
|generation_lm_head_weight_dma_wait_ticks|us|3363.791977|3338.646887|-0.74752%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|14.957961|13.250062|-11.41800%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|126.000000|126.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4008.000000|4008.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|125.000000|125.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1026048.000000|1026048.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|6400.000000|6400.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|198027264.000000|198027264.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|1218.000000|1218.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|336.000000|336.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|1606287360.000000|1606287360.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|3212574720.000000|3212574720.000000|0.00000%|
|weight_dma_ticks|us|31729.509859|31757.280072|0.08752%|
|hmx_compute_ticks|us|13488.036706|13566.897321|0.58467%|
|projection_pack_ticks|us|12.749070|12.760355|0.08851%|
|projection_hmx_wait_ticks|us|1037.216952|1322.599516|27.51426%|
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
|attention_unattributed_ticks|us|790.596292|790.323785|-0.03447%|
|u8_attention_qk_norm_rope_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_k_pack_ticks|us|2132.120102|2132.169829|0.00233%|
|u8_attention_v_pack_ticks|us|1649.670139|1649.703063|0.00200%|
|u8_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_build_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_full_prefix_pack_count|raw|448.000000|448.000000|0.00000%|
|u8_cache_segment_tail_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_segment_sealed_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_append_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_attention_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_full_tile_rmw_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_quartet_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_publish_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_partial_pack_rows|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_v_vtcm_tail_native_load_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_row_update_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_seal_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_cached_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_fallback_head_count|raw|0.000000|0.000000|N/A: zero control|
|u8_cache_k_vtcm_tail_init_bytes|bytes|0.000000|0.000000|N/A: zero control|
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
|u8_attention_qk_hmx_ticks|us|626.781498|627.062748|0.04487%|
|wide_score_mode|raw|8.000000|8.000000|0.00000%|
|paper_format_disable|raw|0.000000|0.000000|N/A: zero control|
|paper_pipeline_disable|raw|0.000000|0.000000|N/A: zero control|
|llama_fp32_residual|raw|1.000000|1.000000|0.00000%|
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
|dense_r4_prepare_ticks|us|0.000000|56.247706|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|279.603609|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|129.249008|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|124.000000|124.000000|0.00000%|
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
|u8_attention_softmax_ticks|us|555.902902|556.021019|0.02125%|
|u8_attention_av_hmx_ticks|us|648.911148|648.978609|0.01040%|
|u8_attention_av_requant_ticks|us|11.351004|11.340898|-0.08904%|
|u8_attention_pipeline_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_hmx_lifetime_ticks|us|24394.031374|24164.258681|-0.94192%|
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
|vtcm_peak_plan_bytes|bytes|8360416.000000|8360416.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1666.000000|1666.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|3143082.666667|3143082.666667|0.00000%|
|weight_dma_descriptor_count|raw|2311.000000|2311.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|1613512704.000000|1613512704.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4965760.000000|4965760.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_count|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_error|raw|0.000000|0.000000|N/A: zero control|
|weight_segment_map_ticks|us|0.000000|0.000000|N/A: zero control|
|weight_segment_unmap_ticks|us|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|676.516342|714.737259|5.64967%|
|Host_wall_us|us|41564.640857|42548.558958|2.36720%|
|generation_lm_head_exclusive_ticks|us|3861.261471|3833.719308|-0.71329%|

## Validation
8686 profiled invocations reconciled exactly. Layer hashes, padding, final hidden, KV payloads, final norm and bounds are in the model evidence. Failed candidate attempts in RECOVERY.md are excluded from all tables. No baseline/default promotion.
