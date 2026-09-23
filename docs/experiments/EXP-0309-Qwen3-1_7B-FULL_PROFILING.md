# EXP-0309 Qwen3-1.7B complete paired profiling

Full model fixed64+42, ordinary W4A8 versus HVX R4. FP32 residual, R3/SP2/INT16 Down off. Timings exclude audit dumping, cold load and external tokenizer. Each timed head agrees with its own independent reference.

Formal: ten alternating pairs, repeat10 per arm. Auxiliary: one warmup repeat1 per arm, descriptive only. Module means reconcile to Host wall; counter tables show medians of per-round phase means. Decode is per step, not all42. Raw engine/worker/wait counters overlap and MUST NOT be summed as an additive ledger. R4 prepare/core/finish are subsets of Activation.

Evidence directory: /mnt/d/llm_exp/results/qwen3-block-htp/exp0309/1.7B; sealed package hash and flags are in per-run protocol.json. Measured source is in the protocol-referenced binary seal. No quality or speed acceptance claim.

## auxiliary prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|249.322917|254.375000|0.69055%|0.60734%|
|Input RMSNorm|2038.125000|2131.562500|5.64501%|5.08927%|
|QKV＋RoPE|7054.114583|7058.281250|19.53783%|16.85218%|
|QK–Softmax–AV|3504.583333|3462.291667|9.70667%|8.26648%|
|O projection|2095.052083|2100.885417|5.80268%|5.01602%|
|Post-attention residual＋RMSNorm|2077.968750|2088.802083|5.75537%|4.98717%|
|Gate/Up＋SwiGLU|6949.218750|13000.104167|19.24730%|31.03873%|
|Down|3573.072917|3563.802083|9.89637%|8.50885%|
|Final residual|2.395833|2.447917|0.00664%|0.00584%|
|KV carrier conversion|136.458333|135.885417|0.37795%|0.32444%|
|KV append DMA|300.885417|296.718750|0.83336%|0.70844%|
|Block orchestration|37.291667|36.875000|0.10329%|0.08804%|
|Layer bookkeeping|22.708333|22.135417|0.06290%|0.05285%|
|Stage-boundary bookkeeping|21.875000|20.781250|0.06059%|0.04962%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|107.708333|108.697917|0.29832%|0.25952%|
|Embedding|60.572917|60.260417|0.16777%|0.14388%|
|Final model RMSNorm|18.593750|14.062500|0.05150%|0.03358%|
|LM head＋greedy（不含 final norm）|5172.968750|5172.552083|14.32761%|12.34986%|
|Host_boundary_us|2681.978333|2352.968167|7.42830%|5.61789%|
|Host_wall_us|36104.895000|41883.489000|100.00000%|100.00000%|

## auxiliary prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|36104895.000000|41883489.000000|16.00502%|
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
|w4u8_o_batch_count|raw|112.000000|112.000000|0.00000%|
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
|w4u8_qkv_ring_pipeline_ticks|us|7015.729167|7020.468750|0.06756%|
|w4u8_qkv_ring_dma_wait_ticks|us|2172.812500|2189.895833|0.78623%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|9.687500|26.822917|176.88172%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|1474.375000|1489.166667|1.00325%|
|w4u8_qkv_ring_hmx_compute_ticks|us|544.739583|545.260417|0.09561%|
|w4u8_qkv_ring_pool_wait_ticks|us|4500.520833|4478.958333|-0.47911%|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|386.354167|1100.156250|184.75330%|
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
|scan_cache_append_ticks|us|300.885417|296.718750|-1.38480%|
|scan_cache_pack_ticks|us|136.458333|135.885417|-0.41985%|
|block_orchestration_ticks|us|37.291667|36.875000|-1.11732%|
|layer_bookkeeping_ticks|us|22.708333|22.135417|-2.52294%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|33369.531250|39475.833333|18.29903%|
|invocation_ticks|us|33422.916667|39530.520833|18.27370%|
|runtime_setup_ticks|us|53.385417|54.687500|2.43902%|
|runtime_teardown_ticks|us|54.322917|54.010417|-0.57526%|
|stage_boundary_ticks|us|21.875000|20.781250|-5.00000%|
|ledger_named_ticks|us|33422.916667|39530.520833|18.27370%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.572917|0.416667|-27.27273%|
|metadata_stage_ticks|us|248.750000|253.958333|2.09380%|
|input_norm_ticks|us|2038.125000|2131.562500|4.58448%|
|qkv_projection_ticks|us|7053.541667|7057.760417|0.05981%|
|qk_norm_rope_ticks|us|0.572917|0.520833|-9.09091%|
|attention_ticks|us|3504.583333|3462.291667|-1.20675%|
|o_projection_ticks|us|2095.052083|2100.885417|0.27843%|
|post_attention_residual_ticks|us|2076.458333|2087.291667|0.52172%|
|post_attention_norm_ticks|us|1.510417|1.510417|0.00000%|
|gate_up_ticks|us|6949.218750|7988.906250|14.96121%|
|activation_ticks|us|0.000000|5011.197917|N/A: zero control|
|down_ticks|us|3573.072917|3563.802083|-0.25946%|
|final_residual_ticks|us|2.395833|2.447917|2.17391%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|60.572917|60.260417|-0.51591%|
|generation_final_norm_ticks|us|18.593750|14.062500|-24.36975%|
|generation_lm_head_ticks|us|5191.562500|5186.614583|-0.09531%|
|generation_lm_head_weight_dma_ticks|us|5134.270833|5132.187500|-0.04058%|
|generation_lm_head_scale_dma_ticks|us|21.927083|23.541667|7.36342%|
|generation_lm_head_expand_ticks|us|3637.760417|3674.531250|1.01081%|
|generation_lm_head_hmx_ticks|us|4534.114583|4529.739583|-0.09649%|
|generation_lm_head_argmax_ticks|us|595.625000|596.093750|0.07870%|
|generation_lm_head_weight_dma_wait_ticks|us|294.479167|297.916667|1.16732%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|195.885417|149.427083|-23.71710%|
|generation_lm_head_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|generation_lm_head_command_count|raw|594.000000|594.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|593.000000|593.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|262400.000000|262400.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|156797952.000000|156797952.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|196.000000|196.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|840.000000|840.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|704643072.000000|704643072.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|1409286144.000000|1409286144.000000|0.00000%|
|weight_dma_ticks|us|18101.302083|18480.156250|2.09297%|
|hmx_compute_ticks|us|12066.041667|11797.291667|-2.22733%|
|projection_pack_ticks|us|4.166667|4.270833|2.50000%|
|projection_hmx_wait_ticks|us|1565.729167|1512.552083|-3.39631%|
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
|u8_attention_qk_norm_rope_ticks|us|24491.354167|24530.833333|0.16120%|
|u8_attention_k_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_v_pack_ticks|us|5063.750000|5054.062500|-0.19131%|
|u8_cache_native_append_update_ticks|us|435.572917|430.312500|-1.20770%|
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
|u8_attention_qk_hmx_ticks|us|575.572917|565.364583|-1.77360%|
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
|dense_r4_prepare_ticks|us|0.000000|506.458333|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|2888.593750|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|1612.604167|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|1.000000|1.000000|0.00000%|
|prefix_group_patch_count|raw|224.000000|224.000000|0.00000%|
|prefix_seed_metadata_read_bytes|bytes|57344.000000|57344.000000|0.00000%|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|8720.364583|8716.041667|-0.04957%|
|u8_attention_av_hmx_ticks|us|626.927083|600.416667|-4.22863%|
|u8_attention_av_requant_ticks|us|1649.843750|1643.958333|-0.35673%|
|u8_attention_pipeline_wait_ticks|us|2055.625000|1948.125000|-5.22955%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|2174.114583|2191.666667|0.80732%|
|w4u8_qkvo_hmx_lifetime_ticks|us|13347.083333|13656.875000|2.32104%|
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
|vtcm_peak_plan_bytes|bytes|8365824.000000|8365824.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1882.000000|1882.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|1687296.000000|1687296.000000|0.00000%|
|weight_dma_descriptor_count|raw|2275.000000|2275.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|866032640.000000|866032640.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|5107072.000000|5107072.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|2681.978333|2352.968167|-12.26744%|
|Host_wall_us|us|36104.895000|41883.489000|16.00502%|
|generation_lm_head_exclusive_ticks|us|5172.968750|5172.552083|-0.00805%|

## auxiliary decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|233.466022|236.878720|1.11072%|1.10004%|
|Input RMSNorm|366.400050|363.996776|1.74316%|1.69036%|
|QKV＋RoPE|2486.940724|2485.155010|11.83170%|11.54077%|
|QK–Softmax–AV|1952.388393|1953.568948|9.28855%|9.07215%|
|O projection|1364.971478|1371.463294|6.49390%|6.36892%|
|Post-attention residual＋RMSNorm|368.573909|372.225942|1.75350%|1.72857%|
|Gate/Up＋SwiGLU|6263.172123|6790.168651|29.79724%|31.53276%|
|Down|3442.996032|3452.058532|16.38016%|16.03096%|
|Final residual|1.775794|1.753472|0.00845%|0.00814%|
|KV carrier conversion|164.342758|164.020337|0.78187%|0.76169%|
|KV append DMA|113.360615|112.997272|0.53932%|0.52475%|
|Block orchestration|29.724702|29.391121|0.14142%|0.13649%|
|Layer bookkeeping|16.423611|16.328125|0.07814%|0.07583%|
|Stage-boundary bookkeeping|1.747272|1.809276|0.00831%|0.00840%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|73.857887|73.726438|0.35138%|0.34238%|
|Embedding|1.405010|1.440972|0.00668%|0.00669%|
|Final model RMSNorm|14.712302|14.556052|0.06999%|0.06760%|
|LM head＋greedy（不含 final norm）|3182.121776|3186.494296|15.13905%|14.79771%|
|Host_boundary_us|940.920163|905.662194|4.47646%|4.20579%|
|Host_wall_us|21019.300619|21533.695429|100.00000%|100.00000%|

## auxiliary decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|21019300.619048|21533695.428571|2.44725%|
|output_mismatches|raw|0.023810|0.095238|300.00000%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.976190|0.904762|-7.31707%|
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
|w4u8_o_batch_count|raw|112.000000|112.000000|0.00000%|
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
|w4u8_qkv_ring_pipeline_ticks|us|2455.939980|2454.236111|-0.06938%|
|w4u8_qkv_ring_dma_wait_ticks|us|2134.077381|2130.742808|-0.15625%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|11.846478|13.598710|14.79117%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|1423.838046|1398.005952|-1.81426%|
|w4u8_qkv_ring_hmx_compute_ticks|us|558.441220|569.928075|2.05695%|
|w4u8_qkv_ring_pool_wait_ticks|us|3.877728|3.865327|-0.31980%|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|141.527778|137.829861|-2.61286%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|519.818948|519.753224|-0.01264%|
|w4u8_gate_up_swiglu_publish_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|537.000248|2332.090774|334.28114%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|5361.521577|3896.938244|-27.31656%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|92.810020|288.379216|210.71992%|
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
|w4u8_decode_swiglu_row4_call_count|raw|5376.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|5376.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|2359296.000000|2359296.000000|0.00000%|
|scan_attention_overlay_required_bytes|bytes|83675.428571|83675.428571|0.00000%|
|scan_cache_dma_descriptor_count|raw|1202.666667|1202.666667|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4559445.333333|4559445.333333|0.00000%|
|scan_cache_ddr_write_bytes|bytes|77312.000000|77312.000000|0.00000%|
|scan_cache_stage_ticks|us|586.341766|586.643105|0.05139%|
|scan_cache_append_ticks|us|113.360615|112.997272|-0.32052%|
|scan_cache_pack_ticks|us|164.342758|164.020337|-0.19619%|
|block_orchestration_ticks|us|29.724702|29.391121|-1.12224%|
|layer_bookkeeping_ticks|us|16.423611|16.328125|-0.58140%|
|scan_dynamic_attention_ticks|us|1946.615823|1947.826141|0.06218%|
|total_ticks|us|20040.317460|20590.006200|2.74291%|
|invocation_ticks|us|20078.380456|20628.033234|2.73754%|
|runtime_setup_ticks|us|38.062996|38.027034|-0.09448%|
|runtime_teardown_ticks|us|35.794891|35.699405|-0.26676%|
|stage_boundary_ticks|us|1.747272|1.809276|3.54862%|
|ledger_named_ticks|us|20078.380456|20628.033234|2.73754%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.401786|0.355903|-11.41975%|
|metadata_stage_ticks|us|233.064236|236.522817|1.48396%|
|input_norm_ticks|us|366.400050|363.996776|-0.65592%|
|qkv_projection_ticks|us|2486.567460|2484.800347|-0.07107%|
|qk_norm_rope_ticks|us|0.373264|0.354663|-4.98339%|
|attention_ticks|us|1952.388393|1953.568948|0.06047%|
|o_projection_ticks|us|1364.971478|1371.463294|0.47560%|
|post_attention_residual_ticks|us|367.477679|371.144593|0.99786%|
|post_attention_norm_ticks|us|1.096230|1.081349|-1.35747%|
|gate_up_ticks|us|6263.172123|6473.283730|3.35472%|
|activation_ticks|us|0.000000|316.884921|N/A: zero control|
|down_ticks|us|3442.996032|3452.058532|0.26322%|
|final_residual_ticks|us|1.775794|1.753472|-1.25698%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|1.405010|1.440972|2.55958%|
|generation_final_norm_ticks|us|14.712302|14.556052|-1.06204%|
|generation_lm_head_ticks|us|3196.834077|3201.050347|0.13189%|
|generation_lm_head_weight_dma_ticks|us|2683.304812|2680.276538|-0.11286%|
|generation_lm_head_scale_dma_ticks|us|20.569196|21.940724|6.66787%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|2673.070437|2674.618056|0.05790%|
|generation_lm_head_argmax_ticks|us|458.896329|459.288194|0.08539%|
|generation_lm_head_weight_dma_wait_ticks|us|2616.174355|2611.719990|-0.17026%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|13.379216|18.866567|41.01400%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|4352.000000|4352.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|156797952.000000|156797952.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|989.000000|989.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|860225536.000000|860225536.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|1720451072.000000|1720451072.000000|0.00000%|
|weight_dma_ticks|us|15525.770089|15509.872272|-0.10240%|
|hmx_compute_ticks|us|8722.896825|8833.555308|1.26860%|
|projection_pack_ticks|us|4.391121|4.394841|0.08472%|
|projection_hmx_wait_ticks|us|744.933036|768.696677|3.19004%|
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
|attention_unattributed_ticks|us|909.636657|909.988839|0.03872%|
|u8_attention_qk_norm_rope_ticks|us|2068.995536|2069.837550|0.04070%|
|u8_attention_k_pack_ticks|us|49.916915|49.931796|0.02981%|
|u8_attention_v_pack_ticks|us|113.344494|113.658234|0.27680%|
|u8_cache_native_append_update_ticks|us|276.116071|275.429067|-0.24881%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|39.583333|39.389881|-0.48872%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|25088.000000|25088.000000|0.00000%|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|147.655010|148.498264|0.57110%|
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
|dense_r4_prepare_ticks|us|0.000000|43.932292|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|173.753720|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|96.965526|N/A: zero control|
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
|prefix_kv_mode|raw|1.000000|1.000000|0.00000%|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|355.182292|355.916419|0.20669%|
|u8_attention_av_hmx_ticks|us|168.505704|167.818700|-0.40770%|
|u8_attention_av_requant_ticks|us|144.434524|143.493304|-0.65166%|
|u8_attention_pipeline_wait_ticks|us|63.712798|64.263393|0.86418%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|2135.162450|2131.768353|-0.15896%|
|w4u8_qkvo_hmx_lifetime_ticks|us|12439.779266|12458.260169|0.14856%|
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
|vtcm_peak_plan_bytes|bytes|8365824.000000|8365824.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1437.000000|1437.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|1691733.333333|1691733.333333|0.00000%|
|weight_dma_descriptor_count|raw|1830.000000|1830.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|866032640.000000|866032640.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4849024.000000|4849024.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|940.920163|905.662194|-3.74718%|
|Host_wall_us|us|21019.300619|21533.695429|2.44725%|
|generation_lm_head_exclusive_ticks|us|3182.121776|3186.494296|0.13741%|

## formal prefill additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|247.168229|248.789583|0.71785%|0.61585%|
|Input RMSNorm|2037.897396|2138.579688|5.91867%|5.29382%|
|QKV＋RoPE|7031.491146|7037.366146|20.42159%|17.42022%|
|QK–Softmax–AV|3499.580729|3496.601562|10.16385%|8.65545%|
|O projection|2090.258854|2096.098437|6.07075%|5.18866%|
|Post-attention residual＋RMSNorm|2074.952604|2097.211458|6.02629%|5.19141%|
|Gate/Up＋SwiGLU|7087.895312|12907.681250|20.58540%|31.95153%|
|Down|3561.191667|3563.416146|10.34278%|8.82084%|
|Final residual|2.641146|2.674479|0.00767%|0.00662%|
|KV carrier conversion|140.695312|138.202604|0.40862%|0.34211%|
|KV append DMA|291.311979|291.921354|0.84606%|0.72262%|
|Block orchestration|39.396354|39.700521|0.11442%|0.09827%|
|Layer bookkeeping|24.419792|24.947917|0.07092%|0.06176%|
|Stage-boundary bookkeeping|20.609896|20.909375|0.05986%|0.05176%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|118.645312|120.991667|0.34458%|0.29950%|
|Embedding|59.725521|60.218750|0.17346%|0.14906%|
|Final model RMSNorm|18.490104|13.684375|0.05370%|0.03387%|
|LM head＋greedy（不含 final norm）|5193.975000|5198.516667|15.08488%|12.86835%|
|Host_boundary_us|891.310406|900.184341|2.58864%|2.22831%|
|Host_wall_us|34431.656760|40397.696320|100.00000%|100.00000%|

## formal prefill full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|0.000000|0.000000|N/A: zero control|
|logical_m|raw|64.000000|64.000000|0.00000%|
|first_position|raw|0.000000|0.000000|N/A: zero control|
|valid_length|raw|64.000000|64.000000|0.00000%|
|host_wall_ns|ns|34447502.600000|40397323.000000|17.27214%|
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
|w4u8_o_batch_count|raw|112.000000|112.000000|0.00000%|
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
|w4u8_qkv_ring_pipeline_ticks|us|6989.205729|6999.276042|0.14408%|
|w4u8_qkv_ring_dma_wait_ticks|us|2166.070313|2162.716146|-0.15485%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|9.690104|9.929688|2.47245%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|1471.859375|1461.088542|-0.73178%|
|w4u8_qkv_ring_hmx_compute_ticks|us|547.986979|549.890625|0.34739%|
|w4u8_qkv_ring_pool_wait_ticks|us|4480.539062|4490.893229|0.23109%|
|w4u8_o_gate_prefetch_start_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_consume_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_o_gate_prefetch_lifetime_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_publish_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_worker_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_gate_up_swiglu_join_wait_ticks|us|391.046875|1174.296875|200.29568%|
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
|scan_cache_append_ticks|us|291.018229|292.182292|0.40000%|
|scan_cache_pack_ticks|us|139.507812|138.239583|-0.90907%|
|block_orchestration_ticks|us|39.156250|39.786458|1.60947%|
|layer_bookkeeping_ticks|us|24.442708|24.960938|2.12018%|
|scan_dynamic_attention_ticks|us|0.000000|0.000000|N/A: zero control|
|total_ticks|us|33484.794271|39429.018229|17.75201%|
|invocation_ticks|us|33553.036458|39498.791667|17.72047%|
|runtime_setup_ticks|us|68.442708|69.059896|0.90176%|
|runtime_teardown_ticks|us|51.000000|51.528646|1.03656%|
|stage_boundary_ticks|us|20.609375|20.911458|1.46576%|
|ledger_named_ticks|us|33553.036458|39498.791667|17.72047%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.518229|0.565104|9.04523%|
|metadata_stage_ticks|us|247.130208|248.828125|0.68705%|
|input_norm_ticks|us|2037.812500|2138.726562|4.95208%|
|qkv_projection_ticks|us|7026.848958|7037.145833|0.14654%|
|qk_norm_rope_ticks|us|0.604167|0.567708|-6.03448%|
|attention_ticks|us|3499.729167|3496.937500|-0.07977%|
|o_projection_ticks|us|2090.898438|2097.367188|0.30938%|
|post_attention_residual_ticks|us|2073.434896|2095.601562|1.06908%|
|post_attention_norm_ticks|us|1.630208|1.664062|2.07668%|
|gate_up_ticks|us|7089.171875|7884.619792|11.22060%|
|activation_ticks|us|0.000000|5021.174479|N/A: zero control|
|down_ticks|us|3563.200521|3566.927083|0.10458%|
|final_residual_ticks|us|2.669271|2.671875|0.09756%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|59.947917|60.197917|0.41703%|
|generation_final_norm_ticks|us|18.450521|13.684896|-25.82922%|
|generation_lm_head_ticks|us|5212.505208|5209.122396|-0.06490%|
|generation_lm_head_weight_dma_ticks|us|5154.924479|5156.127604|0.02334%|
|generation_lm_head_scale_dma_ticks|us|21.580729|21.671875|0.42235%|
|generation_lm_head_expand_ticks|us|3674.804688|3675.690104|0.02409%|
|generation_lm_head_hmx_ticks|us|4554.041667|4554.101562|0.00132%|
|generation_lm_head_argmax_ticks|us|594.786458|595.463542|0.11384%|
|generation_lm_head_weight_dma_wait_ticks|us|302.395833|303.346354|0.31433%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|169.375000|165.992188|-1.99723%|
|generation_lm_head_batch_n_tiles|raw|8.000000|8.000000|0.00000%|
|generation_lm_head_command_count|raw|594.000000|594.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|593.000000|593.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|262400.000000|262400.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|156797952.000000|156797952.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|196.000000|196.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|840.000000|840.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|704643072.000000|704643072.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|1409286144.000000|1409286144.000000|0.00000%|
|weight_dma_ticks|us|18321.330729|18279.171875|-0.23011%|
|hmx_compute_ticks|us|11763.697917|11834.929688|0.60552%|
|projection_pack_ticks|us|4.479167|4.598958|2.67442%|
|projection_hmx_wait_ticks|us|1487.244792|1530.169271|2.88617%|
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
|u8_attention_qk_norm_rope_ticks|us|24482.757812|24484.617188|0.00759%|
|u8_attention_k_pack_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_v_pack_ticks|us|5030.786458|5044.213542|0.26690%|
|u8_cache_native_append_update_ticks|us|429.747396|428.341146|-0.32723%|
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
|u8_attention_qk_hmx_ticks|us|584.273438|582.763021|-0.25851%|
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
|dense_r4_prepare_ticks|us|0.000000|510.401042|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|2892.367188|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|1614.572917|N/A: zero control|
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
|generation_lm_head_direct_slot_join_count|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_rows|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_hmx_calls|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_refined_values|raw|0.000000|0.000000|N/A: zero control|
|dense_r3_total_prepare_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r3_total_finish_ticks|us|0.000000|0.000000|N/A: zero control|
|prefix_kv_mode|raw|1.000000|1.000000|0.00000%|
|prefix_group_patch_count|raw|224.000000|224.000000|0.00000%|
|prefix_seed_metadata_read_bytes|bytes|57344.000000|57344.000000|0.00000%|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|8725.398438|8718.385417|-0.08037%|
|u8_attention_av_hmx_ticks|us|631.085938|628.666667|-0.38335%|
|u8_attention_av_requant_ticks|us|1650.229167|1649.007812|-0.07401%|
|u8_attention_pipeline_wait_ticks|us|2042.593750|2026.020833|-0.81137%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|2167.148438|2163.729167|-0.15778%|
|w4u8_qkvo_hmx_lifetime_ticks|us|13434.906250|13465.419271|0.22712%|
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
|vtcm_peak_plan_bytes|bytes|8365824.000000|8365824.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1882.000000|1882.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|1687296.000000|1687296.000000|0.00000%|
|weight_dma_descriptor_count|raw|2275.000000|2275.000000|0.00000%|
|boundary_dma_descriptor_count|raw|289.000000|289.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|866032640.000000|866032640.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|5107072.000000|5107072.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|885.523442|892.973933|0.84137%|
|Host_wall_us|us|34447.502600|40397.323000|17.27214%|
|generation_lm_head_exclusive_ticks|us|5194.015625|5195.375000|0.02617%|

## formal decode additive overview

|Module|A8 mean us|R4 mean us|A8 Host share|R4 Host share|
|---|---:|---:|---:|---:|
|I/O、metadata|245.325223|245.792882|1.16255%|1.13729%|
|Input RMSNorm|366.546553|363.892336|1.73700%|1.68374%|
|QKV＋RoPE|2521.897148|2516.777269|11.95085%|11.64523%|
|QK–Softmax–AV|1968.974454|1968.293204|9.33064%|9.10737%|
|O projection|1369.507875|1369.440811|6.48987%|6.33646%|
|Post-attention residual＋RMSNorm|368.781994|372.300521|1.74760%|1.72265%|
|Gate/Up＋SwiGLU|6379.135045|6899.122855|30.22964%|31.92252%|
|Down|3489.649082|3492.596156|16.53686%|16.16038%|
|Final residual|1.785503|1.784425|0.00846%|0.00826%|
|KV carrier conversion|164.164707|164.068688|0.77795%|0.75915%|
|KV append DMA|112.626166|113.054315|0.53372%|0.52311%|
|Block orchestration|29.757949|29.279799|0.14102%|0.13548%|
|Layer bookkeeping|16.422793|16.427592|0.07782%|0.07601%|
|Stage-boundary bookkeeping|1.779539|1.769506|0.00843%|0.00819%|
|DSP unattributed|0.000000|0.000000|0.00000%|0.00000%|
|Runtime setup/teardown|73.757168|73.534697|0.34952%|0.34025%|
|Embedding|1.546453|1.545821|0.00733%|0.00715%|
|Final model RMSNorm|14.695089|14.816319|0.06964%|0.06856%|
|LM head＋greedy（不含 final norm）|3243.289645|3236.113157|15.36940%|14.97363%|
|Host_boundary_us|732.606921|731.477888|3.47170%|3.38458%|
|Host_wall_us|21102.249307|21612.088243|100.00000%|100.00000%|

## formal decode full counters

Ticks converted to microseconds; byte/count fields keep raw units. These include overlapping subcounters and metadata, not an additive table. Zero control denominator has no percentage.

|Counter|Unit|A8 median|R4 median|R4 / A8 - 1|
|---|---|---:|---:|---:|
|experiment|raw|218.000000|218.000000|0.00000%|
|generation_step|raw|21.500000|21.500000|0.00000%|
|logical_m|raw|1.000000|1.000000|0.00000%|
|first_position|raw|84.500000|84.500000|0.00000%|
|valid_length|raw|85.500000|85.500000|0.00000%|
|host_wall_ns|ns|21134800.217857|21608117.671429|2.23952%|
|output_mismatches|raw|0.023810|0.095238|300.00000%|
|output_max_abs|raw|0.000000|0.000000|N/A: zero control|
|output_cosine|raw|0.976190|0.904762|-7.31707%|
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
|w4u8_o_batch_count|raw|112.000000|112.000000|0.00000%|
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
|w4u8_qkv_ring_pipeline_ticks|us|2491.434028|2487.154328|-0.17178%|
|w4u8_qkv_ring_dma_wait_ticks|us|2170.689732|2165.549851|-0.23679%|
|w4u8_qkv_ring_producer_slot_wait_ticks|us|11.550471|11.066902|-4.18657%|
|w4u8_qkv_ring_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkv_ring_hmx_ready_wait_ticks|us|1473.718440|1462.573599|-0.75624%|
|w4u8_qkv_ring_hmx_compute_ticks|us|555.344804|556.970424|0.29272%|
|w4u8_qkv_ring_pool_wait_ticks|us|3.895151|3.898996|0.09869%|
|w4u8_o_gate_prefetch_start_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_consume_count|raw|28.000000|28.000000|0.00000%|
|w4u8_o_gate_prefetch_wait_ticks|us|151.130952|146.300657|-3.19610%|
|w4u8_o_gate_prefetch_lifetime_ticks|us|529.669023|528.324777|-0.25379%|
|w4u8_gate_up_swiglu_publish_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_consume_count|raw|168.000000|168.000000|0.00000%|
|w4u8_gate_up_swiglu_overlap_observed|raw|1.000000|1.000000|0.00000%|
|w4u8_gate_up_swiglu_worker_ticks|us|546.673735|2291.045449|319.08826%|
|w4u8_gate_up_swiglu_ready_wait_ticks|us|5455.385913|4056.852927|-25.63582%|
|w4u8_gate_up_swiglu_join_wait_ticks|us|92.580357|288.794457|211.93923%|
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
|w4u8_decode_swiglu_row4_call_count|raw|5376.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_vector_count|raw|5376.000000|0.000000|-100.00000%|
|w4u8_decode_swiglu_padding_poison_count|raw|0.000000|0.000000|N/A: zero control|
|w4u8_decode_swiglu_valid_row_hash|raw|0.000000|0.000000|N/A: zero control|
|dsp_status|raw|3.000000|3.000000|0.00000%|
|numerical_status|raw|1.000000|1.000000|0.00000%|
|scan_logical_m_observed|raw|1.000000|1.000000|0.00000%|
|scan_total_kv_length|raw|85.500000|85.500000|0.00000%|
|scan_padded_kv_length|raw|103.619048|103.619048|0.00000%|
|scan_attention_overlay_capacity_bytes|bytes|2359296.000000|2359296.000000|0.00000%|
|scan_attention_overlay_required_bytes|bytes|83675.428571|83675.428571|0.00000%|
|scan_cache_dma_descriptor_count|raw|1202.666667|1202.666667|0.00000%|
|scan_cache_append_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|scan_cache_ddr_read_bytes|bytes|4559445.333333|4559445.333333|0.00000%|
|scan_cache_ddr_write_bytes|bytes|77312.000000|77312.000000|0.00000%|
|scan_cache_stage_ticks|us|594.058656|594.064608|0.00100%|
|scan_cache_append_ticks|us|112.730655|113.030568|0.26604%|
|scan_cache_pack_ticks|us|164.173487|164.069320|-0.06345%|
|block_orchestration_ticks|us|29.770027|29.285342|-1.62810%|
|layer_bookkeeping_ticks|us|16.419147|16.416171|-0.01813%|
|scan_dynamic_attention_ticks|us|1962.526972|1963.199095|0.03425%|
|total_ticks|us|20355.636223|20839.416109|2.37664%|
|invocation_ticks|us|20393.594680|20877.422681|2.37245%|
|runtime_setup_ticks|us|37.972780|37.998884|0.06874%|
|runtime_teardown_ticks|us|35.772941|35.534846|-0.66557%|
|stage_boundary_ticks|us|1.778460|1.768663|-0.55085%|
|ledger_named_ticks|us|20393.594680|20877.422681|2.37245%|
|ledger_unattributed_ticks|us|0.000000|0.000000|N/A: zero control|
|input_stage_ticks|us|0.377170|0.372272|-1.29870%|
|metadata_stage_ticks|us|245.103423|245.746900|0.26253%|
|input_norm_ticks|us|366.544457|363.895275|-0.72274%|
|qkv_projection_ticks|us|2522.051215|2517.762773|-0.17004%|
|qk_norm_rope_ticks|us|0.347098|0.345052|-0.58950%|
|attention_ticks|us|1968.301897|1968.971292|0.03401%|
|o_projection_ticks|us|1368.099330|1370.314050|0.16188%|
|post_attention_residual_ticks|us|367.726500|371.245598|0.95699%|
|post_attention_norm_ticks|us|1.080357|1.085379|0.46488%|
|gate_up_ticks|us|6376.908606|6586.141927|3.28111%|
|activation_ticks|us|0.000000|316.880580|N/A: zero control|
|down_ticks|us|3489.536458|3493.020151|0.09983%|
|final_residual_ticks|us|1.786582|1.783358|-0.18047%|
|output_stage_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_embedding_ticks|us|1.544829|1.546379|0.10034%|
|generation_final_norm_ticks|us|14.710379|14.790117|0.54205%|
|generation_lm_head_ticks|us|3267.246714|3252.463728|-0.45246%|
|generation_lm_head_weight_dma_ticks|us|2753.534350|2738.128100|-0.55951%|
|generation_lm_head_scale_dma_ticks|us|21.658110|21.669147|0.05096%|
|generation_lm_head_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_ticks|us|2742.914807|2727.859623|-0.54888%|
|generation_lm_head_argmax_ticks|us|458.786086|459.122830|0.07340%|
|generation_lm_head_weight_dma_wait_ticks|us|2686.957155|2671.455853|-0.57691%|
|generation_lm_head_scale_init_ticks|us|0.000000|0.000000|N/A: zero control|
|generation_lm_head_hmx_tail_wait_ticks|us|14.342324|14.118428|-1.56109%|
|generation_lm_head_batch_n_tiles|raw|32.000000|32.000000|0.00000%|
|generation_lm_head_command_count|raw|149.000000|149.000000|0.00000%|
|generation_lm_head_n_tiles|raw|4748.000000|4748.000000|0.00000%|
|generation_lm_head_prefetch_count|raw|148.000000|148.000000|0.00000%|
|generation_lm_head_scale_resident_bytes|bytes|1215488.000000|1215488.000000|0.00000%|
|generation_embedding_ddr_read_bytes|bytes|4352.000000|4352.000000|0.00000%|
|generation_lm_head_ddr_read_bytes|bytes|156797952.000000|156797952.000000|0.00000%|
|w4u8_decode_direct_n_projection_count|raw|197.000000|197.000000|0.00000%|
|w4u8_decode_direct_n_hmx_command_count|raw|989.000000|989.000000|0.00000%|
|w4u8_mlp_down_hmx_command_count|raw|224.000000|224.000000|0.00000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|bytes|860225536.000000|860225536.000000|0.00000%|
|w4u8_decode_direct_n_expand_bytes_avoided|bytes|1720451072.000000|1720451072.000000|0.00000%|
|weight_dma_ticks|us|15831.436508|15795.808160|-0.22505%|
|hmx_compute_ticks|us|8598.691530|8621.506820|0.26533%|
|projection_pack_ticks|us|4.385045|4.396329|0.25735%|
|projection_hmx_wait_ticks|us|723.809648|731.817212|1.10631%|
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
|attention_unattributed_ticks|us|905.516431|906.448847|0.10297%|
|u8_attention_qk_norm_rope_ticks|us|2041.227493|2043.021267|0.08788%|
|u8_attention_k_pack_ticks|us|49.862971|49.865389|0.00485%|
|u8_attention_v_pack_ticks|us|113.389943|113.628906|0.21074%|
|u8_cache_native_append_update_ticks|us|275.347842|275.520213|0.06260%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|us|39.539435|39.523872|-0.03936%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|raw|196.000000|196.000000|0.00000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|bytes|25088.000000|25088.000000|0.00000%|
|f16_cache_native_prefill_reuse_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_prefill_reused_carrier_bytes|bytes|0.000000|0.000000|N/A: zero control|
|f16_cache_native_incremental_append_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_full_prefix_pack_count|raw|0.000000|0.000000|N/A: zero control|
|f16_cache_native_append_update_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_hmx_ticks|us|152.903894|152.693142|-0.13783%|
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
|dense_r4_prepare_ticks|us|0.000000|43.925471|N/A: zero control|
|dense_r4_matmul_ticks|us|0.000000|0.000000|N/A: zero control|
|dense_r4_layout_ticks|us|0.000000|173.738281|N/A: zero control|
|dense_r4_finish_ticks|us|0.000000|96.963108|N/A: zero control|
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
|prefix_kv_mode|raw|1.000000|1.000000|0.00000%|
|prefix_group_patch_count|raw|0.000000|0.000000|N/A: zero control|
|prefix_seed_metadata_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_qk_requant_ticks|us|0.000000|0.000000|N/A: zero control|
|u8_attention_softmax_ticks|us|357.204179|357.785776|0.16282%|
|u8_attention_av_hmx_ticks|us|174.566902|174.845672|0.15969%|
|u8_attention_av_requant_ticks|us|148.100880|147.129960|-0.65558%|
|u8_attention_pipeline_wait_ticks|us|66.863529|66.896329|0.04906%|
|w4u8_qkvo_weight_expand_ticks|us|0.000000|0.000000|N/A: zero control|
|w4u8_qkvo_prefetch_wait_ticks|us|2171.763331|2166.627418|-0.23649%|
|w4u8_qkvo_hmx_lifetime_ticks|us|12628.044023|12614.007006|-0.11116%|
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
|vtcm_peak_plan_bytes|bytes|8365824.000000|8365824.000000|0.00000%|
|block_invocation_count|raw|28.000000|28.000000|0.00000%|
|hmx_command_count|raw|1437.000000|1437.000000|0.00000%|
|hmx_fp16_tile_pair_count|raw|0.000000|0.000000|N/A: zero control|
|hmx_u8s8_tile_pair_count|raw|1691733.333333|1691733.333333|0.00000%|
|weight_dma_descriptor_count|raw|1830.000000|1830.000000|0.00000%|
|boundary_dma_descriptor_count|raw|226.000000|226.000000|0.00000%|
|intermediate_dma_descriptor_count|raw|0.000000|0.000000|N/A: zero control|
|intermediate_spill_fill_count|raw|0.000000|0.000000|N/A: zero control|
|weight_ddr_read_bytes|bytes|866032640.000000|866032640.000000|0.00000%|
|boundary_ddr_read_bytes|bytes|4849024.000000|4849024.000000|0.00000%|
|boundary_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_read_bytes|bytes|0.000000|0.000000|N/A: zero control|
|intermediate_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_audit_ddr_write_bytes|bytes|0.000000|0.000000|N/A: zero control|
|u8_attention_probability_mask_violation_count|raw|0.000000|0.000000|N/A: zero control|
|u8_attention_fused_k_operand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|w4f16_expand_mismatch_count|raw|0.000000|0.000000|N/A: zero control|
|Host_boundary_us|us|732.761831|730.826450|-0.26412%|
|Host_wall_us|us|21134.800218|21608.117671|2.23952%|
|generation_lm_head_exclusive_ticks|us|3252.589720|3237.565600|-0.46191%|

## Validation
8686 profiled invocations reconciled exactly. Layer hashes, padding, final hidden, KV payloads, final norm and bounds are in the model evidence. Failed candidate attempts in RECOVERY.md are excluded from all tables. No baseline/default promotion.
