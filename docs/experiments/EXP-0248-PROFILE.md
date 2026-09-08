# EXP0248 complete profiling comparison

Scope: one real C64 layer0, EOS+71 tokens; M64 then eight teacher-input M1 steps. Control original A8; candidate dense HMX R3 with guarded HVX direct-dot refinement. F16/W4A16 frozen. No full-model token boundary.
Runtime source bf84233fe77202f0bd201d706d9af69c70d0fc8c. Reporting source 3587939fe7a727153a19f080a9d0d1c219f66083.
Five short and ten alternating paired formal rounds, each repeat1 and repeat10. No discarded warmups/outliers. Primary Host wall is median of ten paired-round means; decode averages eight positions. Paired median ratios,10000 bootstrap resamples seed248. Additive tables use identical two median-Host-ranked rounds for every field in a cell. Engine counters can overlap and are separate.
All2970 timed RPCs match frozen audited per-step hashes; unused legacy output/cache zeros are NOT golden checks. One RPC/step,8MiB requested/granted,peak6682752bytes,zero intermediate DDR/spill,seven native W4 projections. Audit exports are disabled for all timed runs. R3 dense matrix remains explicit; no butterfly/FWHT or scalar oracle fallback.

|Scope|Control us|Refined us|Paired time regression|Ratio95% CI|
|---|---:|---:|---:|---|
|repeat1_prefill|1721.901|36840.391|+2038.50%|[20.10262912965893, 22.112527396461417]|
|repeat1_decode|974.932|1966.875|+102.70%|[1.8697369820354233, 2.1086476969104564]|
|repeat10_prefill|1537.276|36447.513|+2274.75%|[23.511461574487093, 24.322570112324573]|
|repeat10_decode|951.845|1908.047|+100.12%|[1.985447419086651, 2.0116379817901104]|

## prefill: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 refined R3|
|---|---|---|---:|---:|
|I/O、metadata|N/A|N/A|17.6 (1.15%)|17.7 (0.05%)|
|Input RMSNorm|N/A|N/A|19.5 (1.27%)|19.5 (0.05%)|
|QKV＋Q/K Norm-RoPE|N/A|N/A|251.2 (16.34%)|34902.6 (95.76%)|
|QK–Softmax–AV|N/A|N/A|119.7 (7.79%)|120.9 (0.33%)|
|O projection|N/A|N/A|44.0 (2.86%)|43.8 (0.12%)|
|Post-attention residual＋RMSNorm|N/A|N/A|23.6 (1.53%)|23.4 (0.06%)|
|Gate/Up＋SwiGLU|N/A|N/A|493.1 (32.08%)|493.4 (1.35%)|
|Down|N/A|N/A|119.6 (7.78%)|118.8 (0.33%)|
|Final residual|N/A|N/A|6.5 (0.43%)|6.6 (0.02%)|
|KV carrier conversion|N/A|N/A|4.6 (0.30%)|4.4 (0.01%)|
|KV append DMA|N/A|N/A|7.4 (0.48%)|7.6 (0.02%)|
|Block orchestration|N/A|N/A|1.6 (0.11%)|1.7 (0.00%)|
|Layer bookkeeping|N/A|N/A|0.8 (0.05%)|0.8 (0.00%)|
|Stage-boundary bookkeeping|N/A|N/A|1.7 (0.11%)|1.6 (0.00%)|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|
|Runtime setup/teardown|N/A|N/A|74.0 (4.81%)|74.2 (0.20%)|
|Host–DSP boundary|N/A|N/A|352.2 (22.91%)|610.6 (1.68%)|
|Complete Host wall|N/A|N/A|1537.3 (100.00%)|36447.5 (100.00%)|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|

## decode: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 refined R3|
|---|---|---|---:|---:|
|I/O、metadata|N/A|N/A|17.9 (1.89%)|18.0 (0.94%)|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.28%)|
|QKV＋Q/K Norm-RoPE|N/A|N/A|86.9 (9.13%)|1005.1 (52.68%)|
|QK–Softmax–AV|N/A|N/A|66.2 (6.95%)|66.4 (3.48%)|
|O projection|N/A|N/A|43.4 (4.56%)|43.6 (2.28%)|
|Post-attention residual＋RMSNorm|N/A|N/A|7.1 (0.75%)|7.3 (0.38%)|
|Gate/Up＋SwiGLU|N/A|N/A|227.5 (23.90%)|226.4 (11.86%)|
|Down|N/A|N/A|118.0 (12.39%)|116.3 (6.10%)|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.08%)|
|KV carrier conversion|N/A|N/A|10.3 (1.09%)|10.1 (0.53%)|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.2 (0.17%)|
|Block orchestration|N/A|N/A|1.0 (0.11%)|1.0 (0.05%)|
|Layer bookkeeping|N/A|N/A|0.6 (0.06%)|0.6 (0.03%)|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.02%)|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|
|Runtime setup/teardown|N/A|N/A|70.8 (7.44%)|70.7 (3.71%)|
|Host–DSP boundary|N/A|N/A|291.8 (30.66%)|332.3 (17.41%)|
|Complete Host wall|N/A|N/A|951.8 (100.00%)|1908.0 (100.00%)|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|

## Complete counter diagnostics

Time counters below use19.2ticks/us. Counts/bytes retain native units. Independent per-field medians need not sum. Legacy unused golden fields are excluded.

### repeat1_prefill

|Field|Control|Refined|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1721901.5000|36840391.0000|+2039.52%|
|prepared_session_run_index|1.0000|1.0000|+0.00%|
|numerical_audit_enabled|0.0000|0.0000|N/A zero denominator|
|projection_failure_result|0.0000|0.0000|N/A zero denominator|
|projection_failure_index|0.0000|0.0000|N/A zero denominator|
|projection_failure_n_tile|0.0000|0.0000|N/A zero denominator|
|projection_failure_step|0.0000|0.0000|N/A zero denominator|
|kv_cache_k_format|14.0000|14.0000|+0.00%|
|kv_cache_v_format|12.0000|12.0000|+0.00%|
|w4u8_prefill_cache_mode|1.0000|1.0000|+0.00%|
|w4u8_delta_reconstruction_mode|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_lm_head_group_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_o_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_o_batch_n_tiles_observed|16.0000|16.0000|+0.00%|
|w4u8_o_batch_count|4.0000|4.0000|+0.00%|
|w4u8_decode_av_requant_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_av_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_rows_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_vector_count|0.0000|0.0000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_common_op_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_common_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_common_op_rows_observed|64.0000|64.0000|+0.00%|
|w4u8_input_norm_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_common_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_projection_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_mask|63.0000|63.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_continuous|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|5.0000|+0.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|6.0000|+0.00%|
|w4u8_qkv_ring_expand_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|5137.5000|1918.0000|-62.67%|
|w4u8_qkv_ring_dma_wait_ticks|1601.0000|1642.0000|+2.56%|
|w4u8_qkv_ring_producer_slot_wait_ticks|5.0000|4.0000|-20.00%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1149.0000|1169.0000|+1.74%|
|w4u8_qkv_ring_hmx_compute_ticks|395.0000|373.0000|-5.57%|
|w4u8_qkv_ring_pool_wait_ticks|3158.0000|3.0000|-99.91%|
|w4u8_o_gate_prefetch_start_count|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_consume_count|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_consume_count|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_pair_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_pair_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_task_count|16.0000|16.0000|+0.00%|
|w4u8_input_norm_main_work_ticks|358.5000|321.0000|-10.46%|
|w4u8_input_norm_worker_work_ticks|1652.0000|1666.5000|+0.88%|
|w4u8_input_norm_pool_wait_ticks|30.5000|74.0000|+142.62%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|431.0000|405.5000|-5.92%|
|w4u8_post_residual_worker_work_ticks|1951.5000|1835.0000|-5.97%|
|w4u8_post_residual_pool_wait_ticks|14.0000|18.0000|+28.57%|
|w4u8_final_residual_main_work_ticks|69.5000|82.0000|+17.99%|
|w4u8_final_residual_worker_work_ticks|379.0000|371.5000|-1.98%|
|w4u8_final_residual_pool_wait_ticks|18.0000|9.0000|-50.00%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4u8_swiglu_rows_observed|64.0000|64.0000|+0.00%|
|w4u8_decode_swiglu_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|dsp_status|3.0000|3.0000|+0.00%|
|numerical_status|1.0000|1.0000|+0.00%|
|scan_logical_m_observed|64.0000|64.0000|+0.00%|
|scan_total_kv_length|64.0000|64.0000|+0.00%|
|scan_padded_kv_length|64.0000|64.0000|+0.00%|
|scan_attention_overlay_capacity_bytes|0.0000|0.0000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.0000|0.0000|N/A zero denominator|
|scan_cache_dma_descriptor_count|16.0000|16.0000|+0.00%|
|scan_cache_append_mismatch_count|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.0000|143360.0000|+0.00%|
|scan_cache_stage_ticks|0.0000|0.0000|N/A zero denominator|
|scan_cache_append_ticks|147.5000|150.5000|+2.03%|
|scan_cache_pack_ticks|102.0000|103.0000|+0.98%|
|block_orchestration_ticks|122.0000|123.0000|+0.82%|
|layer_bookkeeping_ticks|48.0000|48.0000|+0.00%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|23610.0000|691322.5000|+2828.09%|
|invocation_ticks|24683.5000|692448.5000|+2705.31%|
|runtime_setup_ticks|1105.5000|1113.5000|+0.72%|
|runtime_teardown_ticks|888.0000|878.5000|-1.07%|
|stage_boundary_ticks|107.0000|104.0000|-2.80%|
|ledger_named_ticks|24683.5000|692448.5000|+2705.31%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|145.0000|146.0000|+0.69%|
|metadata_stage_ticks|183.0000|182.5000|-0.27%|
|input_norm_ticks|433.5000|423.5000|-2.31%|
|qkv_projection_ticks|5185.0000|672301.5000|+12866.28%|
|qk_norm_rope_ticks|4.0000|4.0000|+0.00%|
|attention_ticks|2332.0000|2322.0000|-0.43%|
|o_projection_ticks|981.5000|947.0000|-3.52%|
|post_attention_residual_ticks|503.0000|482.5000|-4.08%|
|post_attention_norm_ticks|3.0000|3.0000|+0.00%|
|gate_up_ticks|4851.0000|4800.0000|-1.05%|
|activation_ticks|4883.5000|4926.5000|+0.88%|
|down_ticks|2433.0000|2434.0000|+0.04%|
|final_residual_ticks|139.0000|140.5000|+1.08%|
|generation_embedding_ticks|0.0000|0.0000|N/A zero denominator|
|generation_final_norm_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_command_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.0000|0.0000|N/A zero denominator|
|generation_embedding_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_direct_n_projection_count|7.0000|7.0000|+0.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|30.0000|+0.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|25165824.0000|+0.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|50331648.0000|+0.00%|
|weight_dma_ticks|9517.5000|9442.0000|-0.79%|
|hmx_compute_ticks|4389.0000|4548.0000|+3.62%|
|projection_pack_ticks|17.0000|17.0000|+0.00%|
|projection_hmx_wait_ticks|514.5000|509.0000|-1.07%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|attention_setup_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_softmax_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|attention_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_norm_rope_ticks|17608.5000|670219.5000|+3706.23%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|3029.5000|2957.5000|-2.38%|
|u8_cache_native_append_update_ticks|267.0000|247.5000|-7.30%|
|u8_cache_native_prefill_build_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|1.0000|1.0000|+0.00%|
|u8_cache_native_prefill_reused_carrier_bytes|143360.0000|143360.0000|+0.00%|
|u8_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_tail_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|1.0000|1.0000|+0.00%|
|u8_cache_v_vtcm_tail_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|32768.0000|32768.0000|+0.00%|
|u8_cache_v_vtcm_tail_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|1.0000|1.0000|+0.00%|
|u8_cache_k_vtcm_tail_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_fallback_head_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_bytes|29568.0000|29568.0000|+0.00%|
|u8_cache_k_vtcm_tail_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|420.0000|430.0000|+2.38%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5794.0000|5814.5000|+0.35%|
|u8_attention_av_hmx_ticks|422.5000|428.5000|+1.42%|
|u8_attention_av_requant_ticks|1181.5000|1198.5000|+1.44%|
|u8_attention_pipeline_wait_ticks|1282.5000|1275.0000|-0.58%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1601.0000|1643.0000|+2.62%|
|w4u8_qkvo_hmx_lifetime_ticks|8664.5000|8612.0000|-0.61%|
|w4f16_gate_up_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_down_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|47.0000|+2.17%|
|hmx_fp16_tile_pair_count|0.0000|768.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.0000|49408.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|60.0000|+0.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|25329664.0000|+0.00%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|

### repeat1_decode

|Field|Control|Refined|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|974931.6250|1966874.8750|+101.74%|
|prepared_session_run_index|5.5000|5.5000|+0.00%|
|numerical_audit_enabled|0.0000|0.0000|N/A zero denominator|
|projection_failure_result|0.0000|0.0000|N/A zero denominator|
|projection_failure_index|0.0000|0.0000|N/A zero denominator|
|projection_failure_n_tile|0.0000|0.0000|N/A zero denominator|
|projection_failure_step|0.0000|0.0000|N/A zero denominator|
|kv_cache_k_format|14.0000|14.0000|+0.00%|
|kv_cache_v_format|12.0000|12.0000|+0.00%|
|w4u8_prefill_cache_mode|1.0000|1.0000|+0.00%|
|w4u8_delta_reconstruction_mode|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_lm_head_group_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_o_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_o_batch_n_tiles_observed|16.0000|16.0000|+0.00%|
|w4u8_o_batch_count|4.0000|4.0000|+0.00%|
|w4u8_decode_av_requant_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_av_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_av_requant_call_count|8.0000|8.0000|+0.00%|
|w4u8_av_requant_vector_count|64.0000|64.0000|+0.00%|
|w4u8_av_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_common_op_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_common_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_common_op_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_input_norm_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_post_residual_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_final_residual_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_common_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_projection_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_mask|63.0000|63.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_continuous|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|5.0000|+0.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|6.0000|+0.00%|
|w4u8_qkv_ring_expand_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|1662.8125|1642.8750|-1.20%|
|w4u8_qkv_ring_dma_wait_ticks|1431.3750|1422.3125|-0.63%|
|w4u8_qkv_ring_producer_slot_wait_ticks|10.5000|3.9375|-62.50%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|904.1875|926.5000|+2.47%|
|w4u8_qkv_ring_hmx_compute_ticks|414.2500|350.0000|-15.51%|
|w4u8_qkv_ring_pool_wait_ticks|2.6875|2.5625|-4.65%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|204.6875|213.8125|+4.46%|
|w4u8_o_gate_prefetch_lifetime_ticks|356.1250|359.2500|+0.88%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|517.0625|528.6250|+2.24%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3507.5000|3465.2500|-1.20%|
|w4u8_gate_up_swiglu_join_wait_ticks|85.5625|86.2500|+0.80%|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_q_pair_row4_call_count|8.0000|0.0000|-100.00%|
|w4u8_decode_k_pair_row4_call_count|4.0000|0.0000|-100.00%|
|w4u8_decode_qk_rows_processed|96.0000|0.0000|-100.00%|
|w4u8_decode_k_temp_carrier_skipped_count|8.0000|0.0000|-100.00%|
|w4u8_qk_padding_poison_pair_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_residual_active_contexts|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|8.0000|8.0000|+0.00%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_row4_call_count|192.0000|192.0000|+0.00%|
|w4u8_decode_swiglu_vector_count|192.0000|192.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|dsp_status|3.0000|3.0000|+0.00%|
|numerical_status|1.0000|1.0000|+0.00%|
|scan_logical_m_observed|1.0000|1.0000|+0.00%|
|scan_total_kv_length|68.5000|68.5000|+0.00%|
|scan_padded_kv_length|96.0000|96.0000|+0.00%|
|scan_attention_overlay_capacity_bytes|2752512.0000|2752512.0000|+0.00%|
|scan_attention_overlay_required_bytes|77824.0000|77824.0000|+0.00%|
|scan_cache_dma_descriptor_count|42.0000|42.0000|+0.00%|
|scan_cache_append_mismatch_count|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_read_bytes|143936.0000|143936.0000|+0.00%|
|scan_cache_ddr_write_bytes|1152.0000|1152.0000|+0.00%|
|scan_cache_stage_ticks|340.8750|346.6875|+1.71%|
|scan_cache_append_ticks|56.8750|60.8125|+6.92%|
|scan_cache_pack_ticks|203.3750|198.0000|-2.64%|
|block_orchestration_ticks|21.5000|21.0625|-2.03%|
|layer_bookkeeping_ticks|11.3750|11.3125|-0.55%|
|scan_dynamic_attention_ticks|1280.3750|1291.8125|+0.89%|
|total_ticks|11961.0625|29590.8750|+147.39%|
|invocation_ticks|12683.7500|30311.2500|+138.98%|
|runtime_setup_ticks|721.1250|721.1250|+0.00%|
|runtime_teardown_ticks|639.9375|640.5625|+0.10%|
|stage_boundary_ticks|7.3125|7.4375|+1.71%|
|ledger_named_ticks|12683.7500|30311.2500|+138.98%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.1875|136.0000|-0.14%|
|metadata_stage_ticks|140.0625|139.5625|-0.36%|
|input_norm_ticks|102.3750|102.3125|-0.06%|
|qkv_projection_ticks|1683.2500|19318.0000|+1047.66%|
|qk_norm_rope_ticks|0.2500|0.1250|-50.00%|
|attention_ticks|1285.1875|1296.4375|+0.88%|
|o_projection_ticks|832.4375|833.2500|+0.10%|
|post_attention_residual_ticks|141.1875|136.2500|-3.50%|
|post_attention_norm_ticks|0.5625|0.6250|+11.11%|
|gate_up_ticks|4360.2500|4350.8125|-0.22%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2244.7500|2244.5000|-0.01%|
|final_residual_ticks|29.8750|29.8125|-0.21%|
|generation_embedding_ticks|0.0000|0.0000|N/A zero denominator|
|generation_final_norm_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_command_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.0000|0.0000|N/A zero denominator|
|generation_embedding_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_direct_n_projection_count|7.0000|7.0000|+0.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|30.0000|+0.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|25165824.0000|+0.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|50331648.0000|+0.00%|
|weight_dma_ticks|8743.1875|8721.3125|-0.25%|
|hmx_compute_ticks|4235.3125|4355.3125|+2.83%|
|projection_pack_ticks|3.0000|2.8750|-4.17%|
|projection_hmx_wait_ticks|371.8750|380.7500|+2.39%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|attention_setup_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_softmax_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|attention_unattributed_ticks|574.0625|579.0625|+0.87%|
|u8_attention_qk_norm_rope_ticks|1457.7500|17653.5000|+1111.01%|
|u8_attention_k_pack_ticks|13.6250|13.5625|-0.46%|
|u8_attention_v_pack_ticks|83.8125|84.0000|+0.22%|
|u8_cache_native_append_update_ticks|257.9375|256.2500|-0.65%|
|u8_cache_native_prefill_build_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_incremental_append_count|1.0000|1.0000|+0.00%|
|u8_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_tail_append_count|1.0000|1.0000|+0.00%|
|u8_cache_segment_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_row_update_count|8.0000|8.0000|+0.00%|
|u8_cache_v_vtcm_tail_publish_count|2.0000|2.0000|+0.00%|
|u8_cache_v_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.0000|12.0000|+0.00%|
|u8_cache_v_vtcm_tail_init_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.0000|6144.0000|+0.00%|
|u8_cache_k_vtcm_tail_init_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_row_update_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.0000|1.0000|+0.00%|
|u8_cache_k_vtcm_tail_init_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.0000|28672.0000|+0.00%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.0000|126.0000|+0.00%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|27.0625|26.1250|-3.46%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|94.6250|93.3125|-1.39%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|262.5625|262.4375|-0.05%|
|u8_attention_av_hmx_ticks|113.9375|115.1250|+1.04%|
|u8_attention_av_requant_ticks|99.3125|99.0000|-0.31%|
|u8_attention_pipeline_wait_ticks|45.7500|45.9375|+0.41%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1432.2500|1422.7500|-0.66%|
|w4u8_qkvo_hmx_lifetime_ticks|8242.3125|8205.3750|-0.45%|
|w4f16_gate_up_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_down_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|47.0000|+2.17%|
|hmx_fp16_tile_pair_count|0.0000|16.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.0000|49536.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|60.0000|+0.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|25329664.0000|+0.00%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|

### repeat10_prefill

|Field|Control|Refined|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1537275.9500|36447513.1000|+2270.92%|
|prepared_session_run_index|41.5000|41.5000|+0.00%|
|numerical_audit_enabled|0.0000|0.0000|N/A zero denominator|
|projection_failure_result|0.0000|0.0000|N/A zero denominator|
|projection_failure_index|0.0000|0.0000|N/A zero denominator|
|projection_failure_n_tile|0.0000|0.0000|N/A zero denominator|
|projection_failure_step|0.0000|0.0000|N/A zero denominator|
|kv_cache_k_format|14.0000|14.0000|+0.00%|
|kv_cache_v_format|12.0000|12.0000|+0.00%|
|w4u8_prefill_cache_mode|1.0000|1.0000|+0.00%|
|w4u8_delta_reconstruction_mode|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_lm_head_group_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_o_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_o_batch_n_tiles_observed|16.0000|16.0000|+0.00%|
|w4u8_o_batch_count|4.0000|4.0000|+0.00%|
|w4u8_decode_av_requant_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_av_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_rows_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_vector_count|0.0000|0.0000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_common_op_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_common_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_common_op_rows_observed|64.0000|64.0000|+0.00%|
|w4u8_input_norm_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_common_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_projection_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_mask|63.0000|63.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_continuous|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|5.0000|+0.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|6.0000|+0.00%|
|w4u8_qkv_ring_expand_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|4805.7500|1645.6500|-65.76%|
|w4u8_qkv_ring_dma_wait_ticks|1438.1500|1429.7500|-0.58%|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.8000|3.8500|+1.32%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|957.1500|945.8000|-1.19%|
|w4u8_qkv_ring_hmx_compute_ticks|375.5000|354.6500|-5.55%|
|w4u8_qkv_ring_pool_wait_ticks|3128.6500|2.6500|-99.92%|
|w4u8_o_gate_prefetch_start_count|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_consume_count|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_o_gate_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_consume_count|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_pair_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_pair_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_task_count|16.0000|16.0000|+0.00%|
|w4u8_input_norm_main_work_ticks|324.0500|323.6000|-0.14%|
|w4u8_input_norm_worker_work_ticks|1435.6500|1437.8500|+0.15%|
|w4u8_input_norm_pool_wait_ticks|12.7000|14.3000|+12.60%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|346.0500|360.0000|+4.03%|
|w4u8_post_residual_worker_work_ticks|1806.1500|1780.1500|-1.44%|
|w4u8_post_residual_pool_wait_ticks|69.2000|54.3500|-21.46%|
|w4u8_final_residual_main_work_ticks|75.8000|78.3000|+3.30%|
|w4u8_final_residual_worker_work_ticks|375.0500|371.7500|-0.88%|
|w4u8_final_residual_pool_wait_ticks|15.9000|14.4000|-9.43%|
|w4u8_decode_softmax_hvx_tile4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4u8_swiglu_rows_observed|64.0000|64.0000|+0.00%|
|w4u8_decode_swiglu_row4_call_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|dsp_status|3.0000|3.0000|+0.00%|
|numerical_status|1.0000|1.0000|+0.00%|
|scan_logical_m_observed|64.0000|64.0000|+0.00%|
|scan_total_kv_length|64.0000|64.0000|+0.00%|
|scan_padded_kv_length|64.0000|64.0000|+0.00%|
|scan_attention_overlay_capacity_bytes|0.0000|0.0000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.0000|0.0000|N/A zero denominator|
|scan_cache_dma_descriptor_count|16.0000|16.0000|+0.00%|
|scan_cache_append_mismatch_count|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.0000|143360.0000|+0.00%|
|scan_cache_stage_ticks|0.0000|0.0000|N/A zero denominator|
|scan_cache_append_ticks|141.8500|146.8000|+3.49%|
|scan_cache_pack_ticks|85.1500|85.8000|+0.76%|
|block_orchestration_ticks|31.5500|31.9500|+1.27%|
|layer_bookkeeping_ticks|14.9000|14.7500|-1.01%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|21939.4000|687360.1500|+3032.99%|
|invocation_ticks|22698.2000|688117.6500|+2931.60%|
|runtime_setup_ticks|755.4000|757.1500|+0.23%|
|runtime_teardown_ticks|664.8500|667.4000|+0.38%|
|stage_boundary_ticks|30.7000|30.8000|+0.33%|
|ledger_named_ticks|22698.2000|688117.6500|+2931.60%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|133.6000|133.1500|-0.34%|
|metadata_stage_ticks|145.7000|143.3500|-1.61%|
|input_norm_ticks|375.0500|374.3000|-0.20%|
|qkv_projection_ticks|4829.1500|670080.9000|+13775.75%|
|qk_norm_rope_ticks|0.6000|0.6000|+0.00%|
|attention_ticks|2296.1500|2311.6000|+0.67%|
|o_projection_ticks|844.2500|847.6000|+0.40%|
|post_attention_residual_ticks|451.5500|449.6000|-0.43%|
|post_attention_norm_ticks|0.8500|0.9000|+5.88%|
|gate_up_ticks|4557.1000|4538.9500|-0.40%|
|activation_ticks|4869.3000|4918.2000|+1.00%|
|down_ticks|2283.7000|2274.1500|-0.42%|
|final_residual_ticks|126.4000|127.7000|+1.03%|
|generation_embedding_ticks|0.0000|0.0000|N/A zero denominator|
|generation_final_norm_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_command_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.0000|0.0000|N/A zero denominator|
|generation_embedding_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_direct_n_projection_count|7.0000|7.0000|+0.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|30.0000|+0.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|25165824.0000|+0.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|50331648.0000|+0.00%|
|weight_dma_ticks|8826.2500|8785.7500|-0.46%|
|hmx_compute_ticks|4430.5000|4564.3000|+3.02%|
|projection_pack_ticks|4.0000|4.0500|+1.25%|
|projection_hmx_wait_ticks|509.2000|514.8000|+1.10%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|attention_setup_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_softmax_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|attention_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_norm_rope_ticks|16919.8500|668393.7500|+3850.35%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|2945.8000|2919.4000|-0.90%|
|u8_cache_native_append_update_ticks|225.7500|230.9000|+2.28%|
|u8_cache_native_prefill_build_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|1.0000|1.0000|+0.00%|
|u8_cache_native_prefill_reused_carrier_bytes|143360.0000|143360.0000|+0.00%|
|u8_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_tail_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|1.0000|1.0000|+0.00%|
|u8_cache_v_vtcm_tail_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|32768.0000|32768.0000|+0.00%|
|u8_cache_v_vtcm_tail_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|1.0000|1.0000|+0.00%|
|u8_cache_k_vtcm_tail_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_fallback_head_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_bytes|29568.0000|29568.0000|+0.00%|
|u8_cache_k_vtcm_tail_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|413.6000|424.7500|+2.70%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5785.0500|5798.0000|+0.22%|
|u8_attention_av_hmx_ticks|425.1500|434.4500|+2.19%|
|u8_attention_av_requant_ticks|1194.6500|1202.8500|+0.69%|
|u8_attention_pipeline_wait_ticks|1354.4000|1402.4500|+3.55%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1438.7500|1430.0000|-0.61%|
|w4u8_qkvo_hmx_lifetime_ticks|8081.4500|8035.8000|-0.56%|
|w4f16_gate_up_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_down_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|47.0000|+2.17%|
|hmx_fp16_tile_pair_count|0.0000|768.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.0000|49408.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|60.0000|+0.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|25329664.0000|+0.00%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|

### repeat10_decode

|Field|Control|Refined|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|951845.0563|1908046.8625|+100.46%|
|prepared_session_run_index|46.0000|46.0000|+0.00%|
|numerical_audit_enabled|0.0000|0.0000|N/A zero denominator|
|projection_failure_result|0.0000|0.0000|N/A zero denominator|
|projection_failure_index|0.0000|0.0000|N/A zero denominator|
|projection_failure_n_tile|0.0000|0.0000|N/A zero denominator|
|projection_failure_step|0.0000|0.0000|N/A zero denominator|
|kv_cache_k_format|14.0000|14.0000|+0.00%|
|kv_cache_v_format|12.0000|12.0000|+0.00%|
|w4u8_prefill_cache_mode|1.0000|1.0000|+0.00%|
|w4u8_delta_reconstruction_mode|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_lm_head_group_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_o_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_o_batch_n_tiles_observed|16.0000|16.0000|+0.00%|
|w4u8_o_batch_count|4.0000|4.0000|+0.00%|
|w4u8_decode_av_requant_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_av_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_av_requant_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_av_requant_call_count|8.0000|8.0000|+0.00%|
|w4u8_av_requant_vector_count|64.0000|64.0000|+0.00%|
|w4u8_av_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_common_op_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_common_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_common_op_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_input_norm_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_post_residual_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_final_residual_direct_row4_call_count|1.0000|1.0000|+0.00%|
|w4u8_common_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_projection_mode|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_mask|63.0000|63.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_continuous|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|5.0000|+0.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|6.0000|+0.00%|
|w4u8_qkv_ring_expand_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|1654.6688|1624.3063|-1.83%|
|w4u8_qkv_ring_dma_wait_ticks|1425.7938|1412.0875|-0.96%|
|w4u8_qkv_ring_producer_slot_wait_ticks|16.9937|4.4875|-73.59%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|881.0125|918.4625|+4.25%|
|w4u8_qkv_ring_hmx_compute_ticks|423.6500|356.6812|-15.81%|
|w4u8_qkv_ring_pool_wait_ticks|2.6875|2.6063|-3.02%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|211.5500|212.3625|+0.38%|
|w4u8_o_gate_prefetch_lifetime_ticks|354.9625|357.3813|+0.68%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|526.5250|556.3187|+5.66%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3487.6062|3437.6625|-1.43%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.3125|87.1625|+0.98%|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_q_pair_row4_call_count|8.0000|0.0000|-100.00%|
|w4u8_decode_k_pair_row4_call_count|4.0000|0.0000|-100.00%|
|w4u8_decode_qk_rows_processed|96.0000|0.0000|-100.00%|
|w4u8_decode_k_temp_carrier_skipped_count|8.0000|0.0000|-100.00%|
|w4u8_qk_padding_poison_pair_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_input_norm_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_residual_active_contexts|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_task_count|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_post_residual_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_main_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_worker_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_final_residual_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|8.0000|8.0000|+0.00%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_row4_call_count|192.0000|192.0000|+0.00%|
|w4u8_decode_swiglu_vector_count|192.0000|192.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison_count|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_swiglu_valid_row_hash|0.0000|0.0000|N/A zero denominator|
|dsp_status|3.0000|3.0000|+0.00%|
|numerical_status|1.0000|1.0000|+0.00%|
|scan_logical_m_observed|1.0000|1.0000|+0.00%|
|scan_total_kv_length|68.5000|68.5000|+0.00%|
|scan_padded_kv_length|96.0000|96.0000|+0.00%|
|scan_attention_overlay_capacity_bytes|2752512.0000|2752512.0000|+0.00%|
|scan_attention_overlay_required_bytes|77824.0000|77824.0000|+0.00%|
|scan_cache_dma_descriptor_count|42.0000|42.0000|+0.00%|
|scan_cache_append_mismatch_count|0.0000|0.0000|N/A zero denominator|
|scan_cache_ddr_read_bytes|143936.0000|143936.0000|+0.00%|
|scan_cache_ddr_write_bytes|1152.0000|1152.0000|+0.00%|
|scan_cache_stage_ticks|345.2563|347.8813|+0.76%|
|scan_cache_append_ticks|57.0250|62.0125|+8.75%|
|scan_cache_pack_ticks|198.8688|193.4625|-2.72%|
|block_orchestration_ticks|19.9563|19.8625|-0.47%|
|layer_bookkeeping_ticks|10.9437|11.0375|+0.86%|
|scan_dynamic_attention_ticks|1267.9250|1271.4938|+0.28%|
|total_ticks|11914.4937|29564.8812|+148.14%|
|invocation_ticks|12632.1250|30284.5938|+139.74%|
|runtime_setup_ticks|719.4188|719.9813|+0.08%|
|runtime_teardown_ticks|639.6063|638.7625|-0.13%|
|stage_boundary_ticks|7.4437|7.4000|-0.59%|
|ledger_named_ticks|12632.1250|30284.5938|+139.74%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.3625|136.0125|-0.26%|
|metadata_stage_ticks|140.8562|139.6625|-0.85%|
|input_norm_ticks|101.9375|102.0312|+0.09%|
|qkv_projection_ticks|1674.8750|19311.4688|+1053.01%|
|qk_norm_rope_ticks|0.2188|0.2625|+20.00%|
|attention_ticks|1271.3812|1275.0000|+0.28%|
|o_projection_ticks|832.2625|833.2812|+0.12%|
|post_attention_residual_ticks|137.8500|137.9000|+0.04%|
|post_attention_norm_ticks|0.5625|0.5875|+4.44%|
|gate_up_ticks|4360.5938|4356.4688|-0.09%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2251.0687|2250.9313|-0.01%|
|final_residual_ticks|29.2563|29.2437|-0.04%|
|generation_embedding_ticks|0.0000|0.0000|N/A zero denominator|
|generation_final_norm_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_command_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_n_tiles|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.0000|0.0000|N/A zero denominator|
|generation_embedding_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|w4u8_decode_direct_n_projection_count|7.0000|7.0000|+0.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|30.0000|+0.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|25165824.0000|+0.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|50331648.0000|+0.00%|
|weight_dma_ticks|8725.9125|8700.7188|-0.29%|
|hmx_compute_ticks|4316.7750|4330.8000|+0.32%|
|projection_pack_ticks|2.8937|2.9062|+0.43%|
|projection_hmx_wait_ticks|373.9250|375.0875|+0.31%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_cross_prefetch_lifetime_ticks|0.0000|0.0000|N/A zero denominator|
|attention_setup_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_qk_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_softmax_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_pack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_hmx_ticks|0.0000|0.0000|N/A zero denominator|
|attention_av_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|attention_unattributed_ticks|565.7313|569.7500|+0.71%|
|u8_attention_qk_norm_rope_ticks|1437.2125|17669.2875|+1129.41%|
|u8_attention_k_pack_ticks|13.1750|13.1625|-0.09%|
|u8_attention_v_pack_ticks|82.7188|82.7812|+0.08%|
|u8_cache_native_append_update_ticks|254.4750|254.3313|-0.06%|
|u8_cache_native_prefill_build_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_native_incremental_append_count|1.0000|1.0000|+0.00%|
|u8_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_tail_append_count|1.0000|1.0000|+0.00%|
|u8_cache_segment_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_row_update_count|8.0000|8.0000|+0.00%|
|u8_cache_v_vtcm_tail_publish_count|2.0000|2.0000|+0.00%|
|u8_cache_v_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.0000|12.0000|+0.00%|
|u8_cache_v_vtcm_tail_init_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.0000|6144.0000|+0.00%|
|u8_cache_k_vtcm_tail_init_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_row_update_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_seal_count|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.0000|1.0000|+0.00%|
|u8_cache_k_vtcm_tail_init_bytes|0.0000|0.0000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.0000|28672.0000|+0.00%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.0000|126.0000|+0.00%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.1250|25.6313|-1.89%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|95.1750|95.5687|+0.41%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|256.1812|256.0438|-0.05%|
|u8_attention_av_hmx_ticks|114.0062|113.9688|-0.03%|
|u8_attention_av_requant_ticks|98.6125|98.3125|-0.30%|
|u8_attention_pipeline_wait_ticks|45.6375|45.3438|-0.64%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1426.4688|1412.4375|-0.98%|
|w4u8_qkvo_hmx_lifetime_ticks|8244.9813|8192.6437|-0.63%|
|w4f16_gate_up_weight_dma_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_down_pipeline_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|47.0000|+2.17%|
|hmx_fp16_tile_pair_count|0.0000|16.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.0000|49536.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|60.0000|+0.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|25329664.0000|+0.00%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|

## R3 diagnostics

These counters overlap QKV and must not be added to the ledger.
{
  "prefill": {
    "rows": 1536,
    "hmx_calls": 1,
    "refined_values": 16811,
    "prepare_ticks": 37473.99,
    "matmul_ticks": 593651.16,
    "finish_ticks": 37475.98
  },
  "decode": {
    "rows": 24,
    "hmx_calls": 1,
    "refined_values": 285.375,
    "prepare_ticks": 7385.25375,
    "matmul_ticks": 10092.72625,
    "finish_ticks": 190.0575
  }
}

E2E prefill/decode token/s: N/A. Device PPL: N/A. >10percent speed gate stopped full-model work.
