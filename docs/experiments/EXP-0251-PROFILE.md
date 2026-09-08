# EXP0251 complete profiling comparison

Scope: one real C64 layer0, EOS+71 tokens; M64 then eight teacher-input M1 steps. Control original A8; candidate same no-R3 package with wide score difference repair; plain dense HMX R3 plus wide repair failed the independent whole-layer gate and is not formally timed. F16/W4A16 frozen. No full-model token boundary.
Runtime source 910f6f95c8d144194b0711c974b60c7159d3931b. Reporting source 27972a15d770ada194f712805b1be01e1a44dd68.
Five short and ten alternating paired formal rounds, each repeat1 and repeat10. No discarded warmups/outliers. Primary Host wall is median of ten paired-round means; decode averages eight positions. Paired median ratios,10000 bootstrap resamples seed251. Additive tables use identical two median-Host-ranked rounds for every field in a cell. Engine counters can overlap and are separate.
All2970 timed RPCs match frozen audited per-step hashes; unused legacy output/cache zeros are NOT golden checks. One RPC/step,8MiB requested/granted,peak6682752bytes,zero intermediate DDR/spill,seven native W4 projections. Audit exports are disabled for all timed runs. Both timed arms have R3 disabled. No butterfly/FWHT or scalar oracle fallback.

|Scope|Control us|Wide us|Paired time regression|Ratio95% CI|
|---|---:|---:|---:|---|
|repeat1_prefill|1782.344|1556.693|-9.33%|[0.8504066564859294, 1.0087950923992621]|
|repeat1_decode|962.803|957.438|+0.44%|[0.9297281285191749, 1.029824342047507]|
|repeat10_prefill|1542.820|1525.797|-0.41%|[0.9655866468743339, 1.0246680332308367]|
|repeat10_decode|952.656|960.633|+0.67%|[0.996190203268821, 1.020223299959943]|

## prefill: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 wide repair|W4A16 / W4A8 - 1|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.9 (1.16%)|17.5 (1.14%)|N/A no matched W4A16|
|Input RMSNorm|N/A|N/A|20.6 (1.34%)|21.0 (1.38%)|N/A no matched W4A16|
|QKV＋Q/K Norm-RoPE|N/A|N/A|251.8 (16.32%)|252.6 (16.56%)|N/A no matched W4A16|
|QK–Softmax–AV|N/A|N/A|120.3 (7.80%)|119.8 (7.85%)|N/A no matched W4A16|
|O projection|N/A|N/A|44.5 (2.88%)|44.5 (2.92%)|N/A no matched W4A16|
|Post-attention residual＋RMSNorm|N/A|N/A|23.7 (1.54%)|23.6 (1.55%)|N/A no matched W4A16|
|Gate/Up＋SwiGLU|N/A|N/A|491.5 (31.85%)|492.1 (32.25%)|N/A no matched W4A16|
|Down|N/A|N/A|119.3 (7.73%)|120.8 (7.92%)|N/A no matched W4A16|
|Final residual|N/A|N/A|6.7 (0.44%)|6.8 (0.44%)|N/A no matched W4A16|
|KV carrier conversion|N/A|N/A|4.7 (0.30%)|4.7 (0.31%)|N/A no matched W4A16|
|KV append DMA|N/A|N/A|7.2 (0.47%)|7.1 (0.47%)|N/A no matched W4A16|
|Block orchestration|N/A|N/A|1.7 (0.11%)|1.6 (0.11%)|N/A no matched W4A16|
|Layer bookkeeping|N/A|N/A|0.8 (0.05%)|0.8 (0.05%)|N/A no matched W4A16|
|Stage-boundary bookkeeping|N/A|N/A|1.4 (0.09%)|1.4 (0.09%)|N/A no matched W4A16|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A no matched W4A16|
|Runtime setup/teardown|N/A|N/A|74.3 (4.81%)|74.2 (4.86%)|N/A no matched W4A16|
|Host–DSP boundary|N/A|N/A|356.4 (23.10%)|337.2 (22.10%)|N/A no matched W4A16|
|Complete Host wall|N/A|N/A|1542.8 (100.00%)|1525.8 (100.00%)|N/A no matched W4A16|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|N/A|

## decode: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 wide repair|W4A16 / W4A8 - 1|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.8 (1.87%)|17.9 (1.87%)|N/A no matched W4A16|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.55%)|N/A no matched W4A16|
|QKV＋Q/K Norm-RoPE|N/A|N/A|87.0 (9.13%)|87.1 (9.07%)|N/A no matched W4A16|
|QK–Softmax–AV|N/A|N/A|66.3 (6.96%)|64.4 (6.70%)|N/A no matched W4A16|
|O projection|N/A|N/A|43.2 (4.53%)|43.3 (4.51%)|N/A no matched W4A16|
|Post-attention residual＋RMSNorm|N/A|N/A|7.1 (0.74%)|7.0 (0.73%)|N/A no matched W4A16|
|Gate/Up＋SwiGLU|N/A|N/A|226.6 (23.79%)|226.3 (23.56%)|N/A no matched W4A16|
|Down|N/A|N/A|116.8 (12.26%)|117.2 (12.20%)|N/A no matched W4A16|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.16%)|N/A no matched W4A16|
|KV carrier conversion|N/A|N/A|10.4 (1.09%)|10.4 (1.08%)|N/A no matched W4A16|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.0 (0.31%)|N/A no matched W4A16|
|Block orchestration|N/A|N/A|1.0 (0.11%)|1.0 (0.11%)|N/A no matched W4A16|
|Layer bookkeeping|N/A|N/A|0.6 (0.06%)|0.6 (0.06%)|N/A no matched W4A16|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.04%)|N/A no matched W4A16|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A no matched W4A16|
|Runtime setup/teardown|N/A|N/A|70.7 (7.42%)|70.7 (7.36%)|N/A no matched W4A16|
|Host–DSP boundary|N/A|N/A|295.0 (30.96%)|304.5 (31.70%)|N/A no matched W4A16|
|Complete Host wall|N/A|N/A|952.7 (100.00%)|960.6 (100.00%)|N/A no matched W4A16|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|N/A|

Identity: source branch codex/exp-0251-wide-score-device; evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0251; frozen artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0247/{control,r3}. Project Variant W4U8/native per-channel weight.n. Full correctness, recovery and scope limits in REPORT.md; exact binary/package identity in ARTIFACT_PROVENANCE.json.

## Complete counter diagnostics

Time counters below use19.2ticks/us. Counts/bytes retain native units. Independent per-field medians need not sum. Legacy unused golden fields are excluded.

### repeat1_prefill

|Field|Control|Wide|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1782344.0000|1556692.5000|-12.66%|
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
|w4u8_qkv_ring_pipeline_ticks|5155.5000|5153.0000|-0.05%|
|w4u8_qkv_ring_dma_wait_ticks|1710.0000|1680.0000|-1.75%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0000|4.0000|+0.00%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1146.5000|1216.0000|+6.06%|
|w4u8_qkv_ring_hmx_compute_ticks|404.5000|403.5000|-0.25%|
|w4u8_qkv_ring_pool_wait_ticks|3130.5000|3155.0000|+0.78%|
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
|w4u8_input_norm_main_work_ticks|354.5000|354.5000|+0.00%|
|w4u8_input_norm_worker_work_ticks|1615.0000|1602.5000|-0.77%|
|w4u8_input_norm_pool_wait_ticks|25.0000|25.0000|+0.00%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|429.5000|429.0000|-0.12%|
|w4u8_post_residual_worker_work_ticks|1935.0000|1920.5000|-0.75%|
|w4u8_post_residual_pool_wait_ticks|18.5000|16.0000|-13.51%|
|w4u8_final_residual_main_work_ticks|69.0000|66.0000|-4.35%|
|w4u8_final_residual_worker_work_ticks|371.0000|383.5000|+3.37%|
|w4u8_final_residual_pool_wait_ticks|18.5000|24.5000|+32.43%|
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
|scan_cache_append_ticks|147.0000|147.5000|+0.34%|
|scan_cache_pack_ticks|105.5000|100.5000|-4.74%|
|block_orchestration_ticks|122.0000|119.0000|-2.46%|
|layer_bookkeeping_ticks|52.5000|53.0000|+0.95%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|24262.5000|24009.0000|-1.04%|
|invocation_ticks|25415.5000|25138.5000|-1.09%|
|runtime_setup_ticks|1111.5000|1130.0000|+1.66%|
|runtime_teardown_ticks|885.0000|883.0000|-0.23%|
|stage_boundary_ticks|103.0000|105.0000|+1.94%|
|ledger_named_ticks|25415.5000|25138.5000|-1.09%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|143.5000|146.0000|+1.74%|
|metadata_stage_ticks|166.5000|178.0000|+6.91%|
|input_norm_ticks|428.5000|427.5000|-0.23%|
|qkv_projection_ticks|5208.5000|5205.5000|-0.06%|
|qk_norm_rope_ticks|3.0000|3.0000|+0.00%|
|attention_ticks|2391.5000|2371.0000|-0.86%|
|o_projection_ticks|1030.5000|972.0000|-5.68%|
|post_attention_residual_ticks|503.0000|496.0000|-1.39%|
|post_attention_norm_ticks|3.0000|3.0000|+0.00%|
|gate_up_ticks|5193.5000|5081.5000|-2.16%|
|activation_ticks|4876.0000|4816.0000|-1.23%|
|down_ticks|2519.0000|2579.5000|+2.40%|
|final_residual_ticks|146.5000|145.5000|-0.68%|
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
|weight_dma_ticks|10211.0000|10086.5000|-1.22%|
|hmx_compute_ticks|4388.5000|4413.0000|+0.56%|
|projection_pack_ticks|18.0000|18.0000|+0.00%|
|projection_hmx_wait_ticks|516.5000|511.5000|-0.97%|
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
|u8_attention_qk_norm_rope_ticks|17626.0000|17546.5000|-0.45%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|3054.5000|3002.5000|-1.70%|
|u8_cache_native_append_update_ticks|252.0000|240.5000|-4.56%|
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
|u8_attention_qk_hmx_ticks|423.5000|417.5000|-1.42%|
|wide_score_mode|0.0000|1.0000|N/A zero denominator|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|6009.5000|5931.0000|-1.31%|
|u8_attention_av_hmx_ticks|426.0000|416.5000|-2.23%|
|u8_attention_av_requant_ticks|1190.0000|1191.0000|+0.08%|
|u8_attention_pipeline_wait_ticks|1421.0000|1362.5000|-4.12%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1710.0000|1680.0000|-1.75%|
|w4u8_qkvo_hmx_lifetime_ticks|9318.0000|9164.5000|-1.65%|
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
|hmx_command_count|46.0000|46.0000|+0.00%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
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

|Field|Control|Wide|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|962802.6250|957438.1250|-0.56%|
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
|w4u8_qkv_ring_pipeline_ticks|1659.0000|1645.3750|-0.82%|
|w4u8_qkv_ring_dma_wait_ticks|1436.3125|1426.5000|-0.68%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.5625|3.8750|-15.07%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|916.0625|897.8750|-1.99%|
|w4u8_qkv_ring_hmx_compute_ticks|419.6250|397.3750|-5.30%|
|w4u8_qkv_ring_pool_wait_ticks|2.5625|2.5625|+0.00%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|206.4375|208.5625|+1.03%|
|w4u8_o_gate_prefetch_lifetime_ticks|357.2500|356.2500|-0.28%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|548.6250|515.3750|-6.06%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3464.1875|3498.8125|+1.00%|
|w4u8_gate_up_swiglu_join_wait_ticks|87.4375|86.6250|-0.93%|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_q_pair_row4_call_count|8.0000|8.0000|+0.00%|
|w4u8_decode_k_pair_row4_call_count|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_rows_processed|96.0000|96.0000|+0.00%|
|w4u8_decode_k_temp_carrier_skipped_count|8.0000|8.0000|+0.00%|
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
|scan_cache_stage_ticks|338.3750|335.1875|-0.94%|
|scan_cache_append_ticks|57.5000|57.1875|-0.54%|
|scan_cache_pack_ticks|205.2500|204.6875|-0.27%|
|block_orchestration_ticks|20.3125|21.3125|+4.92%|
|layer_bookkeeping_ticks|11.0000|10.8750|-1.14%|
|scan_dynamic_attention_ticks|1280.3125|1244.0000|-2.84%|
|total_ticks|11950.0000|11904.7500|-0.38%|
|invocation_ticks|12670.3125|12626.8750|-0.34%|
|runtime_setup_ticks|720.3125|721.0000|+0.10%|
|runtime_teardown_ticks|639.0000|638.9375|-0.01%|
|stage_boundary_ticks|7.5625|7.8750|+4.13%|
|ledger_named_ticks|12670.3125|12626.8750|-0.34%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|135.8750|136.1875|+0.23%|
|metadata_stage_ticks|139.6875|141.2500|+1.12%|
|input_norm_ticks|102.5000|102.3750|-0.12%|
|qkv_projection_ticks|1679.3750|1665.5000|-0.83%|
|qk_norm_rope_ticks|0.2500|0.1875|-25.00%|
|attention_ticks|1284.8750|1248.6875|-2.82%|
|o_projection_ticks|836.4375|835.5625|-0.10%|
|post_attention_residual_ticks|139.7500|140.1250|+0.27%|
|post_attention_norm_ticks|0.5000|0.5000|+0.00%|
|gate_up_ticks|4358.6250|4360.8750|+0.05%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2250.8750|2246.2500|-0.21%|
|final_residual_ticks|30.4375|30.1875|-0.82%|
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
|weight_dma_ticks|8743.5000|8742.7500|-0.01%|
|hmx_compute_ticks|4298.7500|4277.1250|-0.50%|
|projection_pack_ticks|3.6875|3.6250|-1.69%|
|projection_hmx_wait_ticks|372.5000|367.8750|-1.24%|
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
|attention_unattributed_ticks|577.5000|570.8750|-1.15%|
|u8_attention_qk_norm_rope_ticks|1424.4375|1401.5000|-1.61%|
|u8_attention_k_pack_ticks|13.4375|13.3125|-0.93%|
|u8_attention_v_pack_ticks|83.0000|83.1250|+0.15%|
|u8_cache_native_append_update_ticks|258.6250|257.8125|-0.31%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|27.1875|27.0000|-0.69%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|93.3125|94.2500|+1.00%|
|wide_score_mode|0.0000|1.0000|N/A zero denominator|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|263.2500|233.3750|-11.35%|
|u8_attention_av_hmx_ticks|110.3125|110.8750|+0.51%|
|u8_attention_av_requant_ticks|96.8750|96.1250|-0.77%|
|u8_attention_pipeline_wait_ticks|44.3750|44.8125|+0.99%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1436.8750|1427.5000|-0.65%|
|w4u8_qkvo_hmx_lifetime_ticks|8249.0625|8228.1250|-0.25%|
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
|hmx_command_count|46.0000|46.0000|+0.00%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
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

|Field|Control|Wide|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1542820.2000|1525796.9500|-1.10%|
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
|w4u8_qkv_ring_pipeline_ticks|4813.9000|4823.5500|+0.20%|
|w4u8_qkv_ring_dma_wait_ticks|1451.2500|1454.6000|+0.23%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0000|3.9000|-2.50%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|986.3500|978.9000|-0.76%|
|w4u8_qkv_ring_hmx_compute_ticks|378.0000|377.1500|-0.22%|
|w4u8_qkv_ring_pool_wait_ticks|3126.5500|3118.3000|-0.26%|
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
|w4u8_input_norm_main_work_ticks|322.9000|323.2000|+0.09%|
|w4u8_input_norm_worker_work_ticks|1494.7000|1504.9500|+0.69%|
|w4u8_input_norm_pool_wait_ticks|33.0500|39.6500|+19.97%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|348.7500|356.6500|+2.27%|
|w4u8_post_residual_worker_work_ticks|1808.6000|1796.0000|-0.70%|
|w4u8_post_residual_pool_wait_ticks|68.6000|59.1500|-13.78%|
|w4u8_final_residual_main_work_ticks|75.4000|75.5000|+0.13%|
|w4u8_final_residual_worker_work_ticks|372.8000|373.3500|+0.15%|
|w4u8_final_residual_pool_wait_ticks|17.5500|15.2500|-13.11%|
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
|scan_cache_append_ticks|136.4500|136.5500|+0.07%|
|scan_cache_pack_ticks|91.2000|90.8500|-0.38%|
|block_orchestration_ticks|31.9500|31.5000|-1.41%|
|layer_bookkeeping_ticks|15.0500|15.3000|+1.66%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|22072.6500|21979.0500|-0.42%|
|invocation_ticks|22831.5500|22746.2500|-0.37%|
|runtime_setup_ticks|759.0000|757.3500|-0.22%|
|runtime_teardown_ticks|669.2000|665.7500|-0.52%|
|stage_boundary_ticks|28.1000|26.9500|-4.09%|
|ledger_named_ticks|22831.5500|22746.2500|-0.37%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|135.4500|135.8000|+0.26%|
|metadata_stage_ticks|143.0000|143.3500|+0.24%|
|input_norm_ticks|393.3500|398.8000|+1.39%|
|qkv_projection_ticks|4838.0500|4848.3000|+0.21%|
|qk_norm_rope_ticks|0.6000|0.5500|-8.33%|
|attention_ticks|2317.2000|2306.4500|-0.46%|
|o_projection_ticks|854.4000|846.5500|-0.92%|
|post_attention_residual_ticks|454.7000|453.5500|-0.25%|
|post_attention_norm_ticks|0.6500|0.7000|+7.69%|
|gate_up_ticks|4588.3000|4587.2000|-0.02%|
|activation_ticks|4871.3000|4818.8000|-1.08%|
|down_ticks|2294.1000|2295.7000|+0.07%|
|final_residual_ticks|128.7500|126.7000|-1.59%|
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
|weight_dma_ticks|8876.8000|8906.9000|+0.34%|
|hmx_compute_ticks|4386.5000|4383.0500|-0.08%|
|projection_pack_ticks|4.3000|4.3000|+0.00%|
|projection_hmx_wait_ticks|505.1500|511.3500|+1.23%|
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
|u8_attention_qk_norm_rope_ticks|16931.7500|16950.5000|+0.11%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|2950.0000|2945.1000|-0.17%|
|u8_cache_native_append_update_ticks|225.9500|223.8000|-0.95%|
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
|u8_attention_qk_hmx_ticks|410.3000|410.8000|+0.12%|
|wide_score_mode|0.0000|1.0000|N/A zero denominator|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5960.7500|5862.4000|-1.65%|
|u8_attention_av_hmx_ticks|419.2500|420.2500|+0.24%|
|u8_attention_av_requant_ticks|1198.7500|1196.4500|-0.19%|
|u8_attention_pipeline_wait_ticks|1365.4000|1374.0500|+0.63%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1451.7000|1455.0000|+0.23%|
|w4u8_qkvo_hmx_lifetime_ticks|8130.8000|8148.1500|+0.21%|
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
|hmx_command_count|46.0000|46.0000|+0.00%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
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

|Field|Control|Wide|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|952655.5875|960633.0875|+0.84%|
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
|w4u8_qkv_ring_pipeline_ticks|1651.0062|1656.2063|+0.31%|
|w4u8_qkv_ring_dma_wait_ticks|1431.5062|1435.3688|+0.27%|
|w4u8_qkv_ring_producer_slot_wait_ticks|6.1125|6.4875|+6.13%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|908.1688|921.4937|+1.47%|
|w4u8_qkv_ring_hmx_compute_ticks|406.7250|398.2437|-2.09%|
|w4u8_qkv_ring_pool_wait_ticks|2.6000|2.5938|-0.24%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|216.3062|215.7125|-0.27%|
|w4u8_o_gate_prefetch_lifetime_ticks|358.2437|357.7750|-0.13%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|511.5000|501.2312|-2.01%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3481.2188|3505.2312|+0.69%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.7063|86.4250|-0.32%|
|w4u8_decode_swiglu_rows|4.0000|4.0000|+0.00%|
|w4u8_decode_swiglu_padding_poison|0.0000|0.0000|N/A zero denominator|
|w4u8_qk_norm_rope_rows_observed|4.0000|4.0000|+0.00%|
|w4u8_decode_q_pair_row4_call_count|8.0000|8.0000|+0.00%|
|w4u8_decode_k_pair_row4_call_count|4.0000|4.0000|+0.00%|
|w4u8_decode_qk_rows_processed|96.0000|96.0000|+0.00%|
|w4u8_decode_k_temp_carrier_skipped_count|8.0000|8.0000|+0.00%|
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
|scan_cache_stage_ticks|344.7000|339.6125|-1.48%|
|scan_cache_append_ticks|57.1125|56.9688|-0.25%|
|scan_cache_pack_ticks|199.0687|198.9438|-0.06%|
|block_orchestration_ticks|19.6562|19.7812|+0.64%|
|layer_bookkeeping_ticks|10.8250|10.8750|+0.46%|
|scan_dynamic_attention_ticks|1267.5938|1233.8875|-2.66%|
|total_ticks|11925.9937|11915.3625|-0.09%|
|invocation_ticks|12643.9688|12635.6875|-0.07%|
|runtime_setup_ticks|719.3312|719.6688|+0.05%|
|runtime_teardown_ticks|639.0750|639.3875|+0.05%|
|stage_boundary_ticks|7.5875|7.6312|+0.58%|
|ledger_named_ticks|12643.9688|12635.6875|-0.07%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.5813|135.2625|-0.97%|
|metadata_stage_ticks|141.4625|140.2063|-0.89%|
|input_norm_ticks|101.9500|101.9250|-0.02%|
|qkv_projection_ticks|1671.4875|1676.7687|+0.32%|
|qk_norm_rope_ticks|0.1750|0.1875|+7.14%|
|attention_ticks|1271.1500|1237.3125|-2.66%|
|o_projection_ticks|832.0750|831.7062|-0.04%|
|post_attention_residual_ticks|135.8062|133.9938|-1.33%|
|post_attention_norm_ticks|0.5312|0.4625|-12.94%|
|gate_up_ticks|4365.7750|4380.2437|+0.33%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2243.0688|2243.0375|-0.00%|
|final_residual_ticks|29.3375|29.3562|+0.06%|
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
|weight_dma_ticks|8759.3187|8769.3625|+0.11%|
|hmx_compute_ticks|4300.4000|4268.3125|-0.75%|
|projection_pack_ticks|3.0562|3.0625|+0.20%|
|projection_hmx_wait_ticks|368.2687|366.0063|-0.61%|
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
|attention_unattributed_ticks|572.1500|566.3438|-1.01%|
|u8_attention_qk_norm_rope_ticks|1421.0875|1439.2438|+1.28%|
|u8_attention_k_pack_ticks|13.0687|13.1188|+0.38%|
|u8_attention_v_pack_ticks|82.2938|82.3250|+0.04%|
|u8_cache_native_append_update_ticks|254.3250|254.1125|-0.08%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.2063|26.1313|-0.29%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|91.2812|93.1000|+1.99%|
|wide_score_mode|0.0000|1.0000|N/A zero denominator|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|256.7250|227.2438|-11.48%|
|u8_attention_av_hmx_ticks|113.5187|112.4688|-0.92%|
|u8_attention_av_requant_ticks|96.9250|97.6188|+0.72%|
|u8_attention_pipeline_wait_ticks|45.5250|45.0938|-0.95%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1432.2000|1436.2000|+0.28%|
|w4u8_qkvo_hmx_lifetime_ticks|8236.7438|8259.6000|+0.28%|
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
|hmx_command_count|46.0000|46.0000|+0.00%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
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

Untimed plain dense HMX R3 numerical audit; not valid formal timing evidence.
{
  "r3_wide_eligible": false,
  "r3_wide_whole_gate": [
    {
      "step": 0,
      "dense_pass": true,
      "dense_max_abs": 0.016828659048769623,
      "qk": {
        "max_lsb": 1,
        "changed": 143,
        "elements": 196608,
        "mean_abs_lsb": 0.0007273356119791666
      },
      "whole": {
        "max_lsb": 11,
        "changed": 47232,
        "elements": 131072,
        "mean_abs_lsb": 0.40093994140625
      },
      "cosine": 0.9927221998267413,
      "gate_pass": false
    },
    {
      "step": 1,
      "dense_pass": true,
      "dense_max_abs": 0.01602412760257721,
      "qk": {
        "max_lsb": 1,
        "changed": 7,
        "elements": 3072,
        "mean_abs_lsb": 0.0022786458333333335
      },
      "whole": {
        "max_lsb": 3,
        "changed": 1104,
        "elements": 2048,
        "mean_abs_lsb": 0.6328125
      },
      "cosine": 0.9888342055132645,
      "gate_pass": false
    },
    {
      "step": 2,
      "dense_pass": true,
      "dense_max_abs": 0.008921598899178207,
      "qk": {
        "max_lsb": 0,
        "changed": 0,
        "elements": 3072,
        "mean_abs_lsb": 0.0
      },
      "whole": {
        "max_lsb": 0,
        "changed": 0,
        "elements": 2048,
        "mean_abs_lsb": 0.0
      },
      "cosine": 1.0,
      "gate_pass": true
    },
    {
      "step": 3,
      "dense_pass": true,
      "dense_max_abs": 0.01427252043504268,
      "qk": {
        "max_lsb": 1,
        "changed": 2,
        "elements": 3072,
        "mean_abs_lsb": 0.0006510416666666666
      },
      "whole": {
        "max_lsb": 6,
        "changed": 1145,
        "elements": 2048,
        "mean_abs_lsb": 0.6865234375
      },
      "cosine": 0.986625258688304,
      "gate_pass": false
    },
    {
      "step": 4,
      "dense_pass": true,
      "dense_max_abs": 0.01641767763067037,
      "qk": {
        "max_lsb": 1,
        "changed": 4,
        "elements": 3072,
        "mean_abs_lsb": 0.0013020833333333333
      },
      "whole": {
        "max_lsb": 7,
        "changed": 1157,
        "elements": 2048,
        "mean_abs_lsb": 0.6708984375
      },
      "cosine": 0.9867298955202695,
      "gate_pass": false
    },
    {
      "step": 5,
      "dense_pass": true,
      "dense_max_abs": 0.015677522867918015,
      "qk": {
        "max_lsb": 1,
        "changed": 3,
        "elements": 3072,
        "mean_abs_lsb": 0.0009765625
      },
      "whole": {
        "max_lsb": 3,
        "changed": 1029,
        "elements": 2048,
        "mean_abs_lsb": 0.5625
      },
      "cosine": 0.9905629276745214,
      "gate_pass": false
    },
    {
      "step": 6,
      "dense_pass": true,
      "dense_max_abs": 0.016517288313480094,
      "qk": {
        "max_lsb": 1,
        "changed": 3,
        "elements": 3072,
        "mean_abs_lsb": 0.0009765625
      },
      "whole": {
        "max_lsb": 6,
        "changed": 1046,
        "elements": 2048,
        "mean_abs_lsb": 0.58203125
      },
      "cosine": 0.9854713300864195,
      "gate_pass": false
    },
    {
      "step": 7,
      "dense_pass": true,
      "dense_max_abs": 0.01607669662917033,
      "qk": {
        "max_lsb": 1,
        "changed": 4,
        "elements": 3072,
        "mean_abs_lsb": 0.0013020833333333333
      },
      "whole": {
        "max_lsb": 3,
        "changed": 1108,
        "elements": 2048,
        "mean_abs_lsb": 0.65771484375
      },
      "cosine": 0.9895540210124775,
      "gate_pass": false
    },
    {
      "step": 8,
      "dense_pass": true,
      "dense_max_abs": 0.01570405624806881,
      "qk": {
        "max_lsb": 1,
        "changed": 3,
        "elements": 3072,
        "mean_abs_lsb": 0.0009765625
      },
      "whole": {
        "max_lsb": 0,
        "changed": 0,
        "elements": 2048,
        "mean_abs_lsb": 0.0
      },
      "cosine": 1.0,
      "gate_pass": true
    }
  ]
}

E2E prefill/decode token/s: N/A. Device PPL: N/A. Full-model work is outside PC065. The score repair does not establish model quality acceptance.
