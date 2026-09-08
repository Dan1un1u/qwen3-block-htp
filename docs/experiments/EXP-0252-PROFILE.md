# EXP0252 complete profiling comparison

Scope: one real C64 layer0, EOS+71 tokens; M64 then eight teacher-input M1 steps. Control wide-score SOLE; candidate same no-R3 package with NR64 reciprocal normalization; plain dense HMX R3 plus NR64 failed the independent whole-layer gate and is not formally timed. F16/W4A16 frozen. No full-model token boundary.
Runtime source dd96c4aa6caed2c425237f1672d7dbec72ddcda0. Reporting source 71f8a62449226127e6befb776abc5f572f1c4a88.
Five short and ten alternating paired formal rounds, each repeat1 and repeat10. No discarded warmups/outliers. Primary Host wall is median of ten paired-round means; decode averages eight positions. Paired median ratios,10000 bootstrap resamples seed252. Additive tables use identical two median-Host-ranked rounds for every field in a cell. Engine counters can overlap and are separate.
All2970 timed RPCs match frozen audited per-step hashes; unused legacy output/cache zeros are NOT golden checks. One RPC/step,8MiB requested/granted,peak6682752bytes,zero intermediate DDR/spill,seven native W4 projections. Audit exports are disabled for all timed runs. Both timed arms have R3 disabled. No butterfly/FWHT or scalar oracle fallback.

|Scope|Control us|Wide us|Paired time regression|Ratio95% CI|
|---|---:|---:|---:|---|
|repeat1_prefill|1689.088|1784.193|+5.67%|[1.0087536476886059, 1.1457275028482432]|
|repeat1_decode|1026.361|982.301|+0.63%|[0.9199372887066766, 1.023728823744185]|
|repeat10_prefill|1548.815|1532.357|-1.40%|[0.9636816535913649, 1.0232566887468437]|
|repeat10_decode|954.346|960.539|+0.69%|[0.9929232788322736, 1.0210908932539067]|

## prefill: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 NR64|W4A16 / W4A8 - 1|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.9 (1.15%)|17.8 (1.16%)|N/A no matched W4A16|
|Input RMSNorm|N/A|N/A|19.4 (1.25%)|19.5 (1.27%)|N/A no matched W4A16|
|QKV＋Q/K Norm-RoPE|N/A|N/A|251.8 (16.26%)|251.1 (16.39%)|N/A no matched W4A16|
|QK–Softmax–AV|N/A|N/A|120.0 (7.75%)|120.8 (7.88%)|N/A no matched W4A16|
|O projection|N/A|N/A|43.9 (2.83%)|43.9 (2.87%)|N/A no matched W4A16|
|Post-attention residual＋RMSNorm|N/A|N/A|23.8 (1.53%)|23.6 (1.54%)|N/A no matched W4A16|
|Gate/Up＋SwiGLU|N/A|N/A|489.0 (31.57%)|487.2 (31.80%)|N/A no matched W4A16|
|Down|N/A|N/A|119.2 (7.70%)|118.6 (7.74%)|N/A no matched W4A16|
|Final residual|N/A|N/A|6.7 (0.43%)|6.5 (0.43%)|N/A no matched W4A16|
|KV carrier conversion|N/A|N/A|4.8 (0.31%)|4.7 (0.31%)|N/A no matched W4A16|
|KV append DMA|N/A|N/A|7.2 (0.47%)|7.2 (0.47%)|N/A no matched W4A16|
|Block orchestration|N/A|N/A|1.6 (0.11%)|1.6 (0.11%)|N/A no matched W4A16|
|Layer bookkeeping|N/A|N/A|0.8 (0.05%)|0.8 (0.05%)|N/A no matched W4A16|
|Stage-boundary bookkeeping|N/A|N/A|1.4 (0.09%)|1.5 (0.10%)|N/A no matched W4A16|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A no matched W4A16|
|Runtime setup/teardown|N/A|N/A|74.5 (4.81%)|74.4 (4.86%)|N/A no matched W4A16|
|Host–DSP boundary|N/A|N/A|366.8 (23.68%)|352.9 (23.03%)|N/A no matched W4A16|
|Complete Host wall|N/A|N/A|1548.8 (100.00%)|1532.4 (100.00%)|N/A no matched W4A16|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|N/A|

## decode: repeat10 additive modules

Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.

|Module|F16A16|W4A16|W4A8 control|W4A8 NR64|W4A16 / W4A8 - 1|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|18.0 (1.89%)|17.9 (1.87%)|N/A no matched W4A16|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.55%)|N/A no matched W4A16|
|QKV＋Q/K Norm-RoPE|N/A|N/A|87.0 (9.11%)|87.2 (9.08%)|N/A no matched W4A16|
|QK–Softmax–AV|N/A|N/A|64.6 (6.76%)|64.9 (6.75%)|N/A no matched W4A16|
|O projection|N/A|N/A|43.5 (4.55%)|43.5 (4.53%)|N/A no matched W4A16|
|Post-attention residual＋RMSNorm|N/A|N/A|7.1 (0.74%)|7.1 (0.74%)|N/A no matched W4A16|
|Gate/Up＋SwiGLU|N/A|N/A|226.8 (23.76%)|227.2 (23.65%)|N/A no matched W4A16|
|Down|N/A|N/A|116.4 (12.19%)|117.1 (12.19%)|N/A no matched W4A16|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.16%)|N/A no matched W4A16|
|KV carrier conversion|N/A|N/A|10.3 (1.08%)|10.4 (1.08%)|N/A no matched W4A16|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.0 (0.31%)|N/A no matched W4A16|
|Block orchestration|N/A|N/A|1.0 (0.10%)|1.0 (0.11%)|N/A no matched W4A16|
|Layer bookkeeping|N/A|N/A|0.6 (0.06%)|0.6 (0.06%)|N/A no matched W4A16|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.04%)|N/A no matched W4A16|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A no matched W4A16|
|Runtime setup/teardown|N/A|N/A|70.7 (7.41%)|70.7 (7.36%)|N/A no matched W4A16|
|Host–DSP boundary|N/A|N/A|298.3 (31.26%)|302.8 (31.53%)|N/A no matched W4A16|
|Complete Host wall|N/A|N/A|954.3 (100.00%)|960.5 (100.00%)|N/A no matched W4A16|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A outside layer|N/A outside layer|N/A|

Identity: source branch codex/exp-0252-normalization-r3; evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0252; frozen artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0247/{control,r3}. Project Variant W4U8/native per-channel weight.n. Full correctness, recovery and scope limits in REPORT.md; exact binary/package identity in ARTIFACT_PROVENANCE.json.

## Complete counter diagnostics

Time counters below use19.2ticks/us. Counts/bytes retain native units. Independent per-field medians need not sum. Legacy unused golden fields are excluded.

### repeat1_prefill

|Field|Control|Wide|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1689088.5000|1784192.5000|+5.63%|
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
|w4u8_qkv_ring_pipeline_ticks|5152.5000|5287.5000|+2.62%|
|w4u8_qkv_ring_dma_wait_ticks|1752.5000|1672.5000|-4.56%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0000|4.0000|+0.00%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1274.0000|1205.0000|-5.42%|
|w4u8_qkv_ring_hmx_compute_ticks|404.0000|413.5000|+2.35%|
|w4u8_qkv_ring_pool_wait_ticks|3117.0000|3155.0000|+1.22%|
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
|w4u8_input_norm_main_work_ticks|352.5000|353.0000|+0.14%|
|w4u8_input_norm_worker_work_ticks|1663.5000|1609.0000|-3.28%|
|w4u8_input_norm_pool_wait_ticks|40.0000|25.0000|-37.50%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|425.5000|431.0000|+1.29%|
|w4u8_post_residual_worker_work_ticks|1935.0000|1953.0000|+0.93%|
|w4u8_post_residual_pool_wait_ticks|18.5000|18.0000|-2.70%|
|w4u8_final_residual_main_work_ticks|71.0000|66.5000|-6.34%|
|w4u8_final_residual_worker_work_ticks|373.0000|381.5000|+2.28%|
|w4u8_final_residual_pool_wait_ticks|20.0000|21.0000|+5.00%|
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
|scan_cache_append_ticks|164.0000|171.0000|+4.27%|
|scan_cache_pack_ticks|100.0000|103.0000|+3.00%|
|block_orchestration_ticks|120.5000|119.5000|-0.83%|
|layer_bookkeeping_ticks|51.5000|54.5000|+5.83%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|24405.0000|24191.0000|-0.88%|
|invocation_ticks|25507.5000|25345.0000|-0.64%|
|runtime_setup_ticks|1096.5000|1144.0000|+4.33%|
|runtime_teardown_ticks|884.0000|911.5000|+3.11%|
|stage_boundary_ticks|103.0000|104.0000|+0.97%|
|ledger_named_ticks|25507.5000|25345.0000|-0.64%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|146.0000|146.0000|+0.00%|
|metadata_stage_ticks|178.5000|182.0000|+1.96%|
|input_norm_ticks|426.0000|427.0000|+0.23%|
|qkv_projection_ticks|5206.0000|5338.5000|+2.55%|
|qk_norm_rope_ticks|3.0000|3.0000|+0.00%|
|attention_ticks|2356.0000|2377.0000|+0.89%|
|o_projection_ticks|1042.5000|1035.5000|-0.67%|
|post_attention_residual_ticks|500.5000|510.0000|+1.90%|
|post_attention_norm_ticks|6.0000|6.0000|+0.00%|
|gate_up_ticks|5319.5000|5040.0000|-5.25%|
|activation_ticks|4826.0000|4829.0000|+0.06%|
|down_ticks|2757.5000|2580.0000|-6.44%|
|final_residual_ticks|146.0000|149.5000|+2.40%|
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
|weight_dma_ticks|10526.0000|10012.0000|-4.88%|
|hmx_compute_ticks|4375.5000|4362.0000|-0.31%|
|projection_pack_ticks|18.0000|18.0000|+0.00%|
|projection_hmx_wait_ticks|508.5000|518.0000|+1.87%|
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
|u8_attention_qk_norm_rope_ticks|17557.5000|17778.5000|+1.26%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|3006.0000|3038.0000|+1.06%|
|u8_cache_native_append_update_ticks|259.5000|268.0000|+3.28%|
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
|u8_attention_qk_hmx_ticks|414.0000|407.0000|-1.69%|
|wide_score_mode|1.0000|4.0000|+300.00%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5896.5000|6094.0000|+3.35%|
|u8_attention_av_hmx_ticks|422.0000|406.5000|-3.67%|
|u8_attention_av_requant_ticks|1191.0000|1168.5000|-1.89%|
|u8_attention_pipeline_wait_ticks|1334.0000|1341.0000|+0.52%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1753.5000|1673.5000|-4.56%|
|w4u8_qkvo_hmx_lifetime_ticks|9548.0000|9103.5000|-4.66%|
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
|host_wall_ns|1026360.5625|982301.3125|-4.29%|
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
|w4u8_qkv_ring_pipeline_ticks|1654.0625|1653.1250|-0.06%|
|w4u8_qkv_ring_dma_wait_ticks|1433.0625|1437.3125|+0.30%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.5625|4.5625|+0.00%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|912.3750|924.8125|+1.36%|
|w4u8_qkv_ring_hmx_compute_ticks|400.7500|394.7500|-1.50%|
|w4u8_qkv_ring_pool_wait_ticks|2.5625|2.4375|-4.88%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|206.1250|204.0000|-1.03%|
|w4u8_o_gate_prefetch_lifetime_ticks|355.0625|356.6875|+0.46%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|503.5000|506.5625|+0.61%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3505.1875|3504.6875|-0.01%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.5000|87.1875|+0.79%|
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
|scan_cache_stage_ticks|335.9375|338.6875|+0.82%|
|scan_cache_append_ticks|57.3125|57.7500|+0.76%|
|scan_cache_pack_ticks|204.3750|204.8750|+0.24%|
|block_orchestration_ticks|20.1875|20.3125|+0.62%|
|layer_bookkeeping_ticks|11.1250|11.2500|+1.12%|
|scan_dynamic_attention_ticks|1250.6250|1258.9375|+0.66%|
|total_ticks|11926.2500|11927.1875|+0.01%|
|invocation_ticks|12646.9375|12653.0000|+0.05%|
|runtime_setup_ticks|718.5000|719.1250|+0.09%|
|runtime_teardown_ticks|639.9375|641.2500|+0.21%|
|stage_boundary_ticks|7.6250|7.7500|+1.64%|
|ledger_named_ticks|12646.9375|12653.0000|+0.05%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.1250|136.6875|+0.41%|
|metadata_stage_ticks|140.0000|141.0625|+0.76%|
|input_norm_ticks|102.3750|102.5625|+0.18%|
|qkv_projection_ticks|1674.5000|1673.4375|-0.06%|
|qk_norm_rope_ticks|0.1250|0.2500|+100.00%|
|attention_ticks|1254.8750|1263.3125|+0.67%|
|o_projection_ticks|836.6250|834.7500|-0.22%|
|post_attention_residual_ticks|140.5000|139.6875|-0.58%|
|post_attention_norm_ticks|0.5000|0.4375|-12.50%|
|gate_up_ticks|4355.9375|4363.6250|+0.18%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2243.8125|2232.0000|-0.53%|
|final_residual_ticks|30.8125|30.5625|-0.81%|
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
|weight_dma_ticks|8757.0000|8746.5625|-0.12%|
|hmx_compute_ticks|4265.3125|4264.4375|-0.02%|
|projection_pack_ticks|3.0000|3.1250|+4.17%|
|projection_hmx_wait_ticks|367.6250|365.1250|-0.68%|
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
|attention_unattributed_ticks|572.0625|580.0625|+1.40%|
|u8_attention_qk_norm_rope_ticks|1430.1875|1431.2500|+0.07%|
|u8_attention_k_pack_ticks|13.6250|13.6875|+0.46%|
|u8_attention_v_pack_ticks|83.0625|83.3750|+0.38%|
|u8_cache_native_append_update_ticks|258.1250|259.0000|+0.34%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.8750|27.0000|+0.47%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|92.5625|92.6875|+0.14%|
|wide_score_mode|1.0000|4.0000|+300.00%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|233.0625|242.6250|+4.10%|
|u8_attention_av_hmx_ticks|112.3125|109.6250|-2.39%|
|u8_attention_av_requant_ticks|98.3125|98.0000|-0.32%|
|u8_attention_pipeline_wait_ticks|45.6250|44.4375|-2.60%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1433.6875|1438.3750|+0.33%|
|w4u8_qkvo_hmx_lifetime_ticks|8243.7500|8245.6875|+0.02%|
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
|host_wall_ns|1548815.1500|1532356.7500|-1.06%|
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
|w4u8_qkv_ring_pipeline_ticks|4817.0000|4802.3000|-0.31%|
|w4u8_qkv_ring_dma_wait_ticks|1453.9000|1436.9500|-1.17%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0500|4.0000|-1.23%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|983.4500|976.4000|-0.72%|
|w4u8_qkv_ring_hmx_compute_ticks|374.3000|374.9000|+0.16%|
|w4u8_qkv_ring_pool_wait_ticks|3125.7000|3130.2000|+0.14%|
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
|w4u8_input_norm_main_work_ticks|322.7500|322.4000|-0.11%|
|w4u8_input_norm_worker_work_ticks|1434.7500|1442.7500|+0.56%|
|w4u8_input_norm_pool_wait_ticks|13.2500|17.5500|+32.45%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|354.9000|367.9500|+3.68%|
|w4u8_post_residual_worker_work_ticks|1801.5000|1794.4500|-0.39%|
|w4u8_post_residual_pool_wait_ticks|65.4500|51.4500|-21.39%|
|w4u8_final_residual_main_work_ticks|75.2500|75.7000|+0.60%|
|w4u8_final_residual_worker_work_ticks|373.5000|372.3500|-0.31%|
|w4u8_final_residual_pool_wait_ticks|16.0000|15.2500|-4.69%|
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
|scan_cache_append_ticks|139.6000|139.9500|+0.25%|
|scan_cache_pack_ticks|93.5000|92.0000|-1.60%|
|block_orchestration_ticks|31.2000|31.2000|-0.00%|
|layer_bookkeeping_ticks|14.9500|15.2000|+1.67%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|21978.4500|21907.0500|-0.32%|
|invocation_ticks|22742.8000|22664.5000|-0.34%|
|runtime_setup_ticks|755.2500|758.6500|+0.45%|
|runtime_teardown_ticks|667.7500|666.6500|-0.16%|
|stage_boundary_ticks|27.6500|28.2500|+2.17%|
|ledger_named_ticks|22742.8000|22664.5000|-0.34%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|137.0000|137.1000|+0.07%|
|metadata_stage_ticks|143.4500|144.4500|+0.70%|
|input_norm_ticks|374.3000|374.8500|+0.15%|
|qkv_projection_ticks|4840.9000|4826.3000|-0.30%|
|qk_norm_rope_ticks|0.5000|0.5500|+10.00%|
|attention_ticks|2303.7000|2326.1000|+0.97%|
|o_projection_ticks|845.2500|850.7500|+0.65%|
|post_attention_residual_ticks|453.3500|451.0500|-0.51%|
|post_attention_norm_ticks|0.9500|1.1000|+15.79%|
|gate_up_ticks|4592.6500|4562.8000|-0.65%|
|activation_ticks|4820.5500|4804.1000|-0.34%|
|down_ticks|2289.1000|2278.2000|-0.48%|
|final_residual_ticks|126.5000|126.5000|+0.00%|
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
|weight_dma_ticks|8879.7500|8833.3500|-0.52%|
|hmx_compute_ticks|4439.9500|4415.1500|-0.56%|
|projection_pack_ticks|4.4000|4.5000|+2.27%|
|projection_hmx_wait_ticks|506.3000|509.8500|+0.70%|
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
|u8_attention_qk_norm_rope_ticks|16949.4500|16871.7500|-0.46%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|2944.7000|2948.1500|+0.12%|
|u8_cache_native_append_update_ticks|228.1000|225.6000|-1.10%|
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
|u8_attention_qk_hmx_ticks|411.1000|408.4500|-0.64%|
|wide_score_mode|1.0000|4.0000|+300.00%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5865.4500|6027.2000|+2.76%|
|u8_attention_av_hmx_ticks|421.4500|420.5000|-0.23%|
|u8_attention_av_requant_ticks|1193.1500|1172.0000|-1.77%|
|u8_attention_pipeline_wait_ticks|1362.7000|1382.8000|+1.48%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1454.3500|1437.6500|-1.15%|
|w4u8_qkvo_hmx_lifetime_ticks|8131.1500|8088.7000|-0.52%|
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
|host_wall_ns|954345.7312|960538.7188|+0.65%|
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
|w4u8_qkv_ring_pipeline_ticks|1656.9938|1645.8375|-0.67%|
|w4u8_qkv_ring_dma_wait_ticks|1427.1375|1422.3187|-0.34%|
|w4u8_qkv_ring_producer_slot_wait_ticks|8.7437|7.1000|-18.80%|
|w4u8_qkv_ring_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|897.7188|880.6000|-1.91%|
|w4u8_qkv_ring_hmx_compute_ticks|416.2063|419.2312|+0.73%|
|w4u8_qkv_ring_pool_wait_ticks|2.6063|2.6063|+0.00%|
|w4u8_o_gate_prefetch_start_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|1.0000|+0.00%|
|w4u8_o_gate_prefetch_wait_ticks|215.5750|211.4375|-1.92%|
|w4u8_o_gate_prefetch_lifetime_ticks|357.9500|354.1375|-1.07%|
|w4u8_gate_up_swiglu_publish_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|6.0000|+0.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|1.0000|+0.00%|
|w4u8_gate_up_swiglu_worker_ticks|520.3000|553.3875|+6.36%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3477.7563|3451.1062|-0.77%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.3500|86.2250|-0.14%|
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
|scan_cache_stage_ticks|341.2563|342.3625|+0.32%|
|scan_cache_append_ticks|57.0250|56.9438|-0.14%|
|scan_cache_pack_ticks|198.7312|198.9500|+0.11%|
|block_orchestration_ticks|19.4875|19.5063|+0.10%|
|layer_bookkeeping_ticks|11.0500|11.0563|+0.06%|
|scan_dynamic_attention_ticks|1237.1625|1244.5000|+0.59%|
|total_ticks|11887.9062|11907.3563|+0.16%|
|invocation_ticks|12606.0625|12625.4188|+0.15%|
|runtime_setup_ticks|718.5312|718.4500|-0.01%|
|runtime_teardown_ticks|640.3625|640.1625|-0.03%|
|stage_boundary_ticks|7.7875|7.7937|+0.08%|
|ledger_named_ticks|12606.0625|12625.4188|+0.15%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.4437|136.5188|+0.05%|
|metadata_stage_ticks|141.4000|141.0188|-0.27%|
|input_norm_ticks|101.9125|102.0062|+0.09%|
|qkv_projection_ticks|1677.4313|1666.2062|-0.67%|
|qk_norm_rope_ticks|0.1875|0.1812|-3.33%|
|attention_ticks|1240.5437|1247.8438|+0.59%|
|o_projection_ticks|831.6375|838.6000|+0.84%|
|post_attention_residual_ticks|134.2750|136.1562|+1.40%|
|post_attention_norm_ticks|0.4750|0.5000|+5.26%|
|gate_up_ticks|4355.9062|4354.4937|-0.03%|
|activation_ticks|0.0000|0.0000|N/A zero denominator|
|down_ticks|2238.8250|2247.3812|+0.38%|
|final_residual_ticks|28.9125|28.8375|-0.26%|
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
|weight_dma_ticks|8727.6313|8761.6562|+0.39%|
|hmx_compute_ticks|4315.0938|4276.3250|-0.90%|
|projection_pack_ticks|3.0063|3.1250|+3.95%|
|projection_hmx_wait_ticks|368.9875|367.4875|-0.41%|
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
|attention_unattributed_ticks|566.2750|570.6188|+0.77%|
|u8_attention_qk_norm_rope_ticks|1427.9812|1430.0625|+0.15%|
|u8_attention_k_pack_ticks|13.3438|13.3562|+0.09%|
|u8_attention_v_pack_ticks|82.4188|82.3750|-0.05%|
|u8_cache_native_append_update_ticks|253.9875|254.1375|+0.06%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.0250|25.9187|-0.41%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|93.4562|91.8125|-1.76%|
|wide_score_mode|1.0000|4.0000|+300.00%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|227.0063|235.0813|+3.56%|
|u8_attention_av_hmx_ticks|112.4437|111.8688|-0.51%|
|u8_attention_av_requant_ticks|97.9125|97.7125|-0.20%|
|u8_attention_pipeline_wait_ticks|45.2812|45.1562|-0.28%|
|w4u8_qkvo_weight_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1427.9000|1422.9375|-0.35%|
|w4u8_qkvo_hmx_lifetime_ticks|8226.0500|8237.7313|+0.14%|
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
  "r3_nr64_eligible": false,
  "r3_nr64_whole_gate": {
    "control": [
      {
        "step": 0,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 143,
          "elements": 196608,
          "mean_abs_lsb": 0.0007273356119791666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 64,
            "changed": 580,
            "elements": 65536,
            "mean_abs_lsb": 0.045166015625,
            "cosine": 0.9989649008984282
          },
          "AV": {
            "max_lsb": 35,
            "changed": 813,
            "elements": 131072,
            "mean_abs_lsb": 0.03231048583984375,
            "cosine": 0.9981011897823377
          },
          "O": {
            "max_lsb": 8,
            "changed": 20096,
            "elements": 131072,
            "mean_abs_lsb": 0.15593719482421875,
            "cosine": 0.9975619573551222
          },
          "residual": {
            "max_lsb": 8,
            "changed": 20096,
            "elements": 131072,
            "mean_abs_lsb": 0.15631103515625,
            "cosine": 0.9975854502718957
          },
          "final": {
            "max_lsb": 11,
            "changed": 47232,
            "elements": 131072,
            "mean_abs_lsb": 0.40093994140625,
            "cosine": 0.9927221998267413
          }
        },
        "gate_pass": false
      },
      {
        "step": 1,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 7,
          "elements": 3072,
          "mean_abs_lsb": 0.0022786458333333335
        },
        "boundaries": {
          "probability": {
            "max_lsb": 20,
            "changed": 33,
            "elements": 1040,
            "mean_abs_lsb": 0.09134615384615384,
            "cosine": 0.9991166405415218
          },
          "AV": {
            "max_lsb": 5,
            "changed": 25,
            "elements": 2048,
            "mean_abs_lsb": 0.06103515625,
            "cosine": 0.9967226892108323
          },
          "O": {
            "max_lsb": 2,
            "changed": 580,
            "elements": 2048,
            "mean_abs_lsb": 0.2841796875,
            "cosine": 0.9955591569171786
          },
          "residual": {
            "max_lsb": 1,
            "changed": 6,
            "elements": 2048,
            "mean_abs_lsb": 0.0029296875,
            "cosine": 0.9999992286483433
          },
          "final": {
            "max_lsb": 3,
            "changed": 1104,
            "elements": 2048,
            "mean_abs_lsb": 0.6328125,
            "cosine": 0.9888342055132645
          }
        },
        "gate_pass": false
      },
      {
        "step": 2,
        "dense_pass": true,
        "qk": {
          "max_lsb": 0,
          "changed": 0,
          "elements": 3072,
          "mean_abs_lsb": 0.0
        },
        "boundaries": {
          "probability": {
            "max_lsb": 1,
            "changed": 1,
            "elements": 1056,
            "mean_abs_lsb": 0.000946969696969697,
            "cosine": 0.9999987081096477
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 0.9999999999999998
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 0.9999999999999999
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      },
      {
        "step": 3,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 2,
          "elements": 3072,
          "mean_abs_lsb": 0.0006510416666666666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 40,
            "changed": 53,
            "elements": 1072,
            "mean_abs_lsb": 0.16791044776119404,
            "cosine": 0.9965700244906336
          },
          "AV": {
            "max_lsb": 30,
            "changed": 41,
            "elements": 2048,
            "mean_abs_lsb": 0.1123046875,
            "cosine": 0.9912303809975445
          },
          "O": {
            "max_lsb": 9,
            "changed": 810,
            "elements": 2048,
            "mean_abs_lsb": 0.416015625,
            "cosine": 0.9923831215562317
          },
          "residual": {
            "max_lsb": 1,
            "changed": 15,
            "elements": 2048,
            "mean_abs_lsb": 0.00732421875,
            "cosine": 0.9999980712900162
          },
          "final": {
            "max_lsb": 6,
            "changed": 1145,
            "elements": 2048,
            "mean_abs_lsb": 0.6865234375,
            "cosine": 0.986625258688304
          }
        },
        "gate_pass": false
      },
      {
        "step": 4,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 34,
            "elements": 1088,
            "mean_abs_lsb": 0.08088235294117647,
            "cosine": 0.9995110439742342
          },
          "AV": {
            "max_lsb": 10,
            "changed": 32,
            "elements": 2048,
            "mean_abs_lsb": 0.08056640625,
            "cosine": 0.9953459148909688
          },
          "O": {
            "max_lsb": 6,
            "changed": 546,
            "elements": 2048,
            "mean_abs_lsb": 0.27294921875,
            "cosine": 0.9959386890023308
          },
          "residual": {
            "max_lsb": 1,
            "changed": 11,
            "elements": 2048,
            "mean_abs_lsb": 0.00537109375,
            "cosine": 0.9999985856363222
          },
          "final": {
            "max_lsb": 7,
            "changed": 1157,
            "elements": 2048,
            "mean_abs_lsb": 0.6708984375,
            "cosine": 0.9867298955202695
          }
        },
        "gate_pass": false
      },
      {
        "step": 5,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 20,
            "changed": 27,
            "elements": 1104,
            "mean_abs_lsb": 0.09329710144927536,
            "cosine": 0.9989122824541424
          },
          "AV": {
            "max_lsb": 5,
            "changed": 17,
            "elements": 2048,
            "mean_abs_lsb": 0.04150390625,
            "cosine": 0.9978275831080234
          },
          "O": {
            "max_lsb": 2,
            "changed": 509,
            "elements": 2048,
            "mean_abs_lsb": 0.2490234375,
            "cosine": 0.9964697827366371
          },
          "residual": {
            "max_lsb": 1,
            "changed": 4,
            "elements": 2048,
            "mean_abs_lsb": 0.001953125,
            "cosine": 0.9999994856865606
          },
          "final": {
            "max_lsb": 3,
            "changed": 1029,
            "elements": 2048,
            "mean_abs_lsb": 0.5625,
            "cosine": 0.9905629276745214
          }
        },
        "gate_pass": false
      },
      {
        "step": 6,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 13,
            "changed": 17,
            "elements": 1120,
            "mean_abs_lsb": 0.059821428571428574,
            "cosine": 0.9990701624944756
          },
          "AV": {
            "max_lsb": 5,
            "changed": 21,
            "elements": 2048,
            "mean_abs_lsb": 0.05126953125,
            "cosine": 0.996664887909371
          },
          "O": {
            "max_lsb": 5,
            "changed": 446,
            "elements": 2048,
            "mean_abs_lsb": 0.2236328125,
            "cosine": 0.9959094473721153
          },
          "residual": {
            "max_lsb": 2,
            "changed": 7,
            "elements": 2048,
            "mean_abs_lsb": 0.00390625,
            "cosine": 0.9999987141842783
          },
          "final": {
            "max_lsb": 6,
            "changed": 1046,
            "elements": 2048,
            "mean_abs_lsb": 0.58203125,
            "cosine": 0.9854713300864195
          }
        },
        "gate_pass": false
      },
      {
        "step": 7,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 64,
            "changed": 17,
            "elements": 1136,
            "mean_abs_lsb": 0.08626760563380281,
            "cosine": 0.9952405165749284
          },
          "AV": {
            "max_lsb": 5,
            "changed": 41,
            "elements": 2048,
            "mean_abs_lsb": 0.10009765625,
            "cosine": 0.9947441212113216
          },
          "O": {
            "max_lsb": 2,
            "changed": 729,
            "elements": 2048,
            "mean_abs_lsb": 0.359375,
            "cosine": 0.9944816810204539
          },
          "residual": {
            "max_lsb": 1,
            "changed": 9,
            "elements": 2048,
            "mean_abs_lsb": 0.00439453125,
            "cosine": 0.9999988428482083
          },
          "final": {
            "max_lsb": 3,
            "changed": 1108,
            "elements": 2048,
            "mean_abs_lsb": 0.65771484375,
            "cosine": 0.9895540210124775
          }
        },
        "gate_pass": false
      },
      {
        "step": 8,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 1152,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      }
    ],
    "exact": [
      {
        "step": 0,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 143,
          "elements": 196608,
          "mean_abs_lsb": 0.0007273356119791666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 42,
            "changed": 983,
            "elements": 65536,
            "mean_abs_lsb": 0.0475616455078125,
            "cosine": 0.9993500890849457
          },
          "AV": {
            "max_lsb": 20,
            "changed": 583,
            "elements": 131072,
            "mean_abs_lsb": 0.0225067138671875,
            "cosine": 0.9989552470031575
          },
          "O": {
            "max_lsb": 6,
            "changed": 17199,
            "elements": 131072,
            "mean_abs_lsb": 0.13210296630859375,
            "cosine": 0.9982525660770613
          },
          "residual": {
            "max_lsb": 6,
            "changed": 17199,
            "elements": 131072,
            "mean_abs_lsb": 0.13243865966796875,
            "cosine": 0.9982647663414121
          },
          "final": {
            "max_lsb": 7,
            "changed": 44031,
            "elements": 131072,
            "mean_abs_lsb": 0.36618804931640625,
            "cosine": 0.9943210342447747
          }
        },
        "gate_pass": false
      },
      {
        "step": 1,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 7,
          "elements": 3072,
          "mean_abs_lsb": 0.0022786458333333335
        },
        "boundaries": {
          "probability": {
            "max_lsb": 14,
            "changed": 37,
            "elements": 1040,
            "mean_abs_lsb": 0.10480769230769231,
            "cosine": 0.9994199052925241
          },
          "AV": {
            "max_lsb": 5,
            "changed": 27,
            "elements": 2048,
            "mean_abs_lsb": 0.06591796875,
            "cosine": 0.9971624259722328
          },
          "O": {
            "max_lsb": 2,
            "changed": 636,
            "elements": 2048,
            "mean_abs_lsb": 0.31201171875,
            "cosine": 0.9960145831369064
          },
          "residual": {
            "max_lsb": 1,
            "changed": 10,
            "elements": 2048,
            "mean_abs_lsb": 0.0048828125,
            "cosine": 0.9999987145440395
          },
          "final": {
            "max_lsb": 3,
            "changed": 1091,
            "elements": 2048,
            "mean_abs_lsb": 0.61328125,
            "cosine": 0.9909133549804592
          }
        },
        "gate_pass": false
      },
      {
        "step": 2,
        "dense_pass": true,
        "qk": {
          "max_lsb": 0,
          "changed": 0,
          "elements": 3072,
          "mean_abs_lsb": 0.0
        },
        "boundaries": {
          "probability": {
            "max_lsb": 1,
            "changed": 1,
            "elements": 1056,
            "mean_abs_lsb": 0.000946969696969697,
            "cosine": 0.9999987751544991
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      },
      {
        "step": 3,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 2,
          "elements": 3072,
          "mean_abs_lsb": 0.0006510416666666666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 35,
            "elements": 1072,
            "mean_abs_lsb": 0.06063432835820896,
            "cosine": 0.9996933023628771
          },
          "AV": {
            "max_lsb": 5,
            "changed": 8,
            "elements": 2048,
            "mean_abs_lsb": 0.01953125,
            "cosine": 0.9991201584139767
          },
          "O": {
            "max_lsb": 2,
            "changed": 300,
            "elements": 2048,
            "mean_abs_lsb": 0.14697265625,
            "cosine": 0.9978051590629161
          },
          "residual": {
            "max_lsb": 1,
            "changed": 3,
            "elements": 2048,
            "mean_abs_lsb": 0.00146484375,
            "cosine": 0.9999996142629365
          },
          "final": {
            "max_lsb": 2,
            "changed": 939,
            "elements": 2048,
            "mean_abs_lsb": 0.48876953125,
            "cosine": 0.9932199266946385
          }
        },
        "gate_pass": false
      },
      {
        "step": 4,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 8,
            "changed": 17,
            "elements": 1088,
            "mean_abs_lsb": 0.022058823529411766,
            "cosine": 0.9999250017913279
          },
          "AV": {
            "max_lsb": 5,
            "changed": 10,
            "elements": 2048,
            "mean_abs_lsb": 0.0244140625,
            "cosine": 0.9989188953571324
          },
          "O": {
            "max_lsb": 1,
            "changed": 320,
            "elements": 2048,
            "mean_abs_lsb": 0.15625,
            "cosine": 0.9981669884407444
          },
          "residual": {
            "max_lsb": 1,
            "changed": 3,
            "elements": 2048,
            "mean_abs_lsb": 0.00146484375,
            "cosine": 0.9999996142801958
          },
          "final": {
            "max_lsb": 2,
            "changed": 903,
            "elements": 2048,
            "mean_abs_lsb": 0.47802734375,
            "cosine": 0.9928665770224511
          }
        },
        "gate_pass": false
      },
      {
        "step": 5,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 29,
            "elements": 1104,
            "mean_abs_lsb": 0.07789855072463768,
            "cosine": 0.9995795922685182
          },
          "AV": {
            "max_lsb": 5,
            "changed": 9,
            "elements": 2048,
            "mean_abs_lsb": 0.02197265625,
            "cosine": 0.9990378848125478
          },
          "O": {
            "max_lsb": 1,
            "changed": 358,
            "elements": 2048,
            "mean_abs_lsb": 0.1748046875,
            "cosine": 0.9979254612429328
          },
          "residual": {
            "max_lsb": 1,
            "changed": 5,
            "elements": 2048,
            "mean_abs_lsb": 0.00244140625,
            "cosine": 0.9999993571719267
          },
          "final": {
            "max_lsb": 3,
            "changed": 940,
            "elements": 2048,
            "mean_abs_lsb": 0.50634765625,
            "cosine": 0.9925070823250015
          }
        },
        "gate_pass": false
      },
      {
        "step": 6,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 54,
            "elements": 1120,
            "mean_abs_lsb": 0.08660714285714285,
            "cosine": 0.9994033866463491
          },
          "AV": {
            "max_lsb": 5,
            "changed": 18,
            "elements": 2048,
            "mean_abs_lsb": 0.0439453125,
            "cosine": 0.9975301099558372
          },
          "O": {
            "max_lsb": 1,
            "changed": 435,
            "elements": 2048,
            "mean_abs_lsb": 0.21240234375,
            "cosine": 0.9966180179014269
          },
          "residual": {
            "max_lsb": 1,
            "changed": 6,
            "elements": 2048,
            "mean_abs_lsb": 0.0029296875,
            "cosine": 0.9999992285248314
          },
          "final": {
            "max_lsb": 4,
            "changed": 1006,
            "elements": 2048,
            "mean_abs_lsb": 0.54931640625,
            "cosine": 0.9894606572187675
          }
        },
        "gate_pass": false
      },
      {
        "step": 7,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 22,
            "changed": 30,
            "elements": 1136,
            "mean_abs_lsb": 0.06778169014084508,
            "cosine": 0.9992246446529393
          },
          "AV": {
            "max_lsb": 5,
            "changed": 20,
            "elements": 2048,
            "mean_abs_lsb": 0.048828125,
            "cosine": 0.9979068735605029
          },
          "O": {
            "max_lsb": 2,
            "changed": 491,
            "elements": 2048,
            "mean_abs_lsb": 0.2412109375,
            "cosine": 0.9969896853123144
          },
          "residual": {
            "max_lsb": 1,
            "changed": 9,
            "elements": 2048,
            "mean_abs_lsb": 0.00439453125,
            "cosine": 0.999998842888641
          },
          "final": {
            "max_lsb": 3,
            "changed": 1014,
            "elements": 2048,
            "mean_abs_lsb": 0.5615234375,
            "cosine": 0.9929701116393117
          }
        },
        "gate_pass": false
      },
      {
        "step": 8,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 1,
            "changed": 1,
            "elements": 1152,
            "mean_abs_lsb": 0.0008680555555555555,
            "cosine": 0.9999992929645146
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 0.9999999999999999
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      }
    ],
    "nr64": [
      {
        "step": 0,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 143,
          "elements": 196608,
          "mean_abs_lsb": 0.0007273356119791666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 42,
            "changed": 983,
            "elements": 65536,
            "mean_abs_lsb": 0.0475616455078125,
            "cosine": 0.9993500428228915
          },
          "AV": {
            "max_lsb": 20,
            "changed": 583,
            "elements": 131072,
            "mean_abs_lsb": 0.0225067138671875,
            "cosine": 0.9989551718446792
          },
          "O": {
            "max_lsb": 6,
            "changed": 17232,
            "elements": 131072,
            "mean_abs_lsb": 0.1323699951171875,
            "cosine": 0.9982479557298253
          },
          "residual": {
            "max_lsb": 6,
            "changed": 17232,
            "elements": 131072,
            "mean_abs_lsb": 0.13269805908203125,
            "cosine": 0.9982604972687144
          },
          "final": {
            "max_lsb": 7,
            "changed": 44081,
            "elements": 131072,
            "mean_abs_lsb": 0.366485595703125,
            "cosine": 0.9943212916662678
          }
        },
        "gate_pass": false
      },
      {
        "step": 1,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 7,
          "elements": 3072,
          "mean_abs_lsb": 0.0022786458333333335
        },
        "boundaries": {
          "probability": {
            "max_lsb": 14,
            "changed": 37,
            "elements": 1040,
            "mean_abs_lsb": 0.10480769230769231,
            "cosine": 0.9994199052925241
          },
          "AV": {
            "max_lsb": 5,
            "changed": 27,
            "elements": 2048,
            "mean_abs_lsb": 0.06591796875,
            "cosine": 0.9971624259722328
          },
          "O": {
            "max_lsb": 2,
            "changed": 636,
            "elements": 2048,
            "mean_abs_lsb": 0.31201171875,
            "cosine": 0.9960145831369064
          },
          "residual": {
            "max_lsb": 1,
            "changed": 10,
            "elements": 2048,
            "mean_abs_lsb": 0.0048828125,
            "cosine": 0.9999987145440395
          },
          "final": {
            "max_lsb": 3,
            "changed": 1091,
            "elements": 2048,
            "mean_abs_lsb": 0.61328125,
            "cosine": 0.9909133549804592
          }
        },
        "gate_pass": false
      },
      {
        "step": 2,
        "dense_pass": true,
        "qk": {
          "max_lsb": 0,
          "changed": 0,
          "elements": 3072,
          "mean_abs_lsb": 0.0
        },
        "boundaries": {
          "probability": {
            "max_lsb": 1,
            "changed": 1,
            "elements": 1056,
            "mean_abs_lsb": 0.000946969696969697,
            "cosine": 0.9999987751544991
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      },
      {
        "step": 3,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 2,
          "elements": 3072,
          "mean_abs_lsb": 0.0006510416666666666
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 35,
            "elements": 1072,
            "mean_abs_lsb": 0.06063432835820896,
            "cosine": 0.9996933023628771
          },
          "AV": {
            "max_lsb": 5,
            "changed": 8,
            "elements": 2048,
            "mean_abs_lsb": 0.01953125,
            "cosine": 0.9991201584139767
          },
          "O": {
            "max_lsb": 2,
            "changed": 300,
            "elements": 2048,
            "mean_abs_lsb": 0.14697265625,
            "cosine": 0.9978051590629161
          },
          "residual": {
            "max_lsb": 1,
            "changed": 3,
            "elements": 2048,
            "mean_abs_lsb": 0.00146484375,
            "cosine": 0.9999996142629365
          },
          "final": {
            "max_lsb": 2,
            "changed": 939,
            "elements": 2048,
            "mean_abs_lsb": 0.48876953125,
            "cosine": 0.9932199266946385
          }
        },
        "gate_pass": false
      },
      {
        "step": 4,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 8,
            "changed": 17,
            "elements": 1088,
            "mean_abs_lsb": 0.022058823529411766,
            "cosine": 0.9999250017913279
          },
          "AV": {
            "max_lsb": 5,
            "changed": 10,
            "elements": 2048,
            "mean_abs_lsb": 0.0244140625,
            "cosine": 0.9989188953571324
          },
          "O": {
            "max_lsb": 1,
            "changed": 320,
            "elements": 2048,
            "mean_abs_lsb": 0.15625,
            "cosine": 0.9981669884407444
          },
          "residual": {
            "max_lsb": 1,
            "changed": 3,
            "elements": 2048,
            "mean_abs_lsb": 0.00146484375,
            "cosine": 0.9999996142801958
          },
          "final": {
            "max_lsb": 2,
            "changed": 903,
            "elements": 2048,
            "mean_abs_lsb": 0.47802734375,
            "cosine": 0.9928665770224511
          }
        },
        "gate_pass": false
      },
      {
        "step": 5,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 29,
            "elements": 1104,
            "mean_abs_lsb": 0.07789855072463768,
            "cosine": 0.9995795922685182
          },
          "AV": {
            "max_lsb": 5,
            "changed": 9,
            "elements": 2048,
            "mean_abs_lsb": 0.02197265625,
            "cosine": 0.9990378848125478
          },
          "O": {
            "max_lsb": 1,
            "changed": 358,
            "elements": 2048,
            "mean_abs_lsb": 0.1748046875,
            "cosine": 0.9979254612429328
          },
          "residual": {
            "max_lsb": 1,
            "changed": 5,
            "elements": 2048,
            "mean_abs_lsb": 0.00244140625,
            "cosine": 0.9999993571719267
          },
          "final": {
            "max_lsb": 3,
            "changed": 940,
            "elements": 2048,
            "mean_abs_lsb": 0.50634765625,
            "cosine": 0.9925070823250015
          }
        },
        "gate_pass": false
      },
      {
        "step": 6,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 10,
            "changed": 54,
            "elements": 1120,
            "mean_abs_lsb": 0.08660714285714285,
            "cosine": 0.9994033866463491
          },
          "AV": {
            "max_lsb": 5,
            "changed": 18,
            "elements": 2048,
            "mean_abs_lsb": 0.0439453125,
            "cosine": 0.9975301099558372
          },
          "O": {
            "max_lsb": 1,
            "changed": 435,
            "elements": 2048,
            "mean_abs_lsb": 0.21240234375,
            "cosine": 0.9966180179014269
          },
          "residual": {
            "max_lsb": 1,
            "changed": 6,
            "elements": 2048,
            "mean_abs_lsb": 0.0029296875,
            "cosine": 0.9999992285248314
          },
          "final": {
            "max_lsb": 4,
            "changed": 1006,
            "elements": 2048,
            "mean_abs_lsb": 0.54931640625,
            "cosine": 0.9894606572187675
          }
        },
        "gate_pass": false
      },
      {
        "step": 7,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 4,
          "elements": 3072,
          "mean_abs_lsb": 0.0013020833333333333
        },
        "boundaries": {
          "probability": {
            "max_lsb": 22,
            "changed": 30,
            "elements": 1136,
            "mean_abs_lsb": 0.06778169014084508,
            "cosine": 0.9992246446529393
          },
          "AV": {
            "max_lsb": 5,
            "changed": 20,
            "elements": 2048,
            "mean_abs_lsb": 0.048828125,
            "cosine": 0.9979068735605029
          },
          "O": {
            "max_lsb": 2,
            "changed": 491,
            "elements": 2048,
            "mean_abs_lsb": 0.2412109375,
            "cosine": 0.9969896853123144
          },
          "residual": {
            "max_lsb": 1,
            "changed": 9,
            "elements": 2048,
            "mean_abs_lsb": 0.00439453125,
            "cosine": 0.999998842888641
          },
          "final": {
            "max_lsb": 3,
            "changed": 1014,
            "elements": 2048,
            "mean_abs_lsb": 0.5615234375,
            "cosine": 0.9929701116393117
          }
        },
        "gate_pass": false
      },
      {
        "step": 8,
        "dense_pass": true,
        "qk": {
          "max_lsb": 1,
          "changed": 3,
          "elements": 3072,
          "mean_abs_lsb": 0.0009765625
        },
        "boundaries": {
          "probability": {
            "max_lsb": 1,
            "changed": 1,
            "elements": 1152,
            "mean_abs_lsb": 0.0008680555555555555,
            "cosine": 0.9999992929645146
          },
          "AV": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 0.9999999999999999
          },
          "O": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          },
          "residual": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0000000000000002
          },
          "final": {
            "max_lsb": 0,
            "changed": 0,
            "elements": 2048,
            "mean_abs_lsb": 0.0,
            "cosine": 1.0
          }
        },
        "gate_pass": true
      }
    ]
  }
}

E2E prefill/decode token/s: N/A. Device PPL: N/A. Full-model work has not started; R3 numerical gate remains failed. The score repair does not establish model quality acceptance.

# EXP-0252 integer attention attribution

conditional full-model software PPL with independently validated integer attention, not DSP PPL

F16 PPL 32.210139; C64 W4A16 PPL 32.948761.

| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |
|---|---:|---:|---:|
| wide_sole | 61.643355 | 43.539341 | 0.7063 [0.6579, 0.7583] |
| wide_exact | 53.344543 | 42.741873 | 0.8012 [0.7493, 0.8567] |
| wide_nr64 | 53.747922 | 41.725860 | 0.7763 [0.7265, 0.8278] |

## Conditional increments (positive means worse)

| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |
|---|---:|---:|
| wide_sole->wide_exact | -0.144594 [-0.208946, -0.078130] | -0.018486 [-0.073808, +0.035200] |
| wide_sole->wide_nr64 | -0.137060 [-0.201051, -0.070192] | -0.042544 [-0.099329, +0.013458] |
| wide_exact->wide_nr64 | +0.007533 [-0.047092, +0.062872] | -0.024058 [-0.073718, +0.026818] |

## Cells

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 19.008885 | 38.529575 | 29.582188 | 49.680878 |
| C64 | 19.970410 | 37.256267 | 31.128092 | 50.888251 |
| OFF_wide_sole | 41.049903 | 72.456460 | 56.449858 | 85.998912 |
| OFF_wide_exact | 34.308976 | 62.790521 | 50.659099 | 74.199448 |
| OFF_wide_nr64 | 34.512722 | 63.007195 | 48.485647 | 79.152412 |
| R3_wide_sole | 28.592469 | 53.359615 | 36.485848 | 64.556295 |
| R3_wide_exact | 26.939451 | 49.378318 | 37.032118 | 67.750205 |
| R3_wide_nr64 | 26.232936 | 47.858344 | 37.118096 | 65.047549 |

Eight prespecified arms. F/C64 and OFF/R3 wide_sole/wide_exact reproduce EXP0250 development scores exactly. Fresh128documents/2048targets; frozen weights, parameters, prefix, independent data audit. Numerical, repeat, causal, CE, immutable weight/prefix checks. This is conditional software PPL, not device PPL. NR64 is a single predefined candidate, no final-data selection.

Full hardware and integration results are recorded separately.

# EXP-0252 normalization and dense R3 joint ablation

NR64 (fixed64-bin Q30 reciprocal plus one integer Newton step) approaches exact integer normalization, passes independent hardware arithmetic and no-R3 speed gates. Dense R3 whole-layer gate remains failed even with exact normalization. No promotion, slice or full-model device execution.

Source 71f8a62449226127e6befb776abc5f572f1c4a88; runtime dd96c4aa6caed2c425237f1672d7dbec72ddcda0. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0252. Frozen original per-output-channel W4 and EXP0246 calibration/prefix/attention scales unchanged. C64 names the retained W4A16 quantization reference; this is not group64 deployment.

## Independent software PPL

Fresh128documents/2048targets,32 per en/zh wiki/news cell; all8 arms frozen before scoring. Existing6 development controls reproduce EXP0250 per-token scores. PPL is conditional software arithmetic, not DSP PPL.

|Configuration|PPL|Ratio vs F16|
|---|---:|---:|
|F|32.210139|1.0000|
|C64|32.948761|1.0229|
|OFF_wide_sole|61.643355|1.9138|
|OFF_wide_exact|53.344543|1.6561|
|OFF_wide_nr64|53.747922|1.6687|
|R3_wide_sole|43.539341|1.3517|
|R3_wide_exact|42.741873|1.3270|
|R3_wide_nr64|41.725860|1.2954|
OFF NR64/SOLE PPL ratio 0.871918,95% CI [0.8178711262660174, 0.9322152204144928]. Improvement supported on this panel.
R3 NR64/SOLE PPL ratio 0.958348,95% CI [0.9054445473728077, 1.0135484848728336]. Interval includes1; PPL improvement is not statistically established on this panel.
R3 NR64 remains29.54% above matched F16. All A8 arms fail unchanged quality point gates. NR64 versus exact PPL differences have intervals including1; a lower point estimate is not evidence that reciprocal approximation improves mathematical accuracy.

Paired document NLL/PPL ratios,95%intervals and language/domain cells: SOFTWARE_REPORT.md. Candidate fixed before any final scoring; no final-data selection. Existing overall5%/eachcell10% quality gates remain unchanged.

## Same-input hardware error amplification

|Normalizer|Probability max LSB|AV max LSB|O max LSB|Residual max LSB|Final max LSB|Final cosine|
|---|---:|---:|---:|---:|---:|---:|
|control|64|35|8|8|11|0.99272220|
|exact|42|20|6|6|7|0.99432103|
|nr64|42|20|6|6|7|0.99432129|

These are actual HMX dense R3 versus independently verified Float64-dense reference, on identical layer input and each declared normalizer. Dense matrix1FP16ULP+minnormal and Q/K1code gates pass, but whole2LSB/cosine0.999 does not. Exact normalization also fails; further reciprocal refinement alone cannot remove this captured residual divergence.

Each of12 arm x9step raw-QK/probability/AV boundary sets independently matches NumPy int64 exactly. Same actual Q/K under SOLE/exact/NR64 checked; O/residual/final are actual DSP captures. NR64 vs exact prefill probability difference<=1LSB, final up to3LSB; that is an approximation diagnostic, not its implementation oracle. HVX NR64 vs independent NR64 arithmetic and untimed scalar complete-layer reference are exact. Repeated NR64 output and independently packed K cache/verified V cache pass. Old EXP0251 controls reproduce.

## Hardware speed

Five short and ten rotated paired formal rounds, repeat1/repeat10,2970timed layerRPCs. Exact division and numerically ineligible R3 arms are audit-only. Primary repeat10 median paired-round mean Host wall, paired ratios/10000bootstrap seed252. No outlier or warmup deletion; repeat1 remains separately reported.

|Scope|SOLE us|NR64 us|Paired change|Ratio95% CI|
|---|---:|---:|---:|---|
|repeat1_prefill|1689.088|1784.193|+5.67%|[1.0087536476886059, 1.1457275028482432]|
|repeat1_decode|1026.361|982.301|+0.63%|[0.9199372887066766, 1.023728823744185]|
|repeat10_prefill|1548.815|1532.357|-1.40%|[0.9636816535913649, 1.0232566887468437]|
|repeat10_decode|954.346|960.539|+0.69%|[0.9929232788322736, 1.0210908932539067]|

Primary repeat10 intervals are below1.1; speed gate passes, no extra pairs required. Both include1, so no demonstrated speedup/penalty. Repeat1 prefill uncertainty is wider and does not replace the declared repeat10 primary.8MiB requested/acquired,peak6,682,752bytes,zero timed intermediate DDR/spill/audit,oneRPC and7nativeW4projections; every additive ledger closes exactly. Constant reciprocal table256bytes, per-row LUT construction and Newton arithmetic included.

## Limits and continuation

Current runtime ABI116, layer0 M64+eightM1, capacity72. No full-model device PPL/text/E2E or baseline promotion. Conditional further integration is not entered while R3 numerical failure and model-quality acceptance remain unresolved. No costly mode5 refinement, butterfly, altered weights/activation scales or hidden scalar fallback in timing.

All execution attempts succeeded. PPL model processes loaded before later hardware/reporting-only commits; core PPL source files are byte-identical since a6a8326. Per-score heads retained and verified. Exact probability division is a reference, not an assumed efficient device candidate. Source/evidence checks in independent_integrity_checks.json, provenance in ARTIFACT_PROVENANCE.json.
