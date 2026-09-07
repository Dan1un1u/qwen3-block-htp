# EXP-0240 — LPBQ32 W4A8 single-layer gate

Source: codex/exp-0240-w4a8-lpbq32-layer-gate @ 633e9da1b77ee3336a334013e10329b6baf34c74. Reporting source: e02d01c0d40603bb80fcc9e292d106a7d72be0b9.
Results: /mnt/d/llm_exp/results/qwen3-block-htp/exp0240. Models/artifacts: /mnt/d/llm_exp/models/qwen3-block-htp/exp0240. Paired direct control: existing per-channel direct-W4 W4U8; candidate: fresh original Qwen3 layer14 LPBQ4/8 G32. F16F16 and W4F16 frozen. No baseline promotion.

Decision: stable regression exceeds the user-selected10% threshold in both modes. Full-model execution stopped before starting. This bounds the current project-owned W4-to-S8 implementation; it is not a theoretical lower bound or a benchmark of Qualcomm QNN kernels.

Each replay executes one real layer14 M64 prefill followed by8 teacher-input M1 steps with self-computed persistent KV. There is one FastRPC run per step. Repeat10 means ten complete replays in the same loaded Prepared Runtime; state/cache reset before each prefill, and each decode sequence advances once per step. It is not a frozen-snapshot repeat and not full-model token throughput. No warmup is silently discarded. Five short and ten alternating formal pairs per repeat scope; no outlier deletion.

Primary values are medians of10 paired-round within-round means. Bootstrap uses50000 resamples of the10 paired ratios, seed240. Decode latency is mean complete Host wall per M1 step across all8 positions. The additive module overview uses the same two median-Host-ranked rounds for every field within each cell, preserving closure; detailed counter tables below independently median each numeric field.

| Scope | Control μs | LPBQ32 μs | Paired regression | Paired ratio95% CI |
|---|---:|---:|---:|---|
|repeat1_prefill|1820.104|4416.824|+145.85%|[2.3369, 2.6066]|
|repeat1_decode|1040.143|3727.513|+260.48%|[3.3645, 3.7166]|
|repeat10_prefill|1676.128|4220.438|+151.77%|[2.4951, 2.5788]|
|repeat10_decode|1020.721|3692.571|+262.37%|[3.5605, 3.6876]|

Correctness: independent pinned AIMET grouping/scale oracle and original/control hashes pass. Fresh S8 reconstruction and seven independently derived bias tables pass. All63 projection/step checks against integer accumulation plus SDK native HMX conversion have0 mismatches across1,474,560 values. Integer products are cross-checked with separate int64 reduction. SIMD vs scalar complete-layer outputs and prefill K/V are exact; unit-multiplier LPBQ vs frozen direct-W4 are exact; all final short/formal output hashes match these frozen controls. Numeric audit is disabled during speed collection. Legacy output/cache comparison fields in exp0240_profile are inapplicable placeholders, explicitly marked historical_reference_used=false; they are not correctness evidence.

Physical gates: exact8MiB requested/acquired for every cell, peak plan6,682,752 bytes, zero intermediate DDR or spill/fill, one HMX owner, standalone FastRPC, no QNN/CPU fallback. Audit-only public projection captures are excluded from formal timing; expanded weights never reside in DDR. Payload overhead is3.125% vs W4 code bytes (0.125bits/weight), and50,331,648 S8 bytes are produced inside VTCM per layer.

Scheduling: LPBQ QKV uses two four-output-tile slots with three HVX expansion workers. O/Gate/Up use double-buffered DMA/HVX batches8, Down batches2; one HMX owner overlaps the previous command with next-slot DMA/expansion. Direct control retains recipe-fastest N4 batch sizes and Gate/Up streaming. The additional same-per-channel-format control disables continuous Gate/Up/prefetch/streaming and uses QKV4, O/Gate/Up8, Down2. Its five repeat10 rounds are unpaired diagnostic measurements, so they do not replace the primary paired control.
Schedule control prefill: 1743.422μs.
Schedule control decode: 1169.124μs.

Recovery record: initial runner required token generation even for the isolated layer replay; the allowance is scoped to the one-layer build. The first decoder had scalar multiplier broadcasts and byte loads; final kernel uses aligned HVX metadata reads/rotation and SIMD multiply. Initial public audit copies lacked explicit DSP cache flush; unchanged complete-layer outputs and the subsequent exact native integer audit identified and resolved this capture defect. Original failed captures/results remain retained. A final telemetry-only repair corrected actual O/MLP batch counts, followed by fresh5short/10formal collections under final_* names. Earlier timings remain diagnostic.

## prefill repeat10 additive module overview

F16F16/W4F16 are N/A for this new equivalent-scope paired run. Historical full-model results have different execution boundaries and are not substituted.

|Module|F16F16|W4F16|W4U8 per-channel|W4U8 LPBQ32|LPBQ time change|
|---|---|---|---:|---:|---:|
|I/O、metadata|N/A frozen|N/A frozen|17.417 (1.04%)|17.690 (0.42%)|+1.57%|
|Input RMSNorm|N/A frozen|N/A frozen|19.904 (1.19%)|19.544 (0.46%)|-1.81%|
|QKV＋Q/K Norm-RoPE|N/A frozen|N/A frozen|251.315 (14.99%)|357.771 (8.48%)|+42.36%|
|QK–Softmax–AV|N/A frozen|N/A frozen|125.091 (7.46%)|121.885 (2.89%)|-2.56%|
|O projection|N/A frozen|N/A frozen|42.529 (2.54%)|289.250 (6.85%)|+580.13%|
|Post-attention residual＋RMSNorm|N/A frozen|N/A frozen|23.414 (1.40%)|23.518 (0.56%)|+0.44%|
|Gate/Up＋SwiGLU|N/A frozen|N/A frozen|605.193 (36.11%)|2110.112 (50.00%)|+248.67%|
|Down|N/A frozen|N/A frozen|117.133 (6.99%)|871.466 (20.65%)|+644.00%|
|Final residual|N/A frozen|N/A frozen|6.544 (0.39%)|6.719 (0.16%)|+2.67%|
|KV carrier conversion|N/A frozen|N/A frozen|4.826 (0.29%)|4.586 (0.11%)|-4.96%|
|KV append DMA|N/A frozen|N/A frozen|7.039 (0.42%)|7.211 (0.17%)|+2.44%|
|Block orchestration|N/A frozen|N/A frozen|1.641 (0.10%)|1.646 (0.04%)|+0.32%|
|Layer bookkeeping|N/A frozen|N/A frozen|0.789 (0.05%)|0.768 (0.02%)|-2.64%|
|Stage-boundary bookkeeping|N/A frozen|N/A frozen|1.602 (0.10%)|1.740 (0.04%)|+8.62%|
|DSP unattributed|N/A frozen|N/A frozen|0.000 (0.00%)|0.000 (0.00%)|N/A zero denominator|
|Runtime setup/teardown|N/A frozen|N/A frozen|74.583 (4.45%)|74.352 (1.76%)|-0.31%|
|Host–DSP boundary|N/A frozen|N/A frozen|377.109 (22.50%)|312.180 (7.40%)|-17.22%|
|Complete Host wall|N/A frozen|N/A frozen|1676.128 (100.00%)|4220.438 (100.00%)|+151.80%|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy excluding final norm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
## decode repeat10 additive module overview

F16F16/W4F16 are N/A for this new equivalent-scope paired run. Historical full-model results have different execution boundaries and are not substituted.

|Module|F16F16|W4F16|W4U8 per-channel|W4U8 LPBQ32|LPBQ time change|
|---|---|---|---:|---:|---:|
|I/O、metadata|N/A frozen|N/A frozen|17.657 (1.73%)|17.878 (0.48%)|+1.25%|
|Input RMSNorm|N/A frozen|N/A frozen|5.311 (0.52%)|5.306 (0.14%)|-0.09%|
|QKV＋Q/K Norm-RoPE|N/A frozen|N/A frozen|85.588 (8.39%)|229.520 (6.22%)|+168.17%|
|QK–Softmax–AV|N/A frozen|N/A frozen|123.745 (12.12%)|123.434 (3.34%)|-0.25%|
|O projection|N/A frozen|N/A frozen|42.868 (4.20%)|288.876 (7.82%)|+573.88%|
|Post-attention residual＋RMSNorm|N/A frozen|N/A frozen|7.059 (0.69%)|6.835 (0.19%)|-3.17%|
|Gate/Up＋SwiGLU|N/A frozen|N/A frozen|227.336 (22.27%)|1754.296 (47.51%)|+671.68%|
|Down|N/A frozen|N/A frozen|116.469 (11.41%)|871.450 (23.60%)|+648.22%|
|Final residual|N/A frozen|N/A frozen|1.517 (0.15%)|1.484 (0.04%)|-2.17%|
|KV carrier conversion|N/A frozen|N/A frozen|10.384 (1.02%)|10.483 (0.28%)|+0.95%|
|KV append DMA|N/A frozen|N/A frozen|2.891 (0.28%)|2.886 (0.08%)|-0.17%|
|Block orchestration|N/A frozen|N/A frozen|1.065 (0.10%)|1.082 (0.03%)|+1.65%|
|Layer bookkeeping|N/A frozen|N/A frozen|0.575 (0.06%)|0.583 (0.02%)|+1.42%|
|Stage-boundary bookkeeping|N/A frozen|N/A frozen|0.398 (0.04%)|0.406 (0.01%)|+1.88%|
|DSP unattributed|N/A frozen|N/A frozen|0.000 (0.00%)|0.000 (0.00%)|N/A zero denominator|
|Runtime setup/teardown|N/A frozen|N/A frozen|70.804 (6.94%)|71.081 (1.92%)|+0.39%|
|Host–DSP boundary|N/A frozen|N/A frozen|307.054 (30.08%)|306.971 (8.31%)|-0.03%|
|Complete Host wall|N/A frozen|N/A frozen|1020.721 (100.00%)|3692.571 (100.00%)|+261.76%|
|Embedding|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|
|LM head + greedy excluding final norm|N/A|N/A|N/A outside layer|N/A outside layer|N/A|

## Complete diagnostics

All engine, expansion, DMA and wait counters can overlap and must not be added to the exclusive ledger. Time fields are in qtimer ticks at19.2ticks/μs; bytes/counts retain native units. Runtime configuration fields are declared controls; actual O/MLP batch counts are corrected in final collection. Missing or out-of-scope external correctness fields are excluded here and explained above.

### Repeat1 prefill

|Field|Control median|LPBQ median|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1820104.0000|4416823.5000|+142.67%|
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
|w4u8_o_batch_n_tiles_observed|16.0000|8.0000|-50.00%|
|w4u8_o_batch_count|4.0000|8.0000|+100.00%|
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
|w4u8_decode_direct_n_gate_up_continuous|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|3.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|2.0000|-60.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|32.0000|+433.33%|
|w4u8_qkv_ring_expand_task_count|0.0000|128.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|5024.0000|7062.0000|+40.57%|
|w4u8_qkv_ring_dma_wait_ticks|1485.0000|2064.0000|+38.99%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0000|2129.5000|+53137.50%|
|w4u8_qkv_ring_expand_ticks|0.0000|9716.5000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1001.5000|3638.5000|+263.31%|
|w4u8_qkv_ring_hmx_compute_ticks|404.0000|314.0000|-22.28%|
|w4u8_qkv_ring_pool_wait_ticks|3232.0000|2378.5000|-26.41%|
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
|w4u8_input_norm_main_work_ticks|311.0000|354.5000|+13.99%|
|w4u8_input_norm_worker_work_ticks|1687.0000|1647.5000|-2.34%|
|w4u8_input_norm_pool_wait_ticks|75.5000|38.0000|-49.67%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|426.0000|426.5000|+0.12%|
|w4u8_post_residual_worker_work_ticks|1946.5000|1924.5000|-1.13%|
|w4u8_post_residual_pool_wait_ticks|24.5000|16.0000|-34.69%|
|w4u8_final_residual_main_work_ticks|73.0000|78.5000|+7.53%|
|w4u8_final_residual_worker_work_ticks|382.5000|367.0000|-4.05%|
|w4u8_final_residual_pool_wait_ticks|18.5000|8.0000|-56.76%|
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
|scan_cache_append_ticks|143.5000|146.5000|+2.09%|
|scan_cache_pack_ticks|103.5000|109.0000|+5.31%|
|block_orchestration_ticks|116.5000|113.5000|-2.58%|
|layer_bookkeeping_ticks|53.0000|50.5000|-4.72%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|25479.0000|75361.0000|+195.78%|
|invocation_ticks|26564.5000|76500.5000|+187.98%|
|runtime_setup_ticks|1093.0000|1123.0000|+2.74%|
|runtime_teardown_ticks|904.0000|898.5000|-0.61%|
|stage_boundary_ticks|107.0000|107.0000|+0.00%|
|ledger_named_ticks|26564.5000|76500.5000|+187.98%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|106.0000|124.0000|+16.98%|
|metadata_stage_ticks|162.0000|163.5000|+0.93%|
|input_norm_ticks|436.5000|438.0000|+0.34%|
|qkv_projection_ticks|5068.0000|7106.5000|+40.22%|
|qk_norm_rope_ticks|3.0000|3.0000|+0.00%|
|attention_ticks|2399.0000|2401.5000|+0.10%|
|o_projection_ticks|902.0000|5640.5000|+525.33%|
|post_attention_residual_ticks|500.0000|496.0000|-0.80%|
|post_attention_norm_ticks|3.0000|3.0000|+0.00%|
|gate_up_ticks|4604.5000|33209.0000|+621.23%|
|activation_ticks|7161.0000|7396.0000|+3.28%|
|down_ticks|2425.5000|16751.5000|+590.64%|
|final_residual_ticks|142.0000|137.5000|-3.17%|
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
|w4u8_decode_direct_n_projection_count|7.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|0.0000|-100.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|32.0000|+300.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|0.0000|-100.00%|
|weight_dma_ticks|9007.0000|11571.5000|+28.47%|
|hmx_compute_ticks|4363.5000|9281.5000|+112.71%|
|projection_pack_ticks|18.0000|18.0000|+0.00%|
|projection_hmx_wait_ticks|519.0000|474.0000|-8.67%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|9716.5000|N/A zero denominator|
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
|u8_attention_qk_norm_rope_ticks|17361.5000|16106.0000|-7.23%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|3506.5000|3518.0000|+0.33%|
|u8_cache_native_append_update_ticks|239.5000|255.5000|+6.68%|
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
|u8_attention_qk_hmx_ticks|423.0000|436.5000|+3.19%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5811.0000|5833.5000|+0.39%|
|u8_attention_av_hmx_ticks|401.5000|415.5000|+3.49%|
|u8_attention_av_requant_ticks|1034.5000|1048.0000|+1.30%|
|u8_attention_pipeline_wait_ticks|1435.5000|1266.5000|-11.77%|
|w4u8_qkvo_weight_expand_ticks|0.0000|14296.0000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1485.5000|2066.5000|+39.11%|
|w4u8_qkvo_hmx_lifetime_ticks|8321.0000|57401.0000|+589.83%|
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
|w4u8_mlp_weight_expand_ticks|0.0000|41088.0000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|136.0000|+195.65%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.0000|49408.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|240.0000|+300.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|26116096.0000|+3.10%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|
### Repeat1 decode

|Field|Control median|LPBQ median|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|1040143.3750|3727512.9375|+258.37%|
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
|w4u8_o_batch_n_tiles_observed|16.0000|8.0000|-50.00%|
|w4u8_o_batch_count|4.0000|8.0000|+100.00%|
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
|w4u8_decode_direct_n_gate_up_continuous|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|3.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|2.0000|-60.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|32.0000|+433.33%|
|w4u8_qkv_ring_expand_task_count|0.0000|128.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|1643.0000|4372.8125|+166.15%|
|w4u8_qkv_ring_dma_wait_ticks|1413.1875|1961.5000|+38.80%|
|w4u8_qkv_ring_producer_slot_wait_ticks|12.5625|2013.6875|+15929.35%|
|w4u8_qkv_ring_expand_ticks|0.0000|9594.8750|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|881.1875|3509.8125|+298.30%|
|w4u8_qkv_ring_hmx_compute_ticks|429.6875|252.6250|-41.21%|
|w4u8_qkv_ring_pool_wait_ticks|2.7500|2.4375|-11.36%|
|w4u8_o_gate_prefetch_start_count|1.0000|0.0000|-100.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|0.0000|-100.00%|
|w4u8_o_gate_prefetch_wait_ticks|198.4375|0.0000|-100.00%|
|w4u8_o_gate_prefetch_lifetime_ticks|351.0625|0.0000|-100.00%|
|w4u8_gate_up_swiglu_publish_count|6.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_worker_ticks|709.8125|0.0000|-100.00%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3342.1875|0.0000|-100.00%|
|w4u8_gate_up_swiglu_join_wait_ticks|128.7500|0.0000|-100.00%|
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
|scan_cache_stage_ticks|361.8750|356.3125|-1.54%|
|scan_cache_append_ticks|55.7500|55.5625|-0.34%|
|scan_cache_pack_ticks|203.6250|205.3750|+0.86%|
|block_orchestration_ticks|24.8750|23.3750|-6.03%|
|layer_bookkeeping_ticks|11.0000|11.1875|+1.70%|
|scan_dynamic_attention_ticks|2386.6875|2378.0625|-0.36%|
|total_ticks|13064.8750|64282.8750|+392.03%|
|invocation_ticks|13795.4375|65007.8125|+371.23%|
|runtime_setup_ticks|724.0000|723.1875|-0.11%|
|runtime_teardown_ticks|638.4375|642.5625|+0.65%|
|stage_boundary_ticks|7.6875|7.6250|-0.81%|
|ledger_named_ticks|13795.4375|65007.8125|+371.23%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.2500|136.1250|-0.09%|
|metadata_stage_ticks|141.6875|140.0625|-1.15%|
|input_norm_ticks|101.7500|102.0625|+0.31%|
|qkv_projection_ticks|1663.5000|4393.1250|+164.09%|
|qk_norm_rope_ticks|0.1250|0.2500|+100.00%|
|attention_ticks|2391.5000|2382.8750|-0.36%|
|o_projection_ticks|836.1250|5543.3125|+562.98%|
|post_attention_residual_ticks|141.5000|132.0000|-6.71%|
|post_attention_norm_ticks|0.5000|0.6250|+25.00%|
|gate_up_ticks|4403.7500|33096.6250|+651.56%|
|activation_ticks|0.0000|577.3750|N/A zero denominator|
|down_ticks|2237.0625|16725.1250|+647.64%|
|final_residual_ticks|29.0625|29.0625|+0.00%|
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
|w4u8_decode_direct_n_projection_count|7.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|0.0000|-100.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|32.0000|+300.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|0.0000|-100.00%|
|weight_dma_ticks|8721.5000|11286.0000|+29.40%|
|hmx_compute_ticks|4332.8125|9172.3750|+111.70%|
|projection_pack_ticks|4.5625|4.2500|-6.85%|
|projection_hmx_wait_ticks|376.9375|460.3125|+22.12%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|9594.8750|N/A zero denominator|
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
|attention_unattributed_ticks|1623.4375|1621.6875|-0.11%|
|u8_attention_qk_norm_rope_ticks|1510.2500|996.9375|-33.99%|
|u8_attention_k_pack_ticks|13.6250|13.6875|+0.46%|
|u8_attention_v_pack_ticks|86.6250|86.1250|-0.58%|
|u8_cache_native_append_update_ticks|256.8750|258.5000|+0.63%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.6250|27.5625|+3.52%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|93.0000|95.6250|+2.82%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|272.3750|272.4375|+0.02%|
|u8_attention_av_hmx_ticks|146.3125|143.3750|-2.01%|
|u8_attention_av_requant_ticks|100.1250|100.1875|+0.06%|
|u8_attention_pipeline_wait_ticks|45.8125|45.9375|+0.27%|
|w4u8_qkvo_weight_expand_ticks|0.0000|14162.5000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1413.8125|1964.5625|+38.95%|
|w4u8_qkvo_hmx_lifetime_ticks|8228.9375|57095.9375|+593.84%|
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
|w4u8_mlp_weight_expand_ticks|0.0000|41090.6250|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|136.0000|+195.65%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.0000|49536.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|240.0000|+300.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|26116096.0000|+3.10%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|
### Repeat10 prefill

|Field|Control median|LPBQ median|Change|
|---|---:|---:|---:|
|logical_m|64.0000|64.0000|+0.00%|
|host_wall_ns|1676127.7000|4220437.5500|+151.80%|
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
|w4u8_o_batch_n_tiles_observed|16.0000|8.0000|-50.00%|
|w4u8_o_batch_count|4.0000|8.0000|+100.00%|
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
|w4u8_decode_direct_n_gate_up_continuous|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|3.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|2.0000|-60.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|32.0000|+433.33%|
|w4u8_qkv_ring_expand_task_count|0.0000|128.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|4801.4500|6865.1000|+42.98%|
|w4u8_qkv_ring_dma_wait_ticks|1412.9500|1976.2000|+39.86%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.0500|2041.9500|+50318.52%|
|w4u8_qkv_ring_expand_ticks|0.0000|9631.8000|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|957.2000|3510.4500|+266.74%|
|w4u8_qkv_ring_hmx_compute_ticks|374.6000|286.9500|-23.40%|
|w4u8_qkv_ring_pool_wait_ticks|3154.4000|2414.7500|-23.45%|
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
|w4u8_input_norm_main_work_ticks|319.7000|316.8000|-0.91%|
|w4u8_input_norm_worker_work_ticks|1461.1500|1450.7500|-0.71%|
|w4u8_input_norm_pool_wait_ticks|25.1000|23.0500|-8.17%|
|w4u8_residual_active_contexts|6.0000|6.0000|+0.00%|
|w4u8_post_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_final_residual_task_count|16.0000|16.0000|+0.00%|
|w4u8_post_residual_main_work_ticks|373.1000|381.1000|+2.14%|
|w4u8_post_residual_worker_work_ticks|1783.2000|1771.1500|-0.68%|
|w4u8_post_residual_pool_wait_ticks|40.9000|32.7000|-20.05%|
|w4u8_final_residual_main_work_ticks|77.1000|78.0500|+1.23%|
|w4u8_final_residual_worker_work_ticks|367.5000|368.6000|+0.30%|
|w4u8_final_residual_pool_wait_ticks|12.7500|12.6500|-0.78%|
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
|scan_cache_append_ticks|136.9500|136.3000|-0.47%|
|scan_cache_pack_ticks|86.7500|86.5500|-0.23%|
|block_orchestration_ticks|30.5000|30.4500|-0.16%|
|layer_bookkeeping_ticks|15.4500|15.2500|-1.29%|
|scan_dynamic_attention_ticks|0.0000|0.0000|N/A zero denominator|
|total_ticks|24214.1000|74236.4000|+206.58%|
|invocation_ticks|24981.7000|74998.0000|+200.21%|
|runtime_setup_ticks|764.3000|759.1500|-0.67%|
|runtime_teardown_ticks|667.9000|668.1000|+0.03%|
|stage_boundary_ticks|30.5000|31.1500|+2.13%|
|ledger_named_ticks|24981.7000|74998.0000|+200.21%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|134.9500|128.7000|-4.63%|
|metadata_stage_ticks|146.3000|144.2000|-1.44%|
|input_norm_ticks|381.0500|377.8000|-0.85%|
|qkv_projection_ticks|4824.8000|6887.8500|+42.76%|
|qk_norm_rope_ticks|0.4500|0.5000|+11.11%|
|attention_ticks|2404.0000|2338.0000|-2.75%|
|o_projection_ticks|826.1000|5552.0500|+572.08%|
|post_attention_residual_ticks|448.4500|449.7500|+0.29%|
|post_attention_norm_ticks|0.7000|0.7000|+0.00%|
|gate_up_ticks|4475.9500|33090.0000|+639.28%|
|activation_ticks|7147.1000|7387.4000|+3.36%|
|down_ticks|2258.4000|16712.8000|+640.03%|
|final_residual_ticks|125.4500|125.8500|+0.32%|
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
|w4u8_decode_direct_n_projection_count|7.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|0.0000|-100.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|32.0000|+300.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|0.0000|-100.00%|
|weight_dma_ticks|8689.7500|11284.8000|+29.86%|
|hmx_compute_ticks|4374.2500|9811.2500|+124.30%|
|projection_pack_ticks|4.6000|4.5000|-2.17%|
|projection_hmx_wait_ticks|500.9500|467.4000|-6.70%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|9631.8000|N/A zero denominator|
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
|u8_attention_qk_norm_rope_ticks|16950.3500|15832.7000|-6.59%|
|u8_attention_k_pack_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_v_pack_ticks|3465.4500|3442.3000|-0.67%|
|u8_cache_native_append_update_ticks|222.6000|222.9000|+0.13%|
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
|u8_attention_qk_hmx_ticks|433.7000|431.8000|-0.44%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|5804.9500|5791.7000|-0.23%|
|u8_attention_av_hmx_ticks|418.0000|419.6000|+0.38%|
|u8_attention_av_requant_ticks|1044.2500|1043.6000|-0.06%|
|u8_attention_pipeline_wait_ticks|1475.5500|1226.1500|-16.90%|
|w4u8_qkvo_weight_expand_ticks|0.0000|14200.4000|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1413.5000|1980.5500|+40.12%|
|w4u8_qkvo_hmx_lifetime_ticks|7952.7000|57113.7000|+618.17%|
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
|w4u8_mlp_weight_expand_ticks|0.0000|41086.9500|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|136.0000|+195.65%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.0000|49408.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|240.0000|+300.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|26116096.0000|+3.10%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|
### Repeat10 decode

|Field|Control median|LPBQ median|Change|
|---|---:|---:|---:|
|logical_m|1.0000|1.0000|+0.00%|
|host_wall_ns|1020721.0437|3692571.2750|+261.76%|
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
|w4u8_o_batch_n_tiles_observed|16.0000|8.0000|-50.00%|
|w4u8_o_batch_count|4.0000|8.0000|+100.00%|
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
|w4u8_decode_direct_n_gate_up_continuous|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_o_gate_prefetch|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.0000|16.0000|+0.00%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.0000|32.0000|+0.00%|
|w4u8_decode_direct_n_down_batch_n_tiles|8.0000|8.0000|+0.00%|
|w4u8_decode_direct_n_down_single_dma|1.0000|1.0000|+0.00%|
|w4u8_decode_direct_n_o_single_dma|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_slot_count|2.0000|2.0000|+0.00%|
|w4u8_qkv_ring_expand_worker_count|0.0000|3.0000|N/A zero denominator|
|w4u8_qkv_ring_prep_worker_count|5.0000|2.0000|-60.00%|
|w4u8_qkv_ring_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_batch_count|6.0000|32.0000|+433.33%|
|w4u8_qkv_ring_expand_task_count|0.0000|128.0000|N/A zero denominator|
|w4u8_qkv_ring_hmx_dispatch_count|1.0000|1.0000|+0.00%|
|w4u8_qkv_ring_head_publish_count|24.0000|24.0000|+0.00%|
|w4u8_qkv_ring_pipeline_ticks|1622.8250|4402.7625|+171.30%|
|w4u8_qkv_ring_dma_wait_ticks|1405.1813|1983.2500|+41.14%|
|w4u8_qkv_ring_producer_slot_wait_ticks|5.1437|2012.4125|+39023.45%|
|w4u8_qkv_ring_expand_ticks|0.0000|9577.2500|N/A zero denominator|
|w4u8_qkv_ring_hmx_ready_wait_ticks|908.0438|3510.6937|+286.62%|
|w4u8_qkv_ring_hmx_compute_ticks|402.6500|266.8937|-33.72%|
|w4u8_qkv_ring_pool_wait_ticks|2.6562|2.4312|-8.47%|
|w4u8_o_gate_prefetch_start_count|1.0000|0.0000|-100.00%|
|w4u8_o_gate_prefetch_consume_count|1.0000|0.0000|-100.00%|
|w4u8_o_gate_prefetch_wait_ticks|207.8250|0.0000|-100.00%|
|w4u8_o_gate_prefetch_lifetime_ticks|349.9875|0.0000|-100.00%|
|w4u8_gate_up_swiglu_publish_count|6.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_consume_count|6.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_overlap_observed|1.0000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_worker_ticks|744.4062|0.0000|-100.00%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3266.1000|0.0000|-100.00%|
|w4u8_gate_up_swiglu_join_wait_ticks|129.9375|0.0000|-100.00%|
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
|scan_cache_stage_ticks|365.7875|365.4500|-0.09%|
|scan_cache_append_ticks|55.5312|55.5438|+0.02%|
|scan_cache_pack_ticks|198.9688|200.2438|+0.64%|
|block_orchestration_ticks|20.1750|20.7188|+2.70%|
|layer_bookkeeping_ticks|11.1062|11.0500|-0.51%|
|scan_dynamic_attention_ticks|2370.8500|2371.9125|+0.04%|
|total_ticks|12981.5875|64253.3625|+394.96%|
|invocation_ticks|13704.7000|64975.7312|+374.11%|
|runtime_setup_ticks|722.3125|723.8687|+0.22%|
|runtime_teardown_ticks|639.1750|641.8250|+0.41%|
|stage_boundary_ticks|7.6750|7.6250|-0.65%|
|ledger_named_ticks|13704.7000|64975.7312|+374.11%|
|ledger_unattributed_ticks|0.0000|0.0000|N/A zero denominator|
|input_stage_ticks|136.4250|134.9000|-1.12%|
|metadata_stage_ticks|142.9062|141.3750|-1.07%|
|input_norm_ticks|101.8750|101.9437|+0.07%|
|qkv_projection_ticks|1643.1375|4423.0938|+169.19%|
|qk_norm_rope_ticks|0.1625|0.2000|+23.08%|
|attention_ticks|2374.2875|2375.4312|+0.05%|
|o_projection_ticks|824.1562|5545.7125|+572.90%|
|post_attention_residual_ticks|135.2000|130.7250|-3.31%|
|post_attention_norm_ticks|0.4938|0.4625|-6.33%|
|gate_up_ticks|4363.0500|33081.2875|+658.21%|
|activation_ticks|0.0000|579.5125|N/A zero denominator|
|down_ticks|2237.4563|16708.3500|+646.76%|
|final_residual_ticks|28.4688|28.3750|-0.33%|
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
|w4u8_decode_direct_n_projection_count|7.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_hmx_command_count|30.0000|0.0000|-100.00%|
|w4u8_mlp_down_hmx_command_count|8.0000|32.0000|+300.00%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.0000|0.0000|-100.00%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.0000|0.0000|-100.00%|
|weight_dma_ticks|8667.2687|11289.3250|+30.25%|
|hmx_compute_ticks|4295.8188|9611.8687|+123.75%|
|projection_pack_ticks|3.2125|3.2250|+0.39%|
|projection_hmx_wait_ticks|364.5875|464.1562|+27.31%|
|projection_unpack_ticks|0.0000|0.0000|N/A zero denominator|
|hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_ticks|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_work_ticks|0.0000|9577.2500|N/A zero denominator|
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
|attention_unattributed_ticks|1616.1188|1615.7750|-0.02%|
|u8_attention_qk_norm_rope_ticks|1411.5438|988.5000|-29.97%|
|u8_attention_k_pack_ticks|13.1750|13.2000|+0.19%|
|u8_attention_v_pack_ticks|85.5938|84.9125|-0.80%|
|u8_cache_native_append_update_ticks|253.0813|254.3562|+0.50%|
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
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.1000|26.6812|+2.23%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.0000|7.0000|+0.00%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.0000|896.0000|+0.00%|
|f16_cache_native_prefill_reuse_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_full_prefix_pack_count|0.0000|0.0000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|92.6000|93.5437|+1.02%|
|u8_attention_qk_requant_ticks|0.0000|0.0000|N/A zero denominator|
|u8_attention_softmax_ticks|266.8500|268.7500|+0.71%|
|u8_attention_av_hmx_ticks|150.7688|150.5500|-0.15%|
|u8_attention_av_requant_ticks|102.8688|101.6125|-1.22%|
|u8_attention_pipeline_wait_ticks|45.7437|45.7062|-0.08%|
|w4u8_qkvo_weight_expand_ticks|0.0000|14146.3125|N/A zero denominator|
|w4u8_qkvo_prefetch_wait_ticks|1405.7750|1986.2188|+41.29%|
|w4u8_qkvo_hmx_lifetime_ticks|8156.4000|57087.8375|+599.91%|
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
|w4u8_mlp_weight_expand_ticks|0.0000|41086.5188|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.0000|0.0000|N/A zero denominator|
|vtcm_requested_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_acquired_bytes|8388608.0000|8388608.0000|+0.00%|
|vtcm_peak_plan_bytes|6682752.0000|6682752.0000|+0.00%|
|block_invocation_count|1.0000|1.0000|+0.00%|
|hmx_command_count|46.0000|136.0000|+195.65%|
|hmx_fp16_tile_pair_count|0.0000|0.0000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.0000|49536.0000|+0.00%|
|weight_dma_descriptor_count|60.0000|240.0000|+300.00%|
|boundary_dma_descriptor_count|10.0000|10.0000|+0.00%|
|intermediate_dma_descriptor_count|0.0000|0.0000|N/A zero denominator|
|intermediate_spill_fill_count|0.0000|0.0000|N/A zero denominator|
|weight_ddr_read_bytes|25329664.0000|26116096.0000|+3.10%|
|boundary_ddr_read_bytes|304096.0000|304096.0000|+0.00%|
|boundary_ddr_write_bytes|131072.0000|131072.0000|+0.00%|
|intermediate_ddr_read_bytes|0.0000|0.0000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_audit_ddr_write_bytes|0.0000|0.0000|N/A zero denominator|
|u8_attention_probability_mask_violation_count|0.0000|0.0000|N/A zero denominator|
|u8_attention_fused_k_operand_mismatch_count|0.0000|0.0000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.0000|0.0000|N/A zero denominator|

E2E prefill/decode token/s: N/A. This experiment has no full-model token boundary; full-model work was stopped by the10% single-layer gate. No extrapolation, no PPL/quality claim.
