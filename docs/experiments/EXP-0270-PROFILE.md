# EXP0270: exact Qwen SP2 FP32 residual optimization

Tested source `645b0cdc4d6787dcde7bade9af59c3de3d6d8451`, branch codex/exp-0270-fp32-norm-pipeline. Evidence `/mnt/d/llm_exp/results/qwen3-block-htp/exp0270`; frozen models `/mnt/d/llm_exp/models/qwen3-block-htp/exp0269`. Single-layer M64 then M1/past64/cache128, not complete-model token generation. Five fixed alternating pairs; repeat10 primary and repeat1 auxiliary, paired bootstrap20000 seed270.

Independent implementation gate passes. Prefill short speed gate fails; decode passes. Formal10, chain3/full28/E2E/PPL are N/A because short prefill did not establish eligibility. Original user-selected paper speed baselines remain unchanged.

| Repeat / phase | Control Host us | FP32 Host us | Increase | 95% ratio CI | Gate |
|---|---:|---:|---:|---|---|
| r1_prefill_ns | 1555.771 | 1742.854 | +12.03% | [1.0501478268121174, 1.2229276286029065] | fail (auxiliary) |
| r1_decode_ns | 1032.490 | 1114.521 | +7.94% | [0.8826189590000771, 1.2665320172221786] | fail (auxiliary) |
| r10_prefill_ns | 1357.264 | 1552.975 | +14.42% | [1.1014435544693815, 1.1866591710085206] | fail |
| r10_decode_ns | 1017.179 | 1059.212 | +4.13% | [1.0156070060399227, 1.0822411260104283] | pass |

## Changes and evidence

C1 uses an eight-row register transpose with four ordered columns per vector. It keeps ascending-channel arithmetic, stores sums in VTCM, and has no stack in generated reduction assembly. C2 tried four-row native output, rejected two compiler-spilling attempts before device; a corrected two-row version was exact but slower (diagnostic input Norm101.26us vs C1 74.99us) and was removed. C3 prebuilds three invariant HMX retained-output bias tables (768B within the existing2KiB reserve), replacing repeated scalar rewrites. Control integer SP2 keeps its original path.

Nonpaired selection diagnostics: C1->C1+C3 Down prefill162.81->142.12us, O decode51.82->46.39us. O prefill74.37->75.42us, so no O prefill gain established. These are module diagnoses, not formal or E2E speed improvements. Do not infer whole-model throughput or statistically significant gains versus EXP0269 from separate runs.

Final layers0/14/27 exact FP32 outputs (399360 values), actual Q/K, AV, postnorm, Gate/Up, SP2 live planes and prefill physical KV all match sealed independent references; repeat10 deterministic. Signed24 raw and signed32 merged bounds remain valid. All33 CLI/372 profiles satisfy8MiB, peak6682752B, zero tensorDDR/spill/unattributed and exact additive ledgers. No new numerical failure. All546 EXP0269 sealed evidence files and three reused model manifests reverified.

Assembly: reduce8, O/Down epilogue, HMX worker and streamed raw-output code have no activation stack stores. Generic Norm stack slots contain uniform constants and inverse-RMS coefficient splats, not tensor tiles or partial-sum arrays; all activation tiles/sums stay registers or VTCM. Failed C2 assemblies preserved separately. One C3 compiler API const-argument error repaired before device, recorded in c3-build-attempt.json.

## Stable repeat10 module overview

Current control/candidate means; microseconds (percent of complete Host wall). F16F16 and W4F16 have no equivalent-scope run in this experiment and are N/A (unchanged); no cross-recipe speed ratio is claimed. Historical complete-model measurements cannot fill these single-layer cells.

### prefill

| Module | F16F16 | W4F16 | W4U8-SP2 control | W4U8-SP2 FP32 | W4F16/FP32 speed delta |
|---|---|---|---:|---:|---|
| I/O and metadata | N/A | N/A | 18.63 (1.37%) | 45.97 (2.96%) | N/A |
| Input RMSNorm | N/A | N/A | 20.25 (1.49%) | 75.24 (4.84%) | N/A |
| QKV + Q/K Norm-RoPE | N/A | N/A | 250.63 (18.47%) | 251.48 (16.19%) | N/A |
| QK-Softmax-AV | N/A | N/A | 120.51 (8.88%) | 121.05 (7.79%) | N/A |
| O projection | N/A | N/A | 45.19 (3.33%) | 74.58 (4.80%) | N/A |
| Post-attention residual + RMSNorm | N/A | N/A | 23.55 (1.73%) | 79.05 (5.09%) | N/A |
| Gate/Up + SwiGLU | N/A | N/A | 249.78 (18.40%) | 248.62 (16.01%) | N/A |
| Down projection | N/A | N/A | 159.50 (11.75%) | 140.42 (9.04%) | N/A |
| Final residual | N/A | N/A | 6.64 (0.49%) | 0.12 (0.01%) | N/A |
| KV-cache carrier conversion | N/A | N/A | 4.84 (0.36%) | 4.94 (0.32%) | N/A |
| KV-cache append DMA | N/A | N/A | 8.85 (0.65%) | 9.02 (0.58%) | N/A |
| Block internal orchestration | N/A | N/A | 1.65 (0.12%) | 1.63 (0.10%) | N/A |
| Layer bookkeeping | N/A | N/A | 0.86 (0.06%) | 0.85 (0.05%) | N/A |
| Stage-boundary bookkeeping | N/A | N/A | 1.72 (0.13%) | 1.73 (0.11%) | N/A |
| DSP unattributed residual | N/A | N/A | 0.00 (0.00%) | 0.00 (0.00%) | N/A |
| DSP runtime setup/teardown | N/A | N/A | 74.25 (5.47%) | 74.38 (4.79%) | N/A |
| Token embedding | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| Final model RMSNorm | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| LM head + greedy selection (excluding final norm) | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| True Host-DSP boundary | N/A | N/A | 370.41 (27.29%) | 423.88 (27.29%) | N/A |
| Complete Host wall | N/A | N/A | 1357.26 (100.00%) | 1552.97 (100.00%) | N/A |

### decode

| Module | F16F16 | W4F16 | W4U8-SP2 control | W4U8-SP2 FP32 | W4F16/FP32 speed delta |
|---|---|---|---:|---:|---|
| I/O and metadata | N/A | N/A | 21.22 (2.09%) | 57.84 (5.46%) | N/A |
| Input RMSNorm | N/A | N/A | 5.38 (0.53%) | 14.69 (1.39%) | N/A |
| QKV + Q/K Norm-RoPE | N/A | N/A | 87.91 (8.64%) | 87.04 (8.22%) | N/A |
| QK-Softmax-AV | N/A | N/A | 65.80 (6.47%) | 66.04 (6.23%) | N/A |
| O projection | N/A | N/A | 43.85 (4.31%) | 47.25 (4.46%) | N/A |
| Post-attention residual + RMSNorm | N/A | N/A | 7.41 (0.73%) | 15.03 (1.42%) | N/A |
| Gate/Up + SwiGLU | N/A | N/A | 228.16 (22.43%) | 220.08 (20.78%) | N/A |
| Down projection | N/A | N/A | 123.94 (12.18%) | 122.11 (11.53%) | N/A |
| Final residual | N/A | N/A | 1.59 (0.16%) | 0.07 (0.01%) | N/A |
| KV-cache carrier conversion | N/A | N/A | 10.64 (1.05%) | 10.66 (1.01%) | N/A |
| KV-cache append DMA | N/A | N/A | 3.31 (0.33%) | 3.30 (0.31%) | N/A |
| Block internal orchestration | N/A | N/A | 1.10 (0.11%) | 1.13 (0.11%) | N/A |
| Layer bookkeeping | N/A | N/A | 0.64 (0.06%) | 0.65 (0.06%) | N/A |
| Stage-boundary bookkeeping | N/A | N/A | 0.48 (0.05%) | 0.48 (0.05%) | N/A |
| DSP unattributed residual | N/A | N/A | 0.00 (0.00%) | 0.00 (0.00%) | N/A |
| DSP runtime setup/teardown | N/A | N/A | 70.70 (6.95%) | 70.80 (6.68%) | N/A |
| Token embedding | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| Final model RMSNorm | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| LM head + greedy selection (excluding final norm) | N/A | N/A | N/A: outside replay | N/A: outside replay | N/A |
| True Host-DSP boundary | N/A | N/A | 345.03 (33.92%) | 342.06 (32.29%) | N/A |
| Complete Host wall | N/A | N/A | 1017.18 (100.00%) | 1059.21 (100.00%) | N/A |

## Complete raw-counter comparisons

Each cell uses the median of five per-run means. Tick fields are converted to microseconds; other fields retain their stated units. Engine work/readiness/wait counters overlap and must not be added to the exclusive ledger. PROFILE.json also retains all means. Missing nonnumeric metadata remain in protocol.json/raw JSON. No absent hardware measurement is replaced by zero.

### repeat1 prefill

| Field | Control median | FP32 median | Change |
|---|---:|---:|---:|
| host_us | 1592.34 | 1735.37 | +8.982% |
| host_boundary_us | 446.198 | 443.073 | -0.700% |
| generation_embedding_ticks (us) | 0 | 0 | N/A: zero denominator |
| input_stage_ticks (us) | 8.125 | 28.4896 | +250.641% |
| metadata_stage_ticks (us) | 11.8229 | 11.4062 | -3.524% |
| input_norm_ticks (us) | 23.5417 | 78.9062 | +235.177% |
| qkv_projection_ticks (us) | 274.323 | 276.615 | +0.835% |
| qk_norm_rope_ticks (us) | 0.15625 | 0.15625 | +0.000% |
| attention_ticks (us) | 125.052 | 122.76 | -1.833% |
| o_projection_ticks (us) | 58.8021 | 83.3333 | +41.718% |
| post_attention_residual_ticks (us) | 26.6667 | 79.6875 | +198.828% |
| post_attention_norm_ticks (us) | 0.3125 | 0.3125 | +0.000% |
| gate_up_ticks (us) | 295.417 | 289.688 | -1.939% |
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| down_ticks (us) | 166.094 | 151.562 | -8.749% |
| final_residual_ticks (us) | 7.5 | 0.677083 | -90.972% |
| output_stage_ticks (us) | 4.32292 | 14.0625 | +225.301% |
| scan_cache_pack_ticks (us) | 5.78125 | 5.9375 | +2.703% |
| scan_cache_append_ticks (us) | 12.9688 | 11.875 | -8.434% |
| block_orchestration_ticks (us) | 6.14583 | 6.19792 | +0.847% |
| layer_bookkeeping_ticks (us) | 2.96875 | 2.86458 | -3.509% |
| stage_boundary_ticks (us) | 5.72917 | 5.26042 | -8.182% |
| generation_final_norm_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 0 | 0 | N/A: zero denominator |
| runtime_setup_ticks (us) | 61.8229 | 58.8542 | -4.802% |
| runtime_teardown_ticks (us) | 48.9062 | 47.2917 | -3.301% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| block_invocation_count | 1 | 1 | +0.000% |
| boundary_ddr_read_bytes | 304096 | 697312 | +129.307% |
| boundary_ddr_write_bytes | 131072 | 524288 | +300.000% |
| boundary_dma_descriptor_count | 10 | 10 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 0 | 0 | N/A: zero denominator |
| cache_mismatches | 0 | 0 | N/A: zero denominator |
| cache_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| cache_nonfinite_count | 0 | 0 | N/A: zero denominator |
| cache_prefix_mismatches | 0 | 0 | N/A: zero denominator |
| cache_structure_mismatches | 0 | 0 | N/A: zero denominator |
| cache_tensor_count | 0 | 0 | N/A: zero denominator |
| dense_r3_constant_read_bytes | 0 | 0 | N/A: zero denominator |
| dense_r3_mode | 0 | 0 | N/A: zero denominator |
| dense_r3_optimization | 0 | 0 | N/A: zero denominator |
| dense_r3_total_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_heads | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_refined_values | 0 | 0 | N/A: zero denominator |
| dense_r3_total_rows | 0 | 0 | N/A: zero denominator |
| dense_r4_audit_bytes | 0 | 0 | N/A: zero denominator |
| dense_r4_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_layout_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_mode | 0 | 0 | N/A: zero denominator |
| dense_r4_optimization | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_dispatches | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_batches | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_consume_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_publish_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_rows | 0 | 0 | N/A: zero denominator |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 240 | 240 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| first_position | 0 | 0 | N/A: zero denominator |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| generation_embedding_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_argmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_batch_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_command_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_prefetch_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 46 | 46 | +0.000% |
| hmx_compute_ticks (us) | 340.208 | 361.875 | +6.369% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 61696 | 61696 | +0.000% |
| host_wall_ns | 1.59234e+06 | 1.73536e+06 | +8.982% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 1143.65 | 1281.35 | +12.041% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| ledger_named_ticks (us) | 1143.65 | 1281.35 | +12.041% |
| logical_m | 64 | 64 | +0.000% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 0 | 0 | N/A: zero denominator |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 0 | 0 | N/A: zero denominator |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 1 | 1 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 50.1562 | 52.2396 | +4.154% |
| projection_pack_ticks (us) | 1.09375 | 1.04167 | -4.762% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| repeat_count | 1 | 1 | +0.000% |
| replay_step | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_write_bytes | 210944 | 210944 | +0.000% |
| scan_cache_dma_descriptor_count | 16 | 16 | +0.000% |
| scan_cache_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_dynamic_attention_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_logical_m_observed | 64 | 64 | +0.000% |
| scan_padded_kv_length | 64 | 64 | +0.000% |
| scan_total_kv_length | 64 | 64 | +0.000% |
| total_ticks (us) | 1081.82 | 1221.56 | +12.917% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 20.4167 | 21.6667 | +6.122% |
| u8_attention_av_requant_ticks (us) | 61.3542 | 60.6771 | -1.104% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_pipeline_wait_ticks (us) | 73.9062 | 71.5104 | -3.242% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 21.3021 | 21.6667 | +1.711% |
| u8_attention_qk_norm_rope_ticks (us) | 917.708 | 918.698 | +0.108% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 315 | 314.167 | -0.265% |
| u8_attention_v_pack_ticks (us) | 162.448 | 159.375 | -1.892% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_bytes | 29568 | 29568 | +0.000% |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 17.9688 | 17.0312 | -5.217% |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 1 | 1 | +0.000% |
| u8_cache_native_prefill_reused_carrier_bytes | 143360 | 143360 | +0.000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 32768 | 32768 | +0.000% |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 64 | 64 | +0.000% |
| vtcm_acquired_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| vtcm_peak_plan_bytes | 6.68275e+06 | 6.68275e+06 | +0.000% |
| vtcm_requested_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_decode_audit | 0 | 0 | N/A: zero denominator |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt_calls | 0 | 0 | N/A: zero denominator |
| w4f16_expand_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_av_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_vector_count | 0 | 0 | N/A: zero denominator |
| w4u8_common_op_rows_observed | 64 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 5.03316e+07 | 5.03316e+07 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 30 | 30 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 7 | 7 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.51658e+07 | 2.51658e+07 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A: zero denominator |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_main_work_ticks (us) | 4.11458 | 0 | -100.000% |
| w4u8_final_residual_pool_wait_ticks (us) | 0.78125 | 0 | -100.000% |
| w4u8_final_residual_task_count | 16 | 0 | -100.000% |
| w4u8_final_residual_worker_work_ticks (us) | 20.1042 | 0 | -100.000% |
| w4u8_gate_up_swiglu_consume_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 12.6562 | 13.2292 | +4.527% |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_publish_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_main_work_ticks (us) | 15.1562 | 0 | -100.000% |
| w4u8_input_norm_pool_wait_ticks (us) | 6.25 | 0 | -100.000% |
| w4u8_input_norm_task_count | 16 | 0 | -100.000% |
| w4u8_input_norm_worker_work_ticks (us) | 90.4167 | 0 | -100.000% |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 8 | 8 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 4 | 4 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_main_work_ticks (us) | 16.3542 | 0 | -100.000% |
| w4u8_post_residual_pool_wait_ticks (us) | 7.70833 | 0 | -100.000% |
| w4u8_post_residual_task_count | 16 | 0 | -100.000% |
| w4u8_post_residual_worker_work_ticks (us) | 109.844 | 0 | -100.000% |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 6 | 6 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 97.6042 | 97.7604 | +0.160% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 24 | 24 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 21.5104 | 20.625 | -4.116% |
| w4u8_qkv_ring_hmx_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 72.2917 | 73.125 | +1.153% |
| w4u8_qkv_ring_pipeline_ticks (us) | 271.615 | 273.49 | +0.690% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 164.635 | 159.948 | -2.847% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 0.260417 | 0.208333 | -20.000% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 552.5 | 559.219 | +1.216% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 97.8125 | 97.9688 | +0.160% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 6 | 0 | -100.000% |
| w4u8_swiglu_rows_observed | 64 | 64 | +0.000% |
| weight_ddr_read_bytes | 2.53297e+07 | 2.53297e+07 | +0.000% |
| weight_dma_descriptor_count | 60 | 60 | +0.000% |
| weight_dma_ticks (us) | 561.406 | 583.49 | +3.934% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat1 decode

| Field | Control median | FP32 median | Change |
|---|---:|---:|---:|
| host_us | 966.302 | 1087.24 | +12.515% |
| host_boundary_us | 280.99 | 351.406 | +25.060% |
| generation_embedding_ticks (us) | 0 | 0 | N/A: zero denominator |
| input_stage_ticks (us) | 7.23958 | 27.3958 | +278.417% |
| metadata_stage_ticks (us) | 7.65625 | 7.55208 | -1.361% |
| input_norm_ticks (us) | 5.36458 | 15 | +179.612% |
| qkv_projection_ticks (us) | 87.1875 | 87.7083 | +0.597% |
| qk_norm_rope_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_ticks (us) | 73.4896 | 72.6042 | -1.205% |
| o_projection_ticks (us) | 43.3333 | 47.3438 | +9.255% |
| post_attention_residual_ticks (us) | 8.28125 | 15.4167 | +86.164% |
| post_attention_norm_ticks (us) | 0 | 0 | N/A: zero denominator |
| gate_up_ticks (us) | 227.344 | 226.302 | -0.458% |
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| down_ticks (us) | 125.417 | 122.708 | -2.159% |
| final_residual_ticks (us) | 2.03125 | 0.104167 | -94.872% |
| output_stage_ticks (us) | 6.04167 | 21.6667 | +258.621% |
| scan_cache_pack_ticks (us) | 12.9167 | 13.2292 | +2.419% |
| scan_cache_append_ticks (us) | 3.48958 | 3.54167 | +1.493% |
| block_orchestration_ticks (us) | 1.5625 | 1.875 | +20.000% |
| layer_bookkeeping_ticks (us) | 1.04167 | 1.04167 | +0.000% |
| stage_boundary_ticks (us) | 0.78125 | 0.677083 | -13.333% |
| generation_final_norm_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 0 | 0 | N/A: zero denominator |
| runtime_setup_ticks (us) | 38.9583 | 38.6979 | -0.668% |
| runtime_teardown_ticks (us) | 33.3333 | 33.4896 | +0.469% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_unattributed_ticks (us) | 35.3125 | 33.75 | -4.425% |
| block_invocation_count | 1 | 1 | +0.000% |
| boundary_ddr_read_bytes | 304096 | 697312 | +129.307% |
| boundary_ddr_write_bytes | 131072 | 524288 | +300.000% |
| boundary_dma_descriptor_count | 10 | 10 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 0 | 0 | N/A: zero denominator |
| cache_mismatches | 0 | 0 | N/A: zero denominator |
| cache_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| cache_nonfinite_count | 0 | 0 | N/A: zero denominator |
| cache_prefix_mismatches | 0 | 0 | N/A: zero denominator |
| cache_structure_mismatches | 0 | 0 | N/A: zero denominator |
| cache_tensor_count | 0 | 0 | N/A: zero denominator |
| dense_r3_constant_read_bytes | 0 | 0 | N/A: zero denominator |
| dense_r3_mode | 0 | 0 | N/A: zero denominator |
| dense_r3_optimization | 0 | 0 | N/A: zero denominator |
| dense_r3_total_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_heads | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_refined_values | 0 | 0 | N/A: zero denominator |
| dense_r3_total_rows | 0 | 0 | N/A: zero denominator |
| dense_r4_audit_bytes | 0 | 0 | N/A: zero denominator |
| dense_r4_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_layout_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_mode | 0 | 0 | N/A: zero denominator |
| dense_r4_optimization | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_dispatches | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_batches | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_consume_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_publish_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_rows | 0 | 0 | N/A: zero denominator |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 240 | 240 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| first_position | 64 | 64 | +0.000% |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| generation_embedding_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_argmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_batch_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_command_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_prefetch_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 46 | 46 | +0.000% |
| hmx_compute_ticks (us) | 267.5 | 266.406 | -0.409% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 49536 | 49536 | +0.000% |
| host_wall_ns | 966302 | 1.08724e+06 | +12.515% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 694.427 | 735.833 | +5.963% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| ledger_named_ticks (us) | 694.427 | 735.833 | +5.963% |
| logical_m | 1 | 1 | +0.000% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 0 | 0 | N/A: zero denominator |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 0 | 0 | N/A: zero denominator |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 2 | 2 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 27.6042 | 26.875 | -2.642% |
| projection_pack_ticks (us) | 0.15625 | 0.15625 | +0.000% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| repeat_count | 1 | 1 | +0.000% |
| replay_step | 1 | 1 | +0.000% |
| scan_attention_overlay_capacity_bytes | 2.75251e+06 | 2.3593e+06 | -14.286% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | +0.000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_read_bytes | 143488 | 143488 | +0.000% |
| scan_cache_ddr_write_bytes | 1152 | 1152 | +0.000% |
| scan_cache_dma_descriptor_count | 42 | 42 | +0.000% |
| scan_cache_stage_ticks (us) | 16.5104 | 16.3021 | -1.262% |
| scan_dynamic_attention_ticks (us) | 72.8125 | 71.875 | -1.288% |
| scan_logical_m_observed | 1 | 1 | +0.000% |
| scan_padded_kv_length | 96 | 96 | +0.000% |
| scan_total_kv_length | 65 | 65 | +0.000% |
| total_ticks (us) | 655.677 | 697.135 | +6.323% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 5.78125 | 5.52083 | -4.505% |
| u8_attention_av_requant_ticks (us) | 5.20833 | 5.10417 | -2.000% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0.364583 | 0.364583 | +0.000% |
| u8_attention_pipeline_wait_ticks (us) | 2.23958 | 2.29167 | +2.326% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 4.94792 | 5 | +1.053% |
| u8_attention_qk_norm_rope_ticks (us) | 75.3125 | 73.3854 | -2.559% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 14.5312 | 14.9479 | +2.867% |
| u8_attention_v_pack_ticks (us) | 5.52083 | 5.57292 | +0.943% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 28 | 28 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 896 | 896 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 1.71875 | 1.71875 | +0.000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_native_load_bytes | 28672 | 28672 | +0.000% |
| u8_cache_k_vtcm_tail_row_update_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 15.5208 | 15.5729 | +0.336% |
| u8_cache_native_incremental_append_count | 1 | 1 | +0.000% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 1 | 1 | +0.000% |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_native_load_bytes | 4096 | 4096 | +0.000% |
| u8_cache_v_vtcm_tail_partial_pack_rows | 8 | 8 | +0.000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 8 | 8 | +0.000% |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 65 | 65 | +0.000% |
| vtcm_acquired_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| vtcm_peak_plan_bytes | 6.68275e+06 | 6.68275e+06 | +0.000% |
| vtcm_requested_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_decode_audit | 0 | 0 | N/A: zero denominator |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt_calls | 0 | 0 | N/A: zero denominator |
| w4f16_expand_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_av_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_call_count | 8 | 8 | +0.000% |
| w4u8_av_requant_rows_observed | 4 | 4 | +0.000% |
| w4u8_av_requant_vector_count | 64 | 64 | +0.000% |
| w4u8_common_op_rows_observed | 4 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 5.03316e+07 | 5.03316e+07 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 30 | 30 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 7 | 7 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.51658e+07 | 2.51658e+07 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 4 | 4 | +0.000% |
| w4u8_decode_k_temp_carrier_skipped_count | 8 | 8 | +0.000% |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 8 | 8 | +0.000% |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 96 | 96 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_call_count | 8 | 8 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 192 | 192 | +0.000% |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 192 | 192 | +0.000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_consume_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 4.27083 | 5.83333 | +36.585% |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | +0.000% |
| w4u8_gate_up_swiglu_publish_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 178.073 | 179.323 | +0.702% |
| w4u8_gate_up_swiglu_worker_ticks (us) | 24.4792 | 33.5417 | +37.021% |
| w4u8_input_norm_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 8 | 8 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 4 | 4 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 1 | 1 | +0.000% |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 18.4896 | 18.3854 | -0.563% |
| w4u8_o_gate_prefetch_start_count | 1 | 1 | +0.000% |
| w4u8_o_gate_prefetch_wait_ticks (us) | 9.84375 | 2.60417 | -73.545% |
| w4u8_post_residual_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | +0.000% |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 6 | 6 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 74.4271 | 74.7917 | +0.490% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 24 | 24 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 20.1562 | 23.0729 | +14.470% |
| w4u8_qkv_ring_hmx_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 45.0521 | 47.8125 | +6.127% |
| w4u8_qkv_ring_pipeline_ticks (us) | 85.7812 | 86.3021 | +0.607% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 0.15625 | 0.15625 | +0.000% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 0.208333 | 0.208333 | +0.000% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 436.771 | 442.448 | +1.300% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 74.4792 | 74.8438 | +0.490% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 0 | 0 | N/A: zero denominator |
| w4u8_swiglu_rows_observed | 4 | 4 | +0.000% |
| weight_ddr_read_bytes | 2.53297e+07 | 2.53297e+07 | +0.000% |
| weight_dma_descriptor_count | 60 | 60 | +0.000% |
| weight_dma_ticks (us) | 455.99 | 480.625 | +5.403% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat10 prefill

| Field | Control median | FP32 median | Change |
|---|---:|---:|---:|
| host_us | 1392.15 | 1551.91 | +11.476% |
| host_boundary_us | 402.662 | 421.922 | +4.783% |
| generation_embedding_ticks (us) | 0 | 0 | N/A: zero denominator |
| input_stage_ticks (us) | 7.3125 | 27.349 | +274.003% |
| metadata_stage_ticks (us) | 8.16667 | 8.22396 | +0.702% |
| input_norm_ticks (us) | 20.0781 | 75.2292 | +274.682% |
| qkv_projection_ticks (us) | 250.625 | 251.708 | +0.432% |
| qk_norm_rope_ticks (us) | 0.0260417 | 0.0208333 | -20.000% |
| attention_ticks (us) | 120.375 | 120.781 | +0.337% |
| o_projection_ticks (us) | 45.0365 | 74.1615 | +64.670% |
| post_attention_residual_ticks (us) | 23.4531 | 79.1094 | +237.308% |
| post_attention_norm_ticks (us) | 0.0520833 | 0.0625 | +20.000% |
| gate_up_ticks (us) | 251.651 | 248.609 | -1.209% |
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| down_ticks (us) | 158.979 | 139.521 | -12.240% |
| final_residual_ticks (us) | 6.61979 | 0.119792 | -98.190% |
| output_stage_ticks (us) | 3.13021 | 10.2188 | +226.456% |
| scan_cache_pack_ticks (us) | 4.84896 | 4.82812 | -0.430% |
| scan_cache_append_ticks (us) | 8.80208 | 9.03646 | +2.663% |
| block_orchestration_ticks (us) | 1.65104 | 1.64583 | -0.315% |
| layer_bookkeeping_ticks (us) | 0.859375 | 0.859375 | +0.000% |
| stage_boundary_ticks (us) | 1.70313 | 1.70833 | +0.306% |
| generation_final_norm_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 0 | 0 | N/A: zero denominator |
| runtime_setup_ticks (us) | 39.5052 | 39.6719 | +0.422% |
| runtime_teardown_ticks (us) | 34.625 | 34.7865 | +0.466% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| block_invocation_count | 1 | 1 | +0.000% |
| boundary_ddr_read_bytes | 304096 | 697312 | +129.307% |
| boundary_ddr_write_bytes | 131072 | 524288 | +300.000% |
| boundary_dma_descriptor_count | 10 | 10 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 0 | 0 | N/A: zero denominator |
| cache_mismatches | 0 | 0 | N/A: zero denominator |
| cache_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| cache_nonfinite_count | 0 | 0 | N/A: zero denominator |
| cache_prefix_mismatches | 0 | 0 | N/A: zero denominator |
| cache_structure_mismatches | 0 | 0 | N/A: zero denominator |
| cache_tensor_count | 0 | 0 | N/A: zero denominator |
| dense_r3_constant_read_bytes | 0 | 0 | N/A: zero denominator |
| dense_r3_mode | 0 | 0 | N/A: zero denominator |
| dense_r3_optimization | 0 | 0 | N/A: zero denominator |
| dense_r3_total_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_heads | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_refined_values | 0 | 0 | N/A: zero denominator |
| dense_r3_total_rows | 0 | 0 | N/A: zero denominator |
| dense_r4_audit_bytes | 0 | 0 | N/A: zero denominator |
| dense_r4_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_layout_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_mode | 0 | 0 | N/A: zero denominator |
| dense_r4_optimization | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_dispatches | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_batches | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_consume_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_publish_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_rows | 0 | 0 | N/A: zero denominator |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 240 | 240 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| first_position | 0 | 0 | N/A: zero denominator |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| generation_embedding_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_argmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_batch_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_command_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_prefetch_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 46 | 46 | +0.000% |
| hmx_compute_ticks (us) | 325.464 | 348.214 | +6.990% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 61696 | 61696 | +0.000% |
| host_wall_ns | 1.39215e+06 | 1.55191e+06 | +11.476% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 989.422 | 1129.56 | +14.164% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| ledger_named_ticks (us) | 989.422 | 1129.56 | +14.164% |
| logical_m | 64 | 64 | +0.000% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 0 | 0 | N/A: zero denominator |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 0 | 0 | N/A: zero denominator |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 10 | 10 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 57.3073 | 65.5 | +14.296% |
| projection_pack_ticks (us) | 0.239583 | 0.213542 | -10.870% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| repeat_count | 1 | 1 | +0.000% |
| replay_step | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_write_bytes | 210944 | 210944 | +0.000% |
| scan_cache_dma_descriptor_count | 16 | 16 | +0.000% |
| scan_cache_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_dynamic_attention_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_logical_m_observed | 64 | 64 | +0.000% |
| scan_padded_kv_length | 64 | 64 | +0.000% |
| scan_total_kv_length | 64 | 64 | +0.000% |
| total_ticks (us) | 949.984 | 1089.78 | +14.715% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 21.6458 | 22.0365 | +1.805% |
| u8_attention_av_requant_ticks (us) | 61.125 | 61.1094 | -0.026% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_pipeline_wait_ticks (us) | 69.8385 | 71.8073 | +2.819% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 21.0938 | 21.7031 | +2.889% |
| u8_attention_qk_norm_rope_ticks (us) | 875.698 | 879.99 | +0.490% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 311.802 | 312.542 | +0.237% |
| u8_attention_v_pack_ticks (us) | 153.411 | 153.375 | -0.024% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_bytes | 29568 | 29568 | +0.000% |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 13.5417 | 13.7604 | +1.615% |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 1 | 1 | +0.000% |
| u8_cache_native_prefill_reused_carrier_bytes | 143360 | 143360 | +0.000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 32768 | 32768 | +0.000% |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 64 | 64 | +0.000% |
| vtcm_acquired_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| vtcm_peak_plan_bytes | 6.68275e+06 | 6.68275e+06 | +0.000% |
| vtcm_requested_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_decode_audit | 0 | 0 | N/A: zero denominator |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt_calls | 0 | 0 | N/A: zero denominator |
| w4f16_expand_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_av_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_vector_count | 0 | 0 | N/A: zero denominator |
| w4u8_common_op_rows_observed | 64 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 5.03316e+07 | 5.03316e+07 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 30 | 30 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 7 | 7 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.51658e+07 | 2.51658e+07 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A: zero denominator |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_main_work_ticks (us) | 3.51562 | 0 | -100.000% |
| w4u8_final_residual_pool_wait_ticks (us) | 1.19271 | 0 | -100.000% |
| w4u8_final_residual_task_count | 16 | 0 | -100.000% |
| w4u8_final_residual_worker_work_ticks (us) | 19.8698 | 0 | -100.000% |
| w4u8_gate_up_swiglu_consume_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 13.0365 | 13.25 | +1.638% |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_publish_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_main_work_ticks (us) | 16.3906 | 0 | -100.000% |
| w4u8_input_norm_pool_wait_ticks (us) | 1.65104 | 0 | -100.000% |
| w4u8_input_norm_task_count | 16 | 0 | -100.000% |
| w4u8_input_norm_worker_work_ticks (us) | 77.0573 | 0 | -100.000% |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 8 | 8 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 4 | 4 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_main_work_ticks (us) | 18.5677 | 0 | -100.000% |
| w4u8_post_residual_pool_wait_ticks (us) | 3.41667 | 0 | -100.000% |
| w4u8_post_residual_task_count | 16 | 0 | -100.000% |
| w4u8_post_residual_worker_work_ticks (us) | 93.8958 | 0 | -100.000% |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 6 | 6 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 76.6823 | 75.25 | -1.868% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 24 | 24 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 19.5312 | 19.6094 | +0.400% |
| w4u8_qkv_ring_hmx_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 52.8281 | 50.8698 | -3.707% |
| w4u8_qkv_ring_pipeline_ticks (us) | 249.281 | 250.417 | +0.455% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 160.505 | 162.234 | +1.077% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 0.229167 | 0.510417 | +122.727% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 481.359 | 487.021 | +1.176% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 76.7448 | 75.2917 | -1.893% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 6 | 0 | -100.000% |
| w4u8_swiglu_rows_observed | 64 | 64 | +0.000% |
| weight_ddr_read_bytes | 2.53297e+07 | 2.53297e+07 | +0.000% |
| weight_dma_descriptor_count | 60 | 60 | +0.000% |
| weight_dma_ticks (us) | 469.12 | 487.349 | +3.886% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat10 decode

| Field | Control median | FP32 median | Change |
|---|---:|---:|---:|
| host_us | 1012.22 | 1061.81 | +4.899% |
| host_boundary_us | 339.729 | 349.078 | +2.752% |
| generation_embedding_ticks (us) | 0 | 0 | N/A: zero denominator |
| input_stage_ticks (us) | 7.20313 | 27.2917 | +278.886% |
| metadata_stage_ticks (us) | 7.61458 | 7.73958 | +1.642% |
| input_norm_ticks (us) | 5.38021 | 14.6771 | +172.798% |
| qkv_projection_ticks (us) | 87.849 | 86.6458 | -1.370% |
| qk_norm_rope_ticks (us) | 0.00520833 | 0.00520833 | +0.000% |
| attention_ticks (us) | 65.9115 | 65.9115 | +0.000% |
| o_projection_ticks (us) | 43.9688 | 47.1198 | +7.167% |
| post_attention_residual_ticks (us) | 7.30729 | 14.9948 | +105.203% |
| post_attention_norm_ticks (us) | 0.03125 | 0.0208333 | -33.333% |
| gate_up_ticks (us) | 228.938 | 219.667 | -4.050% |
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| down_ticks (us) | 123.802 | 122.224 | -1.275% |
| final_residual_ticks (us) | 1.58854 | 0.0729167 | -95.410% |
| output_stage_ticks (us) | 6.36458 | 22.7188 | +256.956% |
| scan_cache_pack_ticks (us) | 10.6458 | 10.6562 | +0.098% |
| scan_cache_append_ticks (us) | 3.30729 | 3.30729 | +0.000% |
| block_orchestration_ticks (us) | 1.09375 | 1.13542 | +3.810% |
| layer_bookkeeping_ticks (us) | 0.640625 | 0.65625 | +2.439% |
| stage_boundary_ticks (us) | 0.479167 | 0.489583 | +2.174% |
| generation_final_norm_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 0 | 0 | N/A: zero denominator |
| runtime_setup_ticks (us) | 37.3646 | 37.625 | +0.697% |
| runtime_teardown_ticks (us) | 33.276 | 33.1875 | -0.266% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_unattributed_ticks (us) | 29.651 | 29.5312 | -0.404% |
| block_invocation_count | 1 | 1 | +0.000% |
| boundary_ddr_read_bytes | 304096 | 697312 | +129.307% |
| boundary_ddr_write_bytes | 131072 | 524288 | +300.000% |
| boundary_dma_descriptor_count | 10 | 10 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 0 | 0 | N/A: zero denominator |
| cache_mismatches | 0 | 0 | N/A: zero denominator |
| cache_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| cache_nonfinite_count | 0 | 0 | N/A: zero denominator |
| cache_prefix_mismatches | 0 | 0 | N/A: zero denominator |
| cache_structure_mismatches | 0 | 0 | N/A: zero denominator |
| cache_tensor_count | 0 | 0 | N/A: zero denominator |
| dense_r3_constant_read_bytes | 0 | 0 | N/A: zero denominator |
| dense_r3_mode | 0 | 0 | N/A: zero denominator |
| dense_r3_optimization | 0 | 0 | N/A: zero denominator |
| dense_r3_total_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_heads | 0 | 0 | N/A: zero denominator |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r3_total_refined_values | 0 | 0 | N/A: zero denominator |
| dense_r3_total_rows | 0 | 0 | N/A: zero denominator |
| dense_r4_audit_bytes | 0 | 0 | N/A: zero denominator |
| dense_r4_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_finish_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_hmx_calls | 0 | 0 | N/A: zero denominator |
| dense_r4_layout_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_matmul_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_mode | 0 | 0 | N/A: zero denominator |
| dense_r4_optimization | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_dispatches | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A: zero denominator |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_batches | 0 | 0 | N/A: zero denominator |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_consume_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_publish_count | 0 | 0 | N/A: zero denominator |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_prepare_ticks (us) | 0 | 0 | N/A: zero denominator |
| dense_r4_rows | 0 | 0 | N/A: zero denominator |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 240 | 240 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| first_position | 64 | 64 | +0.000% |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| generation_embedding_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_argmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_batch_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_command_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_n_tiles | 0 | 0 | N/A: zero denominator |
| generation_lm_head_prefetch_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 0 | 0 | N/A: zero denominator |
| generation_lm_head_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_weight_dma_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 46 | 46 | +0.000% |
| hmx_compute_ticks (us) | 263.828 | 269.401 | +2.112% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 49536 | 49536 | +0.000% |
| host_wall_ns | 1.01222e+06 | 1.06181e+06 | +4.899% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 672.495 | 717.339 | +6.668% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| ledger_named_ticks (us) | 672.495 | 717.339 | +6.668% |
| logical_m | 1 | 1 | +0.000% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 0 | 0 | N/A: zero denominator |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 0 | 0 | N/A: zero denominator |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 11 | 11 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 25.5938 | 26.3281 | +2.869% |
| projection_pack_ticks (us) | 0.171875 | 0.166667 | -3.030% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| repeat_count | 1 | 1 | +0.000% |
| replay_step | 1 | 1 | +0.000% |
| scan_attention_overlay_capacity_bytes | 2.75251e+06 | 2.3593e+06 | -14.286% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | +0.000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_read_bytes | 143488 | 143488 | +0.000% |
| scan_cache_ddr_write_bytes | 1152 | 1152 | +0.000% |
| scan_cache_dma_descriptor_count | 42 | 42 | +0.000% |
| scan_cache_stage_ticks (us) | 18 | 17.9062 | -0.521% |
| scan_dynamic_attention_ticks (us) | 65.6615 | 65.6198 | -0.063% |
| scan_logical_m_observed | 1 | 1 | +0.000% |
| scan_padded_kv_length | 96 | 96 | +0.000% |
| scan_total_kv_length | 65 | 65 | +0.000% |
| total_ticks (us) | 635.167 | 679.984 | +7.056% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 5.88021 | 6.125 | +4.163% |
| u8_attention_av_requant_ticks (us) | 5.0625 | 5.23958 | +3.498% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0.25 | 0.244792 | -2.083% |
| u8_attention_pipeline_wait_ticks (us) | 2.375 | 2.43229 | +2.412% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 4.78646 | 4.95833 | +3.591% |
| u8_attention_qk_norm_rope_ticks (us) | 75.4427 | 76.3333 | +1.181% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 12.5781 | 12.5677 | -0.083% |
| u8_attention_v_pack_ticks (us) | 5.01042 | 5.02083 | +0.208% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 28 | 28 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 896 | 896 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 1.42708 | 1.45312 | +1.825% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_native_load_bytes | 28672 | 28672 | +0.000% |
| u8_cache_k_vtcm_tail_row_update_count | 7 | 7 | +0.000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 13.849 | 13.8177 | -0.226% |
| u8_cache_native_incremental_append_count | 1 | 1 | +0.000% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 1 | 1 | +0.000% |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_native_load_bytes | 4096 | 4096 | +0.000% |
| u8_cache_v_vtcm_tail_partial_pack_rows | 8 | 8 | +0.000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 8 | 8 | +0.000% |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 65 | 65 | +0.000% |
| vtcm_acquired_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| vtcm_peak_plan_bytes | 6.68275e+06 | 6.68275e+06 | +0.000% |
| vtcm_requested_bytes | 8.38861e+06 | 8.38861e+06 | +0.000% |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_decode_audit | 0 | 0 | N/A: zero denominator |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt | 0 | 0 | N/A: zero denominator |
| w4f16_decode_opt_calls | 0 | 0 | N/A: zero denominator |
| w4f16_expand_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_av_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_av_requant_call_count | 8 | 8 | +0.000% |
| w4u8_av_requant_rows_observed | 4 | 4 | +0.000% |
| w4u8_av_requant_vector_count | 64 | 64 | +0.000% |
| w4u8_common_op_rows_observed | 4 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 5.03316e+07 | 5.03316e+07 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 30 | 30 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 7 | 7 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.51658e+07 | 2.51658e+07 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 4 | 4 | +0.000% |
| w4u8_decode_k_temp_carrier_skipped_count | 8 | 8 | +0.000% |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 8 | 8 | +0.000% |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 96 | 96 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_call_count | 8 | 8 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 192 | 192 | +0.000% |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 192 | 192 | +0.000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_consume_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 4.49479 | 6.02083 | +33.951% |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | +0.000% |
| w4u8_gate_up_swiglu_publish_count | 6 | 6 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 180.146 | 170.99 | -5.083% |
| w4u8_gate_up_swiglu_worker_ticks (us) | 28.0313 | 37.4167 | +33.482% |
| w4u8_input_norm_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 8 | 8 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 4 | 4 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 1 | 1 | +0.000% |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 18.5938 | 18 | -3.193% |
| w4u8_o_gate_prefetch_start_count | 1 | 1 | +0.000% |
| w4u8_o_gate_prefetch_wait_ticks (us) | 10.9062 | 2.59375 | -76.218% |
| w4u8_post_residual_direct_row4_call_count | 1 | 0 | -100.000% |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | +0.000% |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 6 | 6 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 74.0052 | 73.8958 | -0.148% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 24 | 24 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 22.1198 | 22.6198 | +2.260% |
| w4u8_qkv_ring_hmx_dispatch_count | 1 | 1 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 46.5052 | 45.5677 | -2.016% |
| w4u8_qkv_ring_pipeline_ticks (us) | 86.75 | 85.5208 | -1.417% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 0.140625 | 0.135417 | -3.704% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 1.17708 | 0.234375 | -80.088% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 438.391 | 436.24 | -0.491% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 74.0885 | 73.9167 | -0.232% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 0 | 0 | N/A: zero denominator |
| w4u8_swiglu_rows_observed | 4 | 4 | +0.000% |
| weight_ddr_read_bytes | 2.53297e+07 | 2.53297e+07 | +0.000% |
| weight_dma_descriptor_count | 60 | 60 | +0.000% |
| weight_dma_ticks (us) | 459.219 | 475.307 | +3.503% |
| wide_score_mode | 4 | 4 | +0.000% |

## E2E and next direction

New Qwen FP32 residual complete-model prefill/decode: N/A, not run. Selected historical Qwen SP2 integer residual EXP0268 is2030.238/48.173tok/s; selected Llama SP2 FP32 residual L32-0018 is2069.703/42.510tok/s. These separate historical full-model results are not this candidate performance.

The remaining strict FP32 Norm overhead dominates the prefill difference. A future structural candidate could align FP32 O/Down residual production and RMSNorm consumption in one native physical format, avoiding row-major transposition while preserving exact ascending-channel sums. Fullmodel cost also requires actual scope measurement; no inference or gate bypass is made here. Llama remains frozen, new Qwen C1/C3 have not been timed on Llama.
