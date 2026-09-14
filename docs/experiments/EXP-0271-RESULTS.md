# EXP0271: Qwen full-model SP2 FP32 residual speed

Mean complete invocation Host wall: embedding + all28 blocks + finalNorm + LM head/greedy + Host-DSP. Prepared weights, token IDs in/out; excludes cold model load/ADB/WSL tokenizer and optional diagnostic exports. Two fixed repeat10 AB/BA pairs, no statistical promotion; per-arm generated tokens may differ. Repeat1 auxiliary.

Measured source `cb4929b5c42eba39f2ccc507a97ace975de6e480`. Native code unchanged from EXP0270 C1+C3. Same frozen original C64/SP2 weights, scales and offline prefix; original FP16 embedding added to candidate package. Integer control uses original U8 embedding. Llama unchanged.

| Repeat / phase | Integer Host us | FP32 Host us | Integer token/s | FP32 token/s | Host increase | TPS change |
|---|---:|---:|---:|---:|---:|---:|
| r1_prefill | 32361.354 | 35787.760 | 1977.668 | 1788.321 | +10.59% | -9.57% |
| r1_decode | 23542.417 | 23453.452 | 42.477 | 42.638 | -0.38% | +0.38% |
| r10_prefill | 32266.893 | 35725.180 | 1983.457 | 1791.454 | +10.72% | -9.68% |
| r10_decode | 21955.651 | 22405.112 | 45.546 | 44.633 | +2.05% | -2.01% |

## Correctness and scope

Consecutive3 exact FP32 output:133120 values; last-layer Q/K, AV, postnorm, Gate/Up and SP2 live bytes exact; repeat10 deterministic. Full28 all layers/ledgers/cache lengths pass, no hidden tensor DDR or spill/fill. FinalNorm32768 actual A8 codes exact, full151936-vocabulary LM-head CPU reference selects identical token/logit code for all16 steps; head bias exact. Original integer control matches sealed EXP0268 token/code sequence. Candidate audit and timing agree, every repeated sequence deterministic. 9 CLI, 710 profiles; VTCM requested/acquired8MiB, maxplan8365824B. Timed boundary writes0; audit-only FP32 hidden/nativeNorm exports139264B perstep. No whole28 CPU transformer oracle, independent PPL or text-quality acceptance is claimed.

One validator repair: separate offline prefix metadata was incorrectly counted in logical cache_valid. Device returned0/all_steps_pass; actual lengths0->64 then64->65 agree with sealedEXP0268. Original audit raw retained and revalidated; native arithmetic unchanged.

Generated affected assembly: reduce8, epilogue, HMX worker and streamed Down have no vector stack payload stores. Generic Norm uses stack for uniform constants and scalar inverse-RMS splats, matching the already-audited EXP0270 behavior. Assembly extracts archived.

Diagnostic fullmodel prefill remains slightly above10percent Host overhead; decode stays below. Two pairs are insufficient for the formal confidence-interval gate. Prior EXP0270 singlelayer gate remains failed and neither paper baseline is replaced. Historical Qwen integer2030.238/48.173tok/s is a separate session; use current paired control for overhead.

## Stable module overview

Microseconds and share of complete Host wall; repeat10 pooled means,20 prefill and300 decode invocations per arm. W16A16 and W4A16 were not rerun; no substituted historical comparisons. Exclusive ledger sums to Host wall. Overlapping work/wait counters below must not be summed.

### prefill

| Module | W4A8-SP2 integer residual | W4A8-SP2 FP32 residual |
|---|---:|---:|
| I/O and metadata | 270.54 (0.84%) | 284.45 (0.80%) |
| Input RMSNorm | 562.59 (1.74%) | 2119.73 (5.93%) |
| QKV + Q/K Norm-RoPE | 7059.06 (21.88%) | 7076.86 (19.81%) |
| QK-Softmax-AV | 3470.33 (10.76%) | 3484.14 (9.75%) |
| O projection | 1309.08 (4.06%) | 2085.92 (5.84%) |
| Post-attention residual + RMSNorm | 659.47 (2.04%) | 2216.92 (6.21%) |
| Gate/Up + SwiGLU | 7431.15 (23.03%) | 7540.36 (21.11%) |
| Down projection | 4364.24 (13.53%) | 3978.15 (11.14%) |
| Final residual | 185.68 (0.58%) | 2.94 (0.01%) |
| KV-cache carrier conversion | 134.55 (0.42%) | 135.53 (0.38%) |
| KV-cache append DMA | 288.56 (0.89%) | 286.73 (0.80%) |
| Block internal orchestration | 35.68 (0.11%) | 35.21 (0.10%) |
| Layer bookkeeping | 29.03 (0.09%) | 29.15 (0.08%) |
| Stage-boundary bookkeeping | 20.43 (0.06%) | 20.51 (0.06%) |
| DSP unattributed residual | 0.00 (0.00%) | 0.00 (0.00%) |
| DSP runtime setup/teardown | 117.24 (0.36%) | 117.03 (0.33%) |
| Token embedding | 48.26 (0.15%) | 61.31 (0.17%) |
| Final model RMSNorm | 4.26 (0.01%) | 14.71 (0.04%) |
| LM head + greedy selection (excluding final norm) | 5263.78 (16.31%) | 5290.73 (14.81%) |
| True Host-DSP boundary | 1012.95 (3.14%) | 944.79 (2.64%) |
| Complete Host wall | 32266.89 (100.00%) | 35725.18 (100.00%) |

### decode

| Module | W4A8-SP2 integer residual | W4A8-SP2 FP32 residual |
|---|---:|---:|
| I/O and metadata | 283.43 (1.29%) | 284.68 (1.27%) |
| Input RMSNorm | 150.06 (0.68%) | 368.30 (1.64%) |
| QKV + Q/K Norm-RoPE | 2667.47 (12.15%) | 2677.10 (11.95%) |
| QK-Softmax-AV | 1881.46 (8.57%) | 1879.60 (8.39%) |
| O projection | 1312.17 (5.98%) | 1406.36 (6.28%) |
| Post-attention residual + RMSNorm | 198.14 (0.90%) | 368.06 (1.64%) |
| Gate/Up + SwiGLU | 6976.86 (31.78%) | 6901.53 (30.80%) |
| Down projection | 3708.10 (16.89%) | 3724.58 (16.62%) |
| Final residual | 43.20 (0.20%) | 1.81 (0.01%) |
| KV-cache carrier conversion | 290.35 (1.32%) | 290.69 (1.30%) |
| KV-cache append DMA | 156.13 (0.71%) | 160.73 (0.72%) |
| Block internal orchestration | 28.86 (0.13%) | 28.00 (0.12%) |
| Layer bookkeeping | 16.96 (0.08%) | 16.55 (0.07%) |
| Stage-boundary bookkeeping | 1.74 (0.01%) | 1.74 (0.01%) |
| DSP unattributed residual | 0.00 (0.00%) | 0.00 (0.00%) |
| DSP runtime setup/teardown | 73.51 (0.33%) | 73.70 (0.33%) |
| Token embedding | 2.27 (0.01%) | 2.49 (0.01%) |
| Final model RMSNorm | 3.52 (0.02%) | 15.22 (0.07%) |
| LM head + greedy selection (excluding final norm) | 3331.62 (15.17%) | 3363.39 (15.01%) |
| True Host-DSP boundary | 829.78 (3.78%) | 840.59 (3.75%) |
| Complete Host wall | 21955.65 (100.00%) | 22405.11 (100.00%) |

## Raw counters

Means reconstructed from raw JSON. Tick columns converted to microseconds; count/byte units unchanged. No resampling.

### repeat1 prefill

| Field | Integer | FP32 | Change |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_ticks (us) | 3345.781 | 3382.812 | +1.107% |
| attention_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| block_invocation_count | 28 | 28 | +0.000% |
| block_orchestration_ticks (us) | 35.625 | 35.15625 | -1.316% |
| boundary_ddr_read_bytes | 4976000 | 5107072 | +2.634% |
| boundary_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| boundary_dma_descriptor_count | 289 | 289 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 1 | 1 | +0.000% |
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
| down_ticks (us) | 4200.573 | 3703.125 | -11.842% |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 218 | 218 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| final_residual_ticks (us) | 176.0417 | 2.604167 | -98.521% |
| first_position | 0 | 0 | N/A: zero denominator |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| gate_up_ticks (us) | 6827.865 | 6940.833 | +1.655% |
| generation_embedding_ddr_read_bytes | 131328 | 262400 | +99.805% |
| generation_embedding_ticks (us) | 43.59375 | 56.875 | +30.466% |
| generation_final_norm_ticks (us) | 3.854167 | 15.3125 | +297.297% |
| generation_lm_head_argmax_ticks (us) | 605.4688 | 606.6146 | +0.189% |
| generation_lm_head_batch_n_tiles | 8 | 8 | +0.000% |
| generation_lm_head_command_count | 594 | 594 | +0.000% |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | +0.000% |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 5277.604 | 5253.958 | -0.448% |
| generation_lm_head_expand_ticks (us) | 3662.448 | 3638.854 | -0.644% |
| generation_lm_head_hmx_tail_wait_ticks (us) | 275.3646 | 271.25 | -1.494% |
| generation_lm_head_hmx_ticks (us) | 4624.375 | 4599.479 | -0.538% |
| generation_lm_head_n_tiles | 4748 | 4748 | +0.000% |
| generation_lm_head_prefetch_count | 593 | 593 | +0.000% |
| generation_lm_head_scale_dma_ticks (us) | 22.23958 | 22.96875 | +3.279% |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | +0.000% |
| generation_lm_head_ticks (us) | 5281.458 | 5269.271 | -0.231% |
| generation_lm_head_weight_dma_ticks (us) | 5238.229 | 5213.125 | -0.479% |
| generation_lm_head_weight_dma_wait_ticks (us) | 299.6875 | 302.3958 | +0.904% |
| generation_step | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 1882 | 1882 | +0.000% |
| hmx_compute_ticks (us) | 11774.74 | 12483.75 | +6.021% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | +0.000% |
| host_boundary_us | 2254.375 | 2216.927 | -1.661% |
| host_us | 32361.35 | 35787.76 | +10.588% |
| host_wall_ns | 3.236135e+07 | 3.578776e+07 | +10.588% |
| input_norm_ticks (us) | 561.1458 | 2117.656 | +277.381% |
| input_stage_ticks (us) | 0.5208333 | 0.625 | +20.000% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 30106.98 | 33570.83 | +11.505% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| layer_bookkeeping_ticks (us) | 25.78125 | 26.875 | +4.242% |
| ledger_named_ticks (us) | 30106.98 | 33570.83 | +11.505% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| logical_m | 64 | 64 | +0.000% |
| metadata_stage_ticks (us) | 206.9271 | 232.7604 | +12.484% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| o_projection_ticks (us) | 1194.688 | 1995.26 | +67.011% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 1 | 1 | +0.000% |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| output_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| post_attention_norm_ticks (us) | 1.145833 | 0.625 | -45.455% |
| post_attention_residual_ticks (us) | 641.7188 | 2213.906 | +244.996% |
| prefix_group_patch_count | 224 | 224 | +0.000% |
| prefix_kv_mode | 1 | 1 | +0.000% |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | +0.000% |
| prepared_session_run_index | 1 | 1 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 1422.292 | 1588.281 | +11.671% |
| projection_pack_ticks (us) | 4.895833 | 3.802083 | -22.340% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| qk_norm_rope_ticks (us) | 0.3645833 | 0.5208333 | +42.857% |
| qkv_projection_ticks (us) | 7018.75 | 7057.917 | +0.558% |
| repeat_count | 1 | 1 | +0.000% |
| runtime_setup_ticks (us) | 54.53125 | 53.4375 | -2.006% |
| runtime_teardown_ticks (us) | 53.4375 | 54.32292 | +1.657% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_append_ticks (us) | 281.1979 | 274.4792 | -2.389% |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | +0.000% |
| scan_cache_dma_descriptor_count | 448 | 448 | +0.000% |
| scan_cache_pack_ticks (us) | 135.5208 | 131.25 | -3.151% |
| scan_cache_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_dynamic_attention_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_logical_m_observed | 64 | 64 | +0.000% |
| scan_padded_kv_length | 64 | 64 | +0.000% |
| scan_total_kv_length | 64 | 64 | +0.000% |
| stage_boundary_ticks (us) | 20.3125 | 20.52083 | +1.026% |
| total_ticks (us) | 30052.45 | 33517.4 | +11.530% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 517.2396 | 546.9792 | +5.750% |
| u8_attention_av_requant_ticks (us) | 1661.042 | 1652.76 | -0.499% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_pipeline_wait_ticks (us) | 1652.5 | 1720.312 | +4.104% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 478.3333 | 508.5417 | +6.315% |
| u8_attention_qk_norm_rope_ticks (us) | 24543.7 | 24553.07 | +0.038% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 8710.312 | 8711.146 | +0.010% |
| u8_attention_v_pack_ticks (us) | 5007.917 | 5030.208 | +0.445% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | +0.000% |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 415.1042 | 403.8542 | -2.710% |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 28 | 28 | +0.000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | +0.000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | +0.000% |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 64 | 64 | +0.000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | +0.000% |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | +0.000% |
| vtcm_requested_bytes | 8388608 | 8388608 | +0.000% |
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
| w4u8_decode_direct_n_expand_bytes_avoided | 1.409286e+09 | 1.409286e+09 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 840 | 840 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 196 | 196 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.046431e+08 | 7.046431e+08 | +0.000% |
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
| w4u8_final_residual_main_work_ticks (us) | 106.0938 | 0 | -100.000% |
| w4u8_final_residual_pool_wait_ticks (us) | 19.47917 | 0 | -100.000% |
| w4u8_final_residual_task_count | 448 | 0 | -100.000% |
| w4u8_final_residual_worker_work_ticks (us) | 534.3229 | 0 | -100.000% |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 387.1354 | 395.8854 | +2.260% |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_main_work_ticks (us) | 437.5521 | 0 | -100.000% |
| w4u8_input_norm_pool_wait_ticks (us) | 72.03125 | 0 | -100.000% |
| w4u8_input_norm_task_count | 448 | 0 | -100.000% |
| w4u8_input_norm_worker_work_ticks (us) | 2184.688 | 0 | -100.000% |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 112 | 112 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_main_work_ticks (us) | 520.9375 | 0 | -100.000% |
| w4u8_post_residual_pool_wait_ticks (us) | 73.28125 | 0 | -100.000% |
| w4u8_post_residual_task_count | 448 | 0 | -100.000% |
| w4u8_post_residual_worker_work_ticks (us) | 2582.031 | 0 | -100.000% |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 168 | 168 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2089.531 | 2144.792 | +2.645% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 541.6667 | 581.5104 | +7.356% |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1483.854 | 1493.906 | +0.677% |
| w4u8_qkv_ring_pipeline_ticks (us) | 6984.271 | 7020.729 | +0.522% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 4573.906 | 4541.146 | -0.716% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 6.354167 | 9.895833 | +55.738% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 12895.52 | 13334.01 | +3.400% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2090.885 | 2145.312 | +2.603% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 6 | 0 | -100.000% |
| w4u8_swiglu_rows_observed | 64 | 64 | +0.000% |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | +0.000% |
| weight_dma_descriptor_count | 2275 | 2275 | +0.000% |
| weight_dma_ticks (us) | 17817.66 | 18117.81 | +1.685% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat1 decode

| Field | Integer | FP32 | Change |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_ticks (us) | 1910.556 | 1880.208 | -1.588% |
| attention_unattributed_ticks (us) | 886.5382 | 869.3646 | -1.937% |
| block_invocation_count | 28 | 28 | +0.000% |
| block_orchestration_ticks (us) | 28.71875 | 28.38194 | -1.173% |
| boundary_ddr_read_bytes | 4846976 | 4849024 | +0.042% |
| boundary_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| boundary_dma_descriptor_count | 226 | 226 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 1 | 1 | +0.000% |
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
| down_ticks (us) | 3727.931 | 3704.299 | -0.634% |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 218 | 218 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| final_residual_ticks (us) | 43.08333 | 1.777778 | -95.874% |
| first_position | 71 | 71 | +0.000% |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| gate_up_ticks (us) | 6969.212 | 6810.219 | -2.281% |
| generation_embedding_ddr_read_bytes | 2304 | 4352 | +88.889% |
| generation_embedding_ticks (us) | 1.878472 | 2.111111 | +12.384% |
| generation_final_norm_ticks (us) | 3.864583 | 15.81944 | +309.344% |
| generation_lm_head_argmax_ticks (us) | 463.7326 | 465.5174 | +0.385% |
| generation_lm_head_batch_n_tiles | 32 | 32 | +0.000% |
| generation_lm_head_command_count | 149 | 149 | +0.000% |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | +0.000% |
| generation_lm_head_direct_slot_join_count | 147 | 147 | +0.000% |
| generation_lm_head_exclusive_ticks (us) | 3240.306 | 3388.663 | +4.579% |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 10.70486 | 12.82292 | +19.786% |
| generation_lm_head_hmx_ticks (us) | 2724.347 | 2865.878 | +5.195% |
| generation_lm_head_n_tiles | 4748 | 4748 | +0.000% |
| generation_lm_head_prefetch_count | 148 | 148 | +0.000% |
| generation_lm_head_scale_dma_ticks (us) | 21.81944 | 23.47569 | +7.591% |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | +0.000% |
| generation_lm_head_ticks (us) | 3244.17 | 3404.483 | +4.942% |
| generation_lm_head_weight_dma_ticks (us) | 2739.313 | 2881.778 | +5.201% |
| generation_lm_head_weight_dma_wait_ticks (us) | 2672.66 | 2810.778 | +5.168% |
| generation_step | 8 | 8 | +0.000% |
| hmx_command_count | 1437 | 1437 | +0.000% |
| hmx_compute_ticks (us) | 8448.524 | 8506.378 | +0.685% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | +0.000% |
| host_boundary_us | 2492.712 | 2016.372 | -19.109% |
| host_us | 23542.42 | 23453.45 | -0.378% |
| host_wall_ns | 2.354242e+07 | 2.345345e+07 | -0.378% |
| input_norm_ticks (us) | 149.9896 | 368.5312 | +145.705% |
| input_stage_ticks (us) | 0.3159722 | 0.2708333 | -14.286% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 21049.7 | 21437.08 | +1.840% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| layer_bookkeeping_ticks (us) | 16.42708 | 16.22222 | -1.247% |
| ledger_named_ticks (us) | 21049.7 | 21437.08 | +1.840% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| logical_m | 1 | 1 | +0.000% |
| metadata_stage_ticks (us) | 248.3472 | 257.1181 | +3.532% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| o_projection_ticks (us) | 1317.351 | 1405.924 | +6.724% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 1 | 1 | +0.000% |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| output_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| post_attention_norm_ticks (us) | 0.6909722 | 0.6354167 | -8.040% |
| post_attention_residual_ticks (us) | 200.2326 | 367.7882 | +83.680% |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 1 | 1 | +0.000% |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 9 | 9 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 703.4028 | 727.9479 | +3.489% |
| projection_pack_ticks (us) | 4.642361 | 4.604167 | -0.823% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| qk_norm_rope_ticks (us) | 0.3993056 | 0.3993056 | +0.000% |
| qkv_projection_ticks (us) | 2676.115 | 2664.792 | -0.423% |
| repeat_count | 1 | 1 | +0.000% |
| runtime_setup_ticks (us) | 37.875 | 37.81597 | -0.156% |
| runtime_teardown_ticks (us) | 36.17014 | 35.60069 | -1.574% |
| scan_attention_overlay_capacity_bytes | 2752512 | 2359296 | -14.286% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | +0.000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_append_ticks (us) | 148.7361 | 158.6944 | +6.695% |
| scan_cache_ddr_read_bytes | 4042752 | 4042752 | +0.000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | +0.000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | +0.000% |
| scan_cache_pack_ticks (us) | 289.7882 | 290.0868 | +0.103% |
| scan_cache_stage_ticks (us) | 568.3611 | 543.1146 | -4.442% |
| scan_dynamic_attention_ticks (us) | 1905.392 | 1875 | -1.595% |
| scan_logical_m_observed | 1 | 1 | +0.000% |
| scan_padded_kv_length | 96 | 96 | +0.000% |
| scan_total_kv_length | 72 | 72 | +0.000% |
| stage_boundary_ticks (us) | 1.71875 | 1.722222 | +0.202% |
| total_ticks (us) | 21011.83 | 21399.26 | +1.844% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 172.3889 | 168.1736 | -2.445% |
| u8_attention_av_requant_ticks (us) | 143.9653 | 140.691 | -2.274% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 31.59028 | 31.58681 | -0.011% |
| u8_attention_pipeline_wait_ticks (us) | 66.46181 | 66.18056 | -0.423% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 143.3785 | 139.5972 | -2.637% |
| u8_attention_qk_norm_rope_ticks (us) | 1959.684 | 1948.663 | -0.562% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 348.5972 | 347.1285 | -0.421% |
| u8_attention_v_pack_ticks (us) | 117.6354 | 117.4861 | -0.127% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 6272 | 6272 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 39.10417 | 39.26736 | +0.417% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | +0.000% |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 436.3056 | 446.7188 | +2.387% |
| u8_cache_native_incremental_append_count | 28 | 28 | +0.000% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 28 | 28 | +0.000% |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_native_load_bytes | 275251.2 | 275251.2 | +0.000% |
| u8_cache_v_vtcm_tail_partial_pack_rows | 358.4 | 358.4 | +0.000% |
| u8_cache_v_vtcm_tail_publish_count | 44.8 | 44.8 | +0.000% |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | +0.000% |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 72 | 72 | +0.000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | +0.000% |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | +0.000% |
| vtcm_requested_bytes | 8388608 | 8388608 | +0.000% |
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
| w4u8_av_requant_call_count | 224 | 224 | +0.000% |
| w4u8_av_requant_rows_observed | 4 | 4 | +0.000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | +0.000% |
| w4u8_common_op_rows_observed | 4 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.720451e+09 | 1.720451e+09 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 989 | 989 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 8.602255e+08 | 8.602255e+08 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 112 | 112 | +0.000% |
| w4u8_decode_k_temp_carrier_skipped_count | 224 | 224 | +0.000% |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 224 | 224 | +0.000% |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 2688 | 2688 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_call_count | 224 | 224 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 5376 | 5376 | +0.000% |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 5376 | 5376 | +0.000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 130.9653 | 177.75 | +35.723% |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | +0.000% |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 5703.181 | 5453.045 | -4.386% |
| w4u8_gate_up_swiglu_worker_ticks (us) | 695.25 | 953.8993 | +37.202% |
| w4u8_input_norm_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 112 | 112 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | +0.000% |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 564.6458 | 560.8889 | -0.665% |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | +0.000% |
| w4u8_o_gate_prefetch_wait_ticks (us) | 354.8264 | 183.5521 | -48.270% |
| w4u8_post_residual_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | +0.000% |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 168 | 168 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2328.146 | 2316.285 | -0.509% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 539.8125 | 544.6215 | +0.891% |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1669.802 | 1654.92 | -0.891% |
| w4u8_qkv_ring_pipeline_ticks (us) | 2645.792 | 2634.625 | -0.422% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 3.756944 | 3.784722 | +0.739% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 7.104167 | 7.340278 | +3.324% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 13279.42 | 13292.99 | +0.102% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2329.198 | 2317.472 | -0.503% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 0 | 0 | N/A: zero denominator |
| w4u8_swiglu_rows_observed | 4 | 4 | +0.000% |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | +0.000% |
| weight_dma_descriptor_count | 1830 | 1830 | +0.000% |
| weight_dma_ticks (us) | 16589.47 | 16720.81 | +0.792% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat10 prefill

| Field | Integer | FP32 | Change |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_ticks (us) | 3470.328 | 3484.141 | +0.398% |
| attention_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| block_invocation_count | 28 | 28 | +0.000% |
| block_orchestration_ticks (us) | 35.68229 | 35.20833 | -1.328% |
| boundary_ddr_read_bytes | 4976000 | 5107072 | +2.634% |
| boundary_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| boundary_dma_descriptor_count | 289 | 289 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 1 | 1 | +0.000% |
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
| down_ticks (us) | 4364.245 | 3978.154 | -8.847% |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 218 | 218 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| final_residual_ticks (us) | 185.6823 | 2.9375 | -98.418% |
| first_position | 0 | 0 | N/A: zero denominator |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| gate_up_ticks (us) | 7431.148 | 7540.362 | +1.470% |
| generation_embedding_ddr_read_bytes | 131328 | 262400 | +99.805% |
| generation_embedding_ticks (us) | 48.25521 | 61.30729 | +27.048% |
| generation_final_norm_ticks (us) | 4.257812 | 14.71094 | +245.505% |
| generation_lm_head_argmax_ticks (us) | 603.6927 | 603.651 | -0.007% |
| generation_lm_head_batch_n_tiles | 8 | 8 | +0.000% |
| generation_lm_head_command_count | 594 | 594 | +0.000% |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | +0.000% |
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A: zero denominator |
| generation_lm_head_exclusive_ticks (us) | 5263.779 | 5290.734 | +0.512% |
| generation_lm_head_expand_ticks (us) | 3662.294 | 3657.422 | -0.133% |
| generation_lm_head_hmx_tail_wait_ticks (us) | 268.9427 | 302.5625 | +12.501% |
| generation_lm_head_hmx_ticks (us) | 4612.016 | 4638.667 | +0.578% |
| generation_lm_head_n_tiles | 4748 | 4748 | +0.000% |
| generation_lm_head_prefetch_count | 593 | 593 | +0.000% |
| generation_lm_head_scale_dma_ticks (us) | 22.84635 | 23.54688 | +3.066% |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | +0.000% |
| generation_lm_head_ticks (us) | 5268.036 | 5305.445 | +0.710% |
| generation_lm_head_weight_dma_ticks (us) | 5223.445 | 5250.013 | +0.509% |
| generation_lm_head_weight_dma_wait_ticks (us) | 293.9271 | 291.2865 | -0.898% |
| generation_step | 0 | 0 | N/A: zero denominator |
| hmx_command_count | 1882 | 1882 | +0.000% |
| hmx_compute_ticks (us) | 12217.32 | 13029.6 | +6.649% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | +0.000% |
| host_boundary_us | 1012.948 | 944.7916 | -6.729% |
| host_us | 32266.89 | 35725.18 | +10.718% |
| host_wall_ns | 3.226689e+07 | 3.572518e+07 | +10.718% |
| input_norm_ticks (us) | 562.5938 | 2119.732 | +276.778% |
| input_stage_ticks (us) | 0.515625 | 0.4192708 | -18.687% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 31253.95 | 34780.39 | +11.283% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| layer_bookkeeping_ticks (us) | 29.03125 | 29.14844 | +0.404% |
| ledger_named_ticks (us) | 31253.95 | 34780.39 | +11.283% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| logical_m | 64 | 64 | +0.000% |
| metadata_stage_ticks (us) | 270.0234 | 284.026 | +5.186% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| o_projection_ticks (us) | 1309.083 | 2085.924 | +59.342% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 1 | 1 | +0.000% |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| output_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| post_attention_norm_ticks (us) | 1.065104 | 0.984375 | -7.579% |
| post_attention_residual_ticks (us) | 658.4089 | 2215.94 | +236.560% |
| prefix_group_patch_count | 224 | 224 | +0.000% |
| prefix_kv_mode | 1 | 1 | +0.000% |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | +0.000% |
| prepared_session_run_index | 1 | 1 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 1321.443 | 1604.945 | +21.454% |
| projection_pack_ticks (us) | 4.966146 | 4.190104 | -15.627% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| qk_norm_rope_ticks (us) | 0.5703125 | 0.5989583 | +5.023% |
| qkv_projection_ticks (us) | 7058.487 | 7076.258 | +0.252% |
| repeat_count | 1 | 1 | +0.000% |
| runtime_setup_ticks (us) | 65.74479 | 64.54688 | -1.822% |
| runtime_teardown_ticks (us) | 51.5 | 52.47917 | +1.901% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A: zero denominator |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_append_ticks (us) | 288.5625 | 286.7318 | -0.634% |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | +0.000% |
| scan_cache_dma_descriptor_count | 448 | 448 | +0.000% |
| scan_cache_pack_ticks (us) | 134.5469 | 135.5339 | +0.734% |
| scan_cache_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_dynamic_attention_ticks (us) | 0 | 0 | N/A: zero denominator |
| scan_logical_m_observed | 64 | 64 | +0.000% |
| scan_padded_kv_length | 64 | 64 | +0.000% |
| scan_total_kv_length | 64 | 64 | +0.000% |
| stage_boundary_ticks (us) | 20.4349 | 20.51042 | +0.370% |
| total_ticks (us) | 31188.2 | 34715.84 | +11.311% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 617.4115 | 625.9427 | +1.382% |
| u8_attention_av_requant_ticks (us) | 1643.719 | 1644.388 | +0.041% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_pipeline_wait_ticks (us) | 1991.708 | 2019.87 | +1.414% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 572.6979 | 578.8542 | +1.075% |
| u8_attention_qk_norm_rope_ticks (us) | 24356.67 | 24429.45 | +0.299% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 8703.706 | 8702.643 | -0.012% |
| u8_attention_v_pack_ticks (us) | 5008.948 | 5026.357 | +0.348% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | +0.000% |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 421.3359 | 420.4818 | -0.203% |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 28 | 28 | +0.000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | +0.000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | +0.000% |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | +0.000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 64 | 64 | +0.000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | +0.000% |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | +0.000% |
| vtcm_requested_bytes | 8388608 | 8388608 | +0.000% |
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
| w4u8_decode_direct_n_expand_bytes_avoided | 1.409286e+09 | 1.409286e+09 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 840 | 840 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 196 | 196 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.046431e+08 | 7.046431e+08 | +0.000% |
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
| w4u8_final_residual_main_work_ticks (us) | 106.9323 | 0 | -100.000% |
| w4u8_final_residual_pool_wait_ticks (us) | 27.64063 | 0 | -100.000% |
| w4u8_final_residual_task_count | 448 | 0 | -100.000% |
| w4u8_final_residual_worker_work_ticks (us) | 550.5807 | 0 | -100.000% |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 397.362 | 399.7891 | +0.611% |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_main_work_ticks (us) | 431.5807 | 0 | -100.000% |
| w4u8_input_norm_pool_wait_ticks (us) | 79.7526 | 0 | -100.000% |
| w4u8_input_norm_task_count | 448 | 0 | -100.000% |
| w4u8_input_norm_worker_work_ticks (us) | 2175.117 | 0 | -100.000% |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 112 | 112 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A: zero denominator |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_main_work_ticks (us) | 513.1901 | 0 | -100.000% |
| w4u8_post_residual_pool_wait_ticks (us) | 97.79948 | 0 | -100.000% |
| w4u8_post_residual_task_count | 448 | 0 | -100.000% |
| w4u8_post_residual_worker_work_ticks (us) | 2630.344 | 0 | -100.000% |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A: zero denominator |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 168 | 168 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2305.531 | 2334.143 | +1.241% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 551.7656 | 552.6745 | +0.165% |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1645.917 | 1680.578 | +2.106% |
| w4u8_qkv_ring_pipeline_ticks (us) | 7023.018 | 7039.346 | +0.232% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 4377.583 | 4361.508 | -0.367% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 7.057292 | 8.044271 | +13.985% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 13882.57 | 14376.93 | +3.561% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2306.669 | 2335.302 | +1.241% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 6 | 0 | -100.000% |
| w4u8_swiglu_rows_observed | 64 | 64 | +0.000% |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | +0.000% |
| weight_dma_descriptor_count | 2275 | 2275 | +0.000% |
| weight_dma_ticks (us) | 19068.81 | 19355.08 | +1.501% |
| wide_score_mode | 4 | 4 | +0.000% |

### repeat10 decode

| Field | Integer | FP32 | Change |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_av_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_hmx_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_pack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_qk_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_setup_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_softmax_ticks (us) | 0 | 0 | N/A: zero denominator |
| attention_ticks (us) | 1881.461 | 1879.601 | -0.099% |
| attention_unattributed_ticks (us) | 873.4212 | 870.5681 | -0.327% |
| block_invocation_count | 28 | 28 | +0.000% |
| block_orchestration_ticks (us) | 28.86233 | 27.99618 | -3.001% |
| boundary_ddr_read_bytes | 4846976 | 4849024 | +0.042% |
| boundary_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| boundary_dma_descriptor_count | 226 | 226 | +0.000% |
| cache_compared_elements | 0 | 0 | N/A: zero denominator |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A: zero denominator |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | +0.000% |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | +0.000% |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A: zero denominator |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A: zero denominator |
| cache_max_nrmse | 0 | 0 | N/A: zero denominator |
| cache_min_cosine | 1 | 1 | +0.000% |
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
| down_ticks (us) | 3708.1 | 3724.576 | +0.444% |
| dsp_status | 3 | 3 | +0.000% |
| experiment | 218 | 218 | +0.000% |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | N/A: zero denominator |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| final_residual_ticks (us) | 43.20052 | 1.807292 | -95.817% |
| first_position | 71 | 71 | +0.000% |
| fp32_residual | 0 | 1 | N/A: zero denominator |
| gate_up_ticks (us) | 6976.864 | 6901.528 | -1.080% |
| generation_embedding_ddr_read_bytes | 2304 | 4352 | +88.889% |
| generation_embedding_ticks (us) | 2.268229 | 2.493403 | +9.927% |
| generation_final_norm_ticks (us) | 3.516319 | 15.21597 | +332.724% |
| generation_lm_head_argmax_ticks (us) | 464.758 | 465.3498 | +0.127% |
| generation_lm_head_batch_n_tiles | 32 | 32 | +0.000% |
| generation_lm_head_command_count | 149 | 149 | +0.000% |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | +0.000% |
| generation_lm_head_direct_slot_join_count | 147 | 147 | +0.000% |
| generation_lm_head_exclusive_ticks (us) | 3331.622 | 3363.394 | +0.954% |
| generation_lm_head_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_hmx_tail_wait_ticks (us) | 11.64635 | 11.77934 | +1.142% |
| generation_lm_head_hmx_ticks (us) | 2813.736 | 2841.52 | +0.987% |
| generation_lm_head_n_tiles | 4748 | 4748 | +0.000% |
| generation_lm_head_prefetch_count | 148 | 148 | +0.000% |
| generation_lm_head_scale_dma_ticks (us) | 22.52743 | 23.27431 | +3.315% |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | N/A: zero denominator |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | +0.000% |
| generation_lm_head_ticks (us) | 3335.138 | 3378.61 | +1.303% |
| generation_lm_head_weight_dma_ticks (us) | 2827.619 | 2858.125 | +1.079% |
| generation_lm_head_weight_dma_wait_ticks (us) | 2760.414 | 2788.236 | +1.008% |
| generation_step | 8 | 8 | +0.000% |
| hmx_command_count | 1437 | 1437 | +0.000% |
| hmx_compute_ticks (us) | 8300.219 | 8425.062 | +1.504% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A: zero denominator |
| hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | +0.000% |
| host_boundary_us | 829.7787 | 840.5908 | +1.303% |
| host_us | 21955.65 | 22405.11 | +2.047% |
| host_wall_ns | 2.195565e+07 | 2.240511e+07 | +2.047% |
| input_norm_ticks (us) | 150.0644 | 368.3014 | +145.429% |
| input_stage_ticks (us) | 0.3017361 | 0.3019097 | +0.058% |
| intermediate_ddr_read_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| intermediate_dma_descriptor_count | 0 | 0 | N/A: zero denominator |
| intermediate_spill_fill_count | 0 | 0 | N/A: zero denominator |
| invocation_ticks (us) | 21125.87 | 21564.52 | +2.076% |
| kv_cache_k_format | 14 | 14 | +0.000% |
| kv_cache_v_format | 12 | 12 | +0.000% |
| layer_bookkeeping_ticks (us) | 16.96406 | 16.55399 | -2.417% |
| ledger_named_ticks (us) | 21125.87 | 21564.52 | +2.076% |
| ledger_unattributed_ticks (us) | 0 | 0 | N/A: zero denominator |
| logical_m | 1 | 1 | +0.000% |
| metadata_stage_ticks (us) | 283.1266 | 284.3792 | +0.442% |
| numerical_audit_enabled | 0 | 0 | N/A: zero denominator |
| numerical_status | 1 | 1 | +0.000% |
| o_projection_ticks (us) | 1312.169 | 1406.355 | +7.178% |
| output_cosine | 0 | 0 | N/A: zero denominator |
| output_fp16_atol | 0.0625 | 0.0625 | +0.000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | +0.000% |
| output_fp16_rtol | 0.002 | 0.002 | +0.000% |
| output_max_abs | 0 | 0 | N/A: zero denominator |
| output_max_lsb | 0 | 0 | N/A: zero denominator |
| output_max_required_rtol_after_atol | 0 | 0 | N/A: zero denominator |
| output_mismatches | 1 | 1 | +0.000% |
| output_mixed_tolerance_violations | 0 | 0 | N/A: zero denominator |
| output_nonfinite_count | 0 | 0 | N/A: zero denominator |
| output_nrmse | 0 | 0 | N/A: zero denominator |
| output_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| post_attention_norm_ticks (us) | 0.6756944 | 0.6368056 | -5.755% |
| post_attention_residual_ticks (us) | 197.4595 | 367.424 | +86.076% |
| prefix_group_patch_count | 0 | 0 | N/A: zero denominator |
| prefix_kv_mode | 1 | 1 | +0.000% |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A: zero denominator |
| prepared_session_run_index | 9 | 9 | +0.000% |
| projection_failure_index | 0 | 0 | N/A: zero denominator |
| projection_failure_n_tile | 0 | 0 | N/A: zero denominator |
| projection_failure_result | 0 | 0 | N/A: zero denominator |
| projection_failure_step | 0 | 0 | N/A: zero denominator |
| projection_hmx_wait_ticks (us) | 690.3611 | 727.655 | +5.402% |
| projection_pack_ticks (us) | 4.645139 | 4.645833 | +0.015% |
| projection_unpack_ticks (us) | 0 | 0 | N/A: zero denominator |
| qk_norm_rope_ticks (us) | 0.3607639 | 0.3722222 | +3.176% |
| qkv_projection_ticks (us) | 2667.11 | 2676.726 | +0.361% |
| repeat_count | 1 | 1 | +0.000% |
| runtime_setup_ticks (us) | 37.93698 | 37.91406 | -0.060% |
| runtime_teardown_ticks (us) | 35.57691 | 35.78646 | +0.589% |
| scan_attention_overlay_capacity_bytes | 2752512 | 2359296 | -14.286% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | +0.000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A: zero denominator |
| scan_cache_append_ticks (us) | 156.1337 | 160.7264 | +2.942% |
| scan_cache_ddr_read_bytes | 4042752 | 4042752 | +0.000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | +0.000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | +0.000% |
| scan_cache_pack_ticks (us) | 290.3535 | 290.6908 | +0.116% |
| scan_cache_stage_ticks (us) | 545.7293 | 543.788 | -0.356% |
| scan_dynamic_attention_ticks (us) | 1876.286 | 1874.412 | -0.100% |
| scan_logical_m_observed | 1 | 1 | +0.000% |
| scan_padded_kv_length | 96 | 96 | +0.000% |
| scan_total_kv_length | 72 | 72 | +0.000% |
| stage_boundary_ticks (us) | 1.744792 | 1.739757 | -0.289% |
| total_ticks (us) | 21087.94 | 21526.61 | +2.080% |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A: zero denominator |
| u8_attention_av_hmx_ticks (us) | 168.4915 | 168.7283 | +0.141% |
| u8_attention_av_requant_ticks (us) | 140.8043 | 140.616 | -0.134% |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A: zero denominator |
| u8_attention_k_pack_ticks (us) | 31.65313 | 31.63924 | -0.044% |
| u8_attention_pipeline_wait_ticks (us) | 65.68576 | 65.61094 | -0.114% |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A: zero denominator |
| u8_attention_qk_hmx_ticks (us) | 137.5293 | 138.233 | +0.512% |
| u8_attention_qk_norm_rope_ticks (us) | 1947.268 | 1935.943 | -0.582% |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | N/A: zero denominator |
| u8_attention_softmax_ticks (us) | 346.2141 | 346.6934 | +0.138% |
| u8_attention_v_pack_ticks (us) | 117.6622 | 117.512 | -0.128% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 6272 | 6272 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | +0.000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 39.35781 | 39.59601 | +0.605% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | +0.000% |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | +0.000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_append_update_ticks (us) | 444.3543 | 449.3056 | +1.114% |
| u8_cache_native_incremental_append_count | 28 | 28 | +0.000% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A: zero denominator |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_seal_count | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_segment_tail_append_count | 28 | 28 | +0.000% |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A: zero denominator |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A: zero denominator |
| u8_cache_v_vtcm_tail_native_load_bytes | 275251.2 | 275251.2 | +0.000% |
| u8_cache_v_vtcm_tail_partial_pack_rows | 358.4 | 358.4 | +0.000% |
| u8_cache_v_vtcm_tail_publish_count | 44.8 | 44.8 | +0.000% |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | +0.000% |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A: zero denominator |
| valid_length | 72 | 72 | +0.000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | +0.000% |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | +0.000% |
| vtcm_requested_bytes | 8388608 | 8388608 | +0.000% |
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
| w4u8_av_requant_call_count | 224 | 224 | +0.000% |
| w4u8_av_requant_rows_observed | 4 | 4 | +0.000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | +0.000% |
| w4u8_common_op_rows_observed | 4 | 0 | -100.000% |
| w4u8_common_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_av_requant_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_op_rows | 4 | 4 | +0.000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | +0.000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.720451e+09 | 1.720451e+09 | +0.000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_hmx_command_count | 989 | 989 | +0.000% |
| w4u8_decode_direct_n_mask | 63 | 63 | +0.000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | +0.000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | +0.000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | +0.000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 8.602255e+08 | 8.602255e+08 | +0.000% |
| w4u8_decode_k_pair_row4_call_count | 112 | 112 | +0.000% |
| w4u8_decode_k_temp_carrier_skipped_count | 224 | 224 | +0.000% |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | +0.000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | +0.000% |
| w4u8_decode_projection_mode | 1 | 1 | +0.000% |
| w4u8_decode_q_pair_row4_call_count | 224 | 224 | +0.000% |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | +0.000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_qk_rows_processed | 2688 | 2688 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_call_count | 224 | 224 | +0.000% |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_softmax_mode | 1 | 1 | +0.000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_row4_call_count | 5376 | 5376 | +0.000% |
| w4u8_decode_swiglu_rows | 4 | 4 | +0.000% |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A: zero denominator |
| w4u8_decode_swiglu_vector_count | 5376 | 5376 | +0.000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 129.847 | 178.7113 | +37.632% |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | +0.000% |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | +0.000% |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 5725.096 | 5542.668 | -3.186% |
| w4u8_gate_up_swiglu_worker_ticks (us) | 673.0111 | 950.9622 | +41.300% |
| w4u8_input_norm_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | +0.000% |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_o_batch_count | 112 | 112 | +0.000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | +0.000% |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | +0.000% |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 569.9234 | 563.9717 | -1.044% |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | +0.000% |
| w4u8_o_gate_prefetch_wait_ticks (us) | 363.0061 | 187.2399 | -48.420% |
| w4u8_post_residual_direct_row4_call_count | 28 | 0 | -100.000% |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_prefill_cache_mode | 1 | 1 | +0.000% |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | +0.000% |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_batch_count | 168 | 168 | +0.000% |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2319.123 | 2329.57 | +0.450% |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A: zero denominator |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | +0.000% |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 543.1023 | 538.73 | -0.805% |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | +0.000% |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1655.522 | 1675.847 | +1.228% |
| w4u8_qkv_ring_pipeline_ticks (us) | 2636.823 | 2646.417 | +0.364% |
| w4u8_qkv_ring_pool_wait_ticks (us) | 3.755382 | 3.741319 | -0.374% |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | +0.000% |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 7.755556 | 7.640278 | -1.486% |
| w4u8_qkv_ring_slot_count | 2 | 2 | +0.000% |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 13254.57 | 13409.79 | +1.171% |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2320.214 | 2330.625 | +0.449% |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | N/A: zero denominator |
| w4u8_residual_active_contexts | 0 | 0 | N/A: zero denominator |
| w4u8_swiglu_rows_observed | 4 | 4 | +0.000% |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | +0.000% |
| weight_dma_descriptor_count | 1830 | 1830 | +0.000% |
| weight_dma_ticks (us) | 16697.1 | 16850.65 | +0.920% |
| wide_score_mode | 4 | 4 | +0.000% |
