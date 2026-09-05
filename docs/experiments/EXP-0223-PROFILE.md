# EXP-0223 complete profiling comparison

Frozen ABI108 binary, frozen absmax A0/B0/C0 and clipped A/B/C independent software token sequences. One M64 plus15 feedback steps;5short then10six-way rotating formal rounds. Other recipe columns historical nonpaired, changed W4 prevents activation-only attribution. Quality scoring disabled. All numeric fields retained; additive ledger fields exclusive, engine/wait counters overlapping. Host-DSP boundary per record = Host wall minus DSP invocation. Percent changes below candidate/control minus one; positive timing changes are slower.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 clipped GPTQ B EXP-0223 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 380.8 (0.60%) | 247.4 (0.63%) | +53.93% |
| Input RMSNorm | 489.7 (0.61%) | 491.5 (0.78%) | 554.0 (1.40%) | -11.29% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11748.4 (18.62%) | 7052.7 (17.82%) | +66.58% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3960.5 (6.28%) | 3214.5 (8.12%) | +23.21% |
| O projection | 5757.7 (7.14%) | 5057.3 (8.01%) | 1256.4 (3.17%) | +302.52% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 474.2 (0.75%) | 654.0 (1.65%) | -27.48% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22406.6 (35.51%) | 14442.7 (36.49%) | +55.14% |
| Down | 13447.9 (16.67%) | 8643.2 (13.70%) | 3428.5 (8.66%) | +152.10% |
| Final residual | 140.1 (0.17%) | 140.1 (0.22%) | 183.8 (0.46%) | -23.77% |
| KV carrier conversion | 174.0 (0.22%) | 171.9 (0.27%) | 203.2 (0.51%) | -15.42% |
| KV append DMA | 343.6 (0.43%) | 341.3 (0.54%) | 463.9 (1.17%) | -26.43% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.35% |
| Layer bookkeeping | 23.9 (0.03%) | 23.7 (0.04%) | 23.2 (0.06%) | +2.36% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.9 (0.01%) | 22.5 (0.06%) | -60.42% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.4 (0.15%) | 105.6 (0.27%) | -11.52% |
| Embedding | 68.1 (0.08%) | 66.5 (0.11%) | 62.4 (0.16%) | +6.55% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.71% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6697.0 (10.61%) | 5284.7 (13.35%) | +26.72% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2387.8 (3.78%) | 2466.6 (6.23%) | -3.19% |
| 完整 Host wall | 80692.2 (100.00%) | 63099.3 (100.00%) | 39575.9 (100.00%) | +59.44% |


## Absmax A0 versus clipped A prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 13.000000 | 14.000000 | +7.6923% | 12.000000 | 12.000000 | +0.0000% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 76029.000000 | 76044.000000 | +0.0197% | 75939.500000 | 75807.000000 | -0.1745% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 123.000000 | 120.000000 | -2.4390% | 122.000000 | 122.500000 | +0.4098% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76204.000000 | 76221.000000 | +0.0223% | 76116.000000 | 75983.500000 | -0.1741% |
| attention_unattributed_ticks | 52.000000 | 57.000000 | +9.6154% | 56.000000 | 56.000000 | +0.0000% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 365.000000 | 360.000000 | -1.3699% | 374.000000 | 366.500000 | -2.0053% |
| boundary_ddr_read_bytes | 1423616.000000 | 1423616.000000 | +0.0000% | 1423616.000000 | 1423616.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 233.000000 | 233.000000 | +0.0000% | 233.000000 | 233.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 164968.000000 | 165857.000000 | +0.5389% | 164596.500000 | 165556.000000 | +0.5829% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9815.000000 | 9842.000000 | +0.2751% | 9854.500000 | 9867.000000 | +0.1268% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2691.000000 | 2693.000000 | +0.0743% | 2690.500000 | 2689.500000 | -0.0372% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 429401.000000 | 430390.000000 | +0.2303% | 430068.500000 | 431204.500000 | +0.2641% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1269.000000 | 1300.000000 | +2.4429% | 1265.000000 | 1291.500000 | +2.0949% |
| generation_final_norm_ticks | 921.000000 | 919.000000 | -0.2172% | 923.000000 | 920.500000 | -0.2709% |
| generation_lm_head_argmax_ticks | 6303.000000 | 6490.000000 | +2.9668% | 6736.000000 | 6617.000000 | -1.7666% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 127659.000000 | 128127.000000 | +0.3666% | 129647.000000 | 128885.500000 | -0.5874% |
| generation_lm_head_expand_ticks | 110821.000000 | 110733.000000 | -0.0794% | 112334.000000 | 111723.500000 | -0.5435% |
| generation_lm_head_hmx_tail_wait_ticks | 2218.000000 | 2592.000000 | +16.8620% | 2373.500000 | 2308.500000 | -2.7386% |
| generation_lm_head_hmx_ticks | 126656.000000 | 127111.000000 | +0.3592% | 128625.500000 | 127891.500000 | -0.5706% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 246.000000 | 247.000000 | +0.4065% | 238.500000 | 239.500000 | +0.4193% |
| generation_lm_head_scale_init_ticks | 569.000000 | 577.000000 | +1.4060% | 570.000000 | 572.000000 | +0.3509% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128580.000000 | 129046.000000 | +0.3624% | 130575.000000 | 129815.500000 | -0.5817% |
| generation_lm_head_weight_dma_ticks | 126756.000000 | 127244.000000 | +0.3850% | 128757.000000 | 128003.500000 | -0.5852% |
| generation_lm_head_weight_dma_wait_ticks | 2843.000000 | 2719.000000 | -4.3616% | 2833.000000 | 2779.000000 | -1.9061% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 132234.000000 | 138867.000000 | +5.0161% | 130513.000000 | 132904.000000 | +1.8320% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2343.020667 | 2324.270500 | -0.8003% | 2397.734167 | 2351.744375 | -1.9181% |
| host_us | 62992.604000 | 63064.583000 | +0.1143% | 63069.296500 | 63140.286500 | +0.1126% |
| host_wall_ns | 62992604.000000 | 63064583.000000 | +0.1143% | 63069296.500000 | 63140286.500000 | +0.1126% |
| input_norm_ticks | 9453.000000 | 9448.000000 | -0.0529% | 9449.000000 | 9429.000000 | -0.2117% |
| input_stage_ticks | 14.000000 | 12.000000 | -14.2857% | 12.000000 | 10.500000 | -12.5000% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1164472.000000 | 1166214.000000 | +0.1496% | 1165966.000000 | 1167288.500000 | +0.1134% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 438.000000 | 462.000000 | +5.4795% | 456.000000 | 460.000000 | +0.8772% |
| ledger_named_ticks | 1164472.000000 | 1166214.000000 | +0.1496% | 1165966.000000 | 1167288.500000 | +0.1134% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7320.000000 | 7359.000000 | +0.5328% | 7289.500000 | 7264.000000 | -0.3498% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 97160.000000 | 96588.000000 | -0.5887% | 96819.500000 | 97042.500000 | +0.2303% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 26.000000 | 23.000000 | -11.5385% | 24.000000 | 22.500000 | -6.2500% |
| post_attention_residual_ticks | 9119.000000 | 9056.000000 | -0.6909% | 9067.000000 | 9063.500000 | -0.0386% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684213.000000 | 685774.000000 | +0.2281% | 684094.500000 | 685816.500000 | +0.2517% |
| projection_pack_ticks | 3165.000000 | 3163.000000 | -0.0632% | 3163.500000 | 3168.500000 | +0.1581% |
| projection_unpack_ticks | 10926.000000 | 11131.000000 | +1.8763% | 10913.500000 | 11114.500000 | +1.8418% |
| qk_norm_rope_ticks | 23.000000 | 24.000000 | +4.3478% | 23.000000 | 23.500000 | +2.1739% |
| qkv_projection_ticks | 225609.000000 | 225602.000000 | -0.0031% | 225658.500000 | 225451.500000 | -0.0917% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 893.000000 | 900.000000 | +0.7839% | 907.500000 | 906.000000 | -0.1653% |
| runtime_teardown_ticks | 901.000000 | 823.000000 | -8.6570% | 901.000000 | 885.000000 | -1.7758% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6548.000000 | 6559.000000 | +0.1680% | 6554.000000 | 6565.000000 | +0.1678% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3298.000000 | 3316.000000 | +0.5458% | 3332.000000 | 3310.500000 | -0.6453% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 179.000000 | 161.000000 | -10.0559% | 178.500000 | 166.000000 | -7.0028% |
| total_ticks | 1163579.000000 | 1165314.000000 | +0.1491% | 1165079.000000 | 1166372.500000 | +0.1110% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 249879.000000 | 250546.000000 | +0.2669% | 249969.000000 | 250286.500000 | +0.1270% |
| w4f16_cross_prefetch_wait_ticks | 974.000000 | 1080.000000 | +10.8830% | 1017.000000 | 1093.000000 | +7.4730% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74466.000000 | 74339.000000 | -0.1705% | 76500.000000 | 76283.500000 | -0.2830% |
| w4f16_expand_ticks | 617054.000000 | 618845.000000 | +0.2903% | 615801.500000 | 618703.000000 | +0.4712% |
| w4f16_expand_work_ticks | 1693834.000000 | 1698385.000000 | +0.2687% | 1701779.500000 | 1700637.500000 | -0.0671% |
| w4f16_gate_up_expand_pool_wait_ticks | 30050.000000 | 29325.000000 | -2.4126% | 30153.500000 | 30138.000000 | -0.0514% |
| w4f16_gate_up_expand_ticks | 252363.000000 | 253520.000000 | +0.4585% | 252812.500000 | 254448.000000 | +0.6469% |
| w4f16_gate_up_expand_work_ticks | 158977.000000 | 160376.000000 | +0.8800% | 158933.000000 | 159970.500000 | +0.6528% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7737.000000 | 7724.000000 | -0.1680% | 7510.000000 | 7315.500000 | -2.5899% |
| w4f16_gate_up_hmx_wait_ticks | 278340.000000 | 279614.000000 | +0.4577% | 278905.500000 | 280285.500000 | +0.4948% |
| w4f16_gate_up_stream_join_wait_ticks | 620.000000 | 628.000000 | +1.2903% | 587.500000 | 564.500000 | -3.9149% |
| w4f16_gate_up_stream_ready_wait_ticks | 1340.000000 | 1080.000000 | -19.4030% | 1344.500000 | 1335.000000 | -0.7066% |
| w4f16_gate_up_stream_work_ticks | 134434.000000 | 134386.000000 | -0.0357% | 134434.000000 | 134406.000000 | -0.0208% |
| w4f16_gate_up_weight_dma_ticks | 558427.000000 | 560540.000000 | +0.3784% | 559968.500000 | 562429.000000 | +0.4394% |
| w4f16_hmx_tail_wait_ticks | 19685.000000 | 19530.000000 | -0.7874% | 19262.000000 | 18925.000000 | -1.7496% |
| w4f16_prefetch_wait_ticks | 28522.000000 | 28715.000000 | +0.6767% | 30497.500000 | 30135.500000 | -1.1870% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1138305.000000 | 1140443.000000 | +0.1878% | 1140761.500000 | 1143186.500000 | +0.2126% |


## Absmax A0 versus clipped A decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 8.600000 | 9.800000 | +13.9535% | 8.866667 | 8.433333 | -4.8872% |
| attention_av_hmx_ticks | 8007.466667 | 8163.200000 | +1.9449% | 7848.666667 | 7819.600000 | -0.3703% |
| attention_av_pack_ticks | 3032.533333 | 3032.200000 | -0.0110% | 3037.433333 | 3028.633333 | -0.2897% |
| attention_av_unpack_ticks | 3994.400000 | 3999.333333 | +0.1235% | 4001.133333 | 3999.200000 | -0.0483% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8065.933333 | 8186.666667 | +1.4968% | 7941.700000 | 7927.466667 | -0.1792% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6392.666667 | 6375.200000 | -0.2732% | 6384.033333 | 6381.200000 | -0.0444% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132873.600000 | 133394.466667 | +0.3920% | 132830.733333 | 133392.833333 | +0.4232% |
| attention_ticks | 644199.133333 | 644994.600000 | +0.1235% | 643758.300000 | 644273.633333 | +0.0801% |
| attention_unattributed_ticks | 481832.533333 | 481843.533333 | +0.0023% | 481774.400000 | 481748.266667 | -0.0054% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 282.800000 | 286.466667 | +1.2966% | 285.866667 | 285.633333 | -0.0816% |
| boundary_ddr_read_bytes | 1165568.000000 | 1165568.000000 | +0.0000% | 1165568.000000 | 1165568.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 170.000000 | 170.000000 | +0.0000% | 170.000000 | 170.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 165035.400000 | 165973.466667 | +0.5684% | 164740.666667 | 166039.666667 | +0.7885% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29943.466667 | 29938.600000 | -0.0163% | 29917.933333 | 29921.533333 | +0.0120% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2671.533333 | 2669.800000 | -0.0649% | 2671.200000 | 2670.700000 | -0.0187% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 428078.600000 | 428721.600000 | +0.1502% | 428909.700000 | 429954.433333 | +0.2436% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 37.266667 | 41.133333 | +10.3757% | 36.800000 | 36.200000 | -1.6304% |
| generation_final_norm_ticks | 43.200000 | 41.200000 | -4.6296% | 40.933333 | 40.966667 | +0.0814% |
| generation_lm_head_argmax_ticks | 6312.400000 | 6572.466667 | +4.1199% | 6743.433333 | 6645.133333 | -1.4577% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 127538.200000 | 128716.733333 | +0.9241% | 129501.733333 | 128796.700000 | -0.5444% |
| generation_lm_head_expand_ticks | 110809.800000 | 111666.933333 | +0.7735% | 112305.900000 | 111720.366667 | -0.5214% |
| generation_lm_head_hmx_tail_wait_ticks | 2154.866667 | 2280.266667 | +5.8194% | 2272.400000 | 2276.700000 | +0.1892% |
| generation_lm_head_hmx_ticks | 126612.866667 | 127786.333333 | +0.9268% | 128567.233333 | 127875.133333 | -0.5383% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 211.933333 | 215.200000 | +1.5414% | 210.633333 | 204.266667 | -3.0226% |
| generation_lm_head_scale_init_ticks | 571.733333 | 570.066667 | -0.2915% | 568.333333 | 567.066667 | -0.2229% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127581.400000 | 128757.933333 | +0.9222% | 129543.000000 | 128837.600000 | -0.5445% |
| generation_lm_head_weight_dma_ticks | 126719.533333 | 127887.600000 | +0.9218% | 128676.100000 | 127977.700000 | -0.5428% |
| generation_lm_head_weight_dma_wait_ticks | 2905.400000 | 2883.600000 | -0.7503% | 2806.600000 | 2810.733333 | +0.1473% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 129078.666667 | 131765.000000 | +2.0812% | 124424.733333 | 127144.666667 | +2.1860% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2097.503383 | 1905.062511 | -9.1748% | 1693.630189 | 1640.921839 | -3.1122% |
| host_us | 92778.847133 | 92745.368067 | -0.0361% | 92438.574633 | 92521.045167 | +0.0892% |
| host_wall_ns | 92778847.133333 | 92745368.066667 | -0.0361% | 92438574.633333 | 92521045.166667 | +0.0892% |
| input_norm_ticks | 9368.400000 | 9353.266667 | -0.1615% | 9366.900000 | 9363.233333 | -0.0391% |
| input_stage_ticks | 7.400000 | 8.800000 | +18.9189% | 7.766667 | 8.066667 | +3.8627% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741081.800000 | 1744133.866667 | +0.1753% | 1742639.366667 | 1744453.333333 | +0.1041% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 322.333333 | 322.000000 | -0.1034% | 322.366667 | 321.633333 | -0.2275% |
| ledger_named_ticks | 1741081.800000 | 1744133.866667 | +0.1753% | 1742639.366667 | 1744453.333333 | +0.1041% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6024.000000 | 6038.466667 | +0.2402% | 5825.666667 | 5771.100000 | -0.9367% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96556.600000 | 96321.333333 | -0.2437% | 96605.000000 | 96563.900000 | -0.0425% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 17.800000 | 18.733333 | +5.2434% | 17.200000 | 17.066667 | -0.7752% |
| post_attention_residual_ticks | 9022.133333 | 9013.800000 | -0.0924% | 9016.400000 | 9015.766667 | -0.0070% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682428.333333 | 683774.333333 | +0.1972% | 682896.533333 | 685410.500000 | +0.3681% |
| projection_pack_ticks | 421.000000 | 421.466667 | +0.1108% | 420.033333 | 419.733333 | -0.0714% |
| projection_unpack_ticks | 11030.000000 | 11173.000000 | +1.2965% | 10953.733333 | 11269.366667 | +2.8815% |
| qk_norm_rope_ticks | 17.933333 | 16.933333 | -5.5762% | 17.400000 | 17.333333 | -0.3831% |
| qkv_projection_ticks | 225016.933333 | 224761.266667 | -0.1136% | 224894.600000 | 224704.333333 | -0.0846% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 623.533333 | 615.733333 | -1.2509% | 621.000000 | 618.500000 | -0.4026% |
| runtime_teardown_ticks | 578.000000 | 577.866667 | -0.0231% | 576.200000 | 578.100000 | +0.3297% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4275.600000 | 4250.200000 | -0.5941% | 4246.866667 | 4248.566667 | +0.0400% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21314.666667 | 21335.400000 | +0.0973% | 21327.366667 | 21336.466667 | +0.0427% |
| scan_cache_stage_ticks | 13789.266667 | 13784.066667 | -0.0377% | 13766.866667 | 13756.166667 | -0.0777% |
| scan_dynamic_attention_ticks | 644146.266667 | 644944.333333 | +0.1239% | 643709.100000 | 644224.300000 | +0.0800% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 41.733333 | 45.266667 | +8.4665% | 38.033333 | 39.933333 | +4.9956% |
| total_ticks | 1740458.266667 | 1743518.133333 | +0.1758% | 1742017.600000 | 1743835.866667 | +0.1044% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 833786.400000 | 834559.866667 | +0.0928% | 833107.733333 | 833732.233333 | +0.0750% |
| w4f16_cross_prefetch_wait_ticks | 1125.866667 | 1087.400000 | -3.4166% | 1115.233333 | 1117.833333 | +0.2331% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74838.933333 | 76952.266667 | +2.8238% | 77436.033333 | 77203.766667 | -0.2999% |
| w4f16_expand_ticks | 615543.933333 | 617736.666667 | +0.3562% | 615227.066667 | 617929.100000 | +0.4392% |
| w4f16_expand_work_ticks | 1692173.400000 | 1696875.866667 | +0.2779% | 1700404.300000 | 1697175.566667 | -0.1899% |
| w4f16_gate_up_expand_pool_wait_ticks | 30330.600000 | 30788.266667 | +1.5089% | 31087.433333 | 31355.100000 | +0.8610% |
| w4f16_gate_up_expand_ticks | 251609.466667 | 252686.533333 | +0.4281% | 252504.466667 | 253662.166667 | +0.4585% |
| w4f16_gate_up_expand_work_ticks | 158033.200000 | 157941.933333 | -0.0578% | 158019.333333 | 157981.066667 | -0.0242% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7663.333333 | 7583.333333 | -1.0439% | 7431.166667 | 7387.566667 | -0.5867% |
| w4f16_gate_up_hmx_wait_ticks | 277362.133333 | 278021.600000 | +0.2378% | 278416.366667 | 279469.133333 | +0.3781% |
| w4f16_gate_up_stream_join_wait_ticks | 782.400000 | 785.266667 | +0.3664% | 639.133333 | 645.366667 | +0.9753% |
| w4f16_gate_up_stream_ready_wait_ticks | 1237.466667 | 1257.133333 | +1.5893% | 1305.233333 | 1280.033333 | -1.9307% |
| w4f16_gate_up_stream_work_ticks | 134384.733333 | 134393.533333 | +0.0065% | 134397.533333 | 134386.800000 | -0.0080% |
| w4f16_gate_up_weight_dma_ticks | 556208.066667 | 557482.400000 | +0.2291% | 558297.366667 | 560389.533333 | +0.3747% |
| w4f16_hmx_tail_wait_ticks | 19429.200000 | 19359.866667 | -0.3569% | 18977.900000 | 18815.133333 | -0.8577% |
| w4f16_prefetch_wait_ticks | 28655.133333 | 28055.400000 | -2.0929% | 30146.000000 | 29836.966667 | -1.0251% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1132261.733333 | 1135075.466667 | +0.2485% | 1135630.900000 | 1137985.833333 | +0.2074% |


## Absmax B0 versus clipped B prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 19.000000 | 9.000000 | -52.6316% | 12.000000 | 12.000000 | +0.0000% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75851.000000 | 75881.000000 | +0.0396% | 75847.500000 | 75865.500000 | +0.0237% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 124.000000 | 120.000000 | -3.2258% | 123.000000 | 120.500000 | -2.0325% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76036.000000 | 76059.000000 | +0.0302% | 76028.500000 | 76041.500000 | +0.0171% |
| attention_unattributed_ticks | 61.000000 | 58.000000 | -4.9180% | 57.500000 | 56.000000 | -2.6087% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 431.000000 | 371.000000 | -13.9211% | 367.000000 | 369.500000 | +0.6812% |
| boundary_ddr_read_bytes | 1423616.000000 | 1423616.000000 | +0.0000% | 1423616.000000 | 1423616.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 233.000000 | 233.000000 | +0.0000% | 233.000000 | 233.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 164115.000000 | 167209.000000 | +1.8853% | 164252.500000 | 165950.000000 | +1.0335% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9796.000000 | 9772.000000 | -0.2450% | 9848.500000 | 9837.500000 | -0.1117% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2691.000000 | 2685.000000 | -0.2230% | 2691.500000 | 2690.000000 | -0.0557% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 428271.000000 | 430251.000000 | +0.4623% | 429131.500000 | 430196.000000 | +0.2481% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1345.000000 | 1301.000000 | -3.2714% | 1275.500000 | 1277.500000 | +0.1568% |
| generation_final_norm_ticks | 928.000000 | 914.000000 | -1.5086% | 924.500000 | 920.000000 | -0.4867% |
| generation_lm_head_argmax_ticks | 6753.000000 | 6791.000000 | +0.5627% | 6618.000000 | 6656.500000 | +0.5817% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128866.000000 | 129190.000000 | +0.2514% | 128020.500000 | 128583.000000 | +0.4394% |
| generation_lm_head_expand_ticks | 111885.000000 | 111906.000000 | +0.0188% | 111097.000000 | 111617.500000 | +0.4685% |
| generation_lm_head_hmx_tail_wait_ticks | 2320.000000 | 2503.000000 | +7.8879% | 2353.000000 | 2411.000000 | +2.4649% |
| generation_lm_head_hmx_ticks | 127761.000000 | 128184.000000 | +0.3311% | 127000.500000 | 127575.000000 | +0.4524% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 228.000000 | 232.000000 | +1.7544% | 236.000000 | 237.500000 | +0.6356% |
| generation_lm_head_scale_init_ticks | 555.000000 | 557.000000 | +0.3604% | 565.500000 | 562.000000 | -0.6189% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129794.000000 | 130104.000000 | +0.2388% | 128941.000000 | 129498.500000 | +0.4324% |
| generation_lm_head_weight_dma_ticks | 127968.000000 | 128297.000000 | +0.2571% | 127130.000000 | 127708.500000 | +0.4550% |
| generation_lm_head_weight_dma_wait_ticks | 2431.000000 | 2586.000000 | +6.3760% | 2548.500000 | 2543.500000 | -0.1962% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 130349.000000 | 137337.000000 | +5.3610% | 126098.500000 | 135901.500000 | +7.7741% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2289.062583 | 2727.551833 | +19.1558% | 2252.760000 | 2387.838167 | +5.9961% |
| host_us | 62904.323000 | 63598.906000 | +1.1042% | 62837.109000 | 63099.348500 | +0.4173% |
| host_wall_ns | 62904323.000000 | 63598906.000000 | +1.1042% | 62837109.000000 | 63099348.500000 | +0.4173% |
| input_norm_ticks | 9485.000000 | 9439.000000 | -0.4850% | 9439.500000 | 9436.000000 | -0.0371% |
| input_stage_ticks | 12.000000 | 10.000000 | -16.6667% | 11.000000 | 10.000000 | -9.0909% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1163813.000000 | 1168730.000000 | +0.4225% | 1163747.500000 | 1167045.500000 | +0.2834% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 459.000000 | 432.000000 | -5.8824% | 458.000000 | 455.000000 | -0.6550% |
| ledger_named_ticks | 1163813.000000 | 1168730.000000 | +0.4225% | 1163747.500000 | 1167045.500000 | +0.2834% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7427.000000 | 7384.000000 | -0.5790% | 7197.000000 | 7302.500000 | +1.4659% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96834.000000 | 97109.000000 | +0.2840% | 96575.000000 | 97100.000000 | +0.5436% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 25.000000 | 20.000000 | -20.0000% | 22.000000 | 22.000000 | +0.0000% |
| post_attention_residual_ticks | 9073.000000 | 9090.000000 | +0.1874% | 9070.000000 | 9085.000000 | +0.1654% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 680827.000000 | 686756.000000 | +0.8709% | 682196.000000 | 685497.000000 | +0.4839% |
| projection_pack_ticks | 3166.000000 | 3169.000000 | +0.0948% | 3166.500000 | 3168.500000 | +0.0632% |
| projection_unpack_ticks | 10835.000000 | 11454.000000 | +5.7130% | 10898.500000 | 11155.000000 | +2.3535% |
| qk_norm_rope_ticks | 33.000000 | 27.000000 | -18.1818% | 23.000000 | 24.500000 | +6.5217% |
| qkv_projection_ticks | 225820.000000 | 225475.000000 | -0.1528% | 225774.500000 | 225544.500000 | -0.1019% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 981.000000 | 899.000000 | -8.3588% | 895.500000 | 912.000000 | +1.8425% |
| runtime_teardown_ticks | 926.000000 | 895.000000 | -3.3477% | 871.500000 | 887.500000 | +1.8359% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6520.000000 | 6505.000000 | -0.2301% | 6551.500000 | 6552.500000 | +0.0153% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3308.000000 | 3295.000000 | -0.3930% | 3319.000000 | 3300.500000 | -0.5574% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 208.000000 | 161.000000 | -22.5962% | 166.500000 | 171.000000 | +2.7027% |
| total_ticks | 1162832.000000 | 1167831.000000 | +0.4299% | 1162810.500000 | 1166136.000000 | +0.2860% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 250856.000000 | 249502.000000 | -0.5398% | 250409.000000 | 250083.500000 | -0.1300% |
| w4f16_cross_prefetch_wait_ticks | 966.000000 | 1019.000000 | +5.4865% | 1073.500000 | 1104.500000 | +2.8878% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 76753.000000 | 78004.000000 | +1.6299% | 76765.000000 | 76977.500000 | +0.2768% |
| w4f16_expand_ticks | 613488.000000 | 619456.000000 | +0.9728% | 614443.500000 | 618106.000000 | +0.5961% |
| w4f16_expand_work_ticks | 1701246.000000 | 1703394.000000 | +0.1263% | 1698451.000000 | 1702550.000000 | +0.2413% |
| w4f16_gate_up_expand_pool_wait_ticks | 30155.000000 | 29943.000000 | -0.7030% | 30417.500000 | 30025.000000 | -1.2904% |
| w4f16_gate_up_expand_ticks | 251177.000000 | 253232.000000 | +0.8181% | 252568.000000 | 253286.500000 | +0.2845% |
| w4f16_gate_up_expand_work_ticks | 159202.000000 | 159573.000000 | +0.2330% | 159648.500000 | 159727.000000 | +0.0492% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7530.000000 | 7226.000000 | -4.0372% | 7151.500000 | 7397.000000 | +3.4328% |
| w4f16_gate_up_hmx_wait_ticks | 276936.000000 | 279136.000000 | +0.7944% | 278196.000000 | 279183.000000 | +0.3548% |
| w4f16_gate_up_stream_join_wait_ticks | 830.000000 | 721.000000 | -13.1325% | 635.000000 | 663.500000 | +4.4882% |
| w4f16_gate_up_stream_ready_wait_ticks | 1305.000000 | 1512.000000 | +15.8621% | 1312.500000 | 1278.500000 | -2.5905% |
| w4f16_gate_up_stream_work_ticks | 134387.000000 | 134342.000000 | -0.0335% | 134407.500000 | 134418.000000 | +0.0078% |
| w4f16_gate_up_weight_dma_ticks | 556052.000000 | 560515.000000 | +0.8026% | 558346.000000 | 560353.000000 | +0.3595% |
| w4f16_hmx_tail_wait_ticks | 19628.000000 | 19141.000000 | -2.4811% | 18674.500000 | 19070.500000 | +2.1205% |
| w4f16_prefetch_wait_ticks | 29099.000000 | 29106.000000 | +0.0241% | 30383.500000 | 29582.500000 | -2.6363% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1135378.000000 | 1143807.000000 | +0.7424% | 1136995.500000 | 1141041.000000 | +0.3558% |


## Absmax B0 versus clipped B decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.200000 | 9.000000 | -2.1739% | 8.700000 | 8.700000 | +0.0000% |
| attention_av_hmx_ticks | 7948.733333 | 8187.666667 | +3.0059% | 7888.400000 | 7934.533333 | +0.5848% |
| attention_av_pack_ticks | 3055.600000 | 3045.400000 | -0.3338% | 3034.966667 | 3038.333333 | +0.1109% |
| attention_av_unpack_ticks | 4000.533333 | 4007.933333 | +0.1850% | 4000.233333 | 4003.766667 | +0.0883% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8023.400000 | 8292.266667 | +3.3510% | 8018.666667 | 8017.866667 | -0.0100% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6404.666667 | 6374.133333 | -0.4767% | 6387.100000 | 6386.066667 | -0.0162% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 133136.533333 | 132641.666667 | -0.3717% | 133139.700000 | 132625.133333 | -0.3865% |
| attention_ticks | 644437.200000 | 644396.666667 | -0.0063% | 644108.966667 | 643806.533333 | -0.0470% |
| attention_unattributed_ticks | 481867.733333 | 481847.600000 | -0.0042% | 481738.733333 | 481779.900000 | +0.0085% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 283.866667 | 278.000000 | -2.0667% | 286.000000 | 284.066667 | -0.6760% |
| boundary_ddr_read_bytes | 1165568.000000 | 1165568.000000 | +0.0000% | 1165568.000000 | 1165568.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 170.000000 | 170.000000 | +0.0000% | 170.000000 | 170.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 164666.333333 | 166612.666667 | +1.1820% | 164624.900000 | 166191.966667 | +0.9519% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29980.333333 | 29935.533333 | -0.1494% | 29927.800000 | 29928.233333 | +0.0014% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2670.066667 | 2672.066667 | +0.0749% | 2671.100000 | 2670.900000 | -0.0075% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 427297.000000 | 428882.866667 | +0.3711% | 428008.566667 | 429234.033333 | +0.2863% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 40.200000 | 37.666667 | -6.3018% | 36.700000 | 37.633333 | +2.5431% |
| generation_final_norm_ticks | 42.800000 | 40.866667 | -4.5171% | 40.766667 | 40.833333 | +0.1635% |
| generation_lm_head_argmax_ticks | 6605.666667 | 6742.000000 | +2.0639% | 6614.600000 | 6673.566667 | +0.8915% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 127976.333333 | 128880.400000 | +0.7064% | 127834.300000 | 128438.833333 | +0.4729% |
| generation_lm_head_expand_ticks | 111308.200000 | 111963.866667 | +0.5891% | 111122.633333 | 111618.600000 | +0.4463% |
| generation_lm_head_hmx_tail_wait_ticks | 2275.333333 | 2355.666667 | +3.5306% | 2267.033333 | 2313.566667 | +2.0526% |
| generation_lm_head_hmx_ticks | 127060.866667 | 127944.800000 | +0.6957% | 126913.100000 | 127512.833333 | +0.4726% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 204.666667 | 211.466667 | +3.3225% | 205.966667 | 209.100000 | +1.5213% |
| generation_lm_head_scale_init_ticks | 568.266667 | 568.200000 | -0.0117% | 568.833333 | 568.366667 | -0.0820% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128019.133333 | 128921.266667 | +0.7047% | 127876.033333 | 128481.333333 | +0.4733% |
| generation_lm_head_weight_dma_ticks | 127152.266667 | 128057.933333 | +0.7123% | 127008.566667 | 127615.466667 | +0.4778% |
| generation_lm_head_weight_dma_wait_ticks | 2533.666667 | 2493.133333 | -1.5998% | 2535.133333 | 2505.366667 | -1.1742% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 123880.866667 | 130855.200000 | +5.6299% | 119597.766667 | 128757.133333 | +7.6585% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1777.364661 | 1895.579678 | +6.6511% | 1740.142422 | 1715.031178 | -1.4431% |
| host_us | 92407.784800 | 92795.239400 | +0.4193% | 92312.566000 | 92510.418400 | +0.2143% |
| host_wall_ns | 92407784.800000 | 92795239.400000 | +0.4193% | 92312566.000000 | 92510418.400000 | +0.2143% |
| input_norm_ticks | 9344.600000 | 9360.000000 | +0.1648% | 9359.066667 | 9360.166667 | +0.0118% |
| input_stage_ticks | 7.933333 | 7.866667 | -0.8403% | 8.433333 | 8.000000 | -5.1383% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1740104.066667 | 1745273.466667 | +0.2971% | 1740139.900000 | 1743060.766667 | +0.1679% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 320.133333 | 326.133333 | +1.8742% | 323.266667 | 323.166667 | -0.0309% |
| ledger_named_ticks | 1740104.066667 | 1745273.466667 | +0.2971% | 1740139.900000 | 1743060.766667 | +0.1679% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6033.866667 | 6013.133333 | -0.3436% | 5801.966667 | 5826.666667 | +0.4257% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96423.466667 | 96694.866667 | +0.2815% | 96440.300000 | 96637.266667 | +0.2042% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 16.933333 | 17.133333 | +1.1811% | 16.800000 | 16.800000 | +0.0000% |
| post_attention_residual_ticks | 9002.066667 | 9013.400000 | +0.1259% | 9013.200000 | 9012.400000 | -0.0089% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 680029.400000 | 684676.733333 | +0.6834% | 680903.233333 | 684025.300000 | +0.4585% |
| projection_pack_ticks | 416.333333 | 421.933333 | +1.3451% | 420.400000 | 419.666667 | -0.1744% |
| projection_unpack_ticks | 10953.400000 | 11270.866667 | +2.8983% | 10957.966667 | 11259.566667 | +2.7523% |
| qk_norm_rope_ticks | 17.600000 | 19.066667 | +8.3333% | 17.300000 | 18.000000 | +4.0462% |
| qkv_projection_ticks | 224704.866667 | 225188.133333 | +0.2151% | 224745.900000 | 225001.400000 | +0.1137% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 623.733333 | 621.200000 | -0.4062% | 620.233333 | 619.400000 | -0.1344% |
| runtime_teardown_ticks | 573.000000 | 578.933333 | +1.0355% | 575.800000 | 575.900000 | +0.0174% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4253.933333 | 4258.000000 | +0.0956% | 4249.866667 | 4244.266667 | -0.1318% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21319.200000 | 21329.466667 | +0.0482% | 21323.533333 | 21334.800000 | +0.0528% |
| scan_cache_stage_ticks | 13753.066667 | 13806.533333 | +0.3888% | 13747.366667 | 13769.800000 | +0.1632% |
| scan_dynamic_attention_ticks | 644386.266667 | 644345.200000 | -0.0064% | 644059.666667 | 643757.300000 | -0.0469% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 39.733333 | 35.933333 | -9.5638% | 39.266667 | 36.900000 | -6.0272% |
| total_ticks | 1739480.333333 | 1744652.266667 | +0.2973% | 1739518.200000 | 1742439.366667 | +0.1679% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 834418.800000 | 833863.466667 | -0.0666% | 833511.300000 | 833230.700000 | -0.0337% |
| w4f16_cross_prefetch_wait_ticks | 1066.933333 | 1106.066667 | +3.6678% | 1130.066667 | 1109.266667 | -1.8406% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 77023.266667 | 78377.133333 | +1.7577% | 77240.633333 | 77939.366667 | +0.9046% |
| w4f16_expand_ticks | 613717.600000 | 617392.266667 | +0.5988% | 613706.400000 | 616991.600000 | +0.5353% |
| w4f16_expand_work_ticks | 1699254.933333 | 1700554.400000 | +0.0765% | 1698011.866667 | 1700447.566667 | +0.1434% |
| w4f16_gate_up_expand_pool_wait_ticks | 30628.333333 | 30841.800000 | +0.6970% | 31429.933333 | 30916.200000 | -1.6345% |
| w4f16_gate_up_expand_ticks | 251319.333333 | 252432.400000 | +0.4429% | 251912.800000 | 252917.700000 | +0.3989% |
| w4f16_gate_up_expand_work_ticks | 158926.733333 | 157905.866667 | -0.6424% | 158351.233333 | 158264.033333 | -0.0551% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7300.333333 | 7518.666667 | +2.9907% | 7161.733333 | 7420.566667 | +3.6141% |
| w4f16_gate_up_hmx_wait_ticks | 276530.666667 | 278184.133333 | +0.5979% | 277454.200000 | 278730.466667 | +0.4600% |
| w4f16_gate_up_stream_join_wait_ticks | 807.400000 | 702.400000 | -13.0047% | 666.966667 | 643.966667 | -3.4484% |
| w4f16_gate_up_stream_ready_wait_ticks | 1325.866667 | 1465.666667 | +10.5440% | 1306.600000 | 1290.466667 | -1.2348% |
| w4f16_gate_up_stream_work_ticks | 134412.466667 | 134370.866667 | -0.0309% | 134413.100000 | 134404.500000 | -0.0064% |
| w4f16_gate_up_weight_dma_ticks | 554631.466667 | 558113.666667 | +0.6278% | 556586.766667 | 559029.500000 | +0.4389% |
| w4f16_hmx_tail_wait_ticks | 19232.066667 | 19452.866667 | +1.1481% | 18678.800000 | 19042.166667 | +1.9453% |
| w4f16_prefetch_wait_ticks | 28712.333333 | 28828.400000 | +0.4042% | 29986.466667 | 29880.200000 | -0.3544% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1130010.400000 | 1137206.066667 | +0.6368% | 1131737.000000 | 1135997.600000 | +0.3765% |


## Absmax C0 versus clipped C prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 8.000000 | 7.000000 | -12.5000% | 9.500000 | 11.500000 | +21.0526% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75839.000000 | 76076.000000 | +0.3125% | 75837.500000 | 75847.000000 | +0.0125% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 120.000000 | 118.000000 | -1.6667% | 122.000000 | 120.000000 | -1.6393% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76015.000000 | 76255.000000 | +0.3157% | 76017.000000 | 76022.000000 | +0.0066% |
| attention_unattributed_ticks | 56.000000 | 61.000000 | +8.9286% | 55.500000 | 56.000000 | +0.9009% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 367.000000 | 386.000000 | +5.1771% | 377.000000 | 375.000000 | -0.5305% |
| boundary_ddr_read_bytes | 1423616.000000 | 1423616.000000 | +0.0000% | 1423616.000000 | 1423616.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 233.000000 | 233.000000 | +0.0000% | 233.000000 | 233.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 165496.000000 | 165663.000000 | +0.1009% | 165543.000000 | 165970.000000 | +0.2579% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9940.000000 | 9942.000000 | +0.0201% | 9872.000000 | 9857.500000 | -0.1469% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2690.000000 | 2690.000000 | +0.0000% | 2689.500000 | 2690.000000 | +0.0186% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 430491.000000 | 431490.000000 | +0.2321% | 430713.000000 | 431363.000000 | +0.1509% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1329.000000 | 1299.000000 | -2.2573% | 1280.500000 | 1273.500000 | -0.5467% |
| generation_final_norm_ticks | 913.000000 | 923.000000 | +1.0953% | 921.500000 | 923.500000 | +0.2170% |
| generation_lm_head_argmax_ticks | 7012.000000 | 6918.000000 | -1.3406% | 6907.000000 | 6793.500000 | -1.6433% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 130918.000000 | 130194.000000 | -0.5530% | 129808.000000 | 129830.000000 | +0.0169% |
| generation_lm_head_expand_ticks | 112921.000000 | 112565.000000 | -0.3153% | 112143.000000 | 112581.000000 | +0.3906% |
| generation_lm_head_hmx_tail_wait_ticks | 2593.000000 | 2375.000000 | -8.4073% | 2460.500000 | 2394.500000 | -2.6824% |
| generation_lm_head_hmx_ticks | 129902.000000 | 129173.000000 | -0.5612% | 128757.500000 | 128821.000000 | +0.0493% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 236.000000 | 247.000000 | +4.6610% | 238.000000 | 233.500000 | -1.8908% |
| generation_lm_head_scale_init_ticks | 565.000000 | 570.000000 | +0.8850% | 568.000000 | 572.500000 | +0.7923% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 131831.000000 | 131117.000000 | -0.5416% | 130741.500000 | 130755.500000 | +0.0107% |
| generation_lm_head_weight_dma_ticks | 130025.000000 | 129296.000000 | -0.5607% | 128905.500000 | 128945.000000 | +0.0306% |
| generation_lm_head_weight_dma_wait_ticks | 2912.000000 | 2776.000000 | -4.6703% | 2916.500000 | 2824.500000 | -3.1545% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 131124.000000 | 135542.000000 | +3.3693% | 131255.000000 | 133766.000000 | +1.9131% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 733.021000 | 2055.781500 | +180.4533% | 2260.234125 | 2268.984250 | +0.3871% |
| host_us | 61702.396000 | 62977.969000 | +2.0673% | 63101.198000 | 63242.369500 | +0.2237% |
| host_wall_ns | 61702396.000000 | 62977969.000000 | +2.0673% | 63101198.000000 | 63242369.500000 | +0.2237% |
| input_norm_ticks | 9433.000000 | 9450.000000 | +0.1802% | 9441.500000 | 9448.500000 | +0.0741% |
| input_stage_ticks | 12.000000 | 5.000000 | -58.3333% | 11.500000 | 9.000000 | -21.7391% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1170612.000000 | 1169706.000000 | -0.0774% | 1168322.500000 | 1169468.000000 | +0.0980% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 466.000000 | 467.000000 | +0.2146% | 465.500000 | 465.500000 | +0.0000% |
| ledger_named_ticks | 1170612.000000 | 1169706.000000 | -0.0774% | 1168322.500000 | 1169468.000000 | +0.0980% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7855.000000 | 7655.000000 | -2.5461% | 7259.000000 | 7263.500000 | +0.0620% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 97345.000000 | 96576.000000 | -0.7900% | 97023.000000 | 96921.000000 | -0.1051% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 25.000000 | 18.000000 | -28.0000% | 21.500000 | 23.000000 | +6.9767% |
| post_attention_residual_ticks | 9066.000000 | 9098.000000 | +0.3530% | 9069.000000 | 9079.500000 | +0.1158% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 686370.000000 | 687057.000000 | +0.1001% | 686135.500000 | 686755.000000 | +0.0903% |
| projection_pack_ticks | 3158.000000 | 3169.000000 | +0.3483% | 3165.000000 | 3169.500000 | +0.1422% |
| projection_unpack_ticks | 10955.000000 | 11176.000000 | +2.0173% | 11156.000000 | 11211.000000 | +0.4930% |
| qk_norm_rope_ticks | 26.000000 | 29.000000 | +11.5385% | 23.000000 | 24.000000 | +4.3478% |
| qkv_projection_ticks | 226208.000000 | 225548.000000 | -0.2918% | 225464.000000 | 225521.500000 | +0.0255% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 923.000000 | 925.000000 | +0.2167% | 916.000000 | 911.500000 | -0.4913% |
| runtime_teardown_ticks | 876.000000 | 884.000000 | +0.9132% | 873.500000 | 896.500000 | +2.6331% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6606.000000 | 6600.000000 | -0.0908% | 6562.000000 | 6569.000000 | +0.1067% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3362.000000 | 3374.000000 | +0.3569% | 3341.500000 | 3326.000000 | -0.4639% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 182.000000 | 170.000000 | -6.5934% | 171.500000 | 171.000000 | -0.2915% |
| total_ticks | 1169689.000000 | 1168781.000000 | -0.0776% | 1167363.000000 | 1168577.000000 | +0.1040% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 250408.000000 | 250031.000000 | -0.1506% | 250008.000000 | 249692.000000 | -0.1264% |
| w4f16_cross_prefetch_wait_ticks | 930.000000 | 1045.000000 | +12.3656% | 1087.000000 | 1085.500000 | -0.1380% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 78897.000000 | 76646.000000 | -2.8531% | 77332.000000 | 77139.500000 | -0.2489% |
| w4f16_expand_ticks | 618756.000000 | 619011.000000 | +0.0412% | 618325.500000 | 619550.000000 | +0.1980% |
| w4f16_expand_work_ticks | 1704135.000000 | 1705063.000000 | +0.0545% | 1702243.000000 | 1703347.000000 | +0.0649% |
| w4f16_gate_up_expand_pool_wait_ticks | 31066.000000 | 29802.000000 | -4.0688% | 30765.500000 | 30438.500000 | -1.0629% |
| w4f16_gate_up_expand_ticks | 253545.000000 | 254609.000000 | +0.4196% | 254043.000000 | 254491.000000 | +0.1763% |
| w4f16_gate_up_expand_work_ticks | 158557.000000 | 160448.000000 | +1.1926% | 159443.000000 | 159591.500000 | +0.0931% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7647.000000 | 7353.000000 | -3.8446% | 7214.000000 | 7374.500000 | +2.2248% |
| w4f16_gate_up_hmx_wait_ticks | 279386.000000 | 280679.000000 | +0.4628% | 279662.500000 | 280371.000000 | +0.2533% |
| w4f16_gate_up_stream_join_wait_ticks | 514.000000 | 556.000000 | +8.1712% | 637.000000 | 556.000000 | -12.7159% |
| w4f16_gate_up_stream_ready_wait_ticks | 1128.000000 | 1166.000000 | +3.3688% | 1233.000000 | 1369.500000 | +11.0706% |
| w4f16_gate_up_stream_work_ticks | 134449.000000 | 134421.000000 | -0.0208% | 134440.500000 | 134422.500000 | -0.0134% |
| w4f16_gate_up_weight_dma_ticks | 560773.000000 | 562970.000000 | +0.3918% | 561499.000000 | 562741.000000 | +0.2212% |
| w4f16_hmx_tail_wait_ticks | 19805.000000 | 19111.000000 | -3.5042% | 18766.000000 | 18695.000000 | -0.3783% |
| w4f16_prefetch_wait_ticks | 29055.000000 | 29962.000000 | +3.1217% | 29742.000000 | 30189.000000 | +1.5029% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1145172.000000 | 1145760.000000 | +0.0513% | 1142978.500000 | 1145023.500000 | +0.1789% |


## Absmax C0 versus clipped C decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.600000 | 8.933333 | -6.9444% | 9.066667 | 8.966667 | -1.1029% |
| attention_av_hmx_ticks | 7671.266667 | 7897.333333 | +2.9469% | 7916.466667 | 7853.700000 | -0.7929% |
| attention_av_pack_ticks | 3064.466667 | 3026.000000 | -1.2552% | 3040.400000 | 3033.533333 | -0.2258% |
| attention_av_unpack_ticks | 3954.266667 | 4002.600000 | +1.2223% | 3999.400000 | 4002.766667 | +0.0842% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 7637.733333 | 8011.000000 | +4.8871% | 8007.366667 | 7942.800000 | -0.8063% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6388.333333 | 6389.133333 | +0.0125% | 6384.533333 | 6377.533333 | -0.1096% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132789.400000 | 132725.866667 | -0.0478% | 132854.166667 | 132751.600000 | -0.0772% |
| attention_ticks | 642815.000000 | 643800.133333 | +0.1533% | 644028.500000 | 643742.766667 | -0.0444% |
| attention_unattributed_ticks | 481309.533333 | 481748.200000 | +0.0911% | 481769.200000 | 481761.466667 | -0.0016% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.866667 | 284.933333 | -0.6739% | 286.233333 | 284.833333 | -0.4891% |
| boundary_ddr_read_bytes | 1165568.000000 | 1165568.000000 | +0.0000% | 1165568.000000 | 1165568.000000 | +0.0000% |
| boundary_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| boundary_dma_descriptor_count | 170.000000 | 170.000000 | +0.0000% | 170.000000 | 170.000000 | +0.0000% |
| cache_compared_elements | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_composed_cosine_diagnostic_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_fp16_max_violation_fraction | 0.010000 | 0.010000 | +0.0000% | 0.010000 | 0.010000 | +0.0000% |
| cache_fp16_min_cosine | 0.999990 | 0.999990 | +0.0000% | 0.999990 | 0.999990 | +0.0000% |
| cache_legacy_mixed_bound_failure_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_mixed_tolerance_violation_fraction | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_max_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_min_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| cache_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_prefix_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_structure_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| cache_tensor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| down_ticks | 162299.733333 | 165685.200000 | +2.0859% | 165756.933333 | 166182.133333 | +0.2565% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29944.400000 | 29957.266667 | +0.0430% | 29933.600000 | 29924.333333 | -0.0310% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2670.866667 | 2670.933333 | +0.0025% | 2671.133333 | 2671.033333 | -0.0037% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 426778.400000 | 430162.666667 | +0.7930% | 429652.500000 | 430246.500000 | +0.1383% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 37.266667 | 35.266667 | -5.3667% | 38.266667 | 36.500000 | -4.6167% |
| generation_final_norm_ticks | 41.933333 | 44.200000 | +5.4054% | 40.733333 | 40.600000 | -0.3273% |
| generation_lm_head_argmax_ticks | 7205.933333 | 7010.266667 | -2.7154% | 6842.166667 | 6945.333333 | +1.5078% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 130140.000000 | 130789.200000 | +0.4988% | 129948.100000 | 130108.466667 | +0.1234% |
| generation_lm_head_expand_ticks | 111812.466667 | 113318.533333 | +1.3470% | 112156.866667 | 112805.500000 | +0.5783% |
| generation_lm_head_hmx_tail_wait_ticks | 2614.933333 | 2350.800000 | -10.1010% | 2361.666667 | 2358.233333 | -0.1454% |
| generation_lm_head_hmx_ticks | 129216.733333 | 129856.533333 | +0.4951% | 129023.500000 | 129194.566667 | +0.1326% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 212.133333 | 215.333333 | +1.5085% | 209.333333 | 206.600000 | -1.3057% |
| generation_lm_head_scale_init_ticks | 567.666667 | 567.066667 | -0.1057% | 568.033333 | 566.966667 | -0.1878% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 130181.933333 | 130833.400000 | +0.5004% | 129989.233333 | 130150.833333 | +0.1243% |
| generation_lm_head_weight_dma_ticks | 129319.800000 | 129956.866667 | +0.4926% | 129127.166667 | 129296.666667 | +0.1313% |
| generation_lm_head_weight_dma_wait_ticks | 2824.266667 | 2797.600000 | -0.9442% | 2858.966667 | 2849.133333 | -0.3439% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 135123.866667 | 127968.800000 | -5.2952% | 126259.800000 | 128797.066667 | +2.0096% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 624.649228 | 645.930406 | +3.4069% | 1560.307200 | 1621.328125 | +3.9108% |
| host_us | 91027.715200 | 91609.635267 | +0.6393% | 92469.758567 | 92535.239433 | +0.0708% |
| host_wall_ns | 91027715.200000 | 91609635.266667 | +0.6393% | 92469758.566667 | 92535239.433333 | +0.0708% |
| input_norm_ticks | 9359.866667 | 9366.733333 | +0.0734% | 9359.833333 | 9358.400000 | -0.0153% |
| input_stage_ticks | 7.133333 | 8.066667 | +13.0841% | 8.000000 | 8.100000 | +1.2500% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1735738.866667 | 1746503.133333 | +0.6202% | 1744736.000000 | 1745872.833333 | +0.0652% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 323.200000 | 323.266667 | +0.0206% | 321.733333 | 323.300000 | +0.4869% |
| ledger_named_ticks | 1735738.866667 | 1746503.133333 | +0.6202% | 1744736.000000 | 1745872.833333 | +0.0652% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6553.666667 | 6570.666667 | +0.2594% | 5850.333333 | 5778.866667 | -1.2216% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 95336.800000 | 96357.666667 | +1.0708% | 96579.133333 | 96504.000000 | -0.0778% |
| output_cosine | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| output_fp16_atol | 0.062500 | 0.062500 | +0.0000% | 0.062500 | 0.062500 | +0.0000% |
| output_fp16_max_composed_nrmse | 0.003000 | 0.003000 | +0.0000% | 0.003000 | 0.003000 | +0.0000% |
| output_fp16_rtol | 0.002000 | 0.002000 | +0.0000% | 0.002000 | 0.002000 | +0.0000% |
| output_max_abs | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_lsb | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_max_required_rtol_after_atol | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mismatches | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_mixed_tolerance_violations | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nonfinite_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_nrmse | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| output_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| post_attention_norm_ticks | 15.866667 | 17.266667 | +8.8235% | 16.833333 | 16.966667 | +0.7921% |
| post_attention_residual_ticks | 9027.066667 | 9016.600000 | -0.1159% | 9020.333333 | 9018.000000 | -0.0259% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 672399.066667 | 684489.600000 | +1.7981% | 684472.066667 | 685739.466667 | +0.1852% |
| projection_pack_ticks | 422.333333 | 418.400000 | -0.9313% | 420.300000 | 421.500000 | +0.2855% |
| projection_unpack_ticks | 11370.066667 | 11178.000000 | -1.6892% | 11237.466667 | 11287.666667 | +0.4467% |
| qk_norm_rope_ticks | 16.866667 | 17.466667 | +3.5573% | 17.600000 | 17.433333 | -0.9470% |
| qkv_projection_ticks | 223201.466667 | 224507.800000 | +0.5853% | 224793.333333 | 224841.266667 | +0.0213% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 628.066667 | 622.600000 | -0.8704% | 622.833333 | 622.466667 | -0.0589% |
| runtime_teardown_ticks | 576.333333 | 582.933333 | +1.1452% | 574.966667 | 576.966667 | +0.3478% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4269.066667 | 4261.266667 | -0.1827% | 4245.933333 | 4248.733333 | +0.0659% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21303.400000 | 21333.200000 | +0.1399% | 21329.233333 | 21328.300000 | -0.0044% |
| scan_cache_stage_ticks | 13255.933333 | 13716.266667 | +3.4727% | 13737.866667 | 13740.333333 | +0.0180% |
| scan_dynamic_attention_ticks | 642765.066667 | 643751.133333 | +0.1534% | 643977.833333 | 643692.700000 | -0.0443% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 40.400000 | 36.133333 | -10.5611% | 39.766667 | 39.166667 | -1.5088% |
| total_ticks | 1735110.800000 | 1745880.533333 | +0.6207% | 1744110.633333 | 1745251.100000 | +0.0654% |
| u8_attention_audit_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_av_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_fused_k_operand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_k_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_pipeline_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_probability_mask_violation_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_norm_rope_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_qk_requant_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_attention_v_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_k_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_append_update_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_build_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_sealed_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_segment_tail_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_attention_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_quartet_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_init_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_row_update_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| u8_cache_v_vtcm_tail_seal_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| valid_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| vtcm_acquired_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| vtcm_peak_plan_bytes | 8171008.000000 | 8171008.000000 | +0.0000% | 8171008.000000 | 8171008.000000 | +0.0000% |
| vtcm_requested_bytes | 8388608.000000 | 8388608.000000 | +0.0000% | 8388608.000000 | 8388608.000000 | +0.0000% |
| w4f16_cross_prefetch_lifetime_ticks | 834628.533333 | 833280.800000 | -0.1615% | 833333.233333 | 833278.000000 | -0.0066% |
| w4f16_cross_prefetch_wait_ticks | 1103.800000 | 1129.066667 | +2.2891% | 1113.433333 | 1130.500000 | +1.5328% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 73665.333333 | 79122.200000 | +7.4076% | 77787.333333 | 78678.500000 | +1.1456% |
| w4f16_expand_ticks | 609976.200000 | 617360.000000 | +1.2105% | 617069.000000 | 618227.333333 | +0.1877% |
| w4f16_expand_work_ticks | 1693680.533333 | 1703392.933333 | +0.5734% | 1699417.566667 | 1702165.633333 | +0.1617% |
| w4f16_gate_up_expand_pool_wait_ticks | 29303.200000 | 31443.666667 | +7.3045% | 31372.900000 | 31304.700000 | -0.2174% |
| w4f16_gate_up_expand_ticks | 249951.733333 | 253786.666667 | +1.5343% | 253422.200000 | 253798.133333 | +0.1483% |
| w4f16_gate_up_expand_work_ticks | 157048.866667 | 157863.533333 | +0.5187% | 157987.566667 | 157796.933333 | -0.1207% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7243.000000 | 7274.533333 | +0.4354% | 7313.133333 | 7301.400000 | -0.1604% |
| w4f16_gate_up_hmx_wait_ticks | 274179.066667 | 279370.933333 | +1.8936% | 279052.933333 | 279608.433333 | +0.1991% |
| w4f16_gate_up_stream_join_wait_ticks | 2745.133333 | 865.333333 | -68.4775% | 627.733333 | 646.966667 | +3.0639% |
| w4f16_gate_up_stream_ready_wait_ticks | 1186.666667 | 1276.400000 | +7.5618% | 1319.600000 | 1313.566667 | -0.4572% |
| w4f16_gate_up_stream_work_ticks | 134364.333333 | 134383.266667 | +0.0141% | 134394.033333 | 134388.733333 | -0.0039% |
| w4f16_gate_up_weight_dma_ticks | 550100.933333 | 560437.666667 | +1.8791% | 559836.366667 | 560748.866667 | +0.1630% |
| w4f16_hmx_tail_wait_ticks | 18353.066667 | 18688.733333 | +1.8289% | 18866.900000 | 18873.800000 | +0.0366% |
| w4f16_prefetch_wait_ticks | 24832.533333 | 29718.133333 | +19.6742% | 29909.333333 | 29829.266667 | -0.2677% |
| w4u8_av_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_av_requant_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_op_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_common_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_av_requant_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_op_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_common_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_down_batch_n_tiles | 2.000000 | 2.000000 | +0.0000% | 2.000000 | 2.000000 | +0.0000% |
| w4u8_decode_direct_n_down_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_expand_bytes_avoided | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_mask | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_gate_prefetch | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_o_single_dma | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_projection_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_q_batch_n_tiles | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_k_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_lm_head_group_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| w4u8_decode_o_batch_n_tiles | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| w4u8_decode_projection_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_pair_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_q_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_norm_rope_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_qk_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_qk_rows_processed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_softmax_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_padding_poison_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_rows | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| w4u8_decode_swiglu_valid_row_hash | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_decode_swiglu_vector_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_delta_reconstruction_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_final_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_overlap_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_gate_up_swiglu_worker_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_input_norm_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_activation_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_hmx_command_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_down_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_mlp_weight_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_batch_n_tiles_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_consume_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_start_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_o_gate_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_direct_row4_call_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_main_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_post_residual_worker_work_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_prefill_cache_mode | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_norm_rope_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qk_padding_poison_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_batch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_dma_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_task_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_expand_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_head_publish_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_compute_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_dispatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_pool_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_prep_worker_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkv_ring_slot_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_hmx_lifetime_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_prefetch_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_qkvo_weight_expand_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_residual_active_contexts | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4u8_swiglu_rows_observed | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| weight_ddr_read_bytes | 864507392.000000 | 864507392.000000 | +0.0000% | 864507392.000000 | 864507392.000000 | +0.0000% |
| weight_dma_descriptor_count | 3927.000000 | 3927.000000 | +0.0000% | 3927.000000 | 3927.000000 | +0.0000% |
| weight_dma_ticks | 1123426.600000 | 1140352.400000 | +1.5066% | 1138335.000000 | 1139893.833333 | +0.1369% |


## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63069.2965,
      "prefill_tokens_per_second": 1014.7568397246987,
      "decode_tokens": 15,
      "decode_total_host_us": 1386578.6195,
      "decode_tokens_per_second": 10.817994586855088
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 63140.2865,
      "prefill_tokens_per_second": 1013.615926497261,
      "decode_tokens": 15,
      "decode_total_host_us": 1387815.6775,
      "decode_tokens_per_second": 10.80835174525545
    },
    "B0": {
      "prefill_tokens": 64,
      "prefill_host_us": 62837.109,
      "prefill_tokens_per_second": 1018.5064370163816,
      "decode_tokens": 15,
      "decode_total_host_us": 1384688.49,
      "decode_tokens_per_second": 10.832761381586987
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 63099.3485,
      "prefill_tokens_per_second": 1014.2735467387591,
      "decode_tokens": 15,
      "decode_total_host_us": 1387656.2759999998,
      "decode_tokens_per_second": 10.809593311708555
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63101.198000000004,
      "prefill_tokens_per_second": 1014.2438183186315,
      "decode_tokens": 15,
      "decode_total_host_us": 1387046.3785,
      "decode_tokens_per_second": 10.81434639281602
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63242.3695,
      "prefill_tokens_per_second": 1011.9797930721112,
      "decode_tokens": 15,
      "decode_total_host_us": 1388028.5915,
      "decode_tokens_per_second": 10.806693818741845
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.1627881196463843,
      "decode": -0.08914060633274934
    },
    "B": {
      "prefill": -0.33245670972336105,
      "decode": -0.2061125939613384
    },
    "C": {
      "prefill": -0.2367729934462548,
      "decode": -0.1645508532618134
    }
  }
}
```


## Experiment context and checks

# EXP-0223: transformer per-channel clipping before GPTQ

User-approved bounded scale/clipping trial. A uses original coordinates/gamma, B fresh gamma folding, C fresh fixed H2048 R1/H128 R2. All196 transformer projections per variant are generated afresh from verified Qwen3-origin. Each variant retains byte-identical EXP0221 LM head, embedding and all norm files. No LPBQ/group quantization, new rotation, new calibration, other-recipe change or runtime change.

## Observed outcome and limits

Transformer clipping helps the folded/rotated GPTQ variants but hurts original-coordinate A. Actual DSP A: NLL3.873038->3.969016 and tasks8/24->4/24; B:4.601481->3.911864 and0/24->2/24; C:4.320988->3.956556 and2/24->9/24. Predeclared per-variant effectiveness passes B/C and fails A; clipped C beats clipped A on both metrics, so that incremental gate passes. Aggregate local_gate passes because at least one variant improves; this does not certify usable quality or promote any baseline.

C is still worse in NLL than frozen original-coordinate GPTQ A (3.956556 versus3.873038), and its9/24 versus8/24 is only one extra answer on a tiny suite. The C-to-frozen-A NLL gap shrinks from0.447949 to0.083518 (81.36%), an arithmetic description of this suite rather than a statistical significance claim. Compared with clipped B, clipped C has worse NLL (3.956556 versus3.911864) but more correct tasks (9 versus2); no configuration dominates all quality metrics. Frozen F16A16 scored22/24, so recovered rotated quality remains far below that task reference.

Open-text checks reinforce this limitation: clipped A emits immediate EOS on3/4 open prompts; clipped C does so on1/4, and one other C English answer restates the instruction within the16-token cap. B's four open outputs are nonempty but it still gets only2/24 strict tasks. Exact texts and token IDs are retained in quality_summary.json; short outputs are diagnostic, not general usability certification.

All588projection probes show GPTQ improves over clipped RTN on each identical candidate input, yet A's model quality worsens. The scale search minimizes weight-local L2.4 before GPTQ; it does not minimize activation-weighted post-GPTQ or whole-model loss. This objective mismatch and accumulated quantization error are plausible explanations, not independently proven causes. The experiment supports range selection as a material contributor to folded/rotated degradation, not the claim that fixed absmax alone explains it. No learned rotations or complete SpinQuant reproduction were attempted here.

All six current-run speed comparisons differ by less than0.4% in paired median throughput. No speed benefit is claimed. The overview uses B only because the predeclared report rule selects lowest new NLL; its2/24 task score prevents reading that selection as a recommendation. Keep the frozen baseline identities and all candidates for discussion; no automatic further optimization.

## Fixed quantization intervention

For each output row of transformed FP32 weights, evaluate80 ratios p=1-i/100 for i=0..79. Scale=p*absmax(row)/7, nearest-even signed[-7,7]. Select the minimum sum(abs(weight-reconstructed_weight)^2.4); ties choose the largest range. This adapts the weight-range search approach in SpinQuant's utils/quant_utils.py and its w_clip evaluation option to our fixed integer grid. It is not a full SpinQuant reproduction. No activation-weighted clipping search or post-GPTQ grid search is performed. Original unclipped range is a candidate, so selected weight-local objective cannot increase; this does not guarantee model quality.

GPTQ then keeps that chosen scale fixed while doing activation-weighted error compensation. Existing true-sequential QKV->O->Gate/Up->Down ordering,1% damping,FP64 factor/FP32 compensation, act-order/inverse export and computational block128 remain. Calibration is the frozen EXP0221 independent64x128 bilingual training tokens (8192 total); no evaluation or holdout tuning. Each later projection/layer sees the newly quantized predecessor. The old head stays frozen even though transformer outputs change, isolating the transformer intervention.

## Actual DSP quality

Frozen qbh-lite-v1:512 conditional targets,24 strict tasks,4 open prefixes. EXP0221 absmax A/B/C DSP quality is a frozen historical control. All new candidates complete actual DSP quick/full/repeat and independent software quality. Software scores are separate and not assumed bit-exact DSP. This lightweight diagnostic does not certify general model quality.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| absmax GPTQ A（冻结） | 3.8730 | 48.09 | 8/24 | 62.50% |
| absmax GPTQ B（冻结） | 4.6015 | 99.63 | 0/24 | 46.09% |
| absmax GPTQ C（冻结） | 4.3210 | 75.26 | 2/24 | 57.81% |
| clipping GPTQ A | 3.9690 | 52.93 | 4/24 | 62.70% |
| clipping GPTQ B | 3.9119 | 49.99 | 2/24 | 60.16% |
| clipping GPTQ C | 3.9566 | 52.28 | 9/24 | 60.55% |

Predeclared lower-NLL AND more-task effectiveness per variant: {"A": false, "B": true, "C": true}. Incremental clipped C versus clipped A on BOTH metrics: True. No automatic baseline promotion.

## Scale selection

| Variant | Output rows | Rows clipped | Mean selected range / absmax |
|---|---|---|---|
| A | 573440 | 100.00% | 0.7044 |
| B | 573440 | 99.99% | 0.6709 |
| C | 573440 | 100.00% | 0.7197 |

Per-row choices, scales and before/after objectives are retained as hashed NPZ artifacts. Local probe errors use every16th calibration input (512 positions) on identical candidate inputs, comparing clipped RTN and clipped GPTQ against unquantized transformed weights. They are unweighted projection summaries, not whole-model errors, and are not used to select settings. Input Gram hashes inherited from the GPTQ utility identify diagonals, not full Gram matrices.

## Independent checks and identities

NumPy exhaustive scale selection, dense explicit-scale GPTQ elimination, original absmax parity and integer packing oracles pass. All588projection packing checks and168staged/HF forward checks pass. Nonprojection package files remain byte-identical to each EXP0221 variant.

```json
{
  "A": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 0.0,
          "cosine": 0.9999999999997751,
          "max_abs": 0.0,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 0.0,
          "cosine": 0.9999999999997475,
          "max_abs": 0.0,
          "top1_equal": 16,
          "passed": true
        }
      ],
      "original_shards": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
      }
    },
    "forward_checks": 56,
    "max_forward_nrmse": 0.0,
    "frozen_file_count": 883,
    "determinism": {
      "repeat_equal": true,
      "overlap_mismatches": []
    },
    "execution": {
      "source_head": "d6ab67420cc0c823c0705cd6820e323ef5c2f4c6",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
      }
    }
  },
  "B": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 1.3316402355909908e-06,
          "cosine": 0.9999999999988802,
          "max_abs": 8.0108642578125e-05,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 1.2410618960000922e-06,
          "cosine": 0.9999999999990077,
          "max_abs": 3.814697265625e-05,
          "top1_equal": 16,
          "passed": true
        }
      ],
      "original_shards": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
      }
    },
    "forward_checks": 56,
    "max_forward_nrmse": 0.0,
    "frozen_file_count": 883,
    "determinism": {
      "repeat_equal": true,
      "overlap_mismatches": []
    },
    "execution": {
      "source_head": "b396ac62de2d58b29de555432df016878b8982cb",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
      }
    }
  },
  "C": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 1.5193639915537854e-06,
          "cosine": 0.9999999999986376,
          "max_abs": 5.7220458984375e-05,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 2.0345006134564764e-06,
          "cosine": 0.9999999999977212,
          "max_abs": 5.054473876953125e-05,
          "top1_equal": 16,
          "passed": true
        }
      ],
      "original_shards": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
      }
    },
    "forward_checks": 56,
    "max_forward_nrmse": 0.0,
    "frozen_file_count": 883,
    "determinism": {
      "repeat_equal": true,
      "overlap_mismatches": []
    },
    "execution": {
      "source_head": "b396ac62de2d58b29de555432df016878b8982cb",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
      }
    }
  }
}
```

Software diagnostic:

```json
{
  "A": {
    "nll": 3.969574511051178,
    "ppl": 52.96199130284732,
    "tasks_correct": 3,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.630859375,
    "language_nll": {
      "zh": 4.270898096263409,
      "en": 3.6682509258389473
    }
  },
  "B": {
    "nll": 3.912194263190031,
    "ppl": 50.00856362136663,
    "tasks_correct": 2,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.599609375,
    "language_nll": {
      "zh": 4.345175914466381,
      "en": 3.479212611913681
    }
  },
  "C": {
    "nll": 3.957964614033699,
    "ppl": 52.35066363034929,
    "tasks_correct": 9,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.599609375,
    "language_nll": {
      "zh": 4.258320279419422,
      "en": 3.657608948647976
    }
  }
}
```

## Complete profiling scope

All six variants A0/A/B0/B/C0/C complete one warmup,5short and10formal sessions with rotating order. A0/B0/C0 are the corresponding frozen EXP0221 packages, freshly timed here.960formal invocation ledgers and26880layer ledgers close. All16tokens per speed session match independent per-package references. Scope is one M64 prefill plus15feedback decode steps; quality scoring is off. Host wall is complete28-layer token-in/token-out execution with final norm/head/greedy and persistent KV, excluding cold staging and separately logged WSL tokenizer/detokenizer. No per-layer throughput extrapolation. Fixed speed sequences continue after EOS if present; such throughput describes diagnostic execution.

Runtime remains EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108 (embedded218 intentionally retained, outer experiment223).8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC/one HMX owner, no QNN. Offline quantization elapsed time is not DSP inference time. Package scaffolds named layer14 execute all28layers. Historical inherited replay references are not valid new replay references and are not consumed in generation/evaluation.

The overview W4A16 column uses the lowest-NLL clipped candidate under the predeclared tie rule. Other recipe columns are frozen EXP0218 nonpaired references, with different W4 weights, so no activation-only attribution. Full report retains every numeric repeat-one/repeat-ten control/candidate field; additive accounting fields are exclusive, engine/wait counters overlap. Detailed raw evidence is retained.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 clipped GPTQ B EXP-0223 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 380.8 (0.60%) | 247.4 (0.63%) | +53.93% |
| Input RMSNorm | 489.7 (0.61%) | 491.5 (0.78%) | 554.0 (1.40%) | -11.29% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11748.4 (18.62%) | 7052.7 (17.82%) | +66.58% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3960.5 (6.28%) | 3214.5 (8.12%) | +23.21% |
| O projection | 5757.7 (7.14%) | 5057.3 (8.01%) | 1256.4 (3.17%) | +302.52% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 474.2 (0.75%) | 654.0 (1.65%) | -27.48% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22406.6 (35.51%) | 14442.7 (36.49%) | +55.14% |
| Down | 13447.9 (16.67%) | 8643.2 (13.70%) | 3428.5 (8.66%) | +152.10% |
| Final residual | 140.1 (0.17%) | 140.1 (0.22%) | 183.8 (0.46%) | -23.77% |
| KV carrier conversion | 174.0 (0.22%) | 171.9 (0.27%) | 203.2 (0.51%) | -15.42% |
| KV append DMA | 343.6 (0.43%) | 341.3 (0.54%) | 463.9 (1.17%) | -26.43% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.35% |
| Layer bookkeeping | 23.9 (0.03%) | 23.7 (0.04%) | 23.2 (0.06%) | +2.36% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.9 (0.01%) | 22.5 (0.06%) | -60.42% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.4 (0.15%) | 105.6 (0.27%) | -11.52% |
| Embedding | 68.1 (0.08%) | 66.5 (0.11%) | 62.4 (0.16%) | +6.55% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.71% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6697.0 (10.61%) | 5284.7 (13.35%) | +26.72% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2387.8 (3.78%) | 2466.6 (6.23%) | -3.19% |
| 完整 Host wall | 80692.2 (100.00%) | 63099.3 (100.00%) | 39575.9 (100.00%) | +59.44% |

## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63069.2965,
      "prefill_tokens_per_second": 1014.7568397246987,
      "decode_tokens": 15,
      "decode_total_host_us": 1386578.6195,
      "decode_tokens_per_second": 10.817994586855088
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 63140.2865,
      "prefill_tokens_per_second": 1013.615926497261,
      "decode_tokens": 15,
      "decode_total_host_us": 1387815.6775,
      "decode_tokens_per_second": 10.80835174525545
    },
    "B0": {
      "prefill_tokens": 64,
      "prefill_host_us": 62837.109,
      "prefill_tokens_per_second": 1018.5064370163816,
      "decode_tokens": 15,
      "decode_total_host_us": 1384688.49,
      "decode_tokens_per_second": 10.832761381586987
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 63099.3485,
      "prefill_tokens_per_second": 1014.2735467387591,
      "decode_tokens": 15,
      "decode_total_host_us": 1387656.2759999998,
      "decode_tokens_per_second": 10.809593311708555
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63101.198000000004,
      "prefill_tokens_per_second": 1014.2438183186315,
      "decode_tokens": 15,
      "decode_total_host_us": 1387046.3785,
      "decode_tokens_per_second": 10.81434639281602
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63242.3695,
      "prefill_tokens_per_second": 1011.9797930721112,
      "decode_tokens": 15,
      "decode_total_host_us": 1388028.5915,
      "decode_tokens_per_second": 10.806693818741845
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.1627881196463843,
      "decode": -0.08914060633274934
    },
    "B": {
      "prefill": -0.33245670972336105,
      "decode": -0.2061125939613384
    },
    "C": {
      "prefill": -0.2367729934462548,
      "decode": -0.1645508532618134
    }
  }
}
```

## Reproduction and retained evidence

Models: D:/llm_exp/models/qwen3-block-htp/exp0223. Results: D:/llm_exp/results/qwen3-block-htp/exp0223. Frozen dataset/calibration/source/package hashes, source archive, runtime binaries, per-row NPZs,84hidden checkpoints, manifests and full logs are bound by closure/evidence ledgers. experiment_exp0223.py A|B|C generates fresh models; measure_exp0223.py deploy/quick/full/repeat and warmup/short/formal collects results; summarize_exp0223.py and close_exp0223.py retain reports. Fresh output paths and a registered experiment are required; do not overwrite or tune retained outputs. No baseline promotion or automatic further direction.


Environment:

```json
{
  "python": "3.10.12",
  "torch": "2.13.0+cpu",
  "transformers": "4.51.0",
  "numpy": "2.2.6",
  "threads": 16
}
```
