# EXP-0224 complete profiling comparison

Frozen ABI108 binary, frozen A0/C0 and output-selected A/C independent software token sequences. One M64 plus15 feedback steps;5short then10four-way rotating formal rounds. Other recipe columns historical nonpaired, changed W4 prevents activation-only attribution. Quality scoring disabled. All numeric fields retained; additive ledger fields exclusive, engine/wait counters overlapping. Host-DSP boundary per record = Host wall minus DSP invocation. Percent changes below candidate/control minus one; positive timing changes are slower.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 output-selected GPTQ A EXP-0224 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 392.6 (0.63%) | 247.4 (0.63%) | +58.71% |
| Input RMSNorm | 489.7 (0.61%) | 492.6 (0.79%) | 554.0 (1.40%) | -11.08% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11746.1 (18.84%) | 7052.7 (17.82%) | +66.55% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3964.9 (6.36%) | 3214.5 (8.12%) | +23.34% |
| O projection | 5757.7 (7.14%) | 5050.1 (8.10%) | 1256.4 (3.17%) | +301.95% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.7 (0.76%) | 654.0 (1.65%) | -27.57% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22481.7 (36.06%) | 14442.7 (36.49%) | +55.66% |
| Down | 13447.9 (16.67%) | 8639.3 (13.86%) | 3428.5 (8.66%) | +151.99% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.73% |
| KV carrier conversion | 174.0 (0.22%) | 173.2 (0.28%) | 203.2 (0.51%) | -14.79% |
| KV append DMA | 343.6 (0.43%) | 342.4 (0.55%) | 463.9 (1.17%) | -26.17% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.28% |
| Layer bookkeeping | 23.9 (0.03%) | 23.1 (0.04%) | 23.2 (0.06%) | -0.34% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.0 (0.01%) | 22.5 (0.06%) | -60.07% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 94.2 (0.15%) | 105.6 (0.27%) | -10.80% |
| Embedding | 68.1 (0.08%) | 66.6 (0.11%) | 62.4 (0.16%) | +6.59% |
| Final model RMSNorm | 49.7 (0.06%) | 48.0 (0.08%) | 3.7 (0.01%) | +1189.51% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6725.6 (10.79%) | 5284.7 (13.35%) | +27.27% |
| Host–DSP 边界 | 2374.1 (2.94%) | 1533.8 (2.46%) | 2466.6 (6.23%) | -37.82% |
| 完整 Host wall | 80692.2 (100.00%) | 62347.1 (100.00%) | 39575.9 (100.00%) | +57.54% |


## Control A0 versus output-selected A prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.000000 | 8.000000 | -11.1111% | 12.500000 | 11.500000 | -8.0000% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75880.000000 | 75843.000000 | -0.0488% | 75769.000000 | 75946.500000 | +0.2343% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 119.000000 | 123.000000 | +3.3613% | 121.000000 | 122.000000 | +0.8264% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76060.000000 | 76019.000000 | -0.0539% | 75950.500000 | 76127.000000 | +0.2324% |
| attention_unattributed_ticks | 61.000000 | 53.000000 | -13.1148% | 56.500000 | 54.000000 | -4.4248% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 372.000000 | 371.000000 | -0.2688% | 370.500000 | 370.000000 | -0.1350% |
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
| down_ticks | 164293.000000 | 165304.000000 | +0.6154% | 164392.500000 | 165875.000000 | +0.9018% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9822.000000 | 9956.000000 | +1.3643% | 9876.000000 | 9866.000000 | -0.1013% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2697.000000 | 2697.000000 | +0.0000% | 2694.000000 | 2691.500000 | -0.0928% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 430238.000000 | 431406.000000 | +0.2715% | 429971.000000 | 431638.000000 | +0.3877% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1273.000000 | 1288.000000 | +1.1783% | 1274.000000 | 1278.000000 | +0.3140% |
| generation_final_norm_ticks | 918.000000 | 926.000000 | +0.8715% | 920.000000 | 922.000000 | +0.2174% |
| generation_lm_head_argmax_ticks | 6546.000000 | 6815.000000 | +4.1094% | 6570.000000 | 6647.500000 | +1.1796% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128971.000000 | 130223.000000 | +0.9708% | 128885.000000 | 129131.500000 | +0.1913% |
| generation_lm_head_expand_ticks | 112046.000000 | 112905.000000 | +0.7666% | 111729.500000 | 112198.000000 | +0.4193% |
| generation_lm_head_hmx_tail_wait_ticks | 2198.000000 | 2349.000000 | +6.8699% | 2388.000000 | 2308.000000 | -3.3501% |
| generation_lm_head_hmx_ticks | 127952.000000 | 129222.000000 | +0.9926% | 127878.000000 | 128131.500000 | +0.1982% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 242.000000 | 249.000000 | +2.8926% | 233.500000 | 233.000000 | -0.2141% |
| generation_lm_head_scale_init_ticks | 573.000000 | 576.000000 | +0.5236% | 574.000000 | 565.500000 | -1.4808% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129889.000000 | 131149.000000 | +0.9701% | 129800.000000 | 130051.500000 | +0.1938% |
| generation_lm_head_weight_dma_ticks | 128090.000000 | 129332.000000 | +0.9696% | 128004.000000 | 128253.500000 | +0.1949% |
| generation_lm_head_weight_dma_wait_ticks | 2794.000000 | 2732.000000 | -2.2190% | 2886.000000 | 2795.000000 | -3.1532% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 128327.000000 | 132830.000000 | +3.5090% | 130636.500000 | 132864.000000 | +1.7051% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2319.114083 | 478.437500 | -79.3698% | 2286.822667 | 1533.802250 | -32.9287% |
| host_us | 63017.812000 | 61303.750000 | -2.7200% | 63025.937000 | 62347.083500 | -1.0771% |
| host_wall_ns | 63017812.000000 | 61303750.000000 | -2.7200% | 63025937.000000 | 62347083.500000 | -1.0771% |
| input_norm_ticks | 9433.000000 | 9410.000000 | -0.2438% | 9451.500000 | 9457.500000 | +0.0635% |
| input_stage_ticks | 11.000000 | 11.000000 | +0.0000% | 11.000000 | 9.500000 | -13.6364% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1165415.000000 | 1167846.000000 | +0.2086% | 1165781.000000 | 1168744.000000 | +0.2542% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 429.000000 | 452.000000 | +5.3613% | 437.500000 | 443.000000 | +1.2571% |
| ledger_named_ticks | 1165415.000000 | 1167846.000000 | +0.2086% | 1165781.000000 | 1168744.000000 | +0.2542% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7240.000000 | 7296.000000 | +0.7735% | 7514.500000 | 7526.500000 | +0.1597% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96820.000000 | 96632.000000 | -0.1942% | 96887.500000 | 96962.000000 | +0.0769% |
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
| post_attention_norm_ticks | 24.000000 | 27.000000 | +12.5000% | 24.000000 | 24.500000 | +2.0833% |
| post_attention_residual_ticks | 9099.000000 | 9070.000000 | -0.3187% | 9079.000000 | 9069.000000 | -0.1101% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684121.000000 | 685646.000000 | +0.2229% | 683627.500000 | 686983.000000 | +0.4908% |
| projection_pack_ticks | 3172.000000 | 3168.000000 | -0.1261% | 3169.500000 | 3165.500000 | -0.1262% |
| projection_unpack_ticks | 10966.000000 | 11114.000000 | +1.3496% | 10924.500000 | 11132.000000 | +1.8994% |
| qk_norm_rope_ticks | 20.000000 | 22.000000 | +10.0000% | 25.500000 | 23.000000 | -9.8039% |
| qkv_projection_ticks | 225685.000000 | 224703.000000 | -0.4351% | 225483.000000 | 225499.500000 | +0.0073% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 886.000000 | 906.000000 | +2.2573% | 909.000000 | 906.500000 | -0.2750% |
| runtime_teardown_ticks | 900.000000 | 907.000000 | +0.7778% | 900.000000 | 901.000000 | +0.1111% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6532.000000 | 6555.000000 | +0.3521% | 6576.500000 | 6575.000000 | -0.0228% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3322.000000 | 3435.000000 | +3.4016% | 3316.000000 | 3325.000000 | +0.2714% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 183.000000 | 178.000000 | -2.7322% | 180.000000 | 172.500000 | -4.1667% |
| total_ticks | 1164529.000000 | 1166940.000000 | +0.2070% | 1164826.000000 | 1167796.500000 | +0.2550% |
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
| w4f16_cross_prefetch_lifetime_ticks | 249291.000000 | 249904.000000 | +0.2459% | 250012.500000 | 249943.000000 | -0.0278% |
| w4f16_cross_prefetch_wait_ticks | 1119.000000 | 1020.000000 | -8.8472% | 1091.500000 | 1095.000000 | +0.3207% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75938.000000 | 76407.000000 | +0.6176% | 76348.500000 | 76122.000000 | -0.2967% |
| w4f16_expand_ticks | 616084.000000 | 617939.000000 | +0.3011% | 615928.000000 | 619445.500000 | +0.5711% |
| w4f16_expand_work_ticks | 1696636.000000 | 1703217.000000 | +0.3879% | 1697758.000000 | 1701649.500000 | +0.2292% |
| w4f16_gate_up_expand_pool_wait_ticks | 30725.000000 | 30002.000000 | -2.3531% | 30677.000000 | 30505.000000 | -0.5607% |
| w4f16_gate_up_expand_ticks | 253309.000000 | 254595.000000 | +0.5077% | 253150.000000 | 254754.500000 | +0.6338% |
| w4f16_gate_up_expand_work_ticks | 158901.000000 | 160356.000000 | +0.9157% | 159077.500000 | 160235.500000 | +0.7279% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7467.000000 | 7642.000000 | +2.3436% | 7338.500000 | 7238.500000 | -1.3627% |
| w4f16_gate_up_hmx_wait_ticks | 279304.000000 | 280753.000000 | +0.5188% | 278891.000000 | 280703.000000 | +0.6497% |
| w4f16_gate_up_stream_join_wait_ticks | 585.000000 | 643.000000 | +9.9145% | 655.000000 | 636.000000 | -2.9008% |
| w4f16_gate_up_stream_ready_wait_ticks | 1345.000000 | 1176.000000 | -12.5651% | 1259.500000 | 1327.500000 | +5.3990% |
| w4f16_gate_up_stream_work_ticks | 134415.000000 | 134340.000000 | -0.0558% | 134375.500000 | 134388.000000 | +0.0093% |
| w4f16_gate_up_weight_dma_ticks | 560700.000000 | 563044.000000 | +0.4180% | 559905.000000 | 563407.000000 | +0.6255% |
| w4f16_hmx_tail_wait_ticks | 18835.000000 | 19248.000000 | +2.1927% | 18874.500000 | 18757.000000 | -0.6225% |
| w4f16_prefetch_wait_ticks | 30298.000000 | 29574.000000 | -2.3896% | 30037.500000 | 30172.000000 | +0.4478% |
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
| weight_dma_ticks | 1141199.000000 | 1144172.000000 | +0.2605% | 1140093.500000 | 1145240.000000 | +0.4514% |


## Control A0 versus output-selected A decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 10.133333 | 9.000000 | -11.1842% | 9.633333 | 9.300000 | -3.4602% |
| attention_av_hmx_ticks | 7738.066667 | 7714.600000 | -0.3033% | 7747.266667 | 7751.166667 | +0.0503% |
| attention_av_pack_ticks | 3060.000000 | 3021.866667 | -1.2462% | 3035.966667 | 3031.966667 | -0.1318% |
| attention_av_unpack_ticks | 4009.066667 | 4006.733333 | -0.0582% | 4003.133333 | 4006.300000 | +0.0791% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 7853.866667 | 7840.000000 | -0.1766% | 7836.833333 | 7851.866667 | +0.1918% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6374.733333 | 6367.800000 | -0.1088% | 6386.566667 | 6379.666667 | -0.1080% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132928.333333 | 133109.733333 | +0.1365% | 132894.600000 | 133063.566667 | +0.1271% |
| attention_ticks | 643749.066667 | 643828.066667 | +0.0123% | 643701.533333 | 643830.800000 | +0.0201% |
| attention_unattributed_ticks | 481785.000000 | 481767.333333 | -0.0037% | 481784.733333 | 481780.166667 | -0.0009% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.733333 | 281.200000 | -1.9298% | 283.100000 | 281.466667 | -0.5769% |
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
| down_ticks | 164626.600000 | 165963.400000 | +0.8120% | 164682.400000 | 166140.633333 | +0.8855% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29957.333333 | 29932.000000 | -0.0846% | 29936.866667 | 29933.566667 | -0.0110% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2672.400000 | 2672.333333 | -0.0025% | 2671.033333 | 2671.733333 | +0.0262% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 429124.200000 | 430343.666667 | +0.2842% | 428960.200000 | 430278.433333 | +0.3073% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 37.266667 | 37.533333 | +0.7156% | 37.700000 | 37.233333 | -1.2378% |
| generation_final_norm_ticks | 41.466667 | 43.266667 | +4.3408% | 41.666667 | 42.066667 | +0.9600% |
| generation_lm_head_argmax_ticks | 6677.200000 | 6868.533333 | +2.8655% | 6578.133333 | 6697.800000 | +1.8192% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 129336.000000 | 130105.400000 | +0.5949% | 128662.200000 | 129197.833333 | +0.4163% |
| generation_lm_head_expand_ticks | 112213.066667 | 112936.733333 | +0.6449% | 111271.900000 | 112165.800000 | +0.8033% |
| generation_lm_head_hmx_tail_wait_ticks | 2293.733333 | 2253.866667 | -1.7381% | 2287.666667 | 2256.433333 | -1.3653% |
| generation_lm_head_hmx_ticks | 128407.866667 | 129174.866667 | +0.5973% | 127735.933333 | 128270.900000 | +0.4188% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 215.466667 | 214.333333 | -0.5260% | 209.333333 | 208.700000 | -0.3025% |
| generation_lm_head_scale_init_ticks | 568.866667 | 573.133333 | +0.7500% | 569.133333 | 568.933333 | -0.0351% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129377.466667 | 130148.666667 | +0.5961% | 128705.000000 | 129242.533333 | +0.4176% |
| generation_lm_head_weight_dma_ticks | 128507.866667 | 129273.466667 | +0.5958% | 127836.600000 | 128374.533333 | +0.4208% |
| generation_lm_head_weight_dma_wait_ticks | 2830.800000 | 2798.266667 | -1.1493% | 2834.300000 | 2812.000000 | -0.7868% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 124299.266667 | 126712.333333 | +1.9413% | 125297.466667 | 126643.466667 | +1.0742% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1780.986206 | 1639.541661 | -7.9419% | 1763.163153 | 1693.550231 | -3.9482% |
| host_us | 92566.357733 | 92588.628467 | +0.0241% | 92573.072933 | 92565.317633 | -0.0084% |
| host_wall_ns | 92566357.733333 | 92588628.466667 | +0.0241% | 92573072.933333 | 92565317.633333 | -0.0084% |
| input_norm_ticks | 9366.066667 | 9357.733333 | -0.0890% | 9359.766667 | 9360.433333 | +0.0071% |
| input_stage_ticks | 8.400000 | 8.333333 | -0.7937% | 8.133333 | 8.666667 | +6.5574% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1743079.133333 | 1746222.466667 | +0.1803% | 1743024.166667 | 1745701.400000 | +0.1536% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 322.466667 | 322.933333 | +0.1447% | 322.100000 | 323.433333 | +0.4140% |
| ledger_named_ticks | 1743079.133333 | 1746222.466667 | +0.1803% | 1743024.166667 | 1745701.400000 | +0.1536% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6125.333333 | 6125.466667 | +0.0022% | 6123.933333 | 6133.466667 | +0.1557% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96525.333333 | 96523.066667 | -0.0023% | 96423.300000 | 96542.133333 | +0.1232% |
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
| post_attention_norm_ticks | 17.000000 | 17.400000 | +2.3529% | 17.100000 | 16.900000 | -1.1696% |
| post_attention_residual_ticks | 9013.933333 | 9020.800000 | +0.0762% | 9014.133333 | 9019.833333 | +0.0632% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 683137.800000 | 685503.866667 | +0.3464% | 682570.233333 | 685579.433333 | +0.4409% |
| projection_pack_ticks | 418.933333 | 423.066667 | +0.9866% | 419.566667 | 420.900000 | +0.3178% |
| projection_unpack_ticks | 10950.933333 | 11275.333333 | +2.9623% | 10951.500000 | 11272.333333 | +2.9296% |
| qk_norm_rope_ticks | 18.000000 | 17.266667 | -4.0741% | 18.000000 | 17.466667 | -2.9630% |
| qkv_projection_ticks | 224952.466667 | 224720.333333 | -0.1032% | 224834.600000 | 224763.166667 | -0.0318% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 617.133333 | 626.733333 | +1.5556% | 619.800000 | 625.600000 | +0.9358% |
| runtime_teardown_ticks | 580.933333 | 574.733333 | -1.0672% | 577.000000 | 577.000000 | +0.0000% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4278.733333 | 4253.400000 | -0.5921% | 4263.066667 | 4256.066667 | -0.1642% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21329.666667 | 21333.400000 | +0.0175% | 21327.466667 | 21335.800000 | +0.0391% |
| scan_cache_stage_ticks | 13776.333333 | 13759.133333 | -0.1249% | 13772.033333 | 13751.266667 | -0.1508% |
| scan_dynamic_attention_ticks | 643699.866667 | 643778.466667 | +0.0122% | 643651.600000 | 643780.866667 | +0.0201% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 39.800000 | 37.000000 | -7.0352% | 39.800000 | 38.366667 | -3.6013% |
| total_ticks | 1742462.000000 | 1745595.733333 | +0.1798% | 1742400.966667 | 1745079.066667 | +0.1537% |
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
| w4f16_cross_prefetch_lifetime_ticks | 832745.200000 | 833037.000000 | +0.0350% | 833018.633333 | 833194.500000 | +0.0211% |
| w4f16_cross_prefetch_wait_ticks | 1096.333333 | 1112.866667 | +1.5081% | 1104.433333 | 1109.700000 | +0.4769% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 77487.800000 | 78878.933333 | +1.7953% | 77400.466667 | 77604.666667 | +0.2638% |
| w4f16_expand_ticks | 615193.933333 | 617894.333333 | +0.4390% | 614996.466667 | 617957.666667 | +0.4815% |
| w4f16_expand_work_ticks | 1697925.333333 | 1702049.400000 | +0.2429% | 1696305.633333 | 1698113.933333 | +0.1066% |
| w4f16_gate_up_expand_pool_wait_ticks | 31180.600000 | 31388.266667 | +0.6660% | 30982.900000 | 31380.966667 | +1.2848% |
| w4f16_gate_up_expand_ticks | 252717.666667 | 254028.133333 | +0.5185% | 252635.600000 | 253883.500000 | +0.4940% |
| w4f16_gate_up_expand_work_ticks | 158140.933333 | 158338.933333 | +0.1252% | 158172.666667 | 158333.266667 | +0.1015% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7425.466667 | 7324.666667 | -1.3575% | 7468.266667 | 7385.933333 | -1.1024% |
| w4f16_gate_up_hmx_wait_ticks | 278648.466667 | 279808.466667 | +0.4163% | 278415.100000 | 279685.466667 | +0.4563% |
| w4f16_gate_up_stream_join_wait_ticks | 641.733333 | 579.266667 | -9.7341% | 661.566667 | 629.533333 | -4.8420% |
| w4f16_gate_up_stream_ready_wait_ticks | 1280.866667 | 1240.000000 | -3.1905% | 1273.133333 | 1304.833333 | +2.4899% |
| w4f16_gate_up_stream_work_ticks | 134400.533333 | 134379.600000 | -0.0156% | 134384.033333 | 134388.700000 | +0.0035% |
| w4f16_gate_up_weight_dma_ticks | 558883.666667 | 561038.800000 | +0.3856% | 558411.700000 | 560856.300000 | +0.4378% |
| w4f16_hmx_tail_wait_ticks | 18890.133333 | 18757.200000 | -0.7037% | 18927.966667 | 18828.366667 | -0.5262% |
| w4f16_prefetch_wait_ticks | 30178.800000 | 30114.733333 | -0.2123% | 30112.600000 | 29962.800000 | -0.4975% |
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
| weight_dma_ticks | 1136671.400000 | 1140472.600000 | +0.3344% | 1135327.600000 | 1139515.166667 | +0.3688% |


## Control C0 versus output-selected C prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.000000 | 12.000000 | +33.3333% | 11.500000 | 13.000000 | +13.0435% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75944.000000 | 75032.000000 | -1.2009% | 75869.500000 | 75680.000000 | -0.2498% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 118.000000 | 123.000000 | +4.2373% | 120.000000 | 121.000000 | +0.8333% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76122.000000 | 75204.000000 | -1.2060% | 76049.000000 | 75860.000000 | -0.2485% |
| attention_unattributed_ticks | 60.000000 | 49.000000 | -18.3333% | 57.500000 | 55.500000 | -3.4783% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 361.000000 | 376.000000 | +4.1551% | 363.500000 | 368.500000 | +1.3755% |
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
| down_ticks | 165149.000000 | 165029.000000 | -0.0727% | 166069.000000 | 165610.000000 | -0.2764% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9908.000000 | 9772.000000 | -1.3726% | 9869.500000 | 9853.000000 | -0.1672% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2686.000000 | 2690.000000 | +0.1489% | 2688.500000 | 2687.500000 | -0.0372% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 430470.000000 | 429939.000000 | -0.1234% | 431062.000000 | 431399.500000 | +0.0783% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1243.000000 | 1284.000000 | +3.2985% | 1258.000000 | 1273.500000 | +1.2321% |
| generation_final_norm_ticks | 923.000000 | 918.000000 | -0.5417% | 916.500000 | 925.000000 | +0.9274% |
| generation_lm_head_argmax_ticks | 7006.000000 | 7144.000000 | +1.9697% | 6891.500000 | 6701.500000 | -2.7570% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 130709.000000 | 132108.000000 | +1.0703% | 129739.000000 | 129535.500000 | -0.1569% |
| generation_lm_head_expand_ticks | 113015.000000 | 114259.000000 | +1.1007% | 112270.000000 | 112028.000000 | -0.2156% |
| generation_lm_head_hmx_tail_wait_ticks | 2400.000000 | 2587.000000 | +7.7917% | 2397.500000 | 2391.000000 | -0.2711% |
| generation_lm_head_hmx_ticks | 129703.000000 | 131092.000000 | +1.0709% | 128735.000000 | 128535.000000 | -0.1554% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 238.000000 | 233.000000 | -2.1008% | 238.000000 | 232.500000 | -2.3109% |
| generation_lm_head_scale_init_ticks | 577.000000 | 560.000000 | -2.9463% | 568.000000 | 566.000000 | -0.3521% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 131632.000000 | 133026.000000 | +1.0590% | 130659.500000 | 130458.000000 | -0.1542% |
| generation_lm_head_weight_dma_ticks | 129812.000000 | 131210.000000 | +1.0769% | 128863.000000 | 128645.000000 | -0.1692% |
| generation_lm_head_weight_dma_wait_ticks | 2987.000000 | 2757.000000 | -7.7000% | 2904.000000 | 2894.000000 | -0.3444% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 134607.000000 | 139557.000000 | +3.6774% | 135861.500000 | 135450.000000 | -0.3029% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2282.291250 | 2542.812417 | +11.4149% | 2305.234333 | 2563.073042 | +11.1849% |
| host_us | 63125.260000 | 63290.052000 | +0.2611% | 63194.297000 | 63358.516000 | +0.2599% |
| host_wall_ns | 63125260.000000 | 63290052.000000 | +0.2611% | 63194297.000000 | 63358516.000000 | +0.2599% |
| input_norm_ticks | 9441.000000 | 9457.000000 | +0.1695% | 9457.500000 | 9461.000000 | +0.0370% |
| input_stage_ticks | 16.000000 | 12.000000 | -25.0000% | 10.000000 | 10.000000 | +0.0000% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1168185.000000 | 1166347.000000 | -0.1573% | 1168957.500000 | 1167366.000000 | -0.1361% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 455.000000 | 448.000000 | -1.5385% | 450.500000 | 443.000000 | -1.6648% |
| ledger_named_ticks | 1168185.000000 | 1166347.000000 | -0.1573% | 1168957.500000 | 1167366.000000 | -0.1361% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7331.000000 | 7346.000000 | +0.2046% | 7436.500000 | 7494.500000 | +0.7799% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96935.000000 | 96347.000000 | -0.6066% | 96971.000000 | 96853.000000 | -0.1217% |
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
| post_attention_norm_ticks | 24.000000 | 20.000000 | -16.6667% | 23.500000 | 22.500000 | -4.2553% |
| post_attention_residual_ticks | 9073.000000 | 9053.000000 | -0.2204% | 9073.000000 | 9069.500000 | -0.0386% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684969.000000 | 682130.000000 | -0.4145% | 686637.500000 | 686904.500000 | +0.0389% |
| projection_pack_ticks | 3165.000000 | 3158.000000 | -0.2212% | 3166.500000 | 3167.500000 | +0.0316% |
| projection_unpack_ticks | 10994.000000 | 11198.000000 | +1.8556% | 11193.500000 | 11236.500000 | +0.3842% |
| qk_norm_rope_ticks | 23.000000 | 22.000000 | -4.3478% | 22.000000 | 23.000000 | +4.5455% |
| qkv_projection_ticks | 225295.000000 | 224346.000000 | -0.4212% | 225103.000000 | 225357.000000 | +0.1128% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 895.000000 | 882.000000 | -1.4525% | 894.500000 | 903.000000 | +0.9503% |
| runtime_teardown_ticks | 906.000000 | 883.000000 | -2.5386% | 865.500000 | 888.500000 | +2.6574% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6583.000000 | 6506.000000 | -1.1697% | 6574.500000 | 6577.000000 | +0.0380% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3360.000000 | 3300.000000 | -1.7857% | 3353.000000 | 3314.500000 | -1.1482% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 176.000000 | 165.000000 | -6.2500% | 173.000000 | 168.500000 | -2.6012% |
| total_ticks | 1167290.000000 | 1165465.000000 | -0.1563% | 1168056.500000 | 1166462.000000 | -0.1365% |
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
| w4f16_cross_prefetch_lifetime_ticks | 249933.000000 | 249472.000000 | -0.1844% | 249936.500000 | 249798.500000 | -0.0552% |
| w4f16_cross_prefetch_wait_ticks | 1266.000000 | 1041.000000 | -17.7725% | 1064.000000 | 1023.500000 | -3.8064% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 78781.000000 | 78598.000000 | -0.2323% | 78015.500000 | 75530.000000 | -3.1859% |
| w4f16_expand_ticks | 618003.000000 | 616468.000000 | -0.2484% | 618713.500000 | 619189.000000 | +0.0769% |
| w4f16_expand_work_ticks | 1703275.000000 | 1705371.000000 | +0.1231% | 1698908.000000 | 1699765.000000 | +0.0504% |
| w4f16_gate_up_expand_pool_wait_ticks | 31323.000000 | 30169.000000 | -3.6842% | 31045.500000 | 30165.000000 | -2.8362% |
| w4f16_gate_up_expand_ticks | 253904.000000 | 253251.000000 | -0.2572% | 254112.500000 | 254443.500000 | +0.1303% |
| w4f16_gate_up_expand_work_ticks | 158146.000000 | 158853.000000 | +0.4471% | 158640.500000 | 159423.500000 | +0.4936% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7414.000000 | 7177.000000 | -3.1967% | 7409.500000 | 7321.000000 | -1.1944% |
| w4f16_gate_up_hmx_wait_ticks | 279476.000000 | 278451.000000 | -0.3668% | 279887.500000 | 280218.000000 | +0.1181% |
| w4f16_gate_up_stream_join_wait_ticks | 740.000000 | 966.000000 | +30.5405% | 646.500000 | 651.500000 | +0.7734% |
| w4f16_gate_up_stream_ready_wait_ticks | 1327.000000 | 1364.000000 | +2.7882% | 1250.500000 | 1362.000000 | +8.9164% |
| w4f16_gate_up_stream_work_ticks | 134369.000000 | 134327.000000 | -0.0313% | 134395.000000 | 134369.500000 | -0.0190% |
| w4f16_gate_up_weight_dma_ticks | 560871.000000 | 559511.000000 | -0.2425% | 562199.000000 | 562662.500000 | +0.0824% |
| w4f16_hmx_tail_wait_ticks | 18947.000000 | 18645.000000 | -1.5939% | 18960.500000 | 18813.500000 | -0.7753% |
| w4f16_prefetch_wait_ticks | 29681.000000 | 28589.000000 | -3.6791% | 30059.500000 | 29878.500000 | -0.6021% |
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
| weight_dma_ticks | 1143261.000000 | 1141393.000000 | -0.1634% | 1144338.500000 | 1143672.500000 | -0.0582% |


## Control C0 versus output-selected C decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.000000 | 8.400000 | -6.6667% | 9.066667 | 9.200000 | +1.4706% |
| attention_av_hmx_ticks | 7682.266667 | 7677.000000 | -0.0686% | 7755.766667 | 7839.066667 | +1.0740% |
| attention_av_pack_ticks | 3024.733333 | 3035.466667 | +0.3549% | 3041.533333 | 3043.300000 | +0.0581% |
| attention_av_unpack_ticks | 4011.200000 | 4004.000000 | -0.1795% | 4010.766667 | 4007.133333 | -0.0906% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 7785.933333 | 7795.266667 | +0.1199% | 7838.400000 | 7918.933333 | +1.0274% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6388.933333 | 6404.800000 | +0.2483% | 6384.533333 | 6390.900000 | +0.0997% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132671.933333 | 132546.466667 | -0.0946% | 132694.666667 | 132599.333333 | -0.0718% |
| attention_ticks | 643296.466667 | 643272.133333 | -0.0038% | 643563.500000 | 643530.300000 | -0.0052% |
| attention_unattributed_ticks | 481731.466667 | 481809.133333 | +0.0161% | 481781.366667 | 481808.166667 | +0.0056% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 284.066667 | 284.400000 | +0.1173% | 282.766667 | 285.000000 | +0.7898% |
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
| down_ticks | 165761.600000 | 165967.200000 | +0.1240% | 166042.766667 | 166035.600000 | -0.0043% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29954.800000 | 29959.666667 | +0.0162% | 29953.766667 | 29946.266667 | -0.0250% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2672.200000 | 2671.600000 | -0.0225% | 2671.533333 | 2671.000000 | -0.0200% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 429911.533333 | 430314.600000 | +0.0938% | 430143.166667 | 430184.833333 | +0.0097% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 31.466667 | 33.866667 | +7.6271% | 37.833333 | 36.866667 | -2.5551% |
| generation_final_norm_ticks | 43.000000 | 44.866667 | +4.3411% | 43.066667 | 40.866667 | -5.1084% |
| generation_lm_head_argmax_ticks | 6964.933333 | 7175.800000 | +3.0275% | 6841.900000 | 6726.166667 | -1.6915% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 130665.800000 | 131541.133333 | +0.6699% | 129936.933333 | 129520.833333 | -0.3202% |
| generation_lm_head_expand_ticks | 113313.066667 | 113910.866667 | +0.5276% | 112682.166667 | 112236.400000 | -0.3956% |
| generation_lm_head_hmx_tail_wait_ticks | 2306.200000 | 2416.933333 | +4.8015% | 2349.100000 | 2313.633333 | -1.5098% |
| generation_lm_head_hmx_ticks | 129745.066667 | 130624.466667 | +0.6778% | 129009.500000 | 128603.766667 | -0.3145% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 205.333333 | 198.866667 | -3.1494% | 210.600000 | 203.533333 | -3.3555% |
| generation_lm_head_scale_init_ticks | 568.133333 | 569.000000 | +0.1525% | 569.800000 | 568.500000 | -0.2282% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 130708.800000 | 131586.000000 | +0.6711% | 129980.400000 | 129561.000000 | -0.3227% |
| generation_lm_head_weight_dma_ticks | 129848.866667 | 130728.600000 | +0.6775% | 129116.300000 | 128705.433333 | -0.3182% |
| generation_lm_head_weight_dma_wait_ticks | 2875.333333 | 2804.733333 | -2.4554% | 2861.433333 | 2861.233333 | -0.0070% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 127960.533333 | 127803.533333 | -0.1227% | 128229.666667 | 128746.566667 | +0.4031% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2253.503472 | 1832.913200 | -18.6638% | 1785.860917 | 1755.011994 | -1.7274% |
| host_us | 93137.583333 | 92813.913200 | -0.3475% | 92681.572800 | 92659.765533 | -0.0235% |
| host_wall_ns | 93137583.333333 | 92813913.200000 | -0.3475% | 92681572.800000 | 92659765.533333 | -0.0235% |
| input_norm_ticks | 9345.800000 | 9351.200000 | +0.0578% | 9354.866667 | 9356.566667 | +0.0182% |
| input_stage_ticks | 7.133333 | 7.533333 | +5.6075% | 8.033333 | 7.766667 | -3.3195% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1744974.333333 | 1746835.200000 | +0.1066% | 1746038.000000 | 1745044.066667 | -0.0569% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 319.400000 | 321.600000 | +0.6888% | 322.066667 | 321.866667 | -0.0621% |
| ledger_named_ticks | 1744974.333333 | 1746835.200000 | +0.1066% | 1746038.000000 | 1745044.066667 | -0.0569% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 5772.066667 | 6191.466667 | +7.2660% | 6153.566667 | 6139.233333 | -0.2329% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96512.466667 | 96422.133333 | -0.0936% | 96506.566667 | 96506.300000 | -0.0003% |
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
| post_attention_norm_ticks | 16.466667 | 17.266667 | +4.8583% | 17.066667 | 17.266667 | +1.1719% |
| post_attention_residual_ticks | 9011.266667 | 9023.200000 | +0.1324% | 9015.000000 | 9017.000000 | +0.0222% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684790.466667 | 684974.333333 | +0.0269% | 685473.733333 | 685303.200000 | -0.0249% |
| projection_pack_ticks | 419.066667 | 421.600000 | +0.6045% | 418.400000 | 421.000000 | +0.6214% |
| projection_unpack_ticks | 11197.000000 | 11272.200000 | +0.6716% | 11237.533333 | 11232.433333 | -0.0454% |
| qk_norm_rope_ticks | 18.266667 | 18.933333 | +3.6496% | 17.533333 | 17.500000 | -0.1901% |
| qkv_projection_ticks | 224487.200000 | 224528.800000 | +0.0185% | 224663.166667 | 224766.066667 | +0.0458% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 617.666667 | 617.000000 | -0.1079% | 620.800000 | 620.766667 | -0.0054% |
| runtime_teardown_ticks | 577.200000 | 578.866667 | +0.2888% | 576.633333 | 577.033333 | +0.0694% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4237.266667 | 4245.333333 | +0.1904% | 4258.100000 | 4258.833333 | +0.0172% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21335.933333 | 21334.066667 | -0.0087% | 21333.966667 | 21326.766667 | -0.0337% |
| scan_cache_stage_ticks | 13685.000000 | 13754.533333 | +0.5081% | 13737.533333 | 13756.066667 | +0.1349% |
| scan_dynamic_attention_ticks | 643247.600000 | 643221.600000 | -0.0040% | 643513.700000 | 643480.200000 | -0.0052% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 41.066667 | 39.600000 | -3.5714% | 38.666667 | 39.300000 | +1.6379% |
| total_ticks | 1744356.666667 | 1746218.200000 | +0.1067% | 1745417.400000 | 1744419.133333 | -0.0572% |
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
| w4f16_cross_prefetch_lifetime_ticks | 832287.133333 | 832587.533333 | +0.0361% | 832777.500000 | 832810.400000 | +0.0040% |
| w4f16_cross_prefetch_wait_ticks | 1103.133333 | 1176.733333 | +6.6719% | 1130.900000 | 1133.733333 | +0.2505% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 79287.666667 | 80363.600000 | +1.3570% | 79135.266667 | 77570.566667 | -1.9772% |
| w4f16_expand_ticks | 617522.666667 | 617898.466667 | +0.0609% | 618156.466667 | 617957.933333 | -0.0321% |
| w4f16_expand_work_ticks | 1703626.466667 | 1707808.200000 | +0.2455% | 1702441.266667 | 1698980.166667 | -0.2033% |
| w4f16_gate_up_expand_pool_wait_ticks | 31278.200000 | 31419.933333 | +0.4531% | 31391.666667 | 31361.300000 | -0.0967% |
| w4f16_gate_up_expand_ticks | 253810.133333 | 254174.333333 | +0.1435% | 253863.033333 | 253992.133333 | +0.0509% |
| w4f16_gate_up_expand_work_ticks | 158177.733333 | 158469.133333 | +0.1842% | 158031.500000 | 158126.366667 | +0.0600% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7253.466667 | 7177.533333 | -1.0469% | 7348.566667 | 7360.966667 | +0.1687% |
| w4f16_gate_up_hmx_wait_ticks | 279389.400000 | 279657.333333 | +0.0959% | 279554.866667 | 279636.866667 | +0.0293% |
| w4f16_gate_up_stream_join_wait_ticks | 653.733333 | 622.600000 | -4.7624% | 653.433333 | 628.766667 | -3.7749% |
| w4f16_gate_up_stream_ready_wait_ticks | 1333.600000 | 1415.666667 | +6.1538% | 1315.566667 | 1331.300000 | +1.1959% |
| w4f16_gate_up_stream_work_ticks | 134394.133333 | 134379.000000 | -0.0113% | 134385.600000 | 134376.500000 | -0.0068% |
| w4f16_gate_up_weight_dma_ticks | 560360.666667 | 560991.533333 | +0.1126% | 560672.700000 | 560730.733333 | +0.0104% |
| w4f16_hmx_tail_wait_ticks | 18718.066667 | 18645.333333 | -0.3886% | 18854.566667 | 18927.500000 | +0.3868% |
| w4f16_prefetch_wait_ticks | 29942.266667 | 29971.800000 | +0.0986% | 29972.800000 | 30042.866667 | +0.2338% |
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
| weight_dma_ticks | 1139762.600000 | 1141615.133333 | +0.1625% | 1140010.533333 | 1139695.433333 | -0.0276% |


## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63025.937,
      "prefill_tokens_per_second": 1015.4549546800074,
      "decode_tokens": 15,
      "decode_total_host_us": 1388596.094,
      "decode_tokens_per_second": 10.802277253129015
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62347.08349999999,
      "prefill_tokens_per_second": 1026.5115288031077,
      "decode_tokens": 15,
      "decode_total_host_us": 1388479.7644999998,
      "decode_tokens_per_second": 10.80318228865337
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63194.297000000006,
      "prefill_tokens_per_second": 1012.7496156813011,
      "decode_tokens": 15,
      "decode_total_host_us": 1390223.592,
      "decode_tokens_per_second": 10.789631312773752
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63358.516,
      "prefill_tokens_per_second": 1010.1246689553145,
      "decode_tokens": 15,
      "decode_total_host_us": 1389896.4829999998,
      "decode_tokens_per_second": 10.79217062814886
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.38375244825754606,
      "decode": 0.018878336428107545
    },
    "C": {
      "prefill": -0.05846968245281081,
      "decode": 0.03303114342818603
    }
  }
}
```


## Experiment context and checks

# EXP-0224: post-GPTQ output-aware per-row scales

Only the approved first direction: original-coordinate A and fresh gamma-fold/fixed H2048 R1/H128 R2 C. Controls are frozen EXP0221 A and EXP0223 C. All196transformer projections per candidate regenerated from verified Qwen3-origin. Head, embedding and all norms byte-identical to the corresponding control. No LPBQ/group scales, learned rotations, calibration expansion, runtime or other-recipe changes.

## Observed outcome and limits

Actual DSP A improves from EXP0221 absmax NLL3.873038/8of24 to3.660316/19of24; C improves from EXP0223 clipping3.956556/9of24 to3.892162/17of24. Both predeclared effectiveness conjunctions pass. New original-coordinate A is better than new rotated C on both metrics; cross-coordinate C>A diagnostic is false. No incremental rotation benefit over the improved unrotated method is established. No baseline promotion.

Frozen F16A16 diagnostic is NLL3.633608/22of24. New A is close in this512target NLL diagnostic but still misses5strict tasks versus2for F16. The fixed lightweight suite cannot certify broad quality. All four open outputs are nonempty for both candidates, but A repeats the English quiet-activity instruction rather than answering, and C's Chinese quiet-activity answer suggests reading in the rain. Exact texts and token IDs are retained; readable output is not general usability acceptance. Software A3.661083/19of24 and C3.888549/16of24 remain distinct from actual DSP (C17of24); no bit-exact software/DSP quality claim.

A selects wider-than-clipping candidates on6492of573440rows (1.1321%); on the same current calibration inputs those rows account for91.5298% of the summed clipping-candidate projection SSE. C selects wider candidates on1681rows (0.2931%), accounting for39.1738%. Both largest within-projection improvements are layer2/down (zero-based, the third layer): A clipping SSE4.328490e9 versus selected7.416848e6; C3.065701e8 versus1.456459e8. This directly shows that weight-local clipping can be damaging for a small set of output rows despite looking favorable for most rows. Cross-projection sums are unnormalized and input scales differ; these diagnostics are not comparable whole-model errors, and no isolated-layer ablation proves that layer2/down alone causes the task improvement.

Most rows selecting clipping does not mean most final model bytes are identical to EXP0223: true-sequential GPTQ observes altered predecessor activations and therefore can alter subsequent integer codes. Local selection plus that propagation is the complete intervention. No scores or candidate settings were adjusted after evaluation.

Current-run paired throughput changes: A prefill -0.3838%, decode+0.0189%; C prefill-0.0585%, decode+0.0330%. No confirmed speed improvement is claimed. A marginal prefill median looks faster (62.347ms versus63.026ms), while the median of paired ratios is slightly slower. These are distinct nonlinear summaries; retained rounds show Host-wall variation and the overview Host-DSP boundary shifts. Ranking uses paired evidence, not the apparent marginal-median gain. No repeat confirmation or baseline promotion was performed.

Detailed selection diagnostics (same current candidate inputs):

```json
{
  "A": {
    "rows": 573440,
    "wider_rows": 6492,
    "wider_percent": 1.1321149553571428,
    "wider_rows_share_of_clipping_sse": 0.9152983548253183,
    "clipping_sse": 4828367734.726038,
    "selected_sse": 481548182.33042884,
    "top_projection_improvements": [
      {
        "projection": "layer2/down",
        "rows": 2048,
        "wider_rows": 523,
        "clipping_sse": 4328489656.702142,
        "selected_sse": 7416847.874061054,
        "improvement": 4321072808.82808
      },
      {
        "projection": "layer27/down",
        "rows": 2048,
        "wider_rows": 13,
        "clipping_sse": 20455562.73708816,
        "selected_sse": 13653027.16876295,
        "improvement": 6802535.568325208
      },
      {
        "projection": "layer26/down",
        "rows": 2048,
        "wider_rows": 6,
        "clipping_sse": 10065354.674345072,
        "selected_sse": 6830363.967734422,
        "improvement": 3234990.7066106503
      },
      {
        "projection": "layer27/up",
        "rows": 6144,
        "wider_rows": 141,
        "clipping_sse": 5457868.742694486,
        "selected_sse": 2504577.7659612563,
        "improvement": 2953290.97673323
      },
      {
        "projection": "layer25/v",
        "rows": 1024,
        "wider_rows": 345,
        "clipping_sse": 26207624.80544918,
        "selected_sse": 24098731.455289103,
        "improvement": 2108893.3501600735
      },
      {
        "projection": "layer25/k",
        "rows": 1024,
        "wider_rows": 234,
        "clipping_sse": 17388131.529697567,
        "selected_sse": 15810209.349759284,
        "improvement": 1577922.1799382842
      },
      {
        "projection": "layer26/v",
        "rows": 1024,
        "wider_rows": 203,
        "clipping_sse": 23173661.167926177,
        "selected_sse": 21852892.444140818,
        "improvement": 1320768.723785357
      },
      {
        "projection": "layer24/k",
        "rows": 1024,
        "wider_rows": 219,
        "clipping_sse": 15167385.239043508,
        "selected_sse": 13852529.8429753,
        "improvement": 1314855.3960682102
      }
    ],
    "scope": "same candidate inputs, unnormalized projection SSE sums, not whole-model attribution"
  },
  "C": {
    "rows": 573440,
    "wider_rows": 1681,
    "wider_percent": 0.2931431361607143,
    "wider_rows_share_of_clipping_sse": 0.3917377221452322,
    "clipping_sse": 608676395.1360729,
    "selected_sse": 447747800.22370183,
    "top_projection_improvements": [
      {
        "projection": "layer2/down",
        "rows": 2048,
        "wider_rows": 1398,
        "clipping_sse": 306570092.2729726,
        "selected_sse": 145645923.2900291,
        "improvement": 160924168.98294353
      },
      {
        "projection": "layer26/o",
        "rows": 2048,
        "wider_rows": 14,
        "clipping_sse": 3589271.7742287135,
        "selected_sse": 3586573.0205577808,
        "improvement": 2698.753670932366
      },
      {
        "projection": "layer23/down",
        "rows": 2048,
        "wider_rows": 3,
        "clipping_sse": 2554597.578367616,
        "selected_sse": 2553551.9824226354,
        "improvement": 1045.5959449807824
      },
      {
        "projection": "layer11/o",
        "rows": 2048,
        "wider_rows": 212,
        "clipping_sse": 38339.49481931769,
        "selected_sse": 38020.48995270871,
        "improvement": 319.0048666089756
      },
      {
        "projection": "layer26/down",
        "rows": 2048,
        "wider_rows": 1,
        "clipping_sse": 3934236.857184209,
        "selected_sse": 3933958.834455242,
        "improvement": 278.02272896730165
      },
      {
        "projection": "layer21/v",
        "rows": 1024,
        "wider_rows": 1,
        "clipping_sse": 3091398.816528756,
        "selected_sse": 3091352.4686924024,
        "improvement": 46.347836353622824
      },
      {
        "projection": "layer7/down",
        "rows": 2048,
        "wider_rows": 3,
        "clipping_sse": 15423.90197140899,
        "selected_sse": 15415.30736804275,
        "improvement": 8.594603366240495
      },
      {
        "projection": "layer23/gate",
        "rows": 6144,
        "wider_rows": 1,
        "clipping_sse": 1139744.6882376792,
        "selected_sse": 1139736.8863008146,
        "improvement": 7.801936864571331
      }
    ],
    "scope": "same candidate inputs, unnormalized projection SSE sums, not whole-model attribution"
  }
}
```

## Fixed method

Three candidates per output row: absmax/7, midpoint between absmax and EXP0223 L2.4 clipping scale, and that clipping scale. Each runs complete GPTQ before selection. Score=sum over all8192calibration positions of squared linear output error using FP16(codes*scale).float() minus original FP32 weights; FP32 GEMM in512token chunks, FP64 sum. Undamped actual-input error, not RTN or GPTQ damped proxy. Ties choose earlier/larger range. Per-row choices, all three scores/scales and selected scores retained.

Frozen64x128 bilingual EXP0221 training calibration, same act-order,1% damping,FP64 factor/FP32 error compensation,block128 computational only. True sequential QKV->O->Gate/Up->Down and layer ordering uses the newly quantized predecessors. Included endpoints guarantee nonincreasing selected local score on these identical current inputs; this does not guarantee whole-model quality or imply that old/new projection inputs are identical. No evaluation/holdout selection or tuning.

## Actual DSP quality

Frozen qbh-lite-v1:512conditional targets,24strict tasks,4open prefixes; a lightweight diagnostic, not general quality certification. Software results remain separate.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| A0 EXP0221 absmax（冻结） | 3.8730 | 48.09 | 8/24 | 62.50% |
| C0 EXP0223 clipping（冻结） | 3.9566 | 52.28 | 9/24 | 60.55% |
| output-selected GPTQ A | 3.6603 | 38.87 | 19/24 | 66.80% |
| output-selected GPTQ C | 3.8922 | 49.02 | 17/24 | 66.21% |

Predeclared lower-NLL AND more-task effectiveness: {"A": true, "C": true}. Aggregate pass iff any effectiveness true. Candidate C versus candidate A both-metric diagnostic: False. No promotion.

## Scale selection

| Variant | Rows | Absmax | Midpoint | L2.4 clip | Mean range/absmax |
|---|---|---|---|---|---|
| A | 573440 | 0.06% | 1.07% | 98.87% | 0.7064 |
| C | 573440 | 0.10% | 0.20% | 99.71% | 0.7204 |

Summed projection SSEs are unnormalized local sums from different projection inputs/shapes, retained for diagnostics rather than whole-model comparison. All selected row scores no worse than both endpoints on identical inputs.

```json
{
  "A": {
    "rows": 573440,
    "candidate_names": [
      "absmax",
      "midpoint",
      "weight_L2.4_clip"
    ],
    "choice_histogram": [
      339,
      6153,
      566948
    ],
    "mean_selected_ratio": 0.706375977654378,
    "summed_projection_sse": [
      837812627.9120908,
      1460933357.9753928,
      4828367734.726042
    ],
    "summed_selected_sse": 481548182.33042884
  },
  "C": {
    "rows": 573440,
    "candidate_names": [
      "absmax",
      "midpoint",
      "weight_L2.4_clip"
    ],
    "choice_histogram": [
      558,
      1123,
      571759
    ],
    "mean_selected_ratio": 0.7203844178340659,
    "summed_projection_sse": [
      1095039788.8033412,
      687558006.5351466,
      608676395.136073
    ],
    "summed_selected_sse": 447747800.22370183
  }
}
```

## Checks and provenance

Independent NumPy full-input scoring and choices, explicit-scale dense GPTQ parity, original absmax parity, zero-row ties and packing pass.392projection packing and112staged/HF forward checks pass. FP32 transformed equivalence, independent16speed tokens and all quick/full/repeat consistency pass.

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
      "source_head": "09aec376d74638703a497625e4e531a77c8db47e",
      "files": {
        "experiment_exp0224.py": "97f7c2092d181549c4d05f521d97163f43f64c938f064eebdb09aa87b4c75533",
        "output_scale_exp0224.py": "fae7c192987ec91124b620acb1d25d62a84326c56e14935e4ae78cb513fc4a2d",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1"
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
      "source_head": "dfee1b1c3b3ef720ba2fc1d224e522f048218bab",
      "files": {
        "experiment_exp0224.py": "97f7c2092d181549c4d05f521d97163f43f64c938f064eebdb09aa87b4c75533",
        "output_scale_exp0224.py": "fae7c192987ec91124b620acb1d25d62a84326c56e14935e4ae78cb513fc4a2d",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1"
      }
    }
  }
}
```

Software diagnostic:

```json
{
  "A": {
    "nll": 3.661082625389099,
    "ppl": 38.903437930257674,
    "tasks_correct": 19,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.66796875,
    "language_nll": {
      "zh": 3.9930894672870636,
      "en": 3.3290757834911346
    }
  },
  "C": {
    "nll": 3.888548944145441,
    "ppl": 48.83996556307159,
    "tasks_correct": 16,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.66796875,
    "language_nll": {
      "zh": 4.127618625760078,
      "en": 3.6494792625308037
    }
  }
}
```

## Complete profiling

A0/A/C0/C rotating four-way1warmup,5short,10formal;640formal invocation and17920layer ledgers. M64 prefill plus15feedback decodes, scoring off; direct complete28-layer Host wall includes final norm/head/greedy/persistent KV, excludes cold staging and separately logged WSL tokenizer/detokenizer. Fixed16tokens continue after EOS if encountered and are diagnostic throughput. No partial-model extrapolation.

Frozen EXP0218 source d981072513d06ed61731c14743c76ac6bc81617f ABI108, embedded218 intentionally retained, outer224. Exact8MiB VTCM,zero timed intermediate hidden/logits DDR/spill,one FastRPC/one HMX owner,no QNN. Offline model work is not DSP timing. Scaffolds named layer14 execute all28layers; inherited replay references remain historical and unused.

Overview candidate chosen by lowest new DSP NLL, ties more tasks then alphabetical; report selection does not promote. F16/U8 are frozen EXP0218 nonpaired references with different weights; no activation-only attribution. Full report retains every repeat1/repeat10 numeric control/candidate field, exclusive additive ledgers and overlapping engine/wait counters.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 output-selected GPTQ A EXP-0224 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 392.6 (0.63%) | 247.4 (0.63%) | +58.71% |
| Input RMSNorm | 489.7 (0.61%) | 492.6 (0.79%) | 554.0 (1.40%) | -11.08% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11746.1 (18.84%) | 7052.7 (17.82%) | +66.55% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3964.9 (6.36%) | 3214.5 (8.12%) | +23.34% |
| O projection | 5757.7 (7.14%) | 5050.1 (8.10%) | 1256.4 (3.17%) | +301.95% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.7 (0.76%) | 654.0 (1.65%) | -27.57% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22481.7 (36.06%) | 14442.7 (36.49%) | +55.66% |
| Down | 13447.9 (16.67%) | 8639.3 (13.86%) | 3428.5 (8.66%) | +151.99% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.73% |
| KV carrier conversion | 174.0 (0.22%) | 173.2 (0.28%) | 203.2 (0.51%) | -14.79% |
| KV append DMA | 343.6 (0.43%) | 342.4 (0.55%) | 463.9 (1.17%) | -26.17% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.28% |
| Layer bookkeeping | 23.9 (0.03%) | 23.1 (0.04%) | 23.2 (0.06%) | -0.34% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.0 (0.01%) | 22.5 (0.06%) | -60.07% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 94.2 (0.15%) | 105.6 (0.27%) | -10.80% |
| Embedding | 68.1 (0.08%) | 66.6 (0.11%) | 62.4 (0.16%) | +6.59% |
| Final model RMSNorm | 49.7 (0.06%) | 48.0 (0.08%) | 3.7 (0.01%) | +1189.51% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6725.6 (10.79%) | 5284.7 (13.35%) | +27.27% |
| Host–DSP 边界 | 2374.1 (2.94%) | 1533.8 (2.46%) | 2466.6 (6.23%) | -37.82% |
| 完整 Host wall | 80692.2 (100.00%) | 62347.1 (100.00%) | 39575.9 (100.00%) | +57.54% |

## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63025.937,
      "prefill_tokens_per_second": 1015.4549546800074,
      "decode_tokens": 15,
      "decode_total_host_us": 1388596.094,
      "decode_tokens_per_second": 10.802277253129015
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62347.08349999999,
      "prefill_tokens_per_second": 1026.5115288031077,
      "decode_tokens": 15,
      "decode_total_host_us": 1388479.7644999998,
      "decode_tokens_per_second": 10.80318228865337
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63194.297000000006,
      "prefill_tokens_per_second": 1012.7496156813011,
      "decode_tokens": 15,
      "decode_total_host_us": 1390223.592,
      "decode_tokens_per_second": 10.789631312773752
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63358.516,
      "prefill_tokens_per_second": 1010.1246689553145,
      "decode_tokens": 15,
      "decode_total_host_us": 1389896.4829999998,
      "decode_tokens_per_second": 10.79217062814886
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.38375244825754606,
      "decode": 0.018878336428107545
    },
    "C": {
      "prefill": -0.05846968245281081,
      "decode": 0.03303114342818603
    }
  }
}
```

## Retained artifacts and reproduction

Models D:/llm_exp/models/qwen3-block-htp/exp0224; results D:/llm_exp/results/qwen3-block-htp/exp0224. experiment_exp0224.py A|C; measure_exp0224.py deploy/quick/full/repeat and warmup/short/formal; summarize_exp0224.py; close_exp0224.py report then commit/sync/archive. Fresh paths and registered experiment required.392row-search NPZs,56hidden checkpoints, source archive, runtime binaries, manifests and all logs bound by closure and evidence ledgers. No overwrite or automatic further optimization.
