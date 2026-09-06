# EXP-0227 complete profiling comparison

Frozen ABI108 binary; A0=EXP0224 A, R0=EXP0225 step100, A/R=their scale-reconstructed packages, each with an independent software token sequence. One M64 plus15 feedback steps;5short then10four-way rotating formal rounds. Other recipe columns historical nonpaired, changed W4 prevents activation-only attribution. Quality scoring disabled. All numeric fields retained; additive ledger fields exclusive, engine/wait counters overlapping. Host-DSP boundary per record = Host wall minus DSP invocation. Percent changes below candidate/control minus one; positive timing changes are slower.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 block scale reconstruction A EXP-0227 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 380.9 (0.60%) | 247.4 (0.63%) | +53.95% |
| Input RMSNorm | 489.7 (0.61%) | 492.4 (0.78%) | 554.0 (1.40%) | -11.11% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11731.7 (18.61%) | 7052.7 (17.82%) | +66.34% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3959.1 (6.28%) | 3214.5 (8.12%) | +23.16% |
| O projection | 5757.7 (7.14%) | 5046.3 (8.01%) | 1256.4 (3.17%) | +301.64% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.4 (0.75%) | 654.0 (1.65%) | -27.62% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22464.7 (35.64%) | 14442.7 (36.49%) | +55.54% |
| Down | 13447.9 (16.67%) | 8653.0 (13.73%) | 3428.5 (8.66%) | +152.39% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.83% |
| KV carrier conversion | 174.0 (0.22%) | 172.5 (0.27%) | 203.2 (0.51%) | -15.13% |
| KV append DMA | 343.6 (0.43%) | 341.9 (0.54%) | 463.9 (1.17%) | -26.29% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.35% |
| Layer bookkeeping | 23.9 (0.03%) | 22.9 (0.04%) | 23.2 (0.06%) | -1.01% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.0 (0.01%) | 22.5 (0.06%) | -60.19% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.2 (0.15%) | 105.6 (0.27%) | -11.76% |
| Embedding | 68.1 (0.08%) | 67.0 (0.11%) | 62.4 (0.16%) | +7.34% |
| Final model RMSNorm | 49.7 (0.06%) | 47.8 (0.08%) | 3.7 (0.01%) | +1182.52% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6630.5 (10.52%) | 5284.7 (13.35%) | +25.47% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2312.0 (3.67%) | 2466.6 (6.23%) | -6.27% |
| 完整 Host wall | 80692.2 (100.00%) | 63034.1 (100.00%) | 39575.9 (100.00%) | +59.27% |


## A0 versus R0 prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 15.000000 | 9.000000 | -40.0000% | 11.500000 | 12.000000 | +4.3478% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75950.000000 | 76145.000000 | +0.2567% | 75863.500000 | 76088.000000 | +0.2959% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 122.000000 | 120.000000 | -1.6393% | 121.000000 | 121.000000 | +0.0000% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76139.000000 | 76320.000000 | +0.2377% | 76047.500000 | 76263.500000 | +0.2840% |
| attention_unattributed_ticks | 67.000000 | 55.000000 | -17.9104% | 58.500000 | 53.500000 | -8.5470% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 435.000000 | 362.000000 | -16.7816% | 371.500000 | 374.000000 | +0.6729% |
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
| down_ticks | 166107.000000 | 166357.000000 | +0.1505% | 165840.000000 | 166347.500000 | +0.3060% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9746.000000 | 9892.000000 | +1.4981% | 9813.000000 | 9839.500000 | +0.2700% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2705.000000 | 2706.000000 | +0.0370% | 2690.000000 | 2690.000000 | +0.0000% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 432095.000000 | 431755.000000 | -0.0787% | 431030.500000 | 430866.000000 | -0.0382% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1310.000000 | 1300.000000 | -0.7634% | 1270.000000 | 1291.000000 | +1.6535% |
| generation_final_norm_ticks | 925.000000 | 926.000000 | +0.1081% | 922.000000 | 920.500000 | -0.1627% |
| generation_lm_head_argmax_ticks | 6073.000000 | 6107.000000 | +0.5599% | 6137.500000 | 6166.000000 | +0.4644% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 126374.000000 | 126816.000000 | +0.3498% | 126691.000000 | 127065.000000 | +0.2952% |
| generation_lm_head_expand_ticks | 109957.000000 | 110176.000000 | +0.1992% | 109896.500000 | 110269.500000 | +0.3394% |
| generation_lm_head_hmx_tail_wait_ticks | 2056.000000 | 2174.000000 | +5.7393% | 2204.000000 | 2196.000000 | -0.3630% |
| generation_lm_head_hmx_ticks | 125350.000000 | 125806.000000 | +0.3638% | 125660.500000 | 126034.000000 | +0.2972% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 232.000000 | 245.000000 | +5.6034% | 241.500000 | 243.500000 | +0.8282% |
| generation_lm_head_scale_init_ticks | 579.000000 | 563.000000 | -2.7634% | 574.000000 | 561.000000 | -2.2648% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127299.000000 | 127742.000000 | +0.3480% | 127616.000000 | 128002.000000 | +0.3025% |
| generation_lm_head_weight_dma_ticks | 125489.000000 | 125928.000000 | +0.3498% | 125806.500000 | 126184.000000 | +0.3001% |
| generation_lm_head_weight_dma_wait_ticks | 2771.000000 | 2935.000000 | +5.9184% | 2890.000000 | 2929.000000 | +1.3495% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 130370.000000 | 132551.000000 | +1.6729% | 134070.500000 | 134523.500000 | +0.3379% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2668.750000 | 2262.343750 | -15.2283% | 2402.031542 | 2286.380250 | -4.8147% |
| host_us | 63408.125000 | 63021.250000 | -0.6101% | 62962.864500 | 63048.411500 | +0.1359% |
| host_wall_ns | 63408125.000000 | 63021250.000000 | -0.6101% | 62962864.500000 | 63048411.500000 | +0.1359% |
| input_norm_ticks | 9491.000000 | 9508.000000 | +0.1791% | 9452.000000 | 9442.500000 | -0.1005% |
| input_stage_ticks | 15.000000 | 11.000000 | -26.6667% | 13.000000 | 11.000000 | -15.3846% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1166196.000000 | 1166571.000000 | +0.0322% | 1164688.500000 | 1166037.500000 | +0.1158% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 476.000000 | 439.000000 | -7.7731% | 462.000000 | 449.000000 | -2.8139% |
| ledger_named_ticks | 1166196.000000 | 1166571.000000 | +0.0322% | 1164688.500000 | 1166037.500000 | +0.1158% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7324.000000 | 7531.000000 | +2.8263% | 7343.000000 | 7361.500000 | +0.2519% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96569.000000 | 96812.000000 | +0.2516% | 96609.500000 | 96830.500000 | +0.2288% |
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
| post_attention_norm_ticks | 26.000000 | 24.000000 | -7.6923% | 25.000000 | 22.000000 | -12.0000% |
| post_attention_residual_ticks | 9103.000000 | 9067.000000 | -0.3955% | 9080.000000 | 9071.500000 | -0.0936% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 686353.000000 | 686585.000000 | +0.0338% | 685682.500000 | 686635.500000 | +0.1390% |
| projection_pack_ticks | 3182.000000 | 3163.000000 | -0.5971% | 3167.500000 | 3167.500000 | +0.0000% |
| projection_unpack_ticks | 11259.000000 | 11144.000000 | -1.0214% | 11156.500000 | 11218.500000 | +0.5557% |
| qk_norm_rope_ticks | 23.000000 | 23.000000 | +0.0000% | 23.500000 | 22.500000 | -4.2553% |
| qkv_projection_ticks | 225124.000000 | 224737.000000 | -0.1719% | 225106.500000 | 225014.500000 | -0.0409% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 982.000000 | 910.000000 | -7.3320% | 906.000000 | 906.500000 | +0.0552% |
| runtime_teardown_ticks | 965.000000 | 865.000000 | -10.3627% | 886.000000 | 885.000000 | -0.1129% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6487.000000 | 6622.000000 | +2.0811% | 6514.000000 | 6560.000000 | +0.7062% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3298.000000 | 3296.000000 | -0.0606% | 3303.000000 | 3300.000000 | -0.0908% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 208.000000 | 175.000000 | -15.8654% | 170.000000 | 171.000000 | +0.5882% |
| total_ticks | 1165214.000000 | 1165661.000000 | +0.0384% | 1163710.000000 | 1165104.000000 | +0.1198% |
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
| w4f16_cross_prefetch_lifetime_ticks | 249953.000000 | 249743.000000 | -0.0840% | 249728.000000 | 249646.500000 | -0.0326% |
| w4f16_cross_prefetch_wait_ticks | 1173.000000 | 1044.000000 | -10.9974% | 1142.000000 | 1036.500000 | -9.2382% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 73469.000000 | 74253.000000 | +1.0671% | 73705.500000 | 74695.000000 | +1.3425% |
| w4f16_expand_ticks | 619005.000000 | 618584.000000 | -0.0680% | 618663.000000 | 619309.500000 | +0.1045% |
| w4f16_expand_work_ticks | 1689565.000000 | 1688807.000000 | -0.0449% | 1690158.000000 | 1689393.000000 | -0.0453% |
| w4f16_gate_up_expand_pool_wait_ticks | 30487.000000 | 31060.000000 | +1.8795% | 30551.500000 | 30921.000000 | +1.2094% |
| w4f16_gate_up_expand_ticks | 255393.000000 | 254939.000000 | -0.1778% | 254402.000000 | 254308.000000 | -0.0369% |
| w4f16_gate_up_expand_work_ticks | 160674.000000 | 159414.000000 | -0.7842% | 159714.500000 | 158937.000000 | -0.4868% |
| w4f16_gate_up_hmx_tail_wait_ticks | 6945.000000 | 6889.000000 | -0.8063% | 7262.500000 | 7235.000000 | -0.3787% |
| w4f16_gate_up_hmx_wait_ticks | 281003.000000 | 280519.000000 | -0.1722% | 280138.000000 | 279976.500000 | -0.0577% |
| w4f16_gate_up_stream_join_wait_ticks | 664.000000 | 729.000000 | +9.7892% | 737.500000 | 637.000000 | -13.6271% |
| w4f16_gate_up_stream_ready_wait_ticks | 1440.000000 | 1424.000000 | -1.1111% | 1400.000000 | 1392.000000 | -0.5714% |
| w4f16_gate_up_stream_work_ticks | 134360.000000 | 134351.000000 | -0.0067% | 134395.000000 | 134396.000000 | +0.0007% |
| w4f16_gate_up_weight_dma_ticks | 564241.000000 | 563598.000000 | -0.1140% | 562142.000000 | 561976.500000 | -0.0294% |
| w4f16_hmx_tail_wait_ticks | 18551.000000 | 18513.000000 | -0.2048% | 18616.500000 | 19075.000000 | +2.4629% |
| w4f16_prefetch_wait_ticks | 30207.000000 | 30844.000000 | +2.1088% | 29926.000000 | 29434.500000 | -1.6424% |
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
| weight_dma_ticks | 1142618.000000 | 1143075.000000 | +0.0400% | 1140087.500000 | 1142519.500000 | +0.2133% |


## A0 versus R0 decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.466667 | 10.866667 | +14.7887% | 9.033333 | 8.766667 | -2.9520% |
| attention_av_hmx_ticks | 7714.266667 | 7876.466667 | +2.1026% | 7950.000000 | 7950.033333 | +0.0004% |
| attention_av_pack_ticks | 3043.066667 | 3034.933333 | -0.2673% | 3037.866667 | 3040.833333 | +0.0977% |
| attention_av_unpack_ticks | 4010.933333 | 4005.933333 | -0.1247% | 4010.466667 | 4002.666667 | -0.1945% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 7789.400000 | 8043.733333 | +3.2651% | 7980.366667 | 8028.933333 | +0.6086% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6379.600000 | 6381.066667 | +0.0230% | 6382.333333 | 6385.000000 | +0.0418% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132943.333333 | 132797.133333 | -0.1100% | 133049.200000 | 132878.733333 | -0.1281% |
| attention_ticks | 643739.066667 | 643886.266667 | +0.0229% | 644239.033333 | 644004.266667 | -0.0364% |
| attention_unattributed_ticks | 481858.466667 | 481747.000000 | -0.0231% | 481802.633333 | 481776.100000 | -0.0055% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.133333 | 284.266667 | -0.6524% | 284.166667 | 284.900000 | +0.2581% |
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
| down_ticks | 165825.333333 | 165654.733333 | -0.1029% | 166197.400000 | 166447.200000 | +0.1503% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29942.066667 | 29901.800000 | -0.1345% | 29926.033333 | 29896.433333 | -0.0989% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2670.400000 | 2670.800000 | +0.0150% | 2671.266667 | 2670.800000 | -0.0175% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 429997.666667 | 430108.933333 | +0.0259% | 430127.100000 | 429901.300000 | -0.0525% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 34.466667 | 34.933333 | +1.3540% | 30.666667 | 32.966667 | +7.5000% |
| generation_final_norm_ticks | 40.933333 | 39.733333 | -2.9316% | 39.933333 | 40.000000 | +0.1669% |
| generation_lm_head_argmax_ticks | 6237.666667 | 6184.733333 | -0.8486% | 6230.366667 | 6265.733333 | +0.5676% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 127004.400000 | 126957.000000 | -0.0373% | 126979.666667 | 127264.700000 | +0.2245% |
| generation_lm_head_expand_ticks | 110410.866667 | 110446.000000 | +0.0318% | 110389.466667 | 110611.633333 | +0.2013% |
| generation_lm_head_hmx_tail_wait_ticks | 2169.800000 | 2122.133333 | -2.1968% | 2180.533333 | 2116.066667 | -2.9565% |
| generation_lm_head_hmx_ticks | 126091.200000 | 126030.466667 | -0.0482% | 126058.800000 | 126340.433333 | +0.2234% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 202.866667 | 212.200000 | +4.6007% | 207.300000 | 209.933333 | +1.2703% |
| generation_lm_head_scale_init_ticks | 571.400000 | 566.200000 | -0.9100% | 569.733333 | 566.266667 | -0.6085% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127045.333333 | 126996.733333 | -0.0383% | 127019.233333 | 127305.000000 | +0.2250% |
| generation_lm_head_weight_dma_ticks | 126187.266667 | 126129.066667 | -0.0461% | 126157.566667 | 126439.633333 | +0.2236% |
| generation_lm_head_weight_dma_wait_ticks | 2877.533333 | 2912.400000 | +1.2117% | 2881.100000 | 2918.300000 | +1.2912% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 125101.866667 | 126296.466667 | +0.9549% | 125811.866667 | 127742.900000 | +1.5349% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1742.295083 | 1779.003450 | +2.1069% | 1790.605706 | 1745.954797 | -2.4936% |
| host_us | 92427.368000 | 92463.888867 | +0.0395% | 92538.857567 | 92494.206533 | -0.0483% |
| host_wall_ns | 92427368.000000 | 92463888.866667 | +0.0395% | 92538857.566667 | 92494206.533333 | -0.0483% |
| input_norm_ticks | 9354.266667 | 9364.733333 | +0.1119% | 9355.633333 | 9363.433333 | +0.0834% |
| input_stage_ticks | 7.933333 | 7.600000 | -4.2017% | 7.966667 | 8.566667 | +7.5314% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741153.400000 | 1741149.800000 | -0.0002% | 1741709.700000 | 1741718.100000 | +0.0005% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 320.533333 | 324.800000 | +1.3311% | 322.866667 | 322.866667 | +0.0000% |
| ledger_named_ticks | 1741153.400000 | 1741149.800000 | -0.0002% | 1741709.700000 | 1741718.100000 | +0.0005% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 5739.466667 | 5743.600000 | +0.0720% | 5682.866667 | 5681.233333 | -0.0287% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96322.266667 | 96153.733333 | -0.1750% | 96321.500000 | 96199.833333 | -0.1263% |
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
| post_attention_norm_ticks | 17.666667 | 17.000000 | -3.7736% | 17.266667 | 17.000000 | -1.5444% |
| post_attention_residual_ticks | 9005.666667 | 9008.733333 | +0.0341% | 9015.366667 | 9016.800000 | +0.0159% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684471.133333 | 684478.333333 | +0.0011% | 684721.133333 | 685038.766667 | +0.0464% |
| projection_pack_ticks | 417.133333 | 414.000000 | -0.7512% | 420.400000 | 420.200000 | -0.0476% |
| projection_unpack_ticks | 11102.533333 | 11114.866667 | +0.1111% | 11198.166667 | 11227.333333 | +0.2605% |
| qk_norm_rope_ticks | 18.400000 | 17.400000 | -5.4348% | 17.200000 | 17.433333 | +1.3566% |
| qkv_projection_ticks | 223984.666667 | 224071.933333 | +0.0390% | 224308.500000 | 224459.666667 | +0.0674% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 619.000000 | 619.733333 | +0.1185% | 622.066667 | 621.900000 | -0.0268% |
| runtime_teardown_ticks | 575.666667 | 573.266667 | -0.4169% | 575.933333 | 575.266667 | -0.1158% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4215.533333 | 4233.200000 | +0.4191% | 4220.633333 | 4221.633333 | +0.0237% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21325.600000 | 21329.666667 | +0.0191% | 21333.966667 | 21326.300000 | -0.0359% |
| scan_cache_stage_ticks | 13762.266667 | 13743.600000 | -0.1356% | 13763.700000 | 13743.366667 | -0.1477% |
| scan_dynamic_attention_ticks | 643688.400000 | 643836.533333 | +0.0230% | 644189.566667 | 643953.766667 | -0.0366% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 38.866667 | 36.866667 | -5.1458% | 39.066667 | 39.366667 | +0.7679% |
| total_ticks | 1740534.400000 | 1740530.066667 | -0.0002% | 1741090.200000 | 1741094.600000 | +0.0003% |
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
| w4f16_cross_prefetch_lifetime_ticks | 832568.600000 | 832653.600000 | +0.0102% | 833253.233333 | 832932.900000 | -0.0384% |
| w4f16_cross_prefetch_wait_ticks | 1140.866667 | 1117.200000 | -2.0744% | 1143.633333 | 1129.500000 | -1.2358% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75620.466667 | 75179.333333 | -0.5834% | 75515.100000 | 75578.233333 | +0.0836% |
| w4f16_expand_ticks | 617141.733333 | 617260.466667 | +0.0192% | 617708.766667 | 618059.300000 | +0.0567% |
| w4f16_expand_work_ticks | 1688914.466667 | 1688897.066667 | -0.0010% | 1689449.866667 | 1689447.033333 | -0.0002% |
| w4f16_gate_up_expand_pool_wait_ticks | 31747.866667 | 31620.400000 | -0.4015% | 31521.833333 | 31426.000000 | -0.3040% |
| w4f16_gate_up_expand_ticks | 254042.466667 | 254229.133333 | +0.0735% | 254057.733333 | 253854.366667 | -0.0800% |
| w4f16_gate_up_expand_work_ticks | 158048.533333 | 158216.266667 | +0.1061% | 158139.233333 | 158083.066667 | -0.0355% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7140.200000 | 7143.133333 | +0.0411% | 7234.633333 | 7277.566667 | +0.5934% |
| w4f16_gate_up_hmx_wait_ticks | 279495.000000 | 279696.866667 | +0.0722% | 279617.633333 | 279450.033333 | -0.0599% |
| w4f16_gate_up_stream_join_wait_ticks | 651.466667 | 637.466667 | -2.1490% | 649.833333 | 628.766667 | -3.2419% |
| w4f16_gate_up_stream_ready_wait_ticks | 1332.400000 | 1254.800000 | -5.8241% | 1337.466667 | 1312.666667 | -1.8543% |
| w4f16_gate_up_stream_work_ticks | 134385.266667 | 134387.400000 | +0.0016% | 134383.833333 | 134388.000000 | +0.0031% |
| w4f16_gate_up_weight_dma_ticks | 560704.866667 | 561135.133333 | +0.0767% | 560863.866667 | 560582.733333 | -0.0501% |
| w4f16_hmx_tail_wait_ticks | 18526.533333 | 18634.933333 | +0.5851% | 18774.266667 | 18837.266667 | +0.3356% |
| w4f16_prefetch_wait_ticks | 30089.466667 | 30068.666667 | -0.0691% | 29942.366667 | 29780.233333 | -0.5415% |
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
| weight_dma_ticks | 1135881.200000 | 1135990.866667 | +0.0097% | 1135687.700000 | 1136417.166667 | +0.0642% |


## A0 versus A prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 15.000000 | 20.000000 | +33.3333% | 11.500000 | 11.000000 | -4.3478% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75950.000000 | 75831.000000 | -0.1567% | 75863.500000 | 75833.500000 | -0.0395% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 122.000000 | 123.000000 | +0.8197% | 121.000000 | 121.000000 | +0.0000% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76139.000000 | 76016.000000 | -0.1615% | 76047.500000 | 76015.000000 | -0.0427% |
| attention_unattributed_ticks | 67.000000 | 62.000000 | -7.4627% | 58.500000 | 54.000000 | -7.6923% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 435.000000 | 451.000000 | +3.6782% | 371.500000 | 369.500000 | -0.5384% |
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
| down_ticks | 166107.000000 | 165842.000000 | -0.1595% | 165840.000000 | 166138.000000 | +0.1797% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9746.000000 | 9831.000000 | +0.8722% | 9813.000000 | 9845.000000 | +0.3261% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2705.000000 | 2708.000000 | +0.1109% | 2690.000000 | 2688.000000 | -0.0743% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 432095.000000 | 431238.000000 | -0.1983% | 431030.500000 | 431309.000000 | +0.0646% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1310.000000 | 1350.000000 | +3.0534% | 1270.000000 | 1287.000000 | +1.3386% |
| generation_final_norm_ticks | 925.000000 | 923.000000 | -0.2162% | 922.000000 | 917.000000 | -0.5423% |
| generation_lm_head_argmax_ticks | 6073.000000 | 6253.000000 | +2.9639% | 6137.500000 | 6238.500000 | +1.6456% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 126374.000000 | 127512.000000 | +0.9005% | 126691.000000 | 127305.500000 | +0.4850% |
| generation_lm_head_expand_ticks | 109957.000000 | 110653.000000 | +0.6330% | 109896.500000 | 110603.000000 | +0.6429% |
| generation_lm_head_hmx_tail_wait_ticks | 2056.000000 | 2226.000000 | +8.2685% | 2204.000000 | 2157.000000 | -2.1325% |
| generation_lm_head_hmx_ticks | 125350.000000 | 126387.000000 | +0.8273% | 125660.500000 | 126253.000000 | +0.4715% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 232.000000 | 253.000000 | +9.0517% | 241.500000 | 241.000000 | -0.2070% |
| generation_lm_head_scale_init_ticks | 579.000000 | 546.000000 | -5.6995% | 574.000000 | 560.500000 | -2.3519% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127299.000000 | 128435.000000 | +0.8924% | 127616.000000 | 128224.000000 | +0.4764% |
| generation_lm_head_weight_dma_ticks | 125489.000000 | 126570.000000 | +0.8614% | 125806.500000 | 126412.500000 | +0.4817% |
| generation_lm_head_weight_dma_wait_ticks | 2771.000000 | 2848.000000 | +2.7788% | 2890.000000 | 2862.000000 | -0.9689% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 130370.000000 | 131850.000000 | +1.1352% | 134070.500000 | 132181.500000 | -1.4090% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2668.750000 | 2584.791417 | -3.1460% | 2402.031542 | 2312.005208 | -3.7479% |
| host_us | 63408.125000 | 63307.031000 | -0.1594% | 62962.864500 | 63034.088500 | +0.1131% |
| host_wall_ns | 63408125.000000 | 63307031.000000 | -0.1594% | 62962864.500000 | 63034088.500000 | +0.1131% |
| input_norm_ticks | 9491.000000 | 9488.000000 | -0.0316% | 9452.000000 | 9455.000000 | +0.0317% |
| input_stage_ticks | 15.000000 | 12.000000 | -20.0000% | 13.000000 | 11.000000 | -15.3846% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1166196.000000 | 1165867.000000 | -0.0282% | 1164688.500000 | 1165644.500000 | +0.0821% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 476.000000 | 467.000000 | -1.8908% | 462.000000 | 440.000000 | -4.7619% |
| ledger_named_ticks | 1166196.000000 | 1165867.000000 | -0.0282% | 1164688.500000 | 1165644.500000 | +0.0821% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7324.000000 | 7268.000000 | -0.7646% | 7343.000000 | 7301.000000 | -0.5720% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96569.000000 | 96427.000000 | -0.1470% | 96609.500000 | 96888.500000 | +0.2888% |
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
| post_attention_norm_ticks | 26.000000 | 26.000000 | +0.0000% | 25.000000 | 24.000000 | -4.0000% |
| post_attention_residual_ticks | 9103.000000 | 9135.000000 | +0.3515% | 9080.000000 | 9066.000000 | -0.1542% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 686353.000000 | 685605.000000 | -0.1090% | 685682.500000 | 686227.000000 | +0.0794% |
| projection_pack_ticks | 3182.000000 | 3176.000000 | -0.1886% | 3167.500000 | 3164.000000 | -0.1105% |
| projection_unpack_ticks | 11259.000000 | 11010.000000 | -2.2116% | 11156.500000 | 11236.000000 | +0.7126% |
| qk_norm_rope_ticks | 23.000000 | 29.000000 | +26.0870% | 23.500000 | 24.000000 | +2.1277% |
| qkv_projection_ticks | 225124.000000 | 224985.000000 | -0.0617% | 225106.500000 | 225224.500000 | +0.0524% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 982.000000 | 974.000000 | -0.8147% | 906.000000 | 904.000000 | -0.2208% |
| runtime_teardown_ticks | 965.000000 | 920.000000 | -4.6632% | 886.000000 | 890.500000 | +0.5079% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6487.000000 | 6586.000000 | +1.5261% | 6514.000000 | 6565.000000 | +0.7829% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3298.000000 | 3276.000000 | -0.6671% | 3303.000000 | 3311.500000 | +0.2573% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 208.000000 | 214.000000 | +2.8846% | 170.000000 | 172.000000 | +1.1765% |
| total_ticks | 1165214.000000 | 1164893.000000 | -0.0275% | 1163710.000000 | 1164711.500000 | +0.0861% |
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
| w4f16_cross_prefetch_lifetime_ticks | 249953.000000 | 249236.000000 | -0.2869% | 249728.000000 | 249569.500000 | -0.0635% |
| w4f16_cross_prefetch_wait_ticks | 1173.000000 | 1160.000000 | -1.1083% | 1142.000000 | 1115.500000 | -2.3205% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 73469.000000 | 75130.000000 | +2.2608% | 73705.500000 | 75113.000000 | +1.9096% |
| w4f16_expand_ticks | 619005.000000 | 618526.000000 | -0.0774% | 618663.000000 | 618897.500000 | +0.0379% |
| w4f16_expand_work_ticks | 1689565.000000 | 1692587.000000 | +0.1789% | 1690158.000000 | 1691954.500000 | +0.1063% |
| w4f16_gate_up_expand_pool_wait_ticks | 30487.000000 | 30720.000000 | +0.7643% | 30551.500000 | 30938.500000 | +1.2667% |
| w4f16_gate_up_expand_ticks | 255393.000000 | 254575.000000 | -0.3203% | 254402.000000 | 254577.500000 | +0.0690% |
| w4f16_gate_up_expand_work_ticks | 160674.000000 | 159630.000000 | -0.6498% | 159714.500000 | 159096.000000 | -0.3873% |
| w4f16_gate_up_hmx_tail_wait_ticks | 6945.000000 | 7003.000000 | +0.8351% | 7262.500000 | 7160.500000 | -1.4045% |
| w4f16_gate_up_hmx_wait_ticks | 281003.000000 | 280090.000000 | -0.3249% | 280138.000000 | 280194.000000 | +0.0200% |
| w4f16_gate_up_stream_join_wait_ticks | 664.000000 | 652.000000 | -1.8072% | 737.500000 | 651.500000 | -11.6610% |
| w4f16_gate_up_stream_ready_wait_ticks | 1440.000000 | 1274.000000 | -11.5278% | 1400.000000 | 1351.000000 | -3.5000% |
| w4f16_gate_up_stream_work_ticks | 134360.000000 | 134382.000000 | +0.0164% | 134395.000000 | 134399.000000 | +0.0030% |
| w4f16_gate_up_weight_dma_ticks | 564241.000000 | 562801.000000 | -0.2552% | 562142.000000 | 562740.500000 | +0.1065% |
| w4f16_hmx_tail_wait_ticks | 18551.000000 | 18201.000000 | -1.8867% | 18616.500000 | 18749.500000 | +0.7144% |
| w4f16_prefetch_wait_ticks | 30207.000000 | 30227.000000 | +0.0662% | 29926.000000 | 30109.500000 | +0.6132% |
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
| weight_dma_ticks | 1142618.000000 | 1142211.000000 | -0.0356% | 1140087.500000 | 1142474.000000 | +0.2093% |


## A0 versus A decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.466667 | 8.666667 | -8.4507% | 9.033333 | 8.833333 | -2.2140% |
| attention_av_hmx_ticks | 7714.266667 | 7658.200000 | -0.7268% | 7950.000000 | 7931.166667 | -0.2369% |
| attention_av_pack_ticks | 3043.066667 | 3044.866667 | +0.0592% | 3037.866667 | 3041.533333 | +0.1207% |
| attention_av_unpack_ticks | 4010.933333 | 4004.600000 | -0.1579% | 4010.466667 | 4005.333333 | -0.1280% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 7789.400000 | 7789.333333 | -0.0009% | 7980.366667 | 7999.633333 | +0.2414% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6379.600000 | 6391.333333 | +0.1839% | 6382.333333 | 6385.666667 | +0.0522% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132943.333333 | 132678.800000 | -0.1990% | 133049.200000 | 132656.133333 | -0.2954% |
| attention_ticks | 643739.066667 | 643280.266667 | -0.0713% | 644239.033333 | 643779.200000 | -0.0714% |
| attention_unattributed_ticks | 481858.466667 | 481713.133333 | -0.0302% | 481802.633333 | 481799.200000 | -0.0007% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.133333 | 281.933333 | -1.4678% | 284.166667 | 284.566667 | +0.1408% |
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
| down_ticks | 165825.333333 | 165815.800000 | -0.0057% | 166197.400000 | 166221.333333 | +0.0144% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29942.066667 | 29930.533333 | -0.0385% | 29926.033333 | 29905.766667 | -0.0677% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2670.400000 | 2672.066667 | +0.0624% | 2671.266667 | 2671.766667 | +0.0187% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 429997.666667 | 429781.933333 | -0.0502% | 430127.100000 | 429806.066667 | -0.0746% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 34.466667 | 31.733333 | -7.9304% | 30.666667 | 32.066667 | +4.5652% |
| generation_final_norm_ticks | 40.933333 | 39.400000 | -3.7459% | 39.933333 | 39.900000 | -0.0835% |
| generation_lm_head_argmax_ticks | 6237.666667 | 6200.600000 | -0.5942% | 6230.366667 | 6306.900000 | +1.2284% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 127004.400000 | 126767.600000 | -0.1865% | 126979.666667 | 127342.133333 | +0.2855% |
| generation_lm_head_expand_ticks | 110410.866667 | 110307.000000 | -0.0941% | 110389.466667 | 110662.266667 | +0.2471% |
| generation_lm_head_hmx_tail_wait_ticks | 2169.800000 | 2092.533333 | -3.5610% | 2180.533333 | 2153.866667 | -1.2229% |
| generation_lm_head_hmx_ticks | 126091.200000 | 125834.800000 | -0.2033% | 126058.800000 | 126415.166667 | +0.2827% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 202.866667 | 214.000000 | +5.4880% | 207.300000 | 208.033333 | +0.3538% |
| generation_lm_head_scale_init_ticks | 571.400000 | 565.200000 | -1.0851% | 569.733333 | 566.633333 | -0.5441% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127045.333333 | 126807.000000 | -0.1876% | 127019.233333 | 127382.433333 | +0.2859% |
| generation_lm_head_weight_dma_ticks | 126187.266667 | 125936.200000 | -0.1990% | 126157.566667 | 126515.966667 | +0.2841% |
| generation_lm_head_weight_dma_wait_ticks | 2877.533333 | 2874.466667 | -0.1066% | 2881.100000 | 2862.500000 | -0.6456% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 125101.866667 | 125606.733333 | +0.4036% | 125811.866667 | 125631.500000 | -0.1434% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1742.295083 | 1887.461844 | +8.3319% | 1790.605706 | 1877.456536 | +4.8504% |
| host_us | 92427.368000 | 92500.600733 | +0.0792% | 92538.857567 | 92484.163133 | -0.0591% |
| host_wall_ns | 92427368.000000 | 92500600.733333 | +0.0792% | 92538857.566667 | 92484163.133333 | -0.0591% |
| input_norm_ticks | 9354.266667 | 9348.733333 | -0.0592% | 9355.633333 | 9360.166667 | +0.0485% |
| input_stage_ticks | 7.933333 | 7.333333 | -7.5630% | 7.966667 | 7.900000 | -0.8368% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741153.400000 | 1739772.266667 | -0.0793% | 1741709.700000 | 1741523.500000 | -0.0107% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 320.533333 | 324.733333 | +1.3103% | 322.866667 | 322.800000 | -0.0206% |
| ledger_named_ticks | 1741153.400000 | 1739772.266667 | -0.0793% | 1741709.700000 | 1741523.500000 | -0.0107% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 5739.466667 | 5675.866667 | -1.1081% | 5682.866667 | 5637.066667 | -0.8059% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96322.266667 | 95828.600000 | -0.5125% | 96321.500000 | 96312.700000 | -0.0091% |
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
| post_attention_norm_ticks | 17.666667 | 16.600000 | -6.0377% | 17.266667 | 17.133333 | -0.7722% |
| post_attention_residual_ticks | 9005.666667 | 9007.800000 | +0.0237% | 9015.366667 | 9015.366667 | +0.0000% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684471.133333 | 683744.933333 | -0.1061% | 684721.133333 | 684581.400000 | -0.0204% |
| projection_pack_ticks | 417.133333 | 417.266667 | +0.0320% | 420.400000 | 418.833333 | -0.3727% |
| projection_unpack_ticks | 11102.533333 | 11087.133333 | -0.1387% | 11198.166667 | 11211.733333 | +0.1212% |
| qk_norm_rope_ticks | 18.400000 | 18.066667 | -1.8116% | 17.200000 | 17.633333 | +2.5194% |
| qkv_projection_ticks | 223984.666667 | 224047.733333 | +0.0282% | 224308.500000 | 224375.466667 | +0.0299% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 619.000000 | 619.333333 | +0.0539% | 622.066667 | 619.366667 | -0.4340% |
| runtime_teardown_ticks | 575.666667 | 576.333333 | +0.1158% | 575.933333 | 576.166667 | +0.0405% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4215.533333 | 4239.200000 | +0.5614% | 4220.633333 | 4219.633333 | -0.0237% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21325.600000 | 21346.666667 | +0.0988% | 21333.966667 | 21323.200000 | -0.0505% |
| scan_cache_stage_ticks | 13762.266667 | 13690.266667 | -0.5232% | 13763.700000 | 13756.000000 | -0.0559% |
| scan_dynamic_attention_ticks | 643688.400000 | 643229.666667 | -0.0713% | 644189.566667 | 643729.366667 | -0.0714% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 38.866667 | 35.866667 | -7.7187% | 39.066667 | 36.900000 | -5.5461% |
| total_ticks | 1740534.400000 | 1739152.933333 | -0.0794% | 1741090.200000 | 1740900.266667 | -0.0109% |
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
| w4f16_cross_prefetch_lifetime_ticks | 832568.600000 | 832017.600000 | -0.0662% | 833253.233333 | 832841.233333 | -0.0494% |
| w4f16_cross_prefetch_wait_ticks | 1140.866667 | 1195.000000 | +4.7449% | 1143.633333 | 1135.433333 | -0.7170% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75620.466667 | 75145.933333 | -0.6275% | 75515.100000 | 75723.300000 | +0.2757% |
| w4f16_expand_ticks | 617141.733333 | 617087.133333 | -0.0088% | 617708.766667 | 617676.833333 | -0.0052% |
| w4f16_expand_work_ticks | 1688914.466667 | 1688882.466667 | -0.0019% | 1689449.866667 | 1690120.166667 | +0.0397% |
| w4f16_gate_up_expand_pool_wait_ticks | 31747.866667 | 31420.866667 | -1.0300% | 31521.833333 | 31382.300000 | -0.4427% |
| w4f16_gate_up_expand_ticks | 254042.466667 | 254060.933333 | +0.0073% | 254057.733333 | 253850.600000 | -0.0815% |
| w4f16_gate_up_expand_work_ticks | 158048.533333 | 158574.933333 | +0.3331% | 158139.233333 | 158260.066667 | +0.0764% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7140.200000 | 7040.400000 | -1.3977% | 7234.633333 | 7225.000000 | -0.1332% |
| w4f16_gate_up_hmx_wait_ticks | 279495.000000 | 279429.533333 | -0.0234% | 279617.633333 | 279351.100000 | -0.0953% |
| w4f16_gate_up_stream_join_wait_ticks | 651.466667 | 623.600000 | -4.2775% | 649.833333 | 655.366667 | +0.8515% |
| w4f16_gate_up_stream_ready_wait_ticks | 1332.400000 | 1313.133333 | -1.4460% | 1337.466667 | 1375.866667 | +2.8711% |
| w4f16_gate_up_stream_work_ticks | 134385.266667 | 134380.333333 | -0.0037% | 134383.833333 | 134382.833333 | -0.0007% |
| w4f16_gate_up_weight_dma_ticks | 560704.866667 | 560509.400000 | -0.0349% | 560863.866667 | 560299.300000 | -0.1007% |
| w4f16_hmx_tail_wait_ticks | 18526.533333 | 18236.600000 | -1.5650% | 18774.266667 | 18974.633333 | +1.0672% |
| w4f16_prefetch_wait_ticks | 30089.466667 | 29887.600000 | -0.6709% | 29942.366667 | 29872.233333 | -0.2342% |
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
| weight_dma_ticks | 1135881.200000 | 1134958.000000 | -0.0813% | 1135687.700000 | 1135129.266667 | -0.0492% |


## R0 versus R prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.000000 | 12.000000 | +33.3333% | 12.000000 | 13.000000 | +8.3333% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 76145.000000 | 76329.000000 | +0.2416% | 76088.000000 | 75821.500000 | -0.3503% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 120.000000 | 122.000000 | +1.6667% | 121.000000 | 122.000000 | +0.8264% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76320.000000 | 76504.000000 | +0.2411% | 76263.500000 | 75997.000000 | -0.3494% |
| attention_unattributed_ticks | 55.000000 | 53.000000 | -3.6364% | 53.500000 | 56.000000 | +4.6729% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 362.000000 | 378.000000 | +4.4199% | 374.000000 | 373.500000 | -0.1337% |
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
| down_ticks | 166357.000000 | 165799.000000 | -0.3354% | 166347.500000 | 166074.500000 | -0.1641% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9892.000000 | 9688.000000 | -2.0623% | 9839.500000 | 9896.000000 | +0.5742% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2706.000000 | 2690.000000 | -0.5913% | 2690.000000 | 2695.000000 | +0.1859% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 431755.000000 | 432035.000000 | +0.0649% | 430866.000000 | 431632.000000 | +0.1778% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1300.000000 | 1294.000000 | -0.4615% | 1291.000000 | 1280.000000 | -0.8521% |
| generation_final_norm_ticks | 926.000000 | 918.000000 | -0.8639% | 920.500000 | 925.500000 | +0.5432% |
| generation_lm_head_argmax_ticks | 6107.000000 | 6370.000000 | +4.3065% | 6166.000000 | 6422.000000 | +4.1518% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 126816.000000 | 127710.000000 | +0.7050% | 127065.000000 | 128002.500000 | +0.7378% |
| generation_lm_head_expand_ticks | 110176.000000 | 110716.000000 | +0.4901% | 110269.500000 | 111013.500000 | +0.6747% |
| generation_lm_head_hmx_tail_wait_ticks | 2174.000000 | 2280.000000 | +4.8758% | 2196.000000 | 2257.000000 | +2.7778% |
| generation_lm_head_hmx_ticks | 125806.000000 | 126709.000000 | +0.7178% | 126034.000000 | 126987.500000 | +0.7565% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 245.000000 | 239.000000 | -2.4490% | 243.500000 | 237.500000 | -2.4641% |
| generation_lm_head_scale_init_ticks | 563.000000 | 542.000000 | -3.7300% | 561.000000 | 567.000000 | +1.0695% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 127742.000000 | 128628.000000 | +0.6936% | 128002.000000 | 128923.500000 | +0.7199% |
| generation_lm_head_weight_dma_ticks | 125928.000000 | 126823.000000 | +0.7107% | 126184.000000 | 127113.500000 | +0.7366% |
| generation_lm_head_weight_dma_wait_ticks | 2935.000000 | 2905.000000 | -1.0221% | 2929.000000 | 2972.500000 | +1.4851% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 132551.000000 | 131591.000000 | -0.7242% | 134523.500000 | 132763.000000 | -1.3087% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2262.343750 | 2572.760500 | +13.7210% | 2286.380250 | 2354.270917 | +2.9694% |
| host_us | 63021.250000 | 63348.073000 | +0.5186% | 63048.411500 | 63159.479500 | +0.1762% |
| host_wall_ns | 63021250.000000 | 63348073.000000 | +0.5186% | 63048411.500000 | 63159479.500000 | +0.1762% |
| input_norm_ticks | 9508.000000 | 9483.000000 | -0.2629% | 9442.500000 | 9456.500000 | +0.1483% |
| input_stage_ticks | 11.000000 | 9.000000 | -18.1818% | 11.000000 | 11.000000 | +0.0000% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1166571.000000 | 1166886.000000 | +0.0270% | 1166037.500000 | 1167327.000000 | +0.1106% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 439.000000 | 426.000000 | -2.9613% | 449.000000 | 452.000000 | +0.6682% |
| ledger_named_ticks | 1166571.000000 | 1166886.000000 | +0.0270% | 1166037.500000 | 1167327.000000 | +0.1106% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7531.000000 | 7189.000000 | -4.5412% | 7361.500000 | 7366.000000 | +0.0611% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96812.000000 | 96905.000000 | +0.0961% | 96830.500000 | 96618.500000 | -0.2189% |
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
| post_attention_norm_ticks | 24.000000 | 20.000000 | -16.6667% | 22.000000 | 24.000000 | +9.0909% |
| post_attention_residual_ticks | 9067.000000 | 9074.000000 | +0.0772% | 9071.500000 | 9081.500000 | +0.1102% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 686585.000000 | 686667.000000 | +0.0119% | 686635.500000 | 686638.000000 | +0.0004% |
| projection_pack_ticks | 3163.000000 | 3174.000000 | +0.3478% | 3167.500000 | 3166.500000 | -0.0316% |
| projection_unpack_ticks | 11144.000000 | 11226.000000 | +0.7358% | 11218.500000 | 11100.000000 | -1.0563% |
| qk_norm_rope_ticks | 23.000000 | 23.000000 | +0.0000% | 22.500000 | 22.500000 | +0.0000% |
| qkv_projection_ticks | 224737.000000 | 224734.000000 | -0.0013% | 225014.500000 | 224898.000000 | -0.0518% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 910.000000 | 915.000000 | +0.5495% | 906.500000 | 917.500000 | +1.2135% |
| runtime_teardown_ticks | 865.000000 | 878.000000 | +1.5029% | 885.000000 | 899.500000 | +1.6384% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6622.000000 | 6442.000000 | -2.7182% | 6560.000000 | 6565.500000 | +0.0838% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3296.000000 | 3277.000000 | -0.5765% | 3300.000000 | 3335.500000 | +1.0758% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 175.000000 | 171.000000 | -2.2857% | 171.000000 | 171.000000 | +0.0000% |
| total_ticks | 1165661.000000 | 1165971.000000 | +0.0266% | 1165104.000000 | 1166410.500000 | +0.1121% |
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
| w4f16_cross_prefetch_lifetime_ticks | 249743.000000 | 249558.000000 | -0.0741% | 249646.500000 | 249597.500000 | -0.0196% |
| w4f16_cross_prefetch_wait_ticks | 1044.000000 | 1223.000000 | +17.1456% | 1036.500000 | 1095.500000 | +5.6922% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74253.000000 | 74661.000000 | +0.5495% | 74695.000000 | 75037.500000 | +0.4585% |
| w4f16_expand_ticks | 618584.000000 | 619247.000000 | +0.1072% | 619309.500000 | 619217.500000 | -0.0149% |
| w4f16_expand_work_ticks | 1688807.000000 | 1693822.000000 | +0.2970% | 1689393.000000 | 1694678.000000 | +0.3128% |
| w4f16_gate_up_expand_pool_wait_ticks | 31060.000000 | 30667.000000 | -1.2653% | 30921.000000 | 30585.000000 | -1.0866% |
| w4f16_gate_up_expand_ticks | 254939.000000 | 255191.000000 | +0.0988% | 254308.000000 | 255129.000000 | +0.3228% |
| w4f16_gate_up_expand_work_ticks | 159414.000000 | 159971.000000 | +0.3494% | 158937.000000 | 159609.000000 | +0.4228% |
| w4f16_gate_up_hmx_tail_wait_ticks | 6889.000000 | 7135.000000 | +3.5709% | 7235.000000 | 7132.500000 | -1.4167% |
| w4f16_gate_up_hmx_wait_ticks | 280519.000000 | 281045.000000 | +0.1875% | 279976.500000 | 280749.500000 | +0.2761% |
| w4f16_gate_up_stream_join_wait_ticks | 729.000000 | 655.000000 | -10.1509% | 637.000000 | 642.000000 | +0.7849% |
| w4f16_gate_up_stream_ready_wait_ticks | 1424.000000 | 1497.000000 | +5.1264% | 1392.000000 | 1417.500000 | +1.8319% |
| w4f16_gate_up_stream_work_ticks | 134351.000000 | 134332.000000 | -0.0141% | 134396.000000 | 134382.000000 | -0.0104% |
| w4f16_gate_up_weight_dma_ticks | 563598.000000 | 564379.000000 | +0.1386% | 561976.500000 | 563739.000000 | +0.3136% |
| w4f16_hmx_tail_wait_ticks | 18513.000000 | 18555.000000 | +0.2269% | 19075.000000 | 18624.500000 | -2.3617% |
| w4f16_prefetch_wait_ticks | 30844.000000 | 29965.000000 | -2.8498% | 29434.500000 | 30125.000000 | +2.3459% |
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
| weight_dma_ticks | 1143075.000000 | 1144065.000000 | +0.0866% | 1142519.500000 | 1143138.000000 | +0.0541% |


## R0 versus R decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 10.866667 | 8.266667 | -23.9264% | 8.766667 | 8.866667 | +1.1407% |
| attention_av_hmx_ticks | 7876.466667 | 7912.266667 | +0.4545% | 7950.033333 | 7904.000000 | -0.5790% |
| attention_av_pack_ticks | 3034.933333 | 3039.466667 | +0.1494% | 3040.833333 | 3039.400000 | -0.0471% |
| attention_av_unpack_ticks | 4005.933333 | 3998.266667 | -0.1914% | 4002.666667 | 4001.266667 | -0.0350% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8043.733333 | 8080.466667 | +0.4567% | 8028.933333 | 8003.433333 | -0.3176% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6381.066667 | 6381.466667 | +0.0063% | 6385.000000 | 6392.066667 | +0.1107% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132797.133333 | 132428.000000 | -0.2780% | 132878.733333 | 132499.866667 | -0.2851% |
| attention_ticks | 643886.266667 | 643606.600000 | -0.0434% | 644004.266667 | 643582.033333 | -0.0656% |
| attention_unattributed_ticks | 481747.000000 | 481766.666667 | +0.0041% | 481776.100000 | 481738.833333 | -0.0077% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 284.266667 | 284.133333 | -0.0469% | 284.900000 | 283.533333 | -0.4797% |
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
| down_ticks | 165654.733333 | 166249.066667 | +0.3588% | 166447.200000 | 166171.166667 | -0.1658% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29901.800000 | 29924.000000 | +0.0742% | 29896.433333 | 29888.666667 | -0.0260% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2670.800000 | 2671.400000 | +0.0225% | 2670.800000 | 2671.500000 | +0.0262% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 430108.933333 | 430388.800000 | +0.0651% | 429901.300000 | 429812.266667 | -0.0207% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 34.933333 | 34.000000 | -2.6718% | 32.966667 | 33.300000 | +1.0111% |
| generation_final_norm_ticks | 39.733333 | 39.866667 | +0.3356% | 40.000000 | 39.433333 | -1.4167% |
| generation_lm_head_argmax_ticks | 6184.733333 | 6307.866667 | +1.9909% | 6265.733333 | 6395.200000 | +2.0663% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 126957.000000 | 127564.400000 | +0.4784% | 127264.700000 | 127874.066667 | +0.4788% |
| generation_lm_head_expand_ticks | 110446.000000 | 110734.000000 | +0.2608% | 110611.633333 | 110474.966667 | -0.1236% |
| generation_lm_head_hmx_tail_wait_ticks | 2122.133333 | 2246.200000 | +5.8463% | 2116.066667 | 2201.100000 | +4.0185% |
| generation_lm_head_hmx_ticks | 126030.466667 | 126638.133333 | +0.4822% | 126340.433333 | 126945.766667 | +0.4791% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 212.200000 | 211.533333 | -0.3142% | 209.933333 | 208.000000 | -0.9209% |
| generation_lm_head_scale_init_ticks | 566.200000 | 564.933333 | -0.2237% | 566.266667 | 567.700000 | +0.2531% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 126996.733333 | 127604.266667 | +0.4784% | 127305.000000 | 127913.633333 | +0.4781% |
| generation_lm_head_weight_dma_ticks | 126129.066667 | 126736.933333 | +0.4819% | 126439.633333 | 127046.200000 | +0.4797% |
| generation_lm_head_weight_dma_wait_ticks | 2912.400000 | 2924.466667 | +0.4143% | 2918.300000 | 2890.800000 | -0.9423% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 126296.466667 | 127511.200000 | +0.9618% | 127742.900000 | 127396.966667 | -0.2708% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1779.003450 | 1778.010394 | -0.0558% | 1745.954797 | 1911.041669 | +9.4554% |
| host_us | 92463.888867 | 92561.638867 | +0.1057% | 92494.206533 | 92562.204833 | +0.0735% |
| host_wall_ns | 92463888.866667 | 92561638.866667 | +0.1057% | 92494206.533333 | 92562204.833333 | +0.0735% |
| input_norm_ticks | 9364.733333 | 9365.133333 | +0.0043% | 9363.433333 | 9362.933333 | -0.0053% |
| input_stage_ticks | 7.600000 | 8.533333 | +12.2807% | 8.566667 | 8.000000 | -6.6148% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741149.800000 | 1743045.666667 | +0.1089% | 1741718.100000 | 1741486.033333 | -0.0133% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 324.800000 | 322.666667 | -0.6568% | 322.866667 | 322.466667 | -0.1239% |
| ledger_named_ticks | 1741149.800000 | 1743045.666667 | +0.1089% | 1741718.100000 | 1741486.033333 | -0.0133% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 5743.600000 | 5738.866667 | -0.0824% | 5681.233333 | 5685.133333 | +0.0686% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96153.733333 | 96291.133333 | +0.1429% | 96199.833333 | 96280.666667 | +0.0840% |
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
| post_attention_norm_ticks | 17.000000 | 15.800000 | -7.0588% | 17.000000 | 16.900000 | -0.5882% |
| post_attention_residual_ticks | 9008.733333 | 9022.000000 | +0.1473% | 9016.800000 | 9018.800000 | +0.0222% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 684478.333333 | 685625.133333 | +0.1675% | 685038.766667 | 684906.333333 | -0.0193% |
| projection_pack_ticks | 414.000000 | 420.866667 | +1.6586% | 420.200000 | 420.033333 | -0.0397% |
| projection_unpack_ticks | 11114.866667 | 11286.400000 | +1.5433% | 11227.333333 | 11234.666667 | +0.0653% |
| qk_norm_rope_ticks | 17.400000 | 17.733333 | +1.9157% | 17.433333 | 17.333333 | -0.5736% |
| qkv_projection_ticks | 224071.933333 | 224591.933333 | +0.2321% | 224459.666667 | 224386.566667 | -0.0326% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 619.733333 | 634.266667 | +2.3451% | 621.900000 | 621.266667 | -0.1018% |
| runtime_teardown_ticks | 573.266667 | 581.266667 | +1.3955% | 575.266667 | 575.733333 | +0.0811% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4233.200000 | 4223.933333 | -0.2189% | 4221.633333 | 4221.166667 | -0.0111% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21329.666667 | 21346.400000 | +0.0785% | 21326.300000 | 21327.733333 | +0.0067% |
| scan_cache_stage_ticks | 13743.600000 | 13764.600000 | +0.1528% | 13743.366667 | 13745.733333 | +0.0172% |
| scan_dynamic_attention_ticks | 643836.533333 | 643557.466667 | -0.0433% | 643953.766667 | 643531.933333 | -0.0655% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 36.866667 | 39.466667 | +7.0524% | 39.366667 | 39.433333 | +0.1693% |
| total_ticks | 1740530.066667 | 1742411.400000 | +0.1081% | 1741094.600000 | 1740866.133333 | -0.0131% |
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
| w4f16_cross_prefetch_lifetime_ticks | 832653.600000 | 832409.866667 | -0.0293% | 832932.900000 | 832507.166667 | -0.0511% |
| w4f16_cross_prefetch_wait_ticks | 1117.200000 | 1166.666667 | +4.4277% | 1129.500000 | 1165.500000 | +3.1873% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75179.333333 | 75539.533333 | +0.4791% | 75578.233333 | 74837.066667 | -0.9807% |
| w4f16_expand_ticks | 617260.466667 | 617964.933333 | +0.1141% | 618059.300000 | 617890.433333 | -0.0273% |
| w4f16_expand_work_ticks | 1688897.066667 | 1690023.133333 | +0.0667% | 1689447.033333 | 1687565.966667 | -0.1113% |
| w4f16_gate_up_expand_pool_wait_ticks | 31620.400000 | 31521.466667 | -0.3129% | 31426.000000 | 31473.066667 | +0.1498% |
| w4f16_gate_up_expand_ticks | 254229.133333 | 254134.733333 | -0.0371% | 253854.366667 | 253935.700000 | +0.0320% |
| w4f16_gate_up_expand_work_ticks | 158216.266667 | 158023.800000 | -0.1216% | 158083.066667 | 158042.200000 | -0.0259% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7143.133333 | 7327.200000 | +2.5768% | 7277.566667 | 7120.400000 | -2.1596% |
| w4f16_gate_up_hmx_wait_ticks | 279696.866667 | 279930.333333 | +0.0835% | 279450.033333 | 279297.566667 | -0.0546% |
| w4f16_gate_up_stream_join_wait_ticks | 637.466667 | 633.666667 | -0.5961% | 628.766667 | 664.900000 | +5.7467% |
| w4f16_gate_up_stream_ready_wait_ticks | 1254.800000 | 1422.666667 | +13.3780% | 1312.666667 | 1354.866667 | +3.2148% |
| w4f16_gate_up_stream_work_ticks | 134387.400000 | 134376.533333 | -0.0081% | 134388.000000 | 134377.400000 | -0.0079% |
| w4f16_gate_up_weight_dma_ticks | 561135.133333 | 561515.000000 | +0.0677% | 560582.733333 | 560312.033333 | -0.0483% |
| w4f16_hmx_tail_wait_ticks | 18634.933333 | 18755.600000 | +0.6475% | 18837.266667 | 18604.266667 | -1.2369% |
| w4f16_prefetch_wait_ticks | 30068.666667 | 29936.666667 | -0.4390% | 29780.233333 | 29828.433333 | +0.1619% |
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
| weight_dma_ticks | 1135990.866667 | 1138140.533333 | +0.1892% | 1136417.166667 | 1136535.566667 | +0.0104% |


## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 62962.8645,
      "prefill_tokens_per_second": 1016.4721778184028,
      "decode_tokens": 15,
      "decode_total_host_us": 1388082.8635,
      "decode_tokens_per_second": 10.806271292895332
    },
    "R0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63048.4115,
      "prefill_tokens_per_second": 1015.0929813671831,
      "decode_tokens": 15,
      "decode_total_host_us": 1387413.098,
      "decode_tokens_per_second": 10.811487956703722
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 63034.0885,
      "prefill_tokens_per_second": 1015.3236371459548,
      "decode_tokens": 15,
      "decode_total_host_us": 1387262.447,
      "decode_tokens_per_second": 10.812662039859859
    },
    "R": {
      "prefill_tokens": 64,
      "prefill_host_us": 63159.4795,
      "prefill_tokens_per_second": 1013.3079073268804,
      "decode_tokens": 15,
      "decode_total_host_us": 1388433.0724999998,
      "decode_tokens_per_second": 10.803545591859994
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.009098018536590757,
      "decode": -0.1321997389663876
    },
    "R": {
      "prefill": 0.09477758546798665,
      "decode": -0.022917661012378243
    }
  }
}
```
