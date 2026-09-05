# EXP-0221 complete profiling comparison

Frozen ABI108 binary, original RTN and A/B/C independent software token sequences. One M64 plus15 feedback steps;5short then10four-way rotating formal rounds. Other recipe columns historical nonpaired, changed W4 prevents activation-only attribution. Quality scoring disabled. All numeric fields retained; additive ledger fields exclusive, engine/wait counters overlapping. Host-DSP boundary per record = Host wall minus DSP invocation. Percent changes below candidate/control minus one; positive timing changes are slower.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 GPTQ A EXP-0221 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 384.9 (0.61%) | 247.4 (0.63%) | +55.57% |
| Input RMSNorm | 489.7 (0.61%) | 491.6 (0.78%) | 554.0 (1.40%) | -11.26% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11726.5 (18.66%) | 7052.7 (17.82%) | +66.27% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3953.7 (6.29%) | 3214.5 (8.12%) | +22.99% |
| O projection | 5757.7 (7.14%) | 5024.6 (7.99%) | 1256.4 (3.17%) | +299.92% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.5 (0.75%) | 654.0 (1.65%) | -27.60% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22410.4 (35.66%) | 14442.7 (36.49%) | +55.17% |
| Down | 13447.9 (16.67%) | 8570.6 (13.64%) | 3428.5 (8.66%) | +149.98% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.72% |
| KV carrier conversion | 174.0 (0.22%) | 172.6 (0.27%) | 203.2 (0.51%) | -15.08% |
| KV append DMA | 343.6 (0.43%) | 340.2 (0.54%) | 463.9 (1.17%) | -26.65% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.13% |
| Layer bookkeeping | 23.9 (0.03%) | 23.4 (0.04%) | 23.2 (0.06%) | +1.12% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.3 (0.01%) | 22.5 (0.06%) | -62.96% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.7 (0.15%) | 105.6 (0.27%) | -11.27% |
| Embedding | 68.1 (0.08%) | 65.9 (0.10%) | 62.4 (0.16%) | +5.50% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.01% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6695.7 (10.65%) | 5284.7 (13.35%) | +26.70% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2281.9 (3.63%) | 2466.6 (6.23%) | -7.49% |
| 完整 Host wall | 80692.2 (100.00%) | 62850.8 (100.00%) | 39575.9 (100.00%) | +58.81% |


## Original versus A prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 11.000000 | 12.000000 | +9.0909% | 12.000000 | 10.000000 | -16.6667% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75902.000000 | 74897.000000 | -1.3241% | 75883.000000 | 75733.000000 | -0.1977% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 121.000000 | 121.000000 | +0.0000% | 121.500000 | 120.000000 | -1.2346% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76081.000000 | 75071.000000 | -1.3275% | 76059.000000 | 75910.500000 | -0.1952% |
| attention_unattributed_ticks | 58.000000 | 53.000000 | -8.6207% | 56.500000 | 56.500000 | +0.0000% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 382.000000 | 375.000000 | -1.8325% | 368.000000 | 371.000000 | +0.8152% |
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
| down_ticks | 163227.000000 | 163608.000000 | +0.2334% | 164347.000000 | 164555.000000 | +0.1266% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9844.000000 | 9728.000000 | -1.1784% | 9886.000000 | 9826.000000 | -0.6069% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2684.000000 | 2694.000000 | +0.3726% | 2690.500000 | 2692.000000 | +0.0558% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 428620.000000 | 429490.000000 | +0.2030% | 429669.000000 | 430269.500000 | +0.1398% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1150.000000 | 1139.000000 | -0.9565% | 1277.500000 | 1265.000000 | -0.9785% |
| generation_final_norm_ticks | 927.000000 | 918.000000 | -0.9709% | 920.500000 | 919.500000 | -0.1086% |
| generation_lm_head_argmax_ticks | 6518.000000 | 6530.000000 | +0.1841% | 6447.000000 | 6474.500000 | +0.4266% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128595.000000 | 128721.000000 | +0.0980% | 128537.000000 | 128556.500000 | +0.0152% |
| generation_lm_head_expand_ticks | 111553.000000 | 111506.000000 | -0.0421% | 111346.000000 | 111465.000000 | +0.1069% |
| generation_lm_head_hmx_tail_wait_ticks | 2262.000000 | 2363.000000 | +4.4651% | 2308.000000 | 2358.500000 | +2.1880% |
| generation_lm_head_hmx_ticks | 127558.000000 | 127711.000000 | +0.1199% | 127509.500000 | 127525.000000 | +0.0122% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 244.000000 | 231.000000 | -5.3279% | 242.500000 | 237.000000 | -2.2680% |
| generation_lm_head_scale_init_ticks | 568.000000 | 580.000000 | +2.1127% | 562.500000 | 563.500000 | +0.1778% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129522.000000 | 129639.000000 | +0.0903% | 129455.000000 | 129485.000000 | +0.0232% |
| generation_lm_head_weight_dma_ticks | 127708.000000 | 127838.000000 | +0.1018% | 127631.500000 | 127681.000000 | +0.0388% |
| generation_lm_head_weight_dma_wait_ticks | 2814.000000 | 2892.000000 | +2.7719% | 2910.500000 | 2843.000000 | -2.3192% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 133649.000000 | 132655.000000 | -0.7437% | 130935.500000 | 129752.500000 | -0.9035% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2542.761000 | 2281.875000 | -10.2599% | 2369.192417 | 2281.901500 | -3.6844% |
| host_us | 63048.386000 | 62769.375000 | -0.4425% | 62982.682000 | 62850.834000 | -0.2093% |
| host_wall_ns | 63048386.000000 | 62769375.000000 | -0.4425% | 62982682.000000 | 62850834.000000 | -0.2093% |
| input_norm_ticks | 9448.000000 | 9425.000000 | -0.2434% | 9432.000000 | 9438.500000 | +0.0689% |
| input_stage_ticks | 9.000000 | 10.000000 | +11.1111% | 11.000000 | 11.500000 | +4.5455% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1161708.000000 | 1161360.000000 | -0.0300% | 1163737.000000 | 1164703.500000 | +0.0831% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 425.000000 | 435.000000 | +2.3529% | 452.000000 | 449.500000 | -0.5531% |
| ledger_named_ticks | 1161708.000000 | 1161360.000000 | -0.0300% | 1163737.000000 | 1164703.500000 | +0.0831% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7405.000000 | 7266.000000 | -1.8771% | 7389.000000 | 7377.500000 | -0.1556% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96332.000000 | 96220.000000 | -0.1163% | 96595.000000 | 96471.500000 | -0.1279% |
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
| post_attention_norm_ticks | 23.000000 | 21.000000 | -8.6957% | 22.000000 | 21.500000 | -2.2727% |
| post_attention_residual_ticks | 9071.000000 | 9067.000000 | -0.0441% | 9071.000000 | 9068.500000 | -0.0276% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 680392.000000 | 681096.000000 | +0.1035% | 682840.000000 | 683806.500000 | +0.1415% |
| projection_pack_ticks | 3172.000000 | 3166.000000 | -0.1892% | 3170.500000 | 3167.000000 | -0.1104% |
| projection_unpack_ticks | 10920.000000 | 10733.000000 | -1.7125% | 10832.000000 | 10898.500000 | +0.6139% |
| qk_norm_rope_ticks | 20.000000 | 20.000000 | +0.0000% | 22.500000 | 23.000000 | +2.2222% |
| qkv_projection_ticks | 225513.000000 | 225180.000000 | -0.1477% | 225300.000000 | 225129.000000 | -0.0759% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 906.000000 | 888.000000 | -1.9868% | 904.500000 | 905.500000 | +0.1106% |
| runtime_teardown_ticks | 881.000000 | 885.000000 | +0.4540% | 900.500000 | 890.500000 | -1.1105% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6501.000000 | 6521.000000 | +0.3076% | 6540.500000 | 6532.500000 | -0.1223% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3369.000000 | 3238.000000 | -3.8884% | 3345.500000 | 3313.500000 | -0.9565% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 128.000000 | 156.000000 | +21.8750% | 158.500000 | 160.000000 | +0.9464% |
| total_ticks | 1160802.000000 | 1160472.000000 | -0.0284% | 1162811.500000 | 1163781.500000 | +0.0834% |
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
| w4f16_cross_prefetch_lifetime_ticks | 250684.000000 | 249901.000000 | -0.3123% | 250371.500000 | 249799.500000 | -0.2285% |
| w4f16_cross_prefetch_wait_ticks | 969.000000 | 1032.000000 | +6.5015% | 1065.500000 | 1001.000000 | -6.0535% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74343.000000 | 74781.000000 | +0.5892% | 75287.000000 | 74612.500000 | -0.8959% |
| w4f16_expand_ticks | 613839.000000 | 614768.000000 | +0.1513% | 615273.500000 | 615707.000000 | +0.0705% |
| w4f16_expand_work_ticks | 1693868.000000 | 1696326.000000 | +0.1451% | 1694075.500000 | 1696372.000000 | +0.1356% |
| w4f16_gate_up_expand_pool_wait_ticks | 29907.000000 | 29512.000000 | -1.3208% | 30682.000000 | 29862.500000 | -2.6709% |
| w4f16_gate_up_expand_ticks | 252100.000000 | 252971.000000 | +0.3455% | 252772.500000 | 253440.000000 | +0.2641% |
| w4f16_gate_up_expand_work_ticks | 159116.000000 | 160406.000000 | +0.8107% | 158758.500000 | 160473.500000 | +1.0803% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7378.000000 | 7310.000000 | -0.9217% | 7397.500000 | 7299.000000 | -1.3315% |
| w4f16_gate_up_hmx_wait_ticks | 277478.000000 | 278226.000000 | +0.2696% | 278597.500000 | 279299.500000 | +0.2520% |
| w4f16_gate_up_stream_join_wait_ticks | 1026.000000 | 1277.000000 | +24.4639% | 657.000000 | 680.500000 | +3.5769% |
| w4f16_gate_up_stream_ready_wait_ticks | 1152.000000 | 1263.000000 | +9.6354% | 1328.500000 | 1321.500000 | -0.5269% |
| w4f16_gate_up_stream_work_ticks | 134411.000000 | 134418.000000 | +0.0052% | 134418.000000 | 134413.500000 | -0.0033% |
| w4f16_gate_up_weight_dma_ticks | 556740.000000 | 557797.000000 | +0.1899% | 559323.500000 | 560450.000000 | +0.2014% |
| w4f16_hmx_tail_wait_ticks | 19281.000000 | 19037.000000 | -1.2655% | 19309.000000 | 19098.500000 | -1.0902% |
| w4f16_prefetch_wait_ticks | 28379.000000 | 28708.000000 | +1.1593% | 29107.000000 | 30022.000000 | +3.1436% |
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
| weight_dma_ticks | 1134321.000000 | 1135324.000000 | +0.0884% | 1137968.500000 | 1139800.500000 | +0.1610% |


## Original versus A decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.066667 | 9.400000 | +3.6765% | 9.000000 | 9.200000 | +2.2222% |
| attention_av_hmx_ticks | 8014.000000 | 7920.133333 | -1.1713% | 7958.133333 | 7941.733333 | -0.2061% |
| attention_av_pack_ticks | 3031.866667 | 3044.200000 | +0.4068% | 3030.533333 | 3036.700000 | +0.2035% |
| attention_av_unpack_ticks | 4005.666667 | 4008.200000 | +0.0632% | 4006.933333 | 4006.900000 | -0.0008% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8109.333333 | 8034.400000 | -0.9240% | 8040.300000 | 8035.733333 | -0.0568% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6380.200000 | 6383.866667 | +0.0575% | 6383.933333 | 6382.766667 | -0.0183% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132732.933333 | 132938.133333 | +0.1546% | 132714.233333 | 132829.200000 | +0.0866% |
| attention_ticks | 644018.200000 | 644177.933333 | +0.0248% | 643919.066667 | 644102.566667 | +0.0285% |
| attention_unattributed_ticks | 481744.200000 | 481849.000000 | +0.0218% | 481798.433333 | 481842.466667 | +0.0091% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.333333 | 287.133333 | +0.2794% | 283.900000 | 287.133333 | +1.1389% |
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
| down_ticks | 164347.266667 | 164557.466667 | +0.1279% | 164348.433333 | 164791.866667 | +0.2698% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29904.200000 | 29966.466667 | +0.2082% | 29951.100000 | 29960.133333 | +0.0302% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2672.533333 | 2670.133333 | -0.0898% | 2671.533333 | 2670.966667 | -0.0212% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 428466.133333 | 428634.666667 | +0.0393% | 428546.700000 | 428950.033333 | +0.0941% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 40.466667 | 36.466667 | -9.8847% | 34.600000 | 39.766667 | +14.9326% |
| generation_final_norm_ticks | 45.933333 | 44.733333 | -2.6125% | 41.700000 | 41.900000 | +0.4796% |
| generation_lm_head_argmax_ticks | 6428.800000 | 6456.400000 | +0.4293% | 6452.266667 | 6496.333333 | +0.6830% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128080.400000 | 128236.466667 | +0.1219% | 128317.233333 | 128363.233333 | +0.0358% |
| generation_lm_head_expand_ticks | 111145.466667 | 111440.400000 | +0.2654% | 111477.933333 | 111452.600000 | -0.0227% |
| generation_lm_head_hmx_tail_wait_ticks | 2268.933333 | 2182.066667 | -3.8285% | 2229.633333 | 2241.833333 | +0.5472% |
| generation_lm_head_hmx_ticks | 127155.533333 | 127314.600000 | +0.1251% | 127387.700000 | 127439.000000 | +0.0403% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 210.200000 | 207.000000 | -1.5224% | 210.066667 | 210.166667 | +0.0476% |
| generation_lm_head_scale_init_ticks | 567.600000 | 565.733333 | -0.3289% | 570.000000 | 566.666667 | -0.5848% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128126.333333 | 128281.200000 | +0.1209% | 128358.933333 | 128403.800000 | +0.0350% |
| generation_lm_head_weight_dma_ticks | 127258.600000 | 127414.533333 | +0.1225% | 127490.100000 | 127541.933333 | +0.0407% |
| generation_lm_head_weight_dma_wait_ticks | 2874.466667 | 2876.333333 | +0.0649% | 2870.100000 | 2855.100000 | -0.5226% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 124419.400000 | 125016.733333 | +0.4801% | 123533.866667 | 123530.033333 | -0.0031% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1497.215172 | 2224.541639 | +48.5786% | 1940.418278 | 2015.387003 | +3.8635% |
| host_us | 92185.649200 | 92945.642333 | +0.8244% | 92631.223833 | 92754.499900 | +0.1331% |
| host_wall_ns | 92185649.200000 | 92945642.333333 | +0.8244% | 92631223.833333 | 92754499.900000 | +0.1331% |
| input_norm_ticks | 9367.600000 | 9357.333333 | -0.1096% | 9361.066667 | 9360.600000 | -0.0050% |
| input_stage_ticks | 7.266667 | 8.400000 | +15.5963% | 8.366667 | 8.033333 | -3.9841% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741217.933333 | 1741845.133333 | +0.0360% | 1741017.833333 | 1742276.600000 | +0.0723% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 322.800000 | 321.000000 | -0.5576% | 322.866667 | 322.333333 | -0.1652% |
| ledger_named_ticks | 1741217.933333 | 1741845.133333 | +0.0360% | 1741017.833333 | 1742276.600000 | +0.0723% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6288.133333 | 6207.533333 | -1.2818% | 6145.333333 | 6114.066667 | -0.5088% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96520.533333 | 96373.466667 | -0.1524% | 96352.866667 | 96433.200000 | +0.0834% |
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
| post_attention_norm_ticks | 15.866667 | 16.600000 | +4.6218% | 17.166667 | 16.966667 | -1.1650% |
| post_attention_residual_ticks | 9028.200000 | 9008.733333 | -0.2156% | 9017.033333 | 9019.200000 | +0.0240% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682311.533333 | 682764.800000 | +0.0664% | 681700.900000 | 682743.400000 | +0.1529% |
| projection_pack_ticks | 421.800000 | 416.466667 | -1.2644% | 420.933333 | 418.966667 | -0.4672% |
| projection_unpack_ticks | 10830.533333 | 10820.600000 | -0.0917% | 10849.033333 | 10916.133333 | +0.6185% |
| qk_norm_rope_ticks | 17.600000 | 17.000000 | -3.4091% | 17.466667 | 17.233333 | -1.3359% |
| qkv_projection_ticks | 224817.733333 | 225037.933333 | +0.0979% | 224857.566667 | 224938.433333 | +0.0360% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 629.600000 | 618.600000 | -1.7471% | 622.700000 | 624.233333 | +0.2462% |
| runtime_teardown_ticks | 584.333333 | 575.066667 | -1.5859% | 576.900000 | 577.500000 | +0.1040% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4284.066667 | 4292.466667 | +0.1961% | 4283.266667 | 4286.700000 | +0.0802% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21326.466667 | 21319.800000 | -0.0313% | 21325.400000 | 21329.266667 | +0.0181% |
| scan_cache_stage_ticks | 13780.600000 | 13830.733333 | +0.3638% | 13790.100000 | 13821.233333 | +0.2258% |
| scan_dynamic_attention_ticks | 643969.466667 | 644129.133333 | +0.0248% | 643868.533333 | 644051.866667 | +0.0285% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 41.400000 | 36.800000 | -11.1111% | 39.466667 | 36.700000 | -7.0101% |
| total_ticks | 1740588.333333 | 1741226.533333 | +0.0367% | 1740400.000000 | 1741649.933333 | +0.0718% |
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
| w4f16_cross_prefetch_lifetime_ticks | 833316.466667 | 833691.466667 | +0.0450% | 833167.400000 | 833406.000000 | +0.0286% |
| w4f16_cross_prefetch_wait_ticks | 972.733333 | 984.000000 | +1.1582% | 1073.933333 | 1094.333333 | +1.8996% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75684.333333 | 76107.400000 | +0.5590% | 75997.633333 | 76148.166667 | +0.1981% |
| w4f16_expand_ticks | 614641.000000 | 614964.333333 | +0.0526% | 614479.866667 | 615061.266667 | +0.0946% |
| w4f16_expand_work_ticks | 1692794.733333 | 1694393.866667 | +0.0945% | 1694022.433333 | 1694066.033333 | +0.0026% |
| w4f16_gate_up_expand_pool_wait_ticks | 31111.933333 | 30809.266667 | -0.9728% | 30962.433333 | 30913.633333 | -0.1576% |
| w4f16_gate_up_expand_ticks | 252215.200000 | 252360.400000 | +0.0576% | 252361.200000 | 252523.533333 | +0.0643% |
| w4f16_gate_up_expand_work_ticks | 157864.733333 | 158414.333333 | +0.3481% | 158077.466667 | 158112.966667 | +0.0225% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7499.600000 | 7538.400000 | +0.5174% | 7439.766667 | 7406.533333 | -0.4467% |
| w4f16_gate_up_hmx_wait_ticks | 277820.933333 | 278046.733333 | +0.0813% | 277983.633333 | 278295.700000 | +0.1123% |
| w4f16_gate_up_stream_join_wait_ticks | 699.066667 | 624.733333 | -10.6332% | 682.833333 | 630.500000 | -7.6641% |
| w4f16_gate_up_stream_ready_wait_ticks | 1319.800000 | 1298.866667 | -1.5861% | 1342.200000 | 1347.900000 | +0.4247% |
| w4f16_gate_up_stream_work_ticks | 134363.133333 | 134394.133333 | +0.0231% | 134387.966667 | 134394.833333 | +0.0051% |
| w4f16_gate_up_weight_dma_ticks | 557251.066667 | 557553.733333 | +0.0543% | 557614.166667 | 558507.666667 | +0.1602% |
| w4f16_hmx_tail_wait_ticks | 19685.600000 | 19753.266667 | +0.3437% | 19332.266667 | 19126.833333 | -1.0626% |
| w4f16_prefetch_wait_ticks | 29214.000000 | 29551.000000 | +1.1536% | 29668.033333 | 30119.633333 | +1.5222% |
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
| weight_dma_ticks | 1133403.200000 | 1133840.066667 | +0.0385% | 1133383.966667 | 1134914.466667 | +0.1350% |


## Original versus B prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 11.000000 | 7.000000 | -36.3636% | 12.000000 | 11.000000 | -8.3333% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75902.000000 | 74943.000000 | -1.2635% | 75883.000000 | 75771.000000 | -0.1476% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 121.000000 | 125.000000 | +3.3058% | 121.500000 | 121.500000 | +0.0000% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76081.000000 | 75119.000000 | -1.2644% | 76059.000000 | 75945.500000 | -0.1492% |
| attention_unattributed_ticks | 58.000000 | 51.000000 | -12.0690% | 56.500000 | 55.500000 | -1.7699% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 382.000000 | 378.000000 | -1.0471% | 368.000000 | 367.000000 | -0.2717% |
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
| down_ticks | 163227.000000 | 163255.000000 | +0.0172% | 164347.000000 | 164487.000000 | +0.0852% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9844.000000 | 9755.000000 | -0.9041% | 9886.000000 | 9863.500000 | -0.2276% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2684.000000 | 2694.000000 | +0.3726% | 2690.500000 | 2690.500000 | +0.0000% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 428620.000000 | 427773.000000 | -0.1976% | 429669.000000 | 429125.500000 | -0.1265% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1150.000000 | 1159.000000 | +0.7826% | 1277.500000 | 1260.000000 | -1.3699% |
| generation_final_norm_ticks | 927.000000 | 930.000000 | +0.3236% | 920.500000 | 925.000000 | +0.4889% |
| generation_lm_head_argmax_ticks | 6518.000000 | 6418.000000 | -1.5342% | 6447.000000 | 6419.000000 | -0.4343% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128595.000000 | 127088.000000 | -1.1719% | 128537.000000 | 127201.500000 | -1.0390% |
| generation_lm_head_expand_ticks | 111553.000000 | 110461.000000 | -0.9789% | 111346.000000 | 110417.000000 | -0.8343% |
| generation_lm_head_hmx_tail_wait_ticks | 2262.000000 | 2260.000000 | -0.0884% | 2308.000000 | 2353.500000 | +1.9714% |
| generation_lm_head_hmx_ticks | 127558.000000 | 126076.000000 | -1.1618% | 127509.500000 | 126183.500000 | -1.0399% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 244.000000 | 239.000000 | -2.0492% | 242.500000 | 238.000000 | -1.8557% |
| generation_lm_head_scale_init_ticks | 568.000000 | 572.000000 | +0.7042% | 562.500000 | 570.500000 | +1.4222% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129522.000000 | 128018.000000 | -1.1612% | 129455.000000 | 128120.500000 | -1.0309% |
| generation_lm_head_weight_dma_ticks | 127708.000000 | 126183.000000 | -1.1941% | 127631.500000 | 126327.500000 | -1.0217% |
| generation_lm_head_weight_dma_wait_ticks | 2814.000000 | 2575.000000 | -8.4932% | 2910.500000 | 2643.000000 | -9.1909% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 133649.000000 | 130573.000000 | -2.3016% | 130935.500000 | 125237.000000 | -4.3521% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2542.761000 | 2512.864167 | -1.1758% | 2369.192417 | 2346.536333 | -0.9563% |
| host_us | 63048.386000 | 62786.510000 | -0.4154% | 62982.682000 | 62818.255000 | -0.2611% |
| host_wall_ns | 63048386.000000 | 62786510.000000 | -0.4154% | 62982682.000000 | 62818255.000000 | -0.2611% |
| input_norm_ticks | 9448.000000 | 9426.000000 | -0.2329% | 9432.000000 | 9441.500000 | +0.1007% |
| input_stage_ticks | 9.000000 | 14.000000 | +55.5556% | 11.000000 | 9.500000 | -13.6364% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1161708.000000 | 1157254.000000 | -0.3834% | 1163737.000000 | 1161770.500000 | -0.1690% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 425.000000 | 432.000000 | +1.6471% | 452.000000 | 454.000000 | +0.4425% |
| ledger_named_ticks | 1161708.000000 | 1157254.000000 | -0.3834% | 1163737.000000 | 1161770.500000 | -0.1690% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7405.000000 | 7137.000000 | -3.6192% | 7389.000000 | 7325.500000 | -0.8594% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96332.000000 | 95996.000000 | -0.3488% | 96595.000000 | 96475.000000 | -0.1242% |
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
| post_attention_norm_ticks | 23.000000 | 22.000000 | -4.3478% | 22.000000 | 23.000000 | +4.5455% |
| post_attention_residual_ticks | 9071.000000 | 9056.000000 | -0.1654% | 9071.000000 | 9058.000000 | -0.1433% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 680392.000000 | 677824.000000 | -0.3774% | 682840.000000 | 681398.500000 | -0.2111% |
| projection_pack_ticks | 3172.000000 | 3182.000000 | +0.3153% | 3170.500000 | 3167.500000 | -0.0946% |
| projection_unpack_ticks | 10920.000000 | 10966.000000 | +0.4212% | 10832.000000 | 10910.500000 | +0.7247% |
| qk_norm_rope_ticks | 20.000000 | 22.000000 | +10.0000% | 22.500000 | 21.500000 | -4.4444% |
| qkv_projection_ticks | 225513.000000 | 225021.000000 | -0.2182% | 225300.000000 | 225078.500000 | -0.0983% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 906.000000 | 889.000000 | -1.8764% | 904.500000 | 917.500000 | +1.4373% |
| runtime_teardown_ticks | 881.000000 | 893.000000 | +1.3621% | 900.500000 | 903.500000 | +0.3331% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6501.000000 | 6482.000000 | -0.2923% | 6540.500000 | 6512.500000 | -0.4281% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3369.000000 | 3304.000000 | -1.9294% | 3345.500000 | 3341.500000 | -0.1196% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 128.000000 | 157.000000 | +22.6562% | 158.500000 | 170.500000 | +7.5710% |
| total_ticks | 1160802.000000 | 1156365.000000 | -0.3822% | 1162811.500000 | 1160854.000000 | -0.1683% |
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
| w4f16_cross_prefetch_lifetime_ticks | 250684.000000 | 250403.000000 | -0.1121% | 250371.500000 | 250101.000000 | -0.1080% |
| w4f16_cross_prefetch_wait_ticks | 969.000000 | 973.000000 | +0.4128% | 1065.500000 | 1036.000000 | -2.7687% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74343.000000 | 73819.000000 | -0.7048% | 75287.000000 | 74803.500000 | -0.6422% |
| w4f16_expand_ticks | 613839.000000 | 613314.000000 | -0.0855% | 615273.500000 | 614067.500000 | -0.1960% |
| w4f16_expand_work_ticks | 1693868.000000 | 1693895.000000 | +0.0016% | 1694075.500000 | 1695544.500000 | +0.0867% |
| w4f16_gate_up_expand_pool_wait_ticks | 29907.000000 | 29761.000000 | -0.4882% | 30682.000000 | 30257.000000 | -1.3852% |
| w4f16_gate_up_expand_ticks | 252100.000000 | 251943.000000 | -0.0623% | 252772.500000 | 252591.000000 | -0.0718% |
| w4f16_gate_up_expand_work_ticks | 159116.000000 | 160467.000000 | +0.8491% | 158758.500000 | 160343.500000 | +0.9984% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7378.000000 | 6984.000000 | -5.3402% | 7397.500000 | 7177.500000 | -2.9740% |
| w4f16_gate_up_hmx_wait_ticks | 277478.000000 | 276650.000000 | -0.2984% | 278597.500000 | 278186.000000 | -0.1477% |
| w4f16_gate_up_stream_join_wait_ticks | 1026.000000 | 931.000000 | -9.2593% | 657.000000 | 603.500000 | -8.1431% |
| w4f16_gate_up_stream_ready_wait_ticks | 1152.000000 | 1374.000000 | +19.2708% | 1328.500000 | 1357.000000 | +2.1453% |
| w4f16_gate_up_stream_work_ticks | 134411.000000 | 134458.000000 | +0.0350% | 134418.000000 | 134422.000000 | +0.0030% |
| w4f16_gate_up_weight_dma_ticks | 556740.000000 | 555044.000000 | -0.3046% | 559323.500000 | 558430.500000 | -0.1597% |
| w4f16_hmx_tail_wait_ticks | 19281.000000 | 18822.000000 | -2.3806% | 19309.000000 | 18963.000000 | -1.7919% |
| w4f16_prefetch_wait_ticks | 28379.000000 | 27210.000000 | -4.1192% | 29107.000000 | 29873.000000 | +2.6317% |
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
| weight_dma_ticks | 1134321.000000 | 1129482.000000 | -0.4266% | 1137968.500000 | 1135610.000000 | -0.2073% |


## Original versus B decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.066667 | 8.600000 | -5.1471% | 9.000000 | 8.666667 | -3.7037% |
| attention_av_hmx_ticks | 8014.000000 | 7951.933333 | -0.7745% | 7958.133333 | 7941.966667 | -0.2031% |
| attention_av_pack_ticks | 3031.866667 | 3049.600000 | +0.5849% | 3030.533333 | 3042.933333 | +0.4092% |
| attention_av_unpack_ticks | 4005.666667 | 4012.133333 | +0.1614% | 4006.933333 | 4005.600000 | -0.0333% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8109.333333 | 8034.066667 | -0.9281% | 8040.300000 | 8022.533333 | -0.2210% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6380.200000 | 6386.733333 | +0.1024% | 6383.933333 | 6383.466667 | -0.0073% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132732.933333 | 133212.866667 | +0.3616% | 132714.233333 | 133116.300000 | +0.3030% |
| attention_ticks | 644018.200000 | 644488.400000 | +0.0730% | 643919.066667 | 644388.733333 | +0.0729% |
| attention_unattributed_ticks | 481744.200000 | 481841.066667 | +0.0201% | 481798.433333 | 481848.333333 | +0.0104% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.333333 | 288.333333 | +0.6985% | 283.900000 | 284.266667 | +0.1292% |
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
| down_ticks | 164347.266667 | 164991.466667 | +0.3920% | 164348.433333 | 164985.300000 | +0.3875% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29904.200000 | 30010.866667 | +0.3567% | 29951.100000 | 29964.066667 | +0.0433% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2672.533333 | 2671.466667 | -0.0399% | 2671.533333 | 2670.700000 | -0.0312% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 428466.133333 | 427067.000000 | -0.3265% | 428546.700000 | 427828.433333 | -0.1676% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 40.466667 | 42.600000 | +5.2718% | 34.600000 | 39.400000 | +13.8728% |
| generation_final_norm_ticks | 45.933333 | 42.333333 | -7.8374% | 41.700000 | 42.900000 | +2.8777% |
| generation_lm_head_argmax_ticks | 6428.800000 | 6395.466667 | -0.5185% | 6452.266667 | 6418.166667 | -0.5285% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128080.400000 | 127142.200000 | -0.7325% | 128317.233333 | 127301.266667 | -0.7918% |
| generation_lm_head_expand_ticks | 111145.466667 | 110693.200000 | -0.4069% | 111477.933333 | 110758.866667 | -0.6450% |
| generation_lm_head_hmx_tail_wait_ticks | 2268.933333 | 2178.000000 | -4.0078% | 2229.633333 | 2220.566667 | -0.4066% |
| generation_lm_head_hmx_ticks | 127155.533333 | 126213.466667 | -0.7409% | 127387.700000 | 126375.200000 | -0.7948% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 210.200000 | 213.200000 | +1.4272% | 210.066667 | 213.466667 | +1.6185% |
| generation_lm_head_scale_init_ticks | 567.600000 | 569.333333 | +0.3054% | 570.000000 | 568.366667 | -0.2865% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128126.333333 | 127184.533333 | -0.7351% | 128358.933333 | 127346.466667 | -0.7888% |
| generation_lm_head_weight_dma_ticks | 127258.600000 | 126314.400000 | -0.7420% | 127490.100000 | 126478.533333 | -0.7934% |
| generation_lm_head_weight_dma_wait_ticks | 2874.466667 | 2600.733333 | -9.5229% | 2870.100000 | 2580.566667 | -10.0879% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 124419.400000 | 122872.933333 | -1.2429% | 123533.866667 | 119753.766667 | -3.0600% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1497.215172 | 1518.982550 | +1.4539% | 1940.418278 | 1862.345364 | -4.0235% |
| host_us | 92185.649200 | 92145.472133 | -0.0436% | 92631.223833 | 92479.607467 | -0.1637% |
| host_wall_ns | 92185649.200000 | 92145472.133333 | -0.0436% | 92631223.833333 | 92479607.466667 | -0.1637% |
| input_norm_ticks | 9367.600000 | 9356.200000 | -0.1217% | 9361.066667 | 9355.766667 | -0.0566% |
| input_stage_ticks | 7.266667 | 6.933333 | -4.5872% | 8.366667 | 7.833333 | -6.3745% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741217.933333 | 1740028.600000 | -0.0683% | 1741017.833333 | 1740325.766667 | -0.0398% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 322.800000 | 322.533333 | -0.0826% | 322.866667 | 322.833333 | -0.0103% |
| ledger_named_ticks | 1741217.933333 | 1740028.600000 | -0.0683% | 1741017.833333 | 1740325.766667 | -0.0398% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6288.133333 | 6469.800000 | +2.8890% | 6145.333333 | 6159.466667 | +0.2300% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96520.533333 | 96699.266667 | +0.1852% | 96352.866667 | 96403.800000 | +0.0529% |
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
| post_attention_norm_ticks | 15.866667 | 17.266667 | +8.8235% | 17.166667 | 17.200000 | +0.1942% |
| post_attention_residual_ticks | 9028.200000 | 9022.866667 | -0.0591% | 9017.033333 | 9018.433333 | +0.0155% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682311.533333 | 680002.800000 | -0.3384% | 681700.900000 | 681050.533333 | -0.0954% |
| projection_pack_ticks | 421.800000 | 413.733333 | -1.9124% | 420.933333 | 419.600000 | -0.3168% |
| projection_unpack_ticks | 10830.533333 | 11045.466667 | +1.9845% | 10849.033333 | 10948.733333 | +0.9190% |
| qk_norm_rope_ticks | 17.600000 | 16.866667 | -4.1667% | 17.466667 | 17.166667 | -1.7176% |
| qkv_projection_ticks | 224817.733333 | 224500.933333 | -0.1409% | 224857.566667 | 224959.000000 | +0.0451% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 629.600000 | 621.000000 | -1.3659% | 622.700000 | 623.733333 | +0.1659% |
| runtime_teardown_ticks | 584.333333 | 583.133333 | -0.2054% | 576.900000 | 578.266667 | +0.2369% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4284.066667 | 4300.666667 | +0.3875% | 4283.266667 | 4281.933333 | -0.0311% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21326.466667 | 21329.000000 | +0.0119% | 21325.400000 | 21328.766667 | +0.0158% |
| scan_cache_stage_ticks | 13780.600000 | 13771.666667 | -0.0648% | 13790.100000 | 13808.000000 | +0.1298% |
| scan_dynamic_attention_ticks | 643969.466667 | 644438.533333 | +0.0728% | 643868.533333 | 644339.000000 | +0.0731% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 41.400000 | 39.733333 | -4.0258% | 39.466667 | 39.966667 | +1.2669% |
| total_ticks | 1740588.333333 | 1739407.600000 | -0.0678% | 1740400.000000 | 1739697.833333 | -0.0403% |
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
| w4f16_cross_prefetch_lifetime_ticks | 833316.466667 | 834274.800000 | +0.1150% | 833167.400000 | 833938.366667 | +0.0925% |
| w4f16_cross_prefetch_wait_ticks | 972.733333 | 1062.866667 | +9.2660% | 1073.933333 | 1071.700000 | -0.2080% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75684.333333 | 76293.400000 | +0.8047% | 75997.633333 | 76571.266667 | +0.7548% |
| w4f16_expand_ticks | 614641.000000 | 613463.866667 | -0.1915% | 614479.866667 | 613696.500000 | -0.1275% |
| w4f16_expand_work_ticks | 1692794.733333 | 1694560.000000 | +0.1043% | 1694022.433333 | 1694821.866667 | +0.0472% |
| w4f16_gate_up_expand_pool_wait_ticks | 31111.933333 | 31064.733333 | -0.1517% | 30962.433333 | 31350.266667 | +1.2526% |
| w4f16_gate_up_expand_ticks | 252215.200000 | 251102.866667 | -0.4410% | 252361.200000 | 251903.366667 | -0.1814% |
| w4f16_gate_up_expand_work_ticks | 157864.733333 | 158090.733333 | +0.1432% | 158077.466667 | 158320.000000 | +0.1534% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7499.600000 | 7241.800000 | -3.4375% | 7439.766667 | 7118.500000 | -4.3182% |
| w4f16_gate_up_hmx_wait_ticks | 277820.933333 | 276135.200000 | -0.6068% | 277983.633333 | 277208.533333 | -0.2788% |
| w4f16_gate_up_stream_join_wait_ticks | 699.066667 | 882.266667 | +26.2064% | 682.833333 | 667.166667 | -2.2944% |
| w4f16_gate_up_stream_ready_wait_ticks | 1319.800000 | 1362.333333 | +3.2227% | 1342.200000 | 1349.900000 | +0.5737% |
| w4f16_gate_up_stream_work_ticks | 134363.133333 | 134377.800000 | +0.0109% | 134387.966667 | 134396.433333 | +0.0063% |
| w4f16_gate_up_weight_dma_ticks | 557251.066667 | 554085.800000 | -0.5680% | 557614.166667 | 556399.966667 | -0.2177% |
| w4f16_hmx_tail_wait_ticks | 19685.600000 | 19317.533333 | -1.8697% | 19332.266667 | 18932.800000 | -2.0663% |
| w4f16_prefetch_wait_ticks | 29214.000000 | 28709.733333 | -1.7261% | 29668.033333 | 29994.966667 | +1.1020% |
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
| weight_dma_ticks | 1133403.200000 | 1129640.733333 | -0.3320% | 1133383.966667 | 1131606.933333 | -0.1568% |


## Original versus C prefill

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 11.000000 | 7.000000 | -36.3636% | 12.000000 | 11.500000 | -4.1667% |
| attention_av_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_av_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_gqa_pipeline_ticks | 75902.000000 | 74855.000000 | -1.3794% | 75883.000000 | 75661.500000 | -0.2919% |
| attention_qk_hmx_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_setup_ticks | 121.000000 | 118.000000 | -2.4793% | 121.500000 | 119.500000 | -1.6461% |
| attention_softmax_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_ticks | 76081.000000 | 75036.000000 | -1.3735% | 76059.000000 | 75838.500000 | -0.2899% |
| attention_unattributed_ticks | 58.000000 | 63.000000 | +8.6207% | 56.500000 | 57.000000 | +0.8850% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 382.000000 | 411.000000 | +7.5916% | 368.000000 | 371.500000 | +0.9511% |
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
| down_ticks | 163227.000000 | 164232.000000 | +0.6157% | 164347.000000 | 165623.000000 | +0.7764% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 9844.000000 | 9814.000000 | -0.3048% | 9886.000000 | 9838.000000 | -0.4855% |
| f16_cache_native_incremental_append_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reuse_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reused_carrier_bytes | 7340032.000000 | 7340032.000000 | +0.0000% | 7340032.000000 | 7340032.000000 | +0.0000% |
| final_residual_ticks | 2684.000000 | 2698.000000 | +0.5216% | 2690.500000 | 2691.500000 | +0.0372% |
| first_position | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| gate_up_ticks | 428620.000000 | 429430.000000 | +0.1890% | 429669.000000 | 430890.500000 | +0.2843% |
| generation_embedding_ddr_read_bytes | 262400.000000 | 262400.000000 | +0.0000% | 262400.000000 | 262400.000000 | +0.0000% |
| generation_embedding_ticks | 1150.000000 | 1235.000000 | +7.3913% | 1277.500000 | 1262.000000 | -1.2133% |
| generation_final_norm_ticks | 927.000000 | 927.000000 | +0.0000% | 920.500000 | 924.000000 | +0.3802% |
| generation_lm_head_argmax_ticks | 6518.000000 | 6514.000000 | -0.0614% | 6447.000000 | 6506.000000 | +0.9152% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128595.000000 | 128565.000000 | -0.0233% | 128537.000000 | 128553.500000 | +0.0128% |
| generation_lm_head_expand_ticks | 111553.000000 | 111275.000000 | -0.2492% | 111346.000000 | 111338.000000 | -0.0072% |
| generation_lm_head_hmx_tail_wait_ticks | 2262.000000 | 2262.000000 | +0.0000% | 2308.000000 | 2291.000000 | -0.7366% |
| generation_lm_head_hmx_ticks | 127558.000000 | 127470.000000 | -0.0690% | 127509.500000 | 127493.500000 | -0.0125% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 244.000000 | 234.000000 | -4.0984% | 242.500000 | 235.000000 | -3.0928% |
| generation_lm_head_scale_init_ticks | 568.000000 | 563.000000 | -0.8803% | 562.500000 | 571.500000 | +1.6000% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 129522.000000 | 129492.000000 | -0.0232% | 129455.000000 | 129476.000000 | +0.0162% |
| generation_lm_head_weight_dma_ticks | 127708.000000 | 127645.000000 | -0.0493% | 127631.500000 | 127649.500000 | +0.0141% |
| generation_lm_head_weight_dma_wait_ticks | 2814.000000 | 2985.000000 | +6.0768% | 2910.500000 | 2933.000000 | +0.7731% |
| generation_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_command_count | 6418.000000 | 6418.000000 | +0.0000% | 6418.000000 | 6418.000000 | +0.0000% |
| hmx_compute_ticks | 133649.000000 | 134321.000000 | +0.5028% | 130935.500000 | 132096.000000 | +0.8863% |
| hmx_fp16_tile_pair_count | 3070720.000000 | 3070720.000000 | +0.0000% | 3070720.000000 | 3070720.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 2542.761000 | 2522.499750 | -0.7968% | 2369.192417 | 2299.218750 | -2.9535% |
| host_us | 63048.386000 | 63127.031000 | +0.1247% | 62982.682000 | 63085.104000 | +0.1626% |
| host_wall_ns | 63048386.000000 | 63127031.000000 | +0.1247% | 62982682.000000 | 63085104.000000 | +0.1626% |
| input_norm_ticks | 9448.000000 | 9502.000000 | +0.5715% | 9432.000000 | 9449.500000 | +0.1855% |
| input_stage_ticks | 9.000000 | 13.000000 | +44.4444% | 11.000000 | 10.000000 | -9.0909% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1161708.000000 | 1163607.000000 | +0.1635% | 1163737.000000 | 1166895.000000 | +0.2714% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 425.000000 | 473.000000 | +11.2941% | 452.000000 | 450.500000 | -0.3319% |
| ledger_named_ticks | 1161708.000000 | 1163607.000000 | +0.1635% | 1163737.000000 | 1166895.000000 | +0.2714% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| metadata_stage_ticks | 7405.000000 | 7291.000000 | -1.5395% | 7389.000000 | 7350.500000 | -0.5210% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96332.000000 | 96652.000000 | +0.3322% | 96595.000000 | 96859.500000 | +0.2738% |
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
| post_attention_norm_ticks | 23.000000 | 34.000000 | +47.8261% | 22.000000 | 23.000000 | +4.5455% |
| post_attention_residual_ticks | 9071.000000 | 9094.000000 | +0.2536% | 9071.000000 | 9079.000000 | +0.0882% |
| prepared_session_run_index | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 680392.000000 | 681653.000000 | +0.1853% | 682840.000000 | 685945.000000 | +0.4547% |
| projection_pack_ticks | 3172.000000 | 3181.000000 | +0.2837% | 3170.500000 | 3165.500000 | -0.1577% |
| projection_unpack_ticks | 10920.000000 | 11106.000000 | +1.7033% | 10832.000000 | 11134.500000 | +2.7927% |
| qk_norm_rope_ticks | 20.000000 | 27.000000 | +35.0000% | 22.500000 | 24.000000 | +6.6667% |
| qkv_projection_ticks | 225513.000000 | 226031.000000 | +0.2297% | 225300.000000 | 225575.000000 | +0.1221% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 906.000000 | 951.000000 | +4.9669% | 904.500000 | 916.000000 | +1.2714% |
| runtime_teardown_ticks | 881.000000 | 935.000000 | +6.1294% | 900.500000 | 914.500000 | +1.5547% |
| scan_attention_overlay_capacity_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_attention_overlay_required_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 6501.000000 | 6533.000000 | +0.4922% | 6540.500000 | 6552.500000 | +0.1835% |
| scan_cache_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_ddr_write_bytes | 9175040.000000 | 9175040.000000 | +0.0000% | 9175040.000000 | 9175040.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 448.000000 | 448.000000 | +0.0000% | 448.000000 | 448.000000 | +0.0000% |
| scan_cache_pack_ticks | 3369.000000 | 3315.000000 | -1.6028% | 3345.500000 | 3304.000000 | -1.2405% |
| scan_cache_stage_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_dynamic_attention_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_logical_m_observed | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_padded_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| scan_total_kv_length | 64.000000 | 64.000000 | +0.0000% | 64.000000 | 64.000000 | +0.0000% |
| stage_boundary_ticks | 128.000000 | 215.000000 | +67.9688% | 158.500000 | 164.000000 | +3.4700% |
| total_ticks | 1160802.000000 | 1162656.000000 | +0.1597% | 1162811.500000 | 1165986.500000 | +0.2730% |
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
| w4f16_cross_prefetch_lifetime_ticks | 250684.000000 | 250170.000000 | -0.2050% | 250371.500000 | 249860.000000 | -0.2043% |
| w4f16_cross_prefetch_wait_ticks | 969.000000 | 945.000000 | -2.4768% | 1065.500000 | 1073.500000 | +0.7508% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 74343.000000 | 74297.000000 | -0.0619% | 75287.000000 | 75787.500000 | +0.6648% |
| w4f16_expand_ticks | 613839.000000 | 615799.000000 | +0.3193% | 615273.500000 | 618332.500000 | +0.4972% |
| w4f16_expand_work_ticks | 1693868.000000 | 1693207.000000 | -0.0390% | 1694075.500000 | 1695438.000000 | +0.0804% |
| w4f16_gate_up_expand_pool_wait_ticks | 29907.000000 | 30034.000000 | +0.4246% | 30682.000000 | 30645.500000 | -0.1190% |
| w4f16_gate_up_expand_ticks | 252100.000000 | 252649.000000 | +0.2178% | 252772.500000 | 254207.500000 | +0.5677% |
| w4f16_gate_up_expand_work_ticks | 159116.000000 | 158862.000000 | -0.1596% | 158758.500000 | 159249.500000 | +0.3093% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7378.000000 | 7306.000000 | -0.9759% | 7397.500000 | 7333.500000 | -0.8652% |
| w4f16_gate_up_hmx_wait_ticks | 277478.000000 | 278013.000000 | +0.1928% | 278597.500000 | 279886.000000 | +0.4625% |
| w4f16_gate_up_stream_join_wait_ticks | 1026.000000 | 824.000000 | -19.6881% | 657.000000 | 606.500000 | -7.6865% |
| w4f16_gate_up_stream_ready_wait_ticks | 1152.000000 | 1146.000000 | -0.5208% | 1328.500000 | 1297.000000 | -2.3711% |
| w4f16_gate_up_stream_work_ticks | 134411.000000 | 134446.000000 | +0.0260% | 134418.000000 | 134401.000000 | -0.0126% |
| w4f16_gate_up_weight_dma_ticks | 556740.000000 | 558575.000000 | +0.3296% | 559323.500000 | 561817.500000 | +0.4459% |
| w4f16_hmx_tail_wait_ticks | 19281.000000 | 19120.000000 | -0.8350% | 19309.000000 | 19056.500000 | -1.3077% |
| w4f16_prefetch_wait_ticks | 28379.000000 | 27905.000000 | -1.6702% | 29107.000000 | 29829.000000 | +2.4805% |
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
| weight_dma_ticks | 1134321.000000 | 1137038.000000 | +0.2395% | 1137968.500000 | 1142754.000000 | +0.4205% |


## Original versus C decode

| Field | R1 original | R1 candidate | R1 change | R10 original median | R10 candidate median | R10 change |
|---|---|---|---|---|---|---|
| activation_ticks | 9.066667 | 9.266667 | +2.2059% | 9.000000 | 8.800000 | -2.2222% |
| attention_av_hmx_ticks | 8014.000000 | 8004.066667 | -0.1239% | 7958.133333 | 7954.633333 | -0.0440% |
| attention_av_pack_ticks | 3031.866667 | 3028.266667 | -0.1187% | 3030.533333 | 3035.300000 | +0.1573% |
| attention_av_unpack_ticks | 4005.666667 | 4008.000000 | +0.0583% | 4006.933333 | 4007.866667 | +0.0233% |
| attention_gqa_pipeline_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_hmx_ticks | 8109.333333 | 8009.866667 | -1.2266% | 8040.300000 | 8040.033333 | -0.0033% |
| attention_qk_pack_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_qk_unpack_ticks | 6380.200000 | 6386.000000 | +0.0909% | 6383.933333 | 6384.733333 | +0.0125% |
| attention_setup_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| attention_softmax_ticks | 132732.933333 | 132847.466667 | +0.0863% | 132714.233333 | 132878.166667 | +0.1235% |
| attention_ticks | 644018.200000 | 644128.800000 | +0.0172% | 643919.066667 | 644148.100000 | +0.0356% |
| attention_unattributed_ticks | 481744.200000 | 481845.133333 | +0.0210% | 481798.433333 | 481844.566667 | +0.0096% |
| block_invocation_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| block_orchestration_ticks | 286.333333 | 282.866667 | -1.2107% | 283.900000 | 284.400000 | +0.1761% |
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
| down_ticks | 164347.266667 | 166411.200000 | +1.2558% | 164348.433333 | 165994.533333 | +1.0016% |
| dsp_status | 3.000000 | 3.000000 | +0.0000% | 3.000000 | 3.000000 | +0.0000% |
| experiment | 218.000000 | 218.000000 | +0.0000% | 218.000000 | 218.000000 | +0.0000% |
| f16_cache_full_prefix_pack_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_append_update_ticks | 29904.200000 | 29992.666667 | +0.2958% | 29951.100000 | 29976.066667 | +0.0834% |
| f16_cache_native_incremental_append_count | 28.000000 | 28.000000 | +0.0000% | 28.000000 | 28.000000 | +0.0000% |
| f16_cache_native_prefill_reuse_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| final_residual_ticks | 2672.533333 | 2673.333333 | +0.0299% | 2671.533333 | 2670.100000 | -0.0537% |
| first_position | 71.000000 | 71.000000 | +0.0000% | 71.000000 | 71.000000 | +0.0000% |
| gate_up_ticks | 428466.133333 | 428929.666667 | +0.1082% | 428546.700000 | 429739.800000 | +0.2784% |
| generation_embedding_ddr_read_bytes | 4352.000000 | 4352.000000 | +0.0000% | 4352.000000 | 4352.000000 | +0.0000% |
| generation_embedding_ticks | 40.466667 | 39.200000 | -3.1301% | 34.600000 | 39.533333 | +14.2582% |
| generation_final_norm_ticks | 45.933333 | 40.733333 | -11.3208% | 41.700000 | 42.266667 | +1.3589% |
| generation_lm_head_argmax_ticks | 6428.800000 | 6524.733333 | +1.4922% | 6452.266667 | 6523.433333 | +1.1030% |
| generation_lm_head_batch_n_tiles | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| generation_lm_head_command_count | 594.000000 | 594.000000 | +0.0000% | 594.000000 | 594.000000 | +0.0000% |
| generation_lm_head_ddr_read_bytes | 156190208.000000 | 156190208.000000 | +0.0000% | 156190208.000000 | 156190208.000000 | +0.0000% |
| generation_lm_head_exclusive_ticks | 128080.400000 | 128669.666667 | +0.4601% | 128317.233333 | 128560.300000 | +0.1894% |
| generation_lm_head_expand_ticks | 111145.466667 | 111719.466667 | +0.5164% | 111477.933333 | 111549.266667 | +0.0640% |
| generation_lm_head_hmx_tail_wait_ticks | 2268.933333 | 2196.600000 | -3.1880% | 2229.633333 | 2234.400000 | +0.2138% |
| generation_lm_head_hmx_ticks | 127155.533333 | 127745.933333 | +0.4643% | 127387.700000 | 127632.300000 | +0.1920% |
| generation_lm_head_n_tiles | 4748.000000 | 4748.000000 | +0.0000% | 4748.000000 | 4748.000000 | +0.0000% |
| generation_lm_head_prefetch_count | 593.000000 | 593.000000 | +0.0000% | 593.000000 | 593.000000 | +0.0000% |
| generation_lm_head_scale_dma_ticks | 210.200000 | 203.466667 | -3.2033% | 210.066667 | 208.833333 | -0.5871% |
| generation_lm_head_scale_init_ticks | 567.600000 | 566.666667 | -0.1644% | 570.000000 | 569.066667 | -0.1637% |
| generation_lm_head_scale_resident_bytes | 607744.000000 | 607744.000000 | +0.0000% | 607744.000000 | 607744.000000 | +0.0000% |
| generation_lm_head_ticks | 128126.333333 | 128710.400000 | +0.4559% | 128358.933333 | 128603.700000 | +0.1907% |
| generation_lm_head_weight_dma_ticks | 127258.600000 | 127849.466667 | +0.4643% | 127490.100000 | 127736.033333 | +0.1929% |
| generation_lm_head_weight_dma_wait_ticks | 2874.466667 | 2929.933333 | +1.9296% | 2870.100000 | 2908.500000 | +1.3379% |
| generation_step | 8.000000 | 8.000000 | +0.0000% | 8.000000 | 8.000000 | +0.0000% |
| hmx_command_count | 5970.000000 | 5970.000000 | +0.0000% | 5970.000000 | 5970.000000 | +0.0000% |
| hmx_compute_ticks | 124419.400000 | 129036.133333 | +3.7106% | 123533.866667 | 125137.033333 | +1.2978% |
| hmx_fp16_tile_pair_count | 3077888.000000 | 3077888.000000 | +0.0000% | 3077888.000000 | 3077888.000000 | +0.0000% |
| hmx_ready_wait_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| hmx_u8s8_tile_pair_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| host_boundary_us | 1497.215172 | 1622.520900 | +8.3693% | 1940.418278 | 1861.585011 | -4.0627% |
| host_us | 92185.649200 | 92481.333400 | +0.3207% | 92631.223833 | 92733.395800 | +0.1103% |
| host_wall_ns | 92185649.200000 | 92481333.400000 | +0.3207% | 92631223.833333 | 92733395.800000 | +0.1103% |
| input_norm_ticks | 9367.600000 | 9363.266667 | -0.0463% | 9361.066667 | 9363.733333 | +0.0285% |
| input_stage_ticks | 7.266667 | 7.333333 | +0.9174% | 8.366667 | 7.966667 | -4.7809% |
| intermediate_ddr_read_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_ddr_write_bytes | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_dma_descriptor_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| intermediate_spill_fill_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| invocation_ticks | 1741217.933333 | 1744489.200000 | +0.1879% | 1741017.833333 | 1744508.566667 | +0.2005% |
| kv_cache_k_format | 4.000000 | 4.000000 | +0.0000% | 4.000000 | 4.000000 | +0.0000% |
| kv_cache_v_format | 5.000000 | 5.000000 | +0.0000% | 5.000000 | 5.000000 | +0.0000% |
| layer_bookkeeping_ticks | 322.800000 | 322.466667 | -0.1033% | 322.866667 | 322.866667 | +0.0000% |
| ledger_named_ticks | 1741217.933333 | 1744489.200000 | +0.1879% | 1741017.833333 | 1744508.566667 | +0.2005% |
| ledger_unattributed_ticks | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| logical_m | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| metadata_stage_ticks | 6288.133333 | 6326.600000 | +0.6117% | 6145.333333 | 6172.833333 | +0.4475% |
| numerical_audit_enabled | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| numerical_status | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| o_projection_ticks | 96520.533333 | 96309.800000 | -0.2183% | 96352.866667 | 96466.033333 | +0.1175% |
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
| post_attention_norm_ticks | 15.866667 | 16.733333 | +5.4622% | 17.166667 | 16.866667 | -1.7476% |
| post_attention_residual_ticks | 9028.200000 | 9019.600000 | -0.0953% | 9017.033333 | 9015.066667 | -0.0218% |
| prepared_session_run_index | 9.000000 | 9.000000 | +0.0000% | 9.000000 | 9.000000 | +0.0000% |
| projection_failure_index | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_n_tile | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_result | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_failure_step | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| projection_hmx_wait_ticks | 682311.533333 | 684515.266667 | +0.3230% | 681700.900000 | 684605.166667 | +0.4260% |
| projection_pack_ticks | 421.800000 | 422.333333 | +0.1264% | 420.933333 | 421.566667 | +0.1505% |
| projection_unpack_ticks | 10830.533333 | 11292.466667 | +4.2651% | 10849.033333 | 11211.633333 | +3.3422% |
| qk_norm_rope_ticks | 17.600000 | 16.400000 | -6.8182% | 17.466667 | 16.900000 | -3.2443% |
| qkv_projection_ticks | 224817.733333 | 225071.800000 | +0.1130% | 224857.566667 | 224913.466667 | +0.0249% |
| repeat_count | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| runtime_setup_ticks | 629.600000 | 619.266667 | -1.6413% | 622.700000 | 623.733333 | +0.1659% |
| runtime_teardown_ticks | 584.333333 | 576.933333 | -1.2664% | 576.900000 | 576.866667 | -0.0058% |
| scan_attention_overlay_capacity_bytes | 3473408.000000 | 3473408.000000 | +0.0000% | 3473408.000000 | 3473408.000000 | +0.0000% |
| scan_attention_overlay_required_bytes | 73728.000000 | 73728.000000 | +0.0000% | 73728.000000 | 73728.000000 | +0.0000% |
| scan_cache_append_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| scan_cache_append_ticks | 4284.066667 | 4286.066667 | +0.0467% | 4283.266667 | 4284.433333 | +0.0272% |
| scan_cache_ddr_read_bytes | 8257536.000000 | 8257536.000000 | +0.0000% | 8257536.000000 | 8257536.000000 | +0.0000% |
| scan_cache_ddr_write_bytes | 114688.000000 | 114688.000000 | +0.0000% | 114688.000000 | 114688.000000 | +0.0000% |
| scan_cache_dma_descriptor_count | 1344.000000 | 1344.000000 | +0.0000% | 1344.000000 | 1344.000000 | +0.0000% |
| scan_cache_pack_ticks | 21326.466667 | 21329.533333 | +0.0144% | 21325.400000 | 21332.166667 | +0.0317% |
| scan_cache_stage_ticks | 13780.600000 | 13792.600000 | +0.0871% | 13790.100000 | 13801.300000 | +0.0812% |
| scan_dynamic_attention_ticks | 643969.466667 | 644079.133333 | +0.0170% | 643868.533333 | 644098.066667 | +0.0356% |
| scan_logical_m_observed | 1.000000 | 1.000000 | +0.0000% | 1.000000 | 1.000000 | +0.0000% |
| scan_padded_kv_length | 96.000000 | 96.000000 | +0.0000% | 96.000000 | 96.000000 | +0.0000% |
| scan_total_kv_length | 72.000000 | 72.000000 | +0.0000% | 72.000000 | 72.000000 | +0.0000% |
| stage_boundary_ticks | 41.400000 | 38.666667 | -6.6023% | 39.466667 | 39.233333 | -0.5912% |
| total_ticks | 1740588.333333 | 1743869.933333 | +0.1885% | 1740400.000000 | 1743889.600000 | +0.2005% |
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
| w4f16_cross_prefetch_lifetime_ticks | 833316.466667 | 833150.000000 | -0.0200% | 833167.400000 | 833240.000000 | +0.0087% |
| w4f16_cross_prefetch_wait_ticks | 972.733333 | 1029.333333 | +5.8187% | 1073.933333 | 1083.566667 | +0.8970% |
| w4f16_expand_mismatch_count | 0.000000 | 0.000000 | N/A: zero control | 0.000000 | 0.000000 | N/A: zero control |
| w4f16_expand_pool_wait_ticks | 75684.333333 | 76937.733333 | +1.6561% | 75997.633333 | 76898.700000 | +1.1857% |
| w4f16_expand_ticks | 614641.000000 | 617671.666667 | +0.4931% | 614479.866667 | 617279.533333 | +0.4556% |
| w4f16_expand_work_ticks | 1692794.733333 | 1694383.866667 | +0.0939% | 1694022.433333 | 1695019.866667 | +0.0589% |
| w4f16_gate_up_expand_pool_wait_ticks | 31111.933333 | 31269.066667 | +0.5051% | 30962.433333 | 31349.633333 | +1.2505% |
| w4f16_gate_up_expand_ticks | 252215.200000 | 252730.266667 | +0.2042% | 252361.200000 | 253435.633333 | +0.4258% |
| w4f16_gate_up_expand_work_ticks | 157864.733333 | 157571.733333 | -0.1856% | 158077.466667 | 158016.166667 | -0.0388% |
| w4f16_gate_up_hmx_tail_wait_ticks | 7499.600000 | 7514.400000 | +0.1973% | 7439.766667 | 7254.300000 | -2.4929% |
| w4f16_gate_up_hmx_wait_ticks | 277820.933333 | 278262.400000 | +0.1589% | 277983.633333 | 278997.033333 | +0.3646% |
| w4f16_gate_up_stream_join_wait_ticks | 699.066667 | 674.200000 | -3.5571% | 682.833333 | 621.533333 | -8.9773% |
| w4f16_gate_up_stream_ready_wait_ticks | 1319.800000 | 1240.533333 | -6.0060% | 1342.200000 | 1345.800000 | +0.2682% |
| w4f16_gate_up_stream_work_ticks | 134363.133333 | 134373.333333 | +0.0076% | 134387.966667 | 134382.066667 | -0.0044% |
| w4f16_gate_up_weight_dma_ticks | 557251.066667 | 558374.666667 | +0.2016% | 557614.166667 | 559970.066667 | +0.4225% |
| w4f16_hmx_tail_wait_ticks | 19685.600000 | 19554.600000 | -0.6655% | 19332.266667 | 18948.900000 | -1.9830% |
| w4f16_prefetch_wait_ticks | 29214.000000 | 28342.333333 | -2.9837% | 29668.033333 | 29862.633333 | +0.6559% |
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
| weight_dma_ticks | 1133403.200000 | 1137199.266667 | +0.3349% | 1133383.966667 | 1137716.266667 | +0.3822% |


## Direct E2E

```json
{
  "times": {
    "original": {
      "prefill_tokens": 64,
      "prefill_host_us": 62982.682,
      "prefill_tokens_per_second": 1016.1523448620368,
      "decode_tokens": 15,
      "decode_total_host_us": 1389468.3575000002,
      "decode_tokens_per_second": 10.79549593125585
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62850.834,
      "prefill_tokens_per_second": 1018.2840214976304,
      "decode_tokens": 15,
      "decode_total_host_us": 1391317.4985,
      "decode_tokens_per_second": 10.78114809608283
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 62818.255000000005,
      "prefill_tokens_per_second": 1018.8121271436145,
      "decode_tokens": 15,
      "decode_total_host_us": 1387194.112,
      "decode_tokens_per_second": 10.81319468576291
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63085.10400000001,
      "prefill_tokens_per_second": 1014.5025678328119,
      "decode_tokens": 15,
      "decode_total_host_us": 1391000.937,
      "decode_tokens_per_second": 10.78360165044231
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": 0.13881673030360364,
      "decode": -0.14955646429521963
    },
    "B": {
      "prefill": 0.20515272293613052,
      "decode": 0.04212258862772433
    },
    "C": {
      "prefill": -0.18861969605361573,
      "decode": -0.18218790759139125
    }
  }
}
```


## Experiment context and checks

# EXP-0221: per-channel GPTQ with original, folded and rotated weights

This completed experiment tests A=original+GPTQ, B=fresh gamma fold+GPTQ and C=fresh gamma fold+fixed Sylvester H2048 R1/H128 R2+GPTQ. All changed weights originate in verified Qwen3-origin shards. No upstream folded weights, LPBQ/group scales or QNN assets are used. The other two recipes remain frozen. Historical RTN A/B/C quality comes from EXP-0219; these controls were not re-evaluated.

## Outcome

Unrotated GPTQ A is the best candidate in this fixed trial: actual DSP NLL improves from4.231668 to3.873038 while strict short tasks remain8/24. B reaches4.601481 and0/24; C reaches4.3210 and2/24. GPTQ substantially reduces the damage of folded/rotated RTN, but fixed R1/R2 provides no incremental benefit over GPTQ A. Both predeclared conjunction gates fail; evidence remains valid and no baseline is promoted. This does not establish that all rotations or GPTQ configurations fail. Four-way inference speeds are essentially unchanged; no speed improvement is claimed.

## Fixed calibration and quantizer

64 sequences of128 tokens,32 English WikiText-2-raw train rows and32 Chinese Wikipedia20231101.zh train rows. First128 tokens from first32 eligible distinct rows per language, skipping truncated/short rows and exact32-token overlaps with evaluation samples. No padding, chat-template addition, parameter search or evaluation-guided selection. Raw HTTP snapshots, dataset metadata revisions, exact IDs and tokenizer hashes are retained. Eight holdout windows were only included in the duplicate-exclusion check and never scored. This8192-token calibration is a bounded first trial, not evidence of calibration sufficiency across domains or long contexts.

GPTQ method reference: IST-DASLab/gptq at2d65066eeb06a5c9ff5184d8cebdf33662c67faf, Apache-2.0, archived with its license. Adaptation uses CPU, FP64 damped-Hessian factorization and FP32 error compensation. Fixed transformed-row absmax/7 scale, nearest-even signed[-7,7], groupsize=-1, act-order enabled/inverted on export, damping1% of mean Hessian diagonal. Computational blocksize128 does not add quantization groups. No clipping search. Row chunks share the same input factor and preserve per-row independence.

True-sequential FP16 calibration processes QKV, O, Gate/Up, Down in order; subsequent projections/layers observe previously quantized outputs. The final head observes the quantized stack. All transforms precede calibration, so input statistics use the correct folded/rotated coordinates. Original FP32 transformed weights determine scales before quantization; deployed activations and reconstructed weights use FP16. Tied HF embedding/head are explicitly detached. All197 transformer/head projections are GPTQ quantized, embedding stays FP16, Q/K head norm and RoPE semantics stay intact.

## Actual DSP quality

Frozen qbh-lite-v1:512 bilingual conditional targets,24 strict short tasks and4open prefixes. Quick/full/repeat suites are deterministic across all overlapping token/code/NLL/rank/tie/saturation fields. This is a lightweight diagnostic, not broad model-quality certification. Software results are independently computed from exported integer codes/scales and are not assumed bit-exact DSP logits.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| RTN 原始（冻结） | 4.2317 | 68.83 | 8/24 | 55.08% |
| RTN gamma 折叠（冻结） | 5.6017 | 270.88 | 0/24 | 32.62% |
| RTN R1/R2（冻结） | 8.5625 | 5231.70 | 0/24 | 28.52% |
| GPTQ 原始 A | 3.8730 | 48.09 | 8/24 | 62.50% |
| GPTQ gamma 折叠 B | 4.6015 | 99.63 | 0/24 | 46.09% |
| GPTQ R1/R2 C | 4.3210 | 75.26 | 2/24 | 57.81% |

Predeclared gates: A improves original RTN NLL AND short-task count = False; C improves A on BOTH metrics = False. No automatic baseline promotion.

## Calibration-local diagnostics

| Variant | Projections | Mean RTN NRMSE | Mean GPTQ NRMSE | GPTQ lower count |
|---|---|---|---|---|
| A | 197 | 0.156287 | 0.070592 | 197 |
| B | 197 | 0.187472 | 0.097075 | 197 |
| C | 197 | 0.154998 | 0.064659 | 197 |

These errors use every16th calibration activation,512 positions, and compare linear output against unquantized transformed FP32 weights on identical inputs. Means are unweighted across projections, not whole-model errors; they are neither evaluation scores nor used to choose grid/settings. Input-statistic hashes in weight_stats are hashes of the Gram diagonal, not the full Gram.

All dense inverse-Hessian oracle, packed-code roundtrip, FP32 transform invariance and staged-versus-unchanged-HF calibration-forward checks pass. Full details:

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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000467,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000442,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003662,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000003506,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000305,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000002809,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000002343,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000002158,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001652,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000001776,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001432,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000146,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001215,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000001221,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000915,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000968,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000566,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000684,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000469,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000389,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000369,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000315,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000302,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000222,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000218,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000153,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000158,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000075,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000084,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000006,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000033,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000022,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 3.8741880133748055,
      "ppl": 48.14359046418558,
      "tasks_correct": 9,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.630859375,
      "language_nll": {
        "zh": 4.30016877502203,
        "en": 3.448207251727581
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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000004858,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000921,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003788,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000635,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003135,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000484,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000002336,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000346,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001568,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000255,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.00000000000014,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000226,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001215,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000155,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000968,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000124,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000068,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000084,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000475,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000067,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000433,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000058,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000335,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000275,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000003,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000178,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000098,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000001,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000056,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000003,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000016,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000016,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 4.600959472358227,
      "ppl": 99.57981389951235,
      "tasks_correct": 0,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.46875,
      "language_nll": {
        "zh": 5.093510314822197,
        "en": 4.108408629894257
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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000063,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000682,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.00000000000005,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000039,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000402,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000318,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000333,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000253,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000025,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000209,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000209,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000017,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000187,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000012,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000012,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000073,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000056,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000049,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000033,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000024,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000018,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 4.323169119656086,
      "ppl": 75.42728802273471,
      "tasks_correct": 2,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.572265625,
      "language_nll": {
        "zh": 4.538219437003136,
        "en": 4.108118802309036
      }
    }
  }
}
```

## Complete profiling and provenance

Original RTN plus A/B/C each pass one warmup,5short,10formal sessions with rotating order. All640formal invocation ledgers and17920layer ledgers close exactly. Each16-token speed sequence matches its independent software reference. Fixed M64+15 feedback steps continue after EOS if present; timing then describes diagnostic execution, not usable text throughput. Host wall covers the complete model token-in/token-out pass, excluding cold staging and separately logged WSL tokenizer/detokenizer work. Low-NLL GPTQ candidate is chosen solely for the overview column under the predeclared rule; this does not promote a baseline. Other columns are frozen EXP0218 nonpaired references and do not support activation-only attribution with changed W4 weights.

Runtime is unchanged EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108, embeddedlabel218 intentional; outer experiment221. Full28-layer model, final norm/head/greedy, persistent KV;8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC and one HMX owner, no QNN. Offline calibration/quantization costs are not DSP inference timings.

Evidence D:/llm_exp/results/qwen3-block-htp/exp0221; models D:/llm_exp/models/qwen3-block-htp/exp0221. Source archives, package manifests, frozen runtime binaries, dependency hashes and final evidence ledger are identified in closure.json/evidence_sha256.json. Inherited layer-replay references remain historical placeholders and cannot validate new layer replay. Every successful deployment checks1276files and frozen binary hashes. Tokenizer metadata and license-filename recovery preserved downloaded data and exact calibration IDs; see calibration_recovery.json/reference_recovery.json. Downloads used the existing Windows Clash proxy without global proxy changes.

Reproduction requires a newly registered experiment and fresh output paths. gptq_exp0221.py freezes calibration and checks the independent oracle; experiment_exp0221.py generates each candidate; measure_exp0221.py deploys/runs suites; summarize_exp0221.py reports. Do not overwrite retained outputs or tune against this evaluation/holdout.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 GPTQ A EXP-0221 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 384.9 (0.61%) | 247.4 (0.63%) | +55.57% |
| Input RMSNorm | 489.7 (0.61%) | 491.6 (0.78%) | 554.0 (1.40%) | -11.26% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11726.5 (18.66%) | 7052.7 (17.82%) | +66.27% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3953.7 (6.29%) | 3214.5 (8.12%) | +22.99% |
| O projection | 5757.7 (7.14%) | 5024.6 (7.99%) | 1256.4 (3.17%) | +299.92% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.5 (0.75%) | 654.0 (1.65%) | -27.60% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22410.4 (35.66%) | 14442.7 (36.49%) | +55.17% |
| Down | 13447.9 (16.67%) | 8570.6 (13.64%) | 3428.5 (8.66%) | +149.98% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.72% |
| KV carrier conversion | 174.0 (0.22%) | 172.6 (0.27%) | 203.2 (0.51%) | -15.08% |
| KV append DMA | 343.6 (0.43%) | 340.2 (0.54%) | 463.9 (1.17%) | -26.65% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.13% |
| Layer bookkeeping | 23.9 (0.03%) | 23.4 (0.04%) | 23.2 (0.06%) | +1.12% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.3 (0.01%) | 22.5 (0.06%) | -62.96% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.7 (0.15%) | 105.6 (0.27%) | -11.27% |
| Embedding | 68.1 (0.08%) | 65.9 (0.10%) | 62.4 (0.16%) | +5.50% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.01% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6695.7 (10.65%) | 5284.7 (13.35%) | +26.70% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2281.9 (3.63%) | 2466.6 (6.23%) | -7.49% |
| 完整 Host wall | 80692.2 (100.00%) | 62850.8 (100.00%) | 39575.9 (100.00%) | +58.81% |


## Direct E2E

```json
{
  "times": {
    "original": {
      "prefill_tokens": 64,
      "prefill_host_us": 62982.682,
      "prefill_tokens_per_second": 1016.1523448620368,
      "decode_tokens": 15,
      "decode_total_host_us": 1389468.3575000002,
      "decode_tokens_per_second": 10.79549593125585
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62850.834,
      "prefill_tokens_per_second": 1018.2840214976304,
      "decode_tokens": 15,
      "decode_total_host_us": 1391317.4985,
      "decode_tokens_per_second": 10.78114809608283
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 62818.255000000005,
      "prefill_tokens_per_second": 1018.8121271436145,
      "decode_tokens": 15,
      "decode_total_host_us": 1387194.112,
      "decode_tokens_per_second": 10.81319468576291
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63085.10400000001,
      "prefill_tokens_per_second": 1014.5025678328119,
      "decode_tokens": 15,
      "decode_total_host_us": 1391000.937,
      "decode_tokens_per_second": 10.78360165044231
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": 0.13881673030360364,
      "decode": -0.14955646429521963
    },
    "B": {
      "prefill": 0.20515272293613052,
      "decode": 0.04212258862772433
    },
    "C": {
      "prefill": -0.18861969605361573,
      "decode": -0.18218790759139125
    }
  }
}
```


## Source identity

```json
{
  "source_branch": "codex/exp-0221-w4f16-gptq-offline-rotation",
  "reporting_input_source_head": "bd4944cf2401f9ca732a79bca982d2088aa0d8b2",
  "implementation_files": {
    "gptq_exp0221.py": "1bd1df1bcee446d62035c11f6ea571105ee3909c46b921613f8e7067e8a76179",
    "experiment_exp0221.py": "1a677db15826ce409b49d7ceb272758b52f4cd2c10886b5a6c8b55b5b0d1ff92",
    "measure_exp0221.py": "f11dad4a1836e05cf26f8a7defd099a1401f37a132907115bc8db693a974158a",
    "summarize_exp0221.py": "8a9a021e0261bb140690843f56bdff909e3d716eed5c678170a516be983bca6f"
  },
  "calibration_sha256": "e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669",
  "calibration_ids_sha256": "a20ed2093b0e7fb075fe47a9532843402e9f69784c8ad0b0467194bee2fa8efd",
  "final_archive_source_head": "Recorded after report commit in closure.json"
}
```
