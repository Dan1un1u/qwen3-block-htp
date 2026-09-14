# EXP0272: exact FP32 Norm and full-model gate

Source `5f859339a62461b3c6fb7dc99c9dc243c1385c45`; Qwen3-0.6B28layers, nativeW4/SP2mode8, no rotation, original frozen C64/SP2 packages. Full M64+15, cache128 and offlineEOSprefix. Mode0 integer residual,mode1 priorFP32,mode2 interleaved16-row orderedNorm. Candidate stores and arithmetic are unchanged except independent-row instruction scheduling. Llama and other recipes frozen.

Result: phase=formal; full formal gate=True; short gate=True. Primary repeat10, auxiliary repeat1. Fixed5short then10formal only if short passes; pairedbootstrap20000 seed27295CI upper<=1.10 vsinteger for BOTH prefill and decode. No optional resampling or automatic baseline promotion.

| Repeat / phase | Integer Host us / tps | Prior FP32 Host us / tps | Optimized FP32 Host us / tps |
|---|---:|---:|---:|
| r1_prefill | 33298.844 / 1921.989 | 36448.542 / 1755.900 | 36250.448 / 1765.495 |
| r1_decode | 22093.652 / 45.262 | 22472.796 / 44.498 | 22465.197 / 44.513 |
| r10_prefill | 31749.181 / 2015.800 | 34859.908 / 1835.920 | 34605.520 / 1849.416 |
| r10_decode | 20961.335 / 47.707 | 21241.080 / 47.079 | 21244.903 / 47.070 |

| Ratio | Point | 95% CI | Gate |
|---|---:|---|---|
| r1_prefill_ns_vs0 | 1.088639844 | [1.0837761309198846, 1.0932037110728643] | True |
| r1_prefill_ns_vs1 | 0.994565110 | [0.9909787336983458, 0.9982195047516683] | None |
| r1_decode_ns_vs0 | 1.016816798 | [1.0078959953713196, 1.0260959340243392] | True |
| r1_decode_ns_vs1 | 0.999661860 | [0.992817920400473, 1.006776712785116] | None |
| r10_prefill_ns_vs0 | 1.089965757 | [1.0872579579293324, 1.0925781447612364] | True |
| r10_prefill_ns_vs1 | 0.992702575 | [0.9894403466878361, 0.9959956238432883] | None |
| r10_decode_ns_vs0 | 1.013528146 | [1.0101565942381268, 1.0162691078383812] | True |
| r10_decode_ns_vs1 | 1.000179972 | [0.9963177214432706, 1.0031101817325319] | None |

## Attribution and bounded candidates

EXP0270 singlelayer inputNorm20.25->75.24us; EXP0271 full inputNorm562.59->2119.73us (~20.09->75.70us/layer). Norm overhead accumulates almost linearly; lower fullmodel relative regression comes from amortized RPC/boundary work and unchanged head/other work in the denominator, not lossless cancellation of Norm. EXP0270 Host-DSP delta+53.47us per singlelayer; EXP0271 entirefull delta-68.16us. Separate sessions and scopes, not causal subtraction or an E2E extrapolation.

C1 holds two independent8-row FP32 sums and interleaves their exact ascending-channel updates inside one16-row reduction. No reassociation, reciprocal change, extra DDR, weight/quantizer modification or approximate reduction. Singlelayer Norm75.2->71.6us in diagnosis, while complete singlelayer wall is noisy. C2 packed four repair-flag fragments into aligned128-byte writes; it is exact but slower (86-88us Norm after specialization), and was fully removed. Generic C2 branches initially also slowed mode1; dispatch isolation restored it. One compile failed because clang fp pragma followed a local constant; fixed before hardware, failed source retained. No hardware numerical failure. C2 module cost, not a raw best wall sample, drove rejection.

Final selectedlayers0/14/27 independently exact399360 FP32 values plus Q/K,AV,postnorm,GateUp,SP2 and physicalKV; consecutive3 exact133120values; repeats10 deterministic. Full28 all layer ledgers/cache lengths pass,88 captured residual/finalNorm/KV files byte-exact vs sealedEXP0271. Independent finalNorm32768codes and fullvocabhead16steps exact. Integer control matches sealedEXP0268; optimized and priorFP32 token/logit codes identical. 122CLI/8832profiles;8MiB grant,peak8365824B; zero timed boundary/intermediate tensorDDR/spill/unattributed. New reduce16 has no vector stack; retained genericNorm stack holds uniform constants and inverse-RMS splats only. C2 assemblies remain archived, no C2 code in selected build.

Timing denominator is complete Host invocation wall including embedding/allblocks/finalNorm/head/greedy and Host-DSP boundary, with loaded weights. Prefill64tokens; decode15tokens persequence. Excludes cold load/ADB/CPU tokenizer and diagnostic exports. No full28 independent CPU transformer oracle, PPL or quality acceptance.

## Stable module overview

Microseconds (share of complete Host wall),repeat10. F16F16/W4F16 not rerun and thereforeN/A; all three columns here are same-recipe paired controls. Exclusive module ledger sums to Host wall.

### prefill

| Module | Integer SP2 | Prior FP32 SP2 | Optimized FP32 SP2 |
|---|---:|---:|---:|
| I/O and metadata | 247.39 (0.78%) | 254.35 (0.73%) | 255.96 (0.74%) |
| Input RMSNorm | 558.50 (1.76%) | 2121.64 (6.09%) | 2019.04 (5.83%) |
| QKV + Q/K Norm-RoPE | 7016.67 (22.10%) | 7028.59 (20.16%) | 7030.70 (20.32%) |
| QK-Softmax-AV | 3480.04 (10.96%) | 3487.85 (10.01%) | 3482.41 (10.06%) |
| O projection | 1249.92 (3.94%) | 2096.54 (6.01%) | 2095.33 (6.05%) |
| Post-attention residual + RMSNorm | 661.27 (2.08%) | 2219.24 (6.37%) | 2116.19 (6.12%) |
| Gate/Up + SwiGLU | 7031.92 (22.15%) | 7028.78 (20.16%) | 7026.08 (20.30%) |
| Down projection | 4408.83 (13.89%) | 3778.16 (10.84%) | 3774.57 (10.91%) |
| Final residual | 186.43 (0.59%) | 2.88 (0.01%) | 3.04 (0.01%) |
| KV-cache carrier conversion | 135.53 (0.43%) | 136.37 (0.39%) | 136.33 (0.39%) |
| KV-cache append DMA | 287.11 (0.90%) | 294.87 (0.85%) | 294.32 (0.85%) |
| Block internal orchestration | 35.80 (0.11%) | 35.38 (0.10%) | 35.59 (0.10%) |
| Layer bookkeeping | 28.33 (0.09%) | 27.54 (0.08%) | 27.59 (0.08%) |
| Stage-boundary bookkeeping | 20.21 (0.06%) | 20.28 (0.06%) | 20.42 (0.06%) |
| DSP unattributed residual | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| DSP runtime setup/teardown | 116.67 (0.37%) | 115.94 (0.33%) | 116.78 (0.34%) |
| Token embedding | 47.85 (0.15%) | 61.76 (0.18%) | 61.77 (0.18%) |
| Final model RMSNorm | 3.61 (0.01%) | 15.08 (0.04%) | 14.97 (0.04%) |
| LM head + greedy selection (excluding final norm) | 5259.85 (16.57%) | 5198.35 (14.91%) | 5189.27 (15.00%) |
| True Host-DSP boundary | 973.27 (3.07%) | 936.30 (2.69%) | 905.18 (2.62%) |
| Complete Host wall | 31749.18 (100.00%) | 34859.91 (100.00%) | 34605.52 (100.00%) |

### decode

| Module | Integer SP2 | Prior FP32 SP2 | Optimized FP32 SP2 |
|---|---:|---:|---:|
| I/O and metadata | 249.70 (1.19%) | 250.31 (1.18%) | 251.37 (1.18%) |
| Input RMSNorm | 150.17 (0.72%) | 368.49 (1.73%) | 368.52 (1.73%) |
| QKV + Q/K Norm-RoPE | 2504.61 (11.95%) | 2503.13 (11.78%) | 2504.42 (11.79%) |
| QK-Softmax-AV | 1913.69 (9.13%) | 1926.71 (9.07%) | 1924.74 (9.06%) |
| O projection | 1246.18 (5.95%) | 1342.60 (6.32%) | 1339.61 (6.31%) |
| Post-attention residual + RMSNorm | 198.88 (0.95%) | 369.51 (1.74%) | 369.54 (1.74%) |
| Gate/Up + SwiGLU | 6534.48 (31.17%) | 6404.13 (30.15%) | 6395.60 (30.10%) |
| Down projection | 3526.05 (16.82%) | 3487.95 (16.42%) | 3483.64 (16.40%) |
| Final residual | 42.63 (0.20%) | 2.00 (0.01%) | 1.99 (0.01%) |
| KV-cache carrier conversion | 290.88 (1.39%) | 290.84 (1.37%) | 290.94 (1.37%) |
| KV-cache append DMA | 120.67 (0.58%) | 125.25 (0.59%) | 126.10 (0.59%) |
| Block internal orchestration | 29.32 (0.14%) | 28.61 (0.13%) | 28.63 (0.13%) |
| Layer bookkeeping | 16.50 (0.08%) | 16.33 (0.08%) | 16.30 (0.08%) |
| Stage-boundary bookkeeping | 1.75 (0.01%) | 1.73 (0.01%) | 1.73 (0.01%) |
| DSP unattributed residual | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| DSP runtime setup/teardown | 73.35 (0.35%) | 73.37 (0.35%) | 73.46 (0.35%) |
| Token embedding | 2.05 (0.01%) | 2.05 (0.01%) | 2.07 (0.01%) |
| Final model RMSNorm | 3.06 (0.01%) | 14.83 (0.07%) | 14.74 (0.07%) |
| LM head + greedy selection (excluding final norm) | 3218.65 (15.36%) | 3219.78 (15.16%) | 3216.83 (15.14%) |
| True Host-DSP boundary | 838.72 (4.00%) | 813.45 (3.83%) | 834.67 (3.93%) |
| Complete Host wall | 20961.33 (100.00%) | 21241.08 (100.00%) | 21244.90 (100.00%) |

## Complete counters

Per-run mean medians; tick counters converted to microseconds, other units unchanged. Overlapping work/wait/readiness counters must not be added to exclusive ledger. Means also retained in PROFILE.json.

### repeat1 prefill

| Field | Integer | Prior FP32 | Optimized FP32 |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | 0 |
| attention_av_hmx_ticks (us) | 0 | 0 | 0 |
| attention_av_pack_ticks (us) | 0 | 0 | 0 |
| attention_av_unpack_ticks (us) | 0 | 0 | 0 |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | 0 |
| attention_qk_hmx_ticks (us) | 0 | 0 | 0 |
| attention_qk_pack_ticks (us) | 0 | 0 | 0 |
| attention_qk_unpack_ticks (us) | 0 | 0 | 0 |
| attention_setup_ticks (us) | 0 | 0 | 0 |
| attention_softmax_ticks (us) | 0 | 0 | 0 |
| attention_ticks (us) | 3482.266 | 3484.479 | 3475.964 |
| attention_unattributed_ticks (us) | 0 | 0 | 0 |
| block_invocation_count | 28 | 28 | 28 |
| block_orchestration_ticks (us) | 35.26042 | 34.47917 | 35 |
| boundary_ddr_read_bytes | 4976000 | 5107072 | 5107072 |
| boundary_ddr_write_bytes | 0 | 0 | 0 |
| boundary_dma_descriptor_count | 289 | 289 | 289 |
| cache_compared_elements | 0 | 0 | 0 |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | 0 |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.01 |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.99999 |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | 0 |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | 0 |
| cache_max_nrmse | 0 | 0 | 0 |
| cache_min_cosine | 1 | 1 | 1 |
| cache_mismatches | 0 | 0 | 0 |
| cache_mixed_tolerance_violations | 0 | 0 | 0 |
| cache_nonfinite_count | 0 | 0 | 0 |
| cache_prefix_mismatches | 0 | 0 | 0 |
| cache_structure_mismatches | 0 | 0 | 0 |
| cache_tensor_count | 0 | 0 | 0 |
| dense_r3_constant_read_bytes | 0 | 0 | 0 |
| dense_r3_mode | 0 | 0 | 0 |
| dense_r3_optimization | 0 | 0 | 0 |
| dense_r3_total_calls | 0 | 0 | 0 |
| dense_r3_total_finish_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_hmx_calls | 0 | 0 | 0 |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_parallel_heads | 0 | 0 | 0 |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_refined_values | 0 | 0 | 0 |
| dense_r3_total_rows | 0 | 0 | 0 |
| dense_r4_audit_bytes | 0 | 0 | 0 |
| dense_r4_calls | 0 | 0 | 0 |
| dense_r4_finish_ticks (us) | 0 | 0 | 0 |
| dense_r4_hmx_calls | 0 | 0 | 0 |
| dense_r4_layout_ticks (us) | 0 | 0 | 0 |
| dense_r4_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r4_mode | 0 | 0 | 0 |
| dense_r4_optimization | 0 | 0 | 0 |
| dense_r4_parallel_dispatches | 0 | 0 | 0 |
| dense_r4_parallel_finish_groups | 0 | 0 | 0 |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_parallel_prepare_tiles | 0 | 0 | 0 |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r4_pipeline_batches | 0 | 0 | 0 |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_consume_count | 0 | 0 | 0 |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_publish_count | 0 | 0 | 0 |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | 0 |
| dense_r4_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r4_rows | 0 | 0 | 0 |
| down_ticks (us) | 4401.797 | 3767.917 | 3796.927 |
| dsp_status | 3 | 3 | 3 |
| experiment | 218 | 218 | 218 |
| f16_cache_full_prefix_pack_count | 0 | 0 | 0 |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | 0 |
| f16_cache_native_incremental_append_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| final_residual_ticks (us) | 185.625 | 2.916667 | 2.760417 |
| first_position | 0 | 0 | 0 |
| fp32_residual | 0 | 1 | 2 |
| gate_up_ticks (us) | 7012.917 | 7054.193 | 7056.771 |
| generation_embedding_ddr_read_bytes | 131328 | 262400 | 262400 |
| generation_embedding_ticks (us) | 49.45312 | 64.0625 | 63.46354 |
| generation_final_norm_ticks (us) | 3.802083 | 15.65104 | 15.88542 |
| generation_lm_head_argmax_ticks (us) | 604.1146 | 605.3906 | 605.599 |
| generation_lm_head_batch_n_tiles | 8 | 8 | 8 |
| generation_lm_head_command_count | 594 | 594 | 594 |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | 1.56798e+08 |
| generation_lm_head_direct_slot_join_count | 0 | 0 | 0 |
| generation_lm_head_exclusive_ticks (us) | 5266.797 | 5202.214 | 5197.37 |
| generation_lm_head_expand_ticks (us) | 3695.755 | 3650.833 | 3649.167 |
| generation_lm_head_hmx_tail_wait_ticks (us) | 248.4375 | 214.5312 | 212.9427 |
| generation_lm_head_hmx_ticks (us) | 4617.109 | 4548.464 | 4546.224 |
| generation_lm_head_n_tiles | 4748 | 4748 | 4748 |
| generation_lm_head_prefetch_count | 593 | 593 | 593 |
| generation_lm_head_scale_dma_ticks (us) | 21.90104 | 22.68229 | 22.31771 |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 1215488 |
| generation_lm_head_ticks (us) | 5270.911 | 5217.63 | 5213.568 |
| generation_lm_head_weight_dma_ticks (us) | 5228.724 | 5163.307 | 5158.88 |
| generation_lm_head_weight_dma_wait_ticks (us) | 301.901 | 292.1615 | 294.5312 |
| generation_step | 0 | 0 | 0 |
| hmx_command_count | 1882 | 1882 | 1882 |
| hmx_compute_ticks (us) | 12263.75 | 12947.63 | 12982.6 |
| hmx_fp16_tile_pair_count | 0 | 0 | 0 |
| hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | 2031360 |
| host_boundary_us | 2471.901 | 2589.427 | 2583.959 |
| host_us | 33241.3 | 36532.73 | 36306.43 |
| host_wall_ns | 3.32413e+07 | 3.653273e+07 | 3.630643e+07 |
| input_norm_ticks (us) | 563.4115 | 2119.583 | 2017.24 |
| input_stage_ticks (us) | 0.3385417 | 0.46875 | 0.4427083 |
| intermediate_ddr_read_bytes | 0 | 0 | 0 |
| intermediate_ddr_write_bytes | 0 | 0 | 0 |
| intermediate_dma_descriptor_count | 0 | 0 | 0 |
| intermediate_spill_fill_count | 0 | 0 | 0 |
| invocation_ticks (us) | 30811.35 | 33950.83 | 33761.46 |
| kv_cache_k_format | 14 | 14 | 14 |
| kv_cache_v_format | 12 | 12 | 12 |
| layer_bookkeeping_ticks (us) | 25.9375 | 25.49479 | 25.46875 |
| ledger_named_ticks (us) | 30811.35 | 33950.83 | 33761.46 |
| ledger_unattributed_ticks (us) | 0 | 0 | 0 |
| logical_m | 64 | 64 | 64 |
| metadata_stage_ticks (us) | 243.0729 | 255.2344 | 251.4583 |
| numerical_audit_enabled | 0 | 0 | 0 |
| numerical_status | 1 | 1 | 1 |
| o_projection_ticks (us) | 1243.854 | 2098.281 | 2073.385 |
| output_cosine | 0 | 0 | 0 |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0625 |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.003 |
| output_fp16_rtol | 0.002 | 0.002 | 0.002 |
| output_max_abs | 0 | 0 | 0 |
| output_max_lsb | 0 | 0 | 0 |
| output_max_required_rtol_after_atol | 0 | 0 | 0 |
| output_mismatches | 1 | 1 | 1 |
| output_mixed_tolerance_violations | 0 | 0 | 0 |
| output_nonfinite_count | 0 | 0 | 0 |
| output_nrmse | 0 | 0 | 0 |
| output_stage_ticks (us) | 0 | 0 | 0 |
| post_attention_norm_ticks (us) | 1.119792 | 1.041667 | 1.041667 |
| post_attention_residual_ticks (us) | 659.8698 | 2218.594 | 2110.703 |
| prefix_group_patch_count | 224 | 224 | 224 |
| prefix_kv_mode | 1 | 1 | 1 |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | 57344 |
| prepared_session_run_index | 1 | 1 | 1 |
| projection_failure_index | 0 | 0 | 0 |
| projection_failure_n_tile | 0 | 0 | 0 |
| projection_failure_result | 0 | 0 | 0 |
| projection_failure_step | 0 | 0 | 0 |
| projection_hmx_wait_ticks (us) | 1530.911 | 1747.656 | 1745.26 |
| projection_pack_ticks (us) | 4.505208 | 3.880208 | 3.828125 |
| projection_unpack_ticks (us) | 0 | 0 | 0 |
| qk_norm_rope_ticks (us) | 0.46875 | 0.546875 | 0.546875 |
| qkv_projection_ticks (us) | 7038.281 | 7038.359 | 7031.172 |
| repeat_count | 1 | 1 | 1 |
| runtime_setup_ticks (us) | 55.02604 | 55.39062 | 54.86979 |
| runtime_teardown_ticks (us) | 52.57812 | 52.34375 | 51.79688 |
| scan_attention_overlay_capacity_bytes | 0 | 0 | 0 |
| scan_attention_overlay_required_bytes | 0 | 0 | 0 |
| scan_cache_append_mismatch_count | 0 | 0 | 0 |
| scan_cache_append_ticks (us) | 293.8542 | 300.4688 | 298.4375 |
| scan_cache_ddr_read_bytes | 0 | 0 | 0 |
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 5906432 |
| scan_cache_dma_descriptor_count | 448 | 448 | 448 |
| scan_cache_pack_ticks (us) | 133.0208 | 137.6562 | 132.4219 |
| scan_cache_stage_ticks (us) | 0 | 0 | 0 |
| scan_dynamic_attention_ticks (us) | 0 | 0 | 0 |
| scan_logical_m_observed | 64 | 64 | 64 |
| scan_padded_kv_length | 64 | 64 | 64 |
| scan_total_kv_length | 64 | 64 | 64 |
| stage_boundary_ticks (us) | 20.67708 | 21.22396 | 21.45833 |
| total_ticks (us) | 30755.91 | 33895.08 | 33705.65 |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | 0 |
| u8_attention_av_hmx_ticks (us) | 624.5573 | 627.9948 | 620.1302 |
| u8_attention_av_requant_ticks (us) | 1642.656 | 1646.016 | 1642.344 |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | 0 |
| u8_attention_k_pack_ticks (us) | 0 | 0 | 0 |
| u8_attention_pipeline_wait_ticks (us) | 2017.448 | 2008.672 | 2006.615 |
| u8_attention_probability_mask_violation_count | 0 | 0 | 0 |
| u8_attention_qk_hmx_ticks (us) | 574.1927 | 580.651 | 572.6562 |
| u8_attention_qk_norm_rope_ticks (us) | 24449.64 | 24493.33 | 24472.55 |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | 0 |
| u8_attention_softmax_ticks (us) | 8688.047 | 8692.292 | 8684.297 |
| u8_attention_v_pack_ticks (us) | 5036.901 | 5059.688 | 5055.807 |
| u8_cache_full_prefix_pack_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 827904 |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 1 |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | 0 |
| u8_cache_native_append_update_ticks (us) | 425.9635 | 434.7396 | 428.5417 |
| u8_cache_native_incremental_append_count | 0 | 0 | 0 |
| u8_cache_native_prefill_build_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 28 |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 4014080 |
| u8_cache_segment_seal_count | 0 | 0 | 0 |
| u8_cache_segment_sealed_bytes | 0 | 0 | 0 |
| u8_cache_segment_tail_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | 0 |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | 0 |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_quartet_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 917504 |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 1 |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | 0 |
| valid_length | 64 | 64 | 64 |
| vtcm_acquired_bytes | 8388608 | 8388608 | 8388608 |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 8365824 |
| vtcm_requested_bytes | 8388608 | 8388608 | 8388608 |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_decode_audit | 0 | 0 | 0 |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | 0 |
| w4f16_decode_opt | 0 | 0 | 0 |
| w4f16_decode_opt_calls | 0 | 0 | 0 |
| w4f16_expand_mismatch_count | 0 | 0 | 0 |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | 0 |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_av_padding_poison_count | 0 | 0 | 0 |
| w4u8_av_requant_call_count | 0 | 0 | 0 |
| w4u8_av_requant_rows_observed | 0 | 0 | 0 |
| w4u8_av_requant_vector_count | 0 | 0 | 0 |
| w4u8_common_op_rows_observed | 64 | 0 | 0 |
| w4u8_common_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_av_padding_poison | 0 | 0 | 0 |
| w4u8_decode_av_requant_rows | 4 | 4 | 4 |
| w4u8_decode_common_op_rows | 4 | 4 | 4 |
| w4u8_decode_common_padding_poison | 0 | 0 | 0 |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 8 |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.409286e+09 | 1.409286e+09 | 1.409286e+09 |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 1 |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 1 |
| w4u8_decode_direct_n_hmx_command_count | 840 | 840 | 840 |
| w4u8_decode_direct_n_mask | 63 | 63 | 63 |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 1 |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_projection_count | 196 | 196 | 196 |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.046431e+08 | 7.046431e+08 | 7.046431e+08 |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | 0 |
| w4u8_decode_k_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 32 |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_projection_mode | 1 | 1 | 1 |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_q_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 4 |
| w4u8_decode_qk_padding_poison | 0 | 0 | 0 |
| w4u8_decode_qk_rows_processed | 0 | 0 | 0 |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | 0 |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | 0 |
| w4u8_decode_softmax_mode | 1 | 1 | 1 |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | 0 |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_rows | 4 | 4 | 4 |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_swiglu_vector_count | 0 | 0 | 0 |
| w4u8_delta_reconstruction_mode | 0 | 0 | 0 |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_final_residual_main_work_ticks (us) | 106.3542 | 0 | 0 |
| w4u8_final_residual_pool_wait_ticks (us) | 27.73438 | 0 | 0 |
| w4u8_final_residual_task_count | 448 | 0 | 0 |
| w4u8_final_residual_worker_work_ticks (us) | 548.4896 | 0 | 0 |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 394.1667 | 401.1719 | 401.5885 |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_input_norm_main_work_ticks (us) | 415.8333 | 0 | 0 |
| w4u8_input_norm_pool_wait_ticks (us) | 96.19792 | 0 | 0 |
| w4u8_input_norm_task_count | 448 | 0 | 0 |
| w4u8_input_norm_worker_work_ticks (us) | 2191.042 | 0 | 0 |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 224 |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | 0 |
| w4u8_o_batch_count | 112 | 112 | 112 |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 16 |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_post_residual_main_work_ticks (us) | 513.724 | 0 | 0 |
| w4u8_post_residual_pool_wait_ticks (us) | 97.26562 | 0 | 0 |
| w4u8_post_residual_task_count | 448 | 0 | 0 |
| w4u8_post_residual_worker_work_ticks (us) | 2633.099 | 0 | 0 |
| w4u8_prefill_cache_mode | 1 | 1 | 1 |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | 0 |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | 0 |
| w4u8_qkv_ring_batch_count | 168 | 168 | 168 |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2171.693 | 2158.099 | 2158.776 |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | 0 |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | 672 |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 555.0781 | 544.375 | 545.4688 |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1494.948 | 1484.219 | 1484.74 |
| w4u8_qkv_ring_pipeline_ticks (us) | 7003.516 | 7001.38 | 6993.854 |
| w4u8_qkv_ring_pool_wait_ticks (us) | 4478.516 | 4506.458 | 4499.714 |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | 5 |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 8.723958 | 6.302083 | 6.458333 |
| w4u8_qkv_ring_slot_count | 2 | 2 | 2 |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 13427.97 | 13619.66 | 13662.55 |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2172.656 | 2159.505 | 2159.896 |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_residual_active_contexts | 6 | 0 | 0 |
| w4u8_swiglu_rows_observed | 64 | 64 | 64 |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | 8.660326e+08 |
| weight_dma_descriptor_count | 2275 | 2275 | 2275 |
| weight_dma_ticks (us) | 18300.96 | 18225.81 | 18235.05 |
| wide_score_mode | 4 | 4 | 4 |

### repeat1 decode

| Field | Integer | Prior FP32 | Optimized FP32 |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | 0 |
| attention_av_hmx_ticks (us) | 0 | 0 | 0 |
| attention_av_pack_ticks (us) | 0 | 0 | 0 |
| attention_av_unpack_ticks (us) | 0 | 0 | 0 |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | 0 |
| attention_qk_hmx_ticks (us) | 0 | 0 | 0 |
| attention_qk_pack_ticks (us) | 0 | 0 | 0 |
| attention_qk_unpack_ticks (us) | 0 | 0 | 0 |
| attention_setup_ticks (us) | 0 | 0 | 0 |
| attention_softmax_ticks (us) | 0 | 0 | 0 |
| attention_ticks (us) | 1911.88 | 1926.814 | 1926.028 |
| attention_unattributed_ticks (us) | 891.5226 | 902.6719 | 900.8003 |
| block_invocation_count | 28 | 28 | 28 |
| block_orchestration_ticks (us) | 29.23958 | 28.6059 | 28.6875 |
| boundary_ddr_read_bytes | 4846976 | 4849024 | 4849024 |
| boundary_ddr_write_bytes | 0 | 0 | 0 |
| boundary_dma_descriptor_count | 226 | 226 | 226 |
| cache_compared_elements | 0 | 0 | 0 |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | 0 |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.01 |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.99999 |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | 0 |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | 0 |
| cache_max_nrmse | 0 | 0 | 0 |
| cache_min_cosine | 1 | 1 | 1 |
| cache_mismatches | 0 | 0 | 0 |
| cache_mixed_tolerance_violations | 0 | 0 | 0 |
| cache_nonfinite_count | 0 | 0 | 0 |
| cache_prefix_mismatches | 0 | 0 | 0 |
| cache_structure_mismatches | 0 | 0 | 0 |
| cache_tensor_count | 0 | 0 | 0 |
| dense_r3_constant_read_bytes | 0 | 0 | 0 |
| dense_r3_mode | 0 | 0 | 0 |
| dense_r3_optimization | 0 | 0 | 0 |
| dense_r3_total_calls | 0 | 0 | 0 |
| dense_r3_total_finish_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_hmx_calls | 0 | 0 | 0 |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_parallel_heads | 0 | 0 | 0 |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_refined_values | 0 | 0 | 0 |
| dense_r3_total_rows | 0 | 0 | 0 |
| dense_r4_audit_bytes | 0 | 0 | 0 |
| dense_r4_calls | 0 | 0 | 0 |
| dense_r4_finish_ticks (us) | 0 | 0 | 0 |
| dense_r4_hmx_calls | 0 | 0 | 0 |
| dense_r4_layout_ticks (us) | 0 | 0 | 0 |
| dense_r4_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r4_mode | 0 | 0 | 0 |
| dense_r4_optimization | 0 | 0 | 0 |
| dense_r4_parallel_dispatches | 0 | 0 | 0 |
| dense_r4_parallel_finish_groups | 0 | 0 | 0 |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_parallel_prepare_tiles | 0 | 0 | 0 |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r4_pipeline_batches | 0 | 0 | 0 |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_consume_count | 0 | 0 | 0 |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_publish_count | 0 | 0 | 0 |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | 0 |
| dense_r4_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r4_rows | 0 | 0 | 0 |
| down_ticks (us) | 3515.089 | 3482.297 | 3480.422 |
| dsp_status | 3 | 3 | 3 |
| experiment | 218 | 218 | 218 |
| f16_cache_full_prefix_pack_count | 0 | 0 | 0 |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | 0 |
| f16_cache_native_incremental_append_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| final_residual_ticks (us) | 42.64063 | 1.980903 | 1.954861 |
| first_position | 71 | 71 | 71 |
| fp32_residual | 0 | 1 | 2 |
| gate_up_ticks (us) | 6506.828 | 6387.865 | 6390.344 |
| generation_embedding_ddr_read_bytes | 2304 | 4352 | 4352 |
| generation_embedding_ticks (us) | 1.852431 | 1.810764 | 1.866319 |
| generation_final_norm_ticks (us) | 3.072917 | 14.75 | 14.67708 |
| generation_lm_head_argmax_ticks (us) | 462.9045 | 463.375 | 463.3385 |
| generation_lm_head_batch_n_tiles | 32 | 32 | 32 |
| generation_lm_head_command_count | 149 | 149 | 149 |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | 1.56798e+08 |
| generation_lm_head_direct_slot_join_count | 147 | 147 | 147 |
| generation_lm_head_exclusive_ticks (us) | 3214.476 | 3208.672 | 3210.28 |
| generation_lm_head_expand_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_hmx_tail_wait_ticks (us) | 13.45486 | 12.17188 | 12.34896 |
| generation_lm_head_hmx_ticks (us) | 2701.087 | 2693.861 | 2694.696 |
| generation_lm_head_n_tiles | 4748 | 4748 | 4748 |
| generation_lm_head_prefetch_count | 148 | 148 | 148 |
| generation_lm_head_scale_dma_ticks (us) | 20.63194 | 21.91146 | 21.7309 |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 1215488 |
| generation_lm_head_ticks (us) | 3217.714 | 3223.484 | 3224.957 |
| generation_lm_head_weight_dma_ticks (us) | 2712.601 | 2707.141 | 2707.748 |
| generation_lm_head_weight_dma_wait_ticks (us) | 2645.957 | 2640.748 | 2640.354 |
| generation_step | 8 | 8 | 8 |
| hmx_command_count | 1437 | 1437 | 1437 |
| hmx_compute_ticks (us) | 8442.997 | 8656.91 | 8639.024 |
| hmx_fp16_tile_pair_count | 0 | 0 | 0 |
| hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | 1690880 |
| host_boundary_us | 2054.594 | 2083.188 | 2110.795 |
| host_us | 22131.38 | 22449.11 | 22473.46 |
| host_wall_ns | 2.213138e+07 | 2.244911e+07 | 2.247346e+07 |
| input_norm_ticks (us) | 150.2847 | 368.6389 | 368.4688 |
| input_stage_ticks (us) | 0.3038194 | 0.3038194 | 0.2986111 |
| intermediate_ddr_read_bytes | 0 | 0 | 0 |
| intermediate_ddr_write_bytes | 0 | 0 | 0 |
| intermediate_dma_descriptor_count | 0 | 0 | 0 |
| intermediate_spill_fill_count | 0 | 0 | 0 |
| invocation_ticks (us) | 20039.27 | 20365.93 | 20354.49 |
| kv_cache_k_format | 14 | 14 | 14 |
| kv_cache_v_format | 12 | 12 | 12 |
| layer_bookkeeping_ticks (us) | 16.45312 | 16.25868 | 16.23958 |
| ledger_named_ticks (us) | 20039.27 | 20365.93 | 20354.49 |
| ledger_unattributed_ticks (us) | 0 | 0 | 0 |
| logical_m | 1 | 1 | 1 |
| metadata_stage_ticks (us) | 218.316 | 222.4653 | 221.1302 |
| numerical_audit_enabled | 0 | 0 | 0 |
| numerical_status | 1 | 1 | 1 |
| o_projection_ticks (us) | 1244.679 | 1341.003 | 1337.436 |
| output_cosine | 0 | 0 | 0 |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0625 |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.003 |
| output_fp16_rtol | 0.002 | 0.002 | 0.002 |
| output_max_abs | 0 | 0 | 0 |
| output_max_lsb | 0 | 0 | 0 |
| output_max_required_rtol_after_atol | 0 | 0 | 0 |
| output_mismatches | 1 | 1 | 1 |
| output_mixed_tolerance_violations | 0 | 0 | 0 |
| output_nonfinite_count | 0 | 0 | 0 |
| output_nrmse | 0 | 0 | 0 |
| output_stage_ticks (us) | 0 | 0 | 0 |
| post_attention_norm_ticks (us) | 0.7795139 | 0.7934028 | 0.8107639 |
| post_attention_residual_ticks (us) | 198.4462 | 368.8507 | 368.5312 |
| prefix_group_patch_count | 0 | 0 | 0 |
| prefix_kv_mode | 1 | 1 | 1 |
| prefix_seed_metadata_read_bytes | 0 | 0 | 0 |
| prepared_session_run_index | 9 | 9 | 9 |
| projection_failure_index | 0 | 0 | 0 |
| projection_failure_n_tile | 0 | 0 | 0 |
| projection_failure_result | 0 | 0 | 0 |
| projection_failure_step | 0 | 0 | 0 |
| projection_hmx_wait_ticks (us) | 699.408 | 746.7587 | 743.4653 |
| projection_pack_ticks (us) | 3.793403 | 3.782986 | 3.791667 |
| projection_unpack_ticks (us) | 0 | 0 | 0 |
| qk_norm_rope_ticks (us) | 0.3628472 | 0.3663194 | 0.3628472 |
| qkv_projection_ticks (us) | 2494.528 | 2501.255 | 2499.814 |
| repeat_count | 1 | 1 | 1 |
| runtime_setup_ticks (us) | 37.88889 | 37.85764 | 37.96528 |
| runtime_teardown_ticks (us) | 35.55556 | 35.49653 | 35.56597 |
| scan_attention_overlay_capacity_bytes | 2752512 | 2359296 | 2359296 |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 77824 |
| scan_cache_append_mismatch_count | 0 | 0 | 0 |
| scan_cache_append_ticks (us) | 120.8976 | 126.6389 | 125.8316 |
| scan_cache_ddr_read_bytes | 4042752 | 4042752 | 4042752 |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 32256 |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 1176 |
| scan_cache_pack_ticks (us) | 290.6458 | 290.7031 | 291.0174 |
| scan_cache_stage_ticks (us) | 569.8872 | 582.5278 | 582.2344 |
| scan_dynamic_attention_ticks (us) | 1906.689 | 1921.613 | 1920.835 |
| scan_logical_m_observed | 1 | 1 | 1 |
| scan_padded_kv_length | 96 | 96 | 96 |
| scan_total_kv_length | 72 | 72 | 72 |
| stage_boundary_ticks (us) | 1.725694 | 1.736111 | 1.767361 |
| total_ticks (us) | 20001.39 | 20328.06 | 20316.27 |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | 0 |
| u8_attention_av_hmx_ticks (us) | 170.5764 | 172.1806 | 172.3767 |
| u8_attention_av_requant_ticks (us) | 143.2292 | 144.3715 | 145.1389 |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | 0 |
| u8_attention_k_pack_ticks (us) | 31.57465 | 31.62326 | 31.66667 |
| u8_attention_pipeline_wait_ticks (us) | 65.875 | 66.09549 | 66.02257 |
| u8_attention_probability_mask_violation_count | 0 | 0 | 0 |
| u8_attention_qk_hmx_ticks (us) | 140.3941 | 141.7378 | 142.3385 |
| u8_attention_qk_norm_rope_ticks (us) | 1980.726 | 1969.66 | 1981.981 |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | 0 |
| u8_attention_softmax_ticks (us) | 349.1215 | 350.3073 | 349.9948 |
| u8_attention_v_pack_ticks (us) | 117.3733 | 117.3125 | 117.3646 |
| u8_cache_full_prefix_pack_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_correction_load_bytes | 6272 | 6272 | 6272 |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 25088 |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 28 |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 39.51389 | 39.74132 | 39.6684 |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 802816 |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | 0 |
| u8_cache_native_append_update_ticks (us) | 409.2691 | 415.5069 | 414.3316 |
| u8_cache_native_incremental_append_count | 28 | 28 | 28 |
| u8_cache_native_prefill_build_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| u8_cache_segment_seal_count | 0 | 0 | 0 |
| u8_cache_segment_sealed_bytes | 0 | 0 | 0 |
| u8_cache_segment_tail_append_count | 28 | 28 | 28 |
| u8_cache_v_quartet_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | 0 |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | 0 |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_quartet_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_native_load_bytes | 275251.2 | 275251.2 | 275251.2 |
| u8_cache_v_vtcm_tail_partial_pack_rows | 358.4 | 358.4 | 358.4 |
| u8_cache_v_vtcm_tail_publish_count | 44.8 | 44.8 | 44.8 |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 224 |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | 0 |
| valid_length | 72 | 72 | 72 |
| vtcm_acquired_bytes | 8388608 | 8388608 | 8388608 |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 8365824 |
| vtcm_requested_bytes | 8388608 | 8388608 | 8388608 |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_decode_audit | 0 | 0 | 0 |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | 0 |
| w4f16_decode_opt | 0 | 0 | 0 |
| w4f16_decode_opt_calls | 0 | 0 | 0 |
| w4f16_expand_mismatch_count | 0 | 0 | 0 |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | 0 |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_av_padding_poison_count | 0 | 0 | 0 |
| w4u8_av_requant_call_count | 224 | 224 | 224 |
| w4u8_av_requant_rows_observed | 4 | 4 | 4 |
| w4u8_av_requant_vector_count | 1792 | 1792 | 1792 |
| w4u8_common_op_rows_observed | 4 | 0 | 0 |
| w4u8_common_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_av_padding_poison | 0 | 0 | 0 |
| w4u8_decode_av_requant_rows | 4 | 4 | 4 |
| w4u8_decode_common_op_rows | 4 | 4 | 4 |
| w4u8_decode_common_padding_poison | 0 | 0 | 0 |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 8 |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.720451e+09 | 1.720451e+09 | 1.720451e+09 |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 1 |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 1 |
| w4u8_decode_direct_n_hmx_command_count | 989 | 989 | 989 |
| w4u8_decode_direct_n_mask | 63 | 63 | 63 |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 1 |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 197 |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 8.602255e+08 | 8.602255e+08 | 8.602255e+08 |
| w4u8_decode_k_pair_row4_call_count | 112 | 112 | 112 |
| w4u8_decode_k_temp_carrier_skipped_count | 224 | 224 | 224 |
| w4u8_decode_k_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 32 |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_projection_mode | 1 | 1 | 1 |
| w4u8_decode_q_pair_row4_call_count | 224 | 224 | 224 |
| w4u8_decode_q_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 4 |
| w4u8_decode_qk_padding_poison | 0 | 0 | 0 |
| w4u8_decode_qk_rows_processed | 2688 | 2688 | 2688 |
| w4u8_decode_softmax_hvx_tile4_call_count | 224 | 224 | 224 |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | 0 |
| w4u8_decode_softmax_mode | 1 | 1 | 1 |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | 0 |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_row4_call_count | 5376 | 5376 | 5376 |
| w4u8_decode_swiglu_rows | 4 | 4 | 4 |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_swiglu_vector_count | 5376 | 5376 | 5376 |
| w4u8_delta_reconstruction_mode | 0 | 0 | 0 |
| w4u8_final_residual_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_final_residual_task_count | 0 | 0 | 0 |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 130.1649 | 178.184 | 178.1771 |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | 1 |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 5233.043 | 5020.158 | 5021.995 |
| w4u8_gate_up_swiglu_worker_ticks (us) | 726.9722 | 998.901 | 1002.438 |
| w4u8_input_norm_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_task_count | 0 | 0 | 0 |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 224 |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | 0 |
| w4u8_o_batch_count | 112 | 112 | 112 |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 16 |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | 28 |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 530.1806 | 522.5122 | 522.1528 |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | 28 |
| w4u8_o_gate_prefetch_wait_ticks (us) | 322.0347 | 144.1285 | 143.7587 |
| w4u8_post_residual_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_task_count | 0 | 0 | 0 |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_prefill_cache_mode | 1 | 1 | 1 |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | 4 |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | 0 |
| w4u8_qkv_ring_batch_count | 168 | 168 | 168 |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2143.578 | 2151.486 | 2149.181 |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | 0 |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | 672 |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 551.3351 | 547.0243 | 551.0885 |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1452.061 | 1469.135 | 1464.953 |
| w4u8_qkv_ring_pipeline_ticks (us) | 2464.307 | 2471.089 | 2469.719 |
| w4u8_qkv_ring_pool_wait_ticks (us) | 3.796875 | 3.770833 | 3.748264 |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | 5 |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 10.08333 | 8.392361 | 9.255208 |
| w4u8_qkv_ring_slot_count | 2 | 2 | 2 |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 12442.94 | 12509.04 | 12506.1 |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2144.594 | 2152.524 | 2150.328 |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_residual_active_contexts | 0 | 0 | 0 |
| w4u8_swiglu_rows_observed | 4 | 4 | 4 |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | 8.660326e+08 |
| weight_dma_descriptor_count | 1830 | 1830 | 1830 |
| weight_dma_ticks (us) | 15608.63 | 15605.12 | 15606.64 |
| wide_score_mode | 4 | 4 | 4 |

### repeat10 prefill

| Field | Integer | Prior FP32 | Optimized FP32 |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | 0 |
| attention_av_hmx_ticks (us) | 0 | 0 | 0 |
| attention_av_pack_ticks (us) | 0 | 0 | 0 |
| attention_av_unpack_ticks (us) | 0 | 0 | 0 |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | 0 |
| attention_qk_hmx_ticks (us) | 0 | 0 | 0 |
| attention_qk_pack_ticks (us) | 0 | 0 | 0 |
| attention_qk_unpack_ticks (us) | 0 | 0 | 0 |
| attention_setup_ticks (us) | 0 | 0 | 0 |
| attention_softmax_ticks (us) | 0 | 0 | 0 |
| attention_ticks (us) | 3479.151 | 3487.867 | 3483.219 |
| attention_unattributed_ticks (us) | 0 | 0 | 0 |
| block_invocation_count | 28 | 28 | 28 |
| block_orchestration_ticks (us) | 35.66667 | 35.30208 | 35.58073 |
| boundary_ddr_read_bytes | 4976000 | 5107072 | 5107072 |
| boundary_ddr_write_bytes | 0 | 0 | 0 |
| boundary_dma_descriptor_count | 289 | 289 | 289 |
| cache_compared_elements | 0 | 0 | 0 |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | 0 |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.01 |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.99999 |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | 0 |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | 0 |
| cache_max_nrmse | 0 | 0 | 0 |
| cache_min_cosine | 1 | 1 | 1 |
| cache_mismatches | 0 | 0 | 0 |
| cache_mixed_tolerance_violations | 0 | 0 | 0 |
| cache_nonfinite_count | 0 | 0 | 0 |
| cache_prefix_mismatches | 0 | 0 | 0 |
| cache_structure_mismatches | 0 | 0 | 0 |
| cache_tensor_count | 0 | 0 | 0 |
| dense_r3_constant_read_bytes | 0 | 0 | 0 |
| dense_r3_mode | 0 | 0 | 0 |
| dense_r3_optimization | 0 | 0 | 0 |
| dense_r3_total_calls | 0 | 0 | 0 |
| dense_r3_total_finish_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_hmx_calls | 0 | 0 | 0 |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_parallel_heads | 0 | 0 | 0 |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_refined_values | 0 | 0 | 0 |
| dense_r3_total_rows | 0 | 0 | 0 |
| dense_r4_audit_bytes | 0 | 0 | 0 |
| dense_r4_calls | 0 | 0 | 0 |
| dense_r4_finish_ticks (us) | 0 | 0 | 0 |
| dense_r4_hmx_calls | 0 | 0 | 0 |
| dense_r4_layout_ticks (us) | 0 | 0 | 0 |
| dense_r4_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r4_mode | 0 | 0 | 0 |
| dense_r4_optimization | 0 | 0 | 0 |
| dense_r4_parallel_dispatches | 0 | 0 | 0 |
| dense_r4_parallel_finish_groups | 0 | 0 | 0 |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_parallel_prepare_tiles | 0 | 0 | 0 |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r4_pipeline_batches | 0 | 0 | 0 |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_consume_count | 0 | 0 | 0 |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_publish_count | 0 | 0 | 0 |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | 0 |
| dense_r4_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r4_rows | 0 | 0 | 0 |
| down_ticks (us) | 4410.336 | 3781.448 | 3775.846 |
| dsp_status | 3 | 3 | 3 |
| experiment | 218 | 218 | 218 |
| f16_cache_full_prefix_pack_count | 0 | 0 | 0 |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | 0 |
| f16_cache_native_incremental_append_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| final_residual_ticks (us) | 186.3932 | 2.799479 | 2.921875 |
| first_position | 0 | 0 | 0 |
| fp32_residual | 0 | 1 | 2 |
| gate_up_ticks (us) | 7032.885 | 7027.12 | 7031.279 |
| generation_embedding_ddr_read_bytes | 131328 | 262400 | 262400 |
| generation_embedding_ticks (us) | 47.92969 | 61.61719 | 61.69271 |
| generation_final_norm_ticks (us) | 3.549479 | 15.00781 | 15.04427 |
| generation_lm_head_argmax_ticks (us) | 603.8438 | 605.4635 | 605.6667 |
| generation_lm_head_batch_n_tiles | 8 | 8 | 8 |
| generation_lm_head_command_count | 594 | 594 | 594 |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | 1.56798e+08 |
| generation_lm_head_direct_slot_join_count | 0 | 0 | 0 |
| generation_lm_head_exclusive_ticks (us) | 5259.346 | 5199.208 | 5193.146 |
| generation_lm_head_expand_ticks (us) | 3696.044 | 3650.497 | 3648.388 |
| generation_lm_head_hmx_tail_wait_ticks (us) | 227.8724 | 212.3203 | 208.0677 |
| generation_lm_head_hmx_ticks (us) | 4608.378 | 4547.547 | 4541.263 |
| generation_lm_head_n_tiles | 4748 | 4748 | 4748 |
| generation_lm_head_prefetch_count | 593 | 593 | 593 |
| generation_lm_head_scale_dma_ticks (us) | 21.28125 | 21.77083 | 21.73177 |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 1215488 |
| generation_lm_head_ticks (us) | 5263.188 | 5214.003 | 5207.909 |
| generation_lm_head_weight_dma_ticks (us) | 5220.411 | 5160.193 | 5154.188 |
| generation_lm_head_weight_dma_wait_ticks (us) | 301.7604 | 293.7552 | 291.8854 |
| generation_step | 0 | 0 | 0 |
| hmx_command_count | 1882 | 1882 | 1882 |
| hmx_compute_ticks (us) | 12257.14 | 12935.86 | 12945.42 |
| hmx_fp16_tile_pair_count | 0 | 0 | 0 |
| hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | 2031360 |
| host_boundary_us | 991.5963 | 890.8854 | 876.5105 |
| host_us | 31768.39 | 34809.73 | 34582.87 |
| host_wall_ns | 3.176839e+07 | 3.480973e+07 | 3.458287e+07 |
| input_norm_ticks (us) | 559.0964 | 2121.617 | 2019.057 |
| input_stage_ticks (us) | 0.4348958 | 0.4401042 | 0.4817708 |
| intermediate_ddr_read_bytes | 0 | 0 | 0 |
| intermediate_ddr_write_bytes | 0 | 0 | 0 |
| intermediate_dma_descriptor_count | 0 | 0 | 0 |
| intermediate_spill_fill_count | 0 | 0 | 0 |
| invocation_ticks (us) | 30773.23 | 33923.53 | 33717.11 |
| kv_cache_k_format | 14 | 14 | 14 |
| kv_cache_v_format | 12 | 12 | 12 |
| layer_bookkeeping_ticks (us) | 28.26042 | 27.5599 | 27.52083 |
| ledger_named_ticks (us) | 30773.23 | 33923.53 | 33717.11 |
| ledger_unattributed_ticks (us) | 0 | 0 | 0 |
| logical_m | 64 | 64 | 64 |
| metadata_stage_ticks (us) | 246.7396 | 254.4792 | 255.8802 |
| numerical_audit_enabled | 0 | 0 | 0 |
| numerical_status | 1 | 1 | 1 |
| o_projection_ticks (us) | 1249.776 | 2094.906 | 2095.44 |
| output_cosine | 0 | 0 | 0 |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0625 |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.003 |
| output_fp16_rtol | 0.002 | 0.002 | 0.002 |
| output_max_abs | 0 | 0 | 0 |
| output_max_lsb | 0 | 0 | 0 |
| output_max_required_rtol_after_atol | 0 | 0 | 0 |
| output_mismatches | 1 | 1 | 1 |
| output_mixed_tolerance_violations | 0 | 0 | 0 |
| output_nonfinite_count | 0 | 0 | 0 |
| output_nrmse | 0 | 0 | 0 |
| output_stage_ticks (us) | 0 | 0 | 0 |
| post_attention_norm_ticks (us) | 1.096354 | 1.101562 | 1.164063 |
| post_attention_residual_ticks (us) | 660.0833 | 2217.839 | 2114.854 |
| prefix_group_patch_count | 224 | 224 | 224 |
| prefix_kv_mode | 1 | 1 | 1 |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | 57344 |
| prepared_session_run_index | 1 | 1 | 1 |
| projection_failure_index | 0 | 0 | 0 |
| projection_failure_n_tile | 0 | 0 | 0 |
| projection_failure_result | 0 | 0 | 0 |
| projection_failure_step | 0 | 0 | 0 |
| projection_hmx_wait_ticks (us) | 1537.049 | 1740.375 | 1732.812 |
| projection_pack_ticks (us) | 4.658854 | 3.963542 | 4.020833 |
| projection_unpack_ticks (us) | 0 | 0 | 0 |
| qk_norm_rope_ticks (us) | 0.4947917 | 0.5442708 | 0.5286458 |
| qkv_projection_ticks (us) | 7013.742 | 7029.323 | 7029.578 |
| repeat_count | 1 | 1 | 1 |
| runtime_setup_ticks (us) | 65.03646 | 65.61198 | 66.8724 |
| runtime_teardown_ticks (us) | 50.82292 | 49.55469 | 50.05729 |
| scan_attention_overlay_capacity_bytes | 0 | 0 | 0 |
| scan_attention_overlay_required_bytes | 0 | 0 | 0 |
| scan_cache_append_mismatch_count | 0 | 0 | 0 |
| scan_cache_append_ticks (us) | 287.0234 | 295.1302 | 294.4896 |
| scan_cache_ddr_read_bytes | 0 | 0 | 0 |
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 5906432 |
| scan_cache_dma_descriptor_count | 448 | 448 | 448 |
| scan_cache_pack_ticks (us) | 135.237 | 136.2031 | 135.8932 |
| scan_cache_stage_ticks (us) | 0 | 0 | 0 |
| scan_dynamic_attention_ticks (us) | 0 | 0 | 0 |
| scan_logical_m_observed | 64 | 64 | 64 |
| scan_padded_kv_length | 64 | 64 | 64 |
| scan_total_kv_length | 64 | 64 | 64 |
| stage_boundary_ticks (us) | 20.21094 | 20.16667 | 20.54427 |
| total_ticks (us) | 30705.84 | 33856.78 | 33651.72 |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | 0 |
| u8_attention_av_hmx_ticks (us) | 626.3125 | 629.6536 | 626.2734 |
| u8_attention_av_requant_ticks (us) | 1644.401 | 1644.841 | 1645.273 |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | 0 |
| u8_attention_k_pack_ticks (us) | 0 | 0 | 0 |
| u8_attention_pipeline_wait_ticks (us) | 2006.328 | 2028.083 | 2015.419 |
| u8_attention_probability_mask_violation_count | 0 | 0 | 0 |
| u8_attention_qk_hmx_ticks (us) | 578.4974 | 581.6302 | 580.099 |
| u8_attention_qk_norm_rope_ticks (us) | 24429.02 | 24474.63 | 24483.74 |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | 0 |
| u8_attention_softmax_ticks (us) | 8704.32 | 8698.104 | 8699.523 |
| u8_attention_v_pack_ticks (us) | 5009.346 | 5025.07 | 5025.951 |
| u8_cache_full_prefix_pack_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 827904 |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 1 |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | 0 |
| u8_cache_native_append_update_ticks (us) | 420.599 | 429.0443 | 428.7031 |
| u8_cache_native_incremental_append_count | 0 | 0 | 0 |
| u8_cache_native_prefill_build_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 28 |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 4014080 |
| u8_cache_segment_seal_count | 0 | 0 | 0 |
| u8_cache_segment_sealed_bytes | 0 | 0 | 0 |
| u8_cache_segment_tail_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | 0 |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | 0 |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_quartet_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 917504 |
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 1 |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | 0 |
| valid_length | 64 | 64 | 64 |
| vtcm_acquired_bytes | 8388608 | 8388608 | 8388608 |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 8365824 |
| vtcm_requested_bytes | 8388608 | 8388608 | 8388608 |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_decode_audit | 0 | 0 | 0 |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | 0 |
| w4f16_decode_opt | 0 | 0 | 0 |
| w4f16_decode_opt_calls | 0 | 0 | 0 |
| w4f16_expand_mismatch_count | 0 | 0 | 0 |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | 0 |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_av_padding_poison_count | 0 | 0 | 0 |
| w4u8_av_requant_call_count | 0 | 0 | 0 |
| w4u8_av_requant_rows_observed | 0 | 0 | 0 |
| w4u8_av_requant_vector_count | 0 | 0 | 0 |
| w4u8_common_op_rows_observed | 64 | 0 | 0 |
| w4u8_common_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_av_padding_poison | 0 | 0 | 0 |
| w4u8_decode_av_requant_rows | 4 | 4 | 4 |
| w4u8_decode_common_op_rows | 4 | 4 | 4 |
| w4u8_decode_common_padding_poison | 0 | 0 | 0 |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 8 |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.409286e+09 | 1.409286e+09 | 1.409286e+09 |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 1 |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 1 |
| w4u8_decode_direct_n_hmx_command_count | 840 | 840 | 840 |
| w4u8_decode_direct_n_mask | 63 | 63 | 63 |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 1 |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_projection_count | 196 | 196 | 196 |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.046431e+08 | 7.046431e+08 | 7.046431e+08 |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | 0 |
| w4u8_decode_k_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 32 |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_projection_mode | 1 | 1 | 1 |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_q_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 4 |
| w4u8_decode_qk_padding_poison | 0 | 0 | 0 |
| w4u8_decode_qk_rows_processed | 0 | 0 | 0 |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | 0 |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | 0 |
| w4u8_decode_softmax_mode | 1 | 1 | 1 |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | 0 |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_rows | 4 | 4 | 4 |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_swiglu_vector_count | 0 | 0 | 0 |
| w4u8_delta_reconstruction_mode | 0 | 0 | 0 |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_final_residual_main_work_ticks (us) | 107.1797 | 0 | 0 |
| w4u8_final_residual_pool_wait_ticks (us) | 27.91406 | 0 | 0 |
| w4u8_final_residual_task_count | 448 | 0 | 0 |
| w4u8_final_residual_worker_work_ticks (us) | 551.2865 | 0 | 0 |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 395.1068 | 399.9401 | 398.2526 |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_worker_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_input_norm_main_work_ticks (us) | 425.7734 | 0 | 0 |
| w4u8_input_norm_pool_wait_ticks (us) | 82.26562 | 0 | 0 |
| w4u8_input_norm_task_count | 448 | 0 | 0 |
| w4u8_input_norm_worker_work_ticks (us) | 2168.391 | 0 | 0 |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 224 |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | 0 |
| w4u8_o_batch_count | 112 | 112 | 112 |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 16 |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | 0 |
| w4u8_o_gate_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | 0 |
| w4u8_post_residual_main_work_ticks (us) | 513.8672 | 0 | 0 |
| w4u8_post_residual_pool_wait_ticks (us) | 98.04167 | 0 | 0 |
| w4u8_post_residual_task_count | 448 | 0 | 0 |
| w4u8_post_residual_worker_work_ticks (us) | 2632.25 | 0 | 0 |
| w4u8_prefill_cache_mode | 1 | 1 | 1 |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | 0 |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | 0 |
| w4u8_qkv_ring_batch_count | 168 | 168 | 168 |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2153.289 | 2145.833 | 2153.625 |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | 0 |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | 672 |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 554.2474 | 550.3203 | 549.388 |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1467.076 | 1459.948 | 1471.391 |
| w4u8_qkv_ring_pipeline_ticks (us) | 6978.698 | 6992.367 | 6992.643 |
| w4u8_qkv_ring_pool_wait_ticks (us) | 4488.021 | 4502.505 | 4498.055 |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | 5 |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 9.690104 | 10.09635 | 8.455729 |
| w4u8_qkv_ring_slot_count | 2 | 2 | 2 |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 13394.07 | 13584.46 | 13584.94 |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2154.375 | 2146.836 | 2154.742 |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_residual_active_contexts | 6 | 0 | 0 |
| w4u8_swiglu_rows_observed | 64 | 64 | 64 |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | 8.660326e+08 |
| weight_dma_descriptor_count | 2275 | 2275 | 2275 |
| weight_dma_ticks (us) | 18267.12 | 18200.4 | 18217.56 |
| wide_score_mode | 4 | 4 | 4 |

### repeat10 decode

| Field | Integer | Prior FP32 | Optimized FP32 |
|---|---:|---:|---:|
| activation_ticks (us) | 0 | 0 | 0 |
| attention_av_hmx_ticks (us) | 0 | 0 | 0 |
| attention_av_pack_ticks (us) | 0 | 0 | 0 |
| attention_av_unpack_ticks (us) | 0 | 0 | 0 |
| attention_gqa_pipeline_ticks (us) | 0 | 0 | 0 |
| attention_qk_hmx_ticks (us) | 0 | 0 | 0 |
| attention_qk_pack_ticks (us) | 0 | 0 | 0 |
| attention_qk_unpack_ticks (us) | 0 | 0 | 0 |
| attention_setup_ticks (us) | 0 | 0 | 0 |
| attention_softmax_ticks (us) | 0 | 0 | 0 |
| attention_ticks (us) | 1912.57 | 1927.851 | 1927.846 |
| attention_unattributed_ticks (us) | 891.7521 | 904.0257 | 903.4908 |
| block_invocation_count | 28 | 28 | 28 |
| block_orchestration_ticks (us) | 29.30608 | 28.61788 | 28.62951 |
| boundary_ddr_read_bytes | 4846976 | 4849024 | 4849024 |
| boundary_ddr_write_bytes | 0 | 0 | 0 |
| boundary_dma_descriptor_count | 226 | 226 | 226 |
| cache_compared_elements | 0 | 0 | 0 |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | 0 |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.01 |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.99999 |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | 0 |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | 0 |
| cache_max_nrmse | 0 | 0 | 0 |
| cache_min_cosine | 1 | 1 | 1 |
| cache_mismatches | 0 | 0 | 0 |
| cache_mixed_tolerance_violations | 0 | 0 | 0 |
| cache_nonfinite_count | 0 | 0 | 0 |
| cache_prefix_mismatches | 0 | 0 | 0 |
| cache_structure_mismatches | 0 | 0 | 0 |
| cache_tensor_count | 0 | 0 | 0 |
| dense_r3_constant_read_bytes | 0 | 0 | 0 |
| dense_r3_mode | 0 | 0 | 0 |
| dense_r3_optimization | 0 | 0 | 0 |
| dense_r3_total_calls | 0 | 0 | 0 |
| dense_r3_total_finish_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_hmx_calls | 0 | 0 | 0 |
| dense_r3_total_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_parallel_heads | 0 | 0 | 0 |
| dense_r3_total_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r3_total_refined_values | 0 | 0 | 0 |
| dense_r3_total_rows | 0 | 0 | 0 |
| dense_r4_audit_bytes | 0 | 0 | 0 |
| dense_r4_calls | 0 | 0 | 0 |
| dense_r4_finish_ticks (us) | 0 | 0 | 0 |
| dense_r4_hmx_calls | 0 | 0 | 0 |
| dense_r4_layout_ticks (us) | 0 | 0 | 0 |
| dense_r4_matmul_ticks (us) | 0 | 0 | 0 |
| dense_r4_mode | 0 | 0 | 0 |
| dense_r4_optimization | 0 | 0 | 0 |
| dense_r4_parallel_dispatches | 0 | 0 | 0 |
| dense_r4_parallel_finish_groups | 0 | 0 | 0 |
| dense_r4_parallel_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_parallel_prepare_tiles | 0 | 0 | 0 |
| dense_r4_parallel_work_ticks (us) | 0 | 0 | 0 |
| dense_r4_pipeline_batches | 0 | 0 | 0 |
| dense_r4_pipeline_hvx_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_consume_count | 0 | 0 | 0 |
| dense_r4_prefill_join_ticks (us) | 0 | 0 | 0 |
| dense_r4_prefill_publish_count | 0 | 0 | 0 |
| dense_r4_prefill_worker_ticks (us) | 0 | 0 | 0 |
| dense_r4_prepare_ticks (us) | 0 | 0 | 0 |
| dense_r4_rows | 0 | 0 | 0 |
| down_ticks (us) | 3527.876 | 3487.444 | 3487.165 |
| dsp_status | 3 | 3 | 3 |
| experiment | 218 | 218 | 218 |
| f16_cache_full_prefix_pack_count | 0 | 0 | 0 |
| f16_cache_native_append_update_ticks (us) | 0 | 0 | 0 |
| f16_cache_native_incremental_append_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| final_residual_ticks (us) | 42.62882 | 1.993403 | 1.988715 |
| first_position | 71 | 71 | 71 |
| fp32_residual | 0 | 1 | 2 |
| gate_up_ticks (us) | 6537.981 | 6403.983 | 6401.541 |
| generation_embedding_ddr_read_bytes | 2304 | 4352 | 4352 |
| generation_embedding_ticks (us) | 2.047917 | 2.055208 | 2.065104 |
| generation_final_norm_ticks (us) | 3.048958 | 14.84028 | 14.65868 |
| generation_lm_head_argmax_ticks (us) | 462.8424 | 463.4193 | 463.4151 |
| generation_lm_head_batch_n_tiles | 32 | 32 | 32 |
| generation_lm_head_command_count | 149 | 149 | 149 |
| generation_lm_head_ddr_read_bytes | 1.56798e+08 | 1.56798e+08 | 1.56798e+08 |
| generation_lm_head_direct_slot_join_count | 147 | 147 | 147 |
| generation_lm_head_exclusive_ticks (us) | 3218.704 | 3219.999 | 3220.153 |
| generation_lm_head_expand_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_hmx_tail_wait_ticks (us) | 12.61615 | 12.12118 | 12.15677 |
| generation_lm_head_hmx_ticks (us) | 2705.644 | 2705.118 | 2704.392 |
| generation_lm_head_n_tiles | 4748 | 4748 | 4748 |
| generation_lm_head_prefetch_count | 148 | 148 | 148 |
| generation_lm_head_scale_dma_ticks (us) | 21.0684 | 21.96111 | 21.77014 |
| generation_lm_head_scale_init_ticks (us) | 0 | 0 | 0 |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 1215488 |
| generation_lm_head_ticks (us) | 3221.71 | 3234.697 | 3234.71 |
| generation_lm_head_weight_dma_ticks (us) | 2717.892 | 2716.751 | 2717.29 |
| generation_lm_head_weight_dma_wait_ticks (us) | 2650.781 | 2650.007 | 2649.984 |
| generation_step | 8 | 8 | 8 |
| hmx_command_count | 1437 | 1437 | 1437 |
| hmx_compute_ticks (us) | 8444.625 | 8635.43 | 8637.236 |
| hmx_fp16_tile_pair_count | 0 | 0 | 0 |
| hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | 1690880 |
| host_boundary_us | 829.4118 | 818.7211 | 834.6429 |
| host_us | 20970.37 | 21248.18 | 21267.62 |
| host_wall_ns | 2.097037e+07 | 2.124818e+07 | 2.126762e+07 |
| input_norm_ticks (us) | 150.1686 | 368.4774 | 368.5278 |
| input_stage_ticks (us) | 0.3097222 | 0.3107639 | 0.3046875 |
| intermediate_ddr_read_bytes | 0 | 0 | 0 |
| intermediate_ddr_write_bytes | 0 | 0 | 0 |
| intermediate_dma_descriptor_count | 0 | 0 | 0 |
| intermediate_spill_fill_count | 0 | 0 | 0 |
| invocation_ticks (us) | 20126.52 | 20422.07 | 20437.45 |
| kv_cache_k_format | 14 | 14 | 14 |
| kv_cache_v_format | 12 | 12 | 12 |
| layer_bookkeeping_ticks (us) | 16.49809 | 16.31944 | 16.29583 |
| ledger_named_ticks (us) | 20126.52 | 20422.07 | 20437.45 |
| ledger_unattributed_ticks (us) | 0 | 0 | 0 |
| logical_m | 1 | 1 | 1 |
| metadata_stage_ticks (us) | 249.4116 | 250.2979 | 251.4408 |
| numerical_audit_enabled | 0 | 0 | 0 |
| numerical_status | 1 | 1 | 1 |
| o_projection_ticks (us) | 1247.053 | 1341.711 | 1340.246 |
| output_cosine | 0 | 0 | 0 |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0625 |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.003 |
| output_fp16_rtol | 0.002 | 0.002 | 0.002 |
| output_max_abs | 0 | 0 | 0 |
| output_max_lsb | 0 | 0 | 0 |
| output_max_required_rtol_after_atol | 0 | 0 | 0 |
| output_mismatches | 1 | 1 | 1 |
| output_mixed_tolerance_violations | 0 | 0 | 0 |
| output_nonfinite_count | 0 | 0 | 0 |
| output_nrmse | 0 | 0 | 0 |
| output_stage_ticks (us) | 0 | 0 | 0 |
| post_attention_norm_ticks (us) | 0.7607639 | 0.8133681 | 0.8128472 |
| post_attention_residual_ticks (us) | 198.1688 | 368.6835 | 368.7333 |
| prefix_group_patch_count | 0 | 0 | 0 |
| prefix_kv_mode | 1 | 1 | 1 |
| prefix_seed_metadata_read_bytes | 0 | 0 | 0 |
| prepared_session_run_index | 9 | 9 | 9 |
| projection_failure_index | 0 | 0 | 0 |
| projection_failure_n_tile | 0 | 0 | 0 |
| projection_failure_result | 0 | 0 | 0 |
| projection_failure_step | 0 | 0 | 0 |
| projection_hmx_wait_ticks (us) | 699.1917 | 748.6778 | 742.7417 |
| projection_pack_ticks (us) | 3.786111 | 3.777604 | 3.780035 |
| projection_unpack_ticks (us) | 0 | 0 | 0 |
| qk_norm_rope_ticks (us) | 0.3758681 | 0.3704861 | 0.3746528 |
| qkv_projection_ticks (us) | 2502.474 | 2501.967 | 2504.733 |
| repeat_count | 1 | 1 | 1 |
| runtime_setup_ticks (us) | 37.91441 | 37.89323 | 37.98247 |
| runtime_teardown_ticks (us) | 35.3849 | 35.47743 | 35.4901 |
| scan_attention_overlay_capacity_bytes | 2752512 | 2359296 | 2359296 |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 77824 |
| scan_cache_append_mismatch_count | 0 | 0 | 0 |
| scan_cache_append_ticks (us) | 120.7273 | 124.9927 | 126.0852 |
| scan_cache_ddr_read_bytes | 4042752 | 4042752 | 4042752 |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 32256 |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 1176 |
| scan_cache_pack_ticks (us) | 290.9092 | 290.8856 | 290.9752 |
| scan_cache_stage_ticks (us) | 569.8828 | 583.1333 | 583.2806 |
| scan_dynamic_attention_ticks (us) | 1907.406 | 1922.684 | 1922.672 |
| scan_logical_m_observed | 1 | 1 | 1 |
| scan_padded_kv_length | 96 | 96 | 96 |
| scan_total_kv_length | 72 | 72 | 72 |
| stage_boundary_ticks (us) | 1.739931 | 1.732118 | 1.732986 |
| total_ticks (us) | 20088.61 | 20384.16 | 20399.57 |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | 0 |
| u8_attention_av_hmx_ticks (us) | 171.9547 | 171.4559 | 172.9257 |
| u8_attention_av_requant_ticks (us) | 144.0575 | 144.4304 | 144.3043 |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | 0 |
| u8_attention_k_pack_ticks (us) | 31.59826 | 31.62778 | 31.64306 |
| u8_attention_pipeline_wait_ticks (us) | 65.78542 | 66.11962 | 66.32674 |
| u8_attention_probability_mask_violation_count | 0 | 0 | 0 |
| u8_attention_qk_hmx_ticks (us) | 141.1677 | 142.4109 | 142.1547 |
| u8_attention_qk_norm_rope_ticks (us) | 1974.598 | 1985.699 | 1984.339 |
| u8_attention_qk_requant_ticks (us) | 0 | 0 | 0 |
| u8_attention_softmax_ticks (us) | 349.0958 | 349.822 | 350.2033 |
| u8_attention_v_pack_ticks (us) | 117.529 | 117.3134 | 117.3304 |
| u8_cache_full_prefix_pack_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_correction_load_bytes | 6272 | 6272 | 6272 |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 25088 |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 28 |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks (us) | 39.61476 | 39.73924 | 39.77066 |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | 0 |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 802816 |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 196 |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | 0 |
| u8_cache_native_append_update_ticks (us) | 409.4069 | 413.5677 | 414.8736 |
| u8_cache_native_incremental_append_count | 28 | 28 | 28 |
| u8_cache_native_prefill_build_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reuse_count | 0 | 0 | 0 |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | 0 |
| u8_cache_segment_seal_count | 0 | 0 | 0 |
| u8_cache_segment_sealed_bytes | 0 | 0 | 0 |
| u8_cache_segment_tail_append_count | 28 | 28 | 28 |
| u8_cache_v_quartet_append_count | 0 | 0 | 0 |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | 0 |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | 0 |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | 0 |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | 0 |
| u8_cache_v_quartet_publish_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | 0 |
| u8_cache_v_vtcm_tail_native_load_bytes | 275251.2 | 275251.2 | 275251.2 |
| u8_cache_v_vtcm_tail_partial_pack_rows | 358.4 | 358.4 | 358.4 |
| u8_cache_v_vtcm_tail_publish_count | 44.8 | 44.8 | 44.8 |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 224 |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | 0 |
| valid_length | 72 | 72 | 72 |
| vtcm_acquired_bytes | 8388608 | 8388608 | 8388608 |
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 8365824 |
| vtcm_requested_bytes | 8388608 | 8388608 | 8388608 |
| w4f16_cross_prefetch_lifetime_ticks (us) | 0 | 0 | 0 |
| w4f16_cross_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_decode_audit | 0 | 0 | 0 |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | 0 |
| w4f16_decode_opt | 0 | 0 | 0 |
| w4f16_decode_opt_calls | 0 | 0 | 0 |
| w4f16_expand_mismatch_count | 0 | 0 | 0 |
| w4f16_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_expand_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_hmx_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_join_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_stream_work_ticks (us) | 0 | 0 | 0 |
| w4f16_gate_up_weight_dma_ticks (us) | 0 | 0 | 0 |
| w4f16_hmx_tail_wait_ticks (us) | 0 | 0 | 0 |
| w4f16_prefetch_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_av_padding_poison_count | 0 | 0 | 0 |
| w4u8_av_requant_call_count | 224 | 224 | 224 |
| w4u8_av_requant_rows_observed | 4 | 4 | 4 |
| w4u8_av_requant_vector_count | 1792 | 1792 | 1792 |
| w4u8_common_op_rows_observed | 4 | 0 | 0 |
| w4u8_common_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_av_padding_poison | 0 | 0 | 0 |
| w4u8_decode_av_requant_rows | 4 | 4 | 4 |
| w4u8_decode_common_op_rows | 4 | 4 | 4 |
| w4u8_decode_common_padding_poison | 0 | 0 | 0 |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 8 |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.720451e+09 | 1.720451e+09 | 1.720451e+09 |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 1 |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 1 |
| w4u8_decode_direct_n_hmx_command_count | 989 | 989 | 989 |
| w4u8_decode_direct_n_mask | 63 | 63 | 63 |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 1 |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 1 |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 197 |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 32 |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 8.602255e+08 | 8.602255e+08 | 8.602255e+08 |
| w4u8_decode_k_pair_row4_call_count | 112 | 112 | 112 |
| w4u8_decode_k_temp_carrier_skipped_count | 224 | 224 | 224 |
| w4u8_decode_k_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 32 |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 16 |
| w4u8_decode_projection_mode | 1 | 1 | 1 |
| w4u8_decode_q_pair_row4_call_count | 224 | 224 | 224 |
| w4u8_decode_q_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 4 |
| w4u8_decode_qk_padding_poison | 0 | 0 | 0 |
| w4u8_decode_qk_rows_processed | 2688 | 2688 | 2688 |
| w4u8_decode_softmax_hvx_tile4_call_count | 224 | 224 | 224 |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | 0 |
| w4u8_decode_softmax_mode | 1 | 1 | 1 |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | 0 |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | 0 |
| w4u8_decode_swiglu_row4_call_count | 5376 | 5376 | 5376 |
| w4u8_decode_swiglu_rows | 4 | 4 | 4 |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | 0 |
| w4u8_decode_swiglu_vector_count | 5376 | 5376 | 5376 |
| w4u8_delta_reconstruction_mode | 0 | 0 | 0 |
| w4u8_final_residual_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_final_residual_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_final_residual_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_final_residual_task_count | 0 | 0 | 0 |
| w4u8_final_residual_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_gate_up_swiglu_consume_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_join_wait_ticks (us) | 130.6576 | 178.7156 | 178.3361 |
| w4u8_gate_up_swiglu_overlap_observed | 1 | 1 | 1 |
| w4u8_gate_up_swiglu_publish_count | 168 | 168 | 168 |
| w4u8_gate_up_swiglu_ready_wait_ticks (us) | 5273.818 | 5041.515 | 5036.613 |
| w4u8_gate_up_swiglu_worker_ticks (us) | 725.249 | 997.1552 | 996.8356 |
| w4u8_input_norm_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_input_norm_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_input_norm_task_count | 0 | 0 | 0 |
| w4u8_input_norm_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_activation_work_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 224 |
| w4u8_mlp_down_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_expanded_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_gate_up_pipeline_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_compute_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_hmx_ready_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_producer_slot_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_mlp_weight_stage_ticks (us) | 0 | 0 | 0 |
| w4u8_o_batch_count | 112 | 112 | 112 |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 16 |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | 28 |
| w4u8_o_gate_prefetch_lifetime_ticks (us) | 532.3451 | 522.7641 | 523.126 |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | 28 |
| w4u8_o_gate_prefetch_wait_ticks (us) | 324.8109 | 144.4915 | 144.9017 |
| w4u8_post_residual_direct_row4_call_count | 28 | 0 | 0 |
| w4u8_post_residual_main_work_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_pool_wait_ticks (us) | 0 | 0 | 0 |
| w4u8_post_residual_task_count | 0 | 0 | 0 |
| w4u8_post_residual_worker_work_ticks (us) | 0 | 0 | 0 |
| w4u8_prefill_cache_mode | 1 | 1 | 1 |
| w4u8_qk_norm_rope_rows_observed | 4 | 4 | 4 |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | 0 |
| w4u8_qkv_ring_batch_count | 168 | 168 | 168 |
| w4u8_qkv_ring_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_dma_wait_ticks (us) | 2151.959 | 2150.076 | 2154.282 |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | 0 |
| w4u8_qkv_ring_head_publish_count | 672 | 672 | 672 |
| w4u8_qkv_ring_hmx_compute_ticks (us) | 554.6613 | 554.241 | 548.403 |
| w4u8_qkv_ring_hmx_dispatch_count | 28 | 28 | 28 |
| w4u8_qkv_ring_hmx_ready_wait_ticks (us) | 1464.4 | 1458.298 | 1471.21 |
| w4u8_qkv_ring_pipeline_ticks (us) | 2472.229 | 2471.859 | 2474.659 |
| w4u8_qkv_ring_pool_wait_ticks (us) | 3.764063 | 3.766319 | 3.771528 |
| w4u8_qkv_ring_prep_worker_count | 5 | 5 | 5 |
| w4u8_qkv_ring_producer_slot_wait_ticks (us) | 9.200868 | 10.28403 | 9.320486 |
| w4u8_qkv_ring_slot_count | 2 | 2 | 2 |
| w4u8_qkvo_hmx_lifetime_ticks (us) | 12492.49 | 12530.75 | 12534.17 |
| w4u8_qkvo_prefetch_wait_ticks (us) | 2153.034 | 2151.148 | 2155.32 |
| w4u8_qkvo_weight_expand_ticks (us) | 0 | 0 | 0 |
| w4u8_residual_active_contexts | 0 | 0 | 0 |
| w4u8_swiglu_rows_observed | 4 | 4 | 4 |
| weight_ddr_read_bytes | 8.660326e+08 | 8.660326e+08 | 8.660326e+08 |
| weight_dma_descriptor_count | 1830 | 1830 | 1830 |
| weight_dma_ticks (us) | 15682.39 | 15669.18 | 15689.4 |
| wide_score_mode | 4 | 4 | 4 |

## Same-session prior FP32 diagnostic

Prior FP32/integer prefill ratio1.097978170,95CI[1.095060869,1.101093893]: point estimate below10percent, confidence gate fails narrowly. Optimized1.089965757,95CI[1.087257958,1.092578145]: passes. Thus do not attribute all change from historical10.718percent to optimization; matched old->new prefill wall improves0.729742percent,95CI ratio[0.989440347,0.995995624]. Old/new decode is statistically tied.
