# Explicit QDQ four-model diagnostic

EXP-0312 and L32-0073; completed 2026-09-24. Original G evidence retained. New final worksheet N_FP与QDQ拆分 in the rebuilt desktop master.

Ten formal paired repeat10 rounds/model after five short rounds. Fixed M64 prefill and one subsequent decode. All figures below are full-model prefill milliseconds, means of round means. QDQ contains input preparation and output scale/round/layout publication.

| Model | Fused Host | Explicit Host | FP | DQ/prep | Q/publish | QDQ / explicit Host |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 49.430566 | 57.052282 | 12.567025 | 4.054676 | 4.846249 | 15.6013% |
| Qwen3-1.7B | 75.998054 | 90.834413 | 23.211254 | 7.419820 | 9.194006 | 18.2902% |
| Llama3.2-1B | 47.582418 | 58.489101 | 18.286077 | 5.792327 | 7.045796 | 21.9496% |
| Llama3.2-3B | 83.503767 | 103.725715 | 31.494575 | 10.091545 | 12.499269 | 21.7794% |

Explicit timers include FP32 intermediate materialization, dispatch and join. They are not a retrospective decomposition of the fused control or production-pipeline critical-path attribution. FP32 intermediate buffers remain in 8 MiB VTCM with no timed intermediate DDR or spill. Softmax DQ/input preparation deliberately preserves integer max/centering to keep rounding exact. SwiGLU 2048-element tiling changes scheduling overhead. Only Softmax and SwiGLU are split; other boundaries, Norm, RoPE, residual remain unchanged. Decode keeps fused Softmax, so its single-step E2E is auxiliary and no full decode split is claimed.

Numerics: exact historical independent-reference layer/head checks at 1/3/full layers, plus byte-exact cross-arm fullmodel hidden/norm/KV payloads. Device model hashes match retained historical manifests. No PPL/model-quality claim, recipe or baseline promotion.

Routine repair log: first Qwen build invocation lacked execute permission, rerun through bash. Runner syntax newline defect fixed before hardware. First Llama1B fused audit falsely flagged historical hex-string vs integer representation; numerical hashes were identical, parser normalized types, failed raw evidence retained and all audits rerun. No threshold relaxation or accepted artifact hash changes.

Reproduction: source EXP0312 scripts/profile_exp0312.py and Llama tools/profile_explicit_qdq.py. Raw protocol.json files retain full commands, fixtures, source-sealed binaries and hashes. Windows ADB -P5038 used because default5037 served an empty WSL daemon. One device owner at a time.

## Qwen3-0.6B

Measured source: 51a00a484e7c0209cdc60441c08a5312c3386c9a

Full report: /mnt/d/llm_exp/results/qwen3-block-htp/exp0312/0.6B/REPORT.md

## Qwen3-1.7B

Measured source: 51a00a484e7c0209cdc60441c08a5312c3386c9a

Full report: /mnt/d/llm_exp/results/qwen3-block-htp/exp0312/1.7B/REPORT.md

## Llama3.2-1B

Measured source: 272298e3958e65bc9e9d89c0a7c9800fb8aad465

Full report: /mnt/d/llm_exp/results/llama32-htp/l32-0073/1B/REPORT.md

## Llama3.2-3B

Measured source: 272298e3958e65bc9e9d89c0a7c9800fb8aad465

Full report: /mnt/d/llm_exp/results/llama32-htp/l32-0073/3B/REPORT.md


# L32-0073 Llama3.2-1B explicit QDQ profiling

Full-model M64 prefill plus one fixed decode. Same binary, weights and token trajectory; alternating AB/BA. Five short paired repeat10 rounds and ten formal paired repeat10 rounds. Two warmup repeat1 measurements are auxiliary only. No model quality or speed acceptance gate. Original G data remains unchanged.

Fused control and explicit DQ/FP/Q preserve vector arithmetic, fixed scales, INT16 Down and FP32 residual. Prefill Softmax and SwiGLU are separated; decode Softmax remains fused, so no complete decode FP/QDQ split is claimed. DQ includes stable integer max/centering before float scaling. Q includes scaling, rounding and consumer-layout publication. Stage timers include materialization and dispatch/join. SwiGLU uses 2048-element tiles, three 16KiB phase scratch regions in dead VTCM. No timed DDR materialization. This is a new diagnostic schedule, not a retrospective decomposition of the historical fused cost.

## prefill

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.118197 (0.248%) | 0.119620 (0.205%) |
| Input RMSNorm | 1.342286 (2.821%) | 1.341814 (2.294%) |
| QKV + Q/K preparation | 3.853307 (8.098%) | 3.851976 (6.586%) |
| QK–Softmax–AV | 9.448278 (19.857%) | 9.754246 (16.677%) |
| O projection | 1.207202 (2.537%) | 1.202481 (2.056%) |
| Post-attention residual＋RMSNorm | 1.310801 (2.755%) | 1.310235 (2.240%) |
| Gate/Up＋SwiGLU | 22.032287 (46.303%) | 32.729513 (55.958%) |
| Down | 3.034650 (6.378%) | 3.038262 (5.195%) |
| Final residual | 0.002002 (0.004%) | 0.001971 (0.003%) |
| KV carrier conversion | 0.026714 (0.056%) | 0.026625 (0.046%) |
| KV append DMA | 0.112020 (0.235%) | 0.112021 (0.192%) |
| Block orchestration | 0.026345 (0.055%) | 0.025982 (0.044%) |
| Layer bookkeeping | 0.013532 (0.028%) | 0.013204 (0.023%) |
| Stage-boundary bookkeeping | 0.005814 (0.012%) | 0.005594 (0.010%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.087761 (0.184%) | 0.085924 (0.147%) |
| Embedding | 0.056105 (0.118%) | 0.056187 (0.096%) |
| Final model RMSNorm | 0.058482 (0.123%) | 0.058478 (0.100%) |
| LM head＋greedy（不含 final norm） | 3.333095 (7.005%) | 3.360880 (5.746%) |
| Host-DSP boundary | 1.513539 (3.181%) | 1.394088 (2.384%) |
| Host wall | 47.582418 | 58.489101 |
| E2E token/s | 1345.0346 | 1094.2210 |

Paired split/fused Host wall: {'wall_ratio': 1.2292166704266725, 'ci95': [1.2227759538744523, 1.2347536127686896]}

## decode_one_step

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.116047 (0.520%) | 0.116764 (0.510%) |
| Input RMSNorm | 0.891973 (3.999%) | 0.892021 (3.899%) |
| QKV + Q/K preparation | 1.656798 (7.427%) | 1.657473 (7.244%) |
| QK–Softmax–AV | 4.701255 (21.075%) | 4.697909 (20.532%) |
| O projection | 0.890632 (3.993%) | 0.885906 (3.872%) |
| Post-attention residual＋RMSNorm | 0.876747 (3.930%) | 0.876854 (3.832%) |
| Gate/Up＋SwiGLU | 5.547096 (24.867%) | 6.232587 (27.240%) |
| Down | 2.574453 (11.541%) | 2.579912 (11.276%) |
| Final residual | 0.001220 (0.005%) | 0.001240 (0.005%) |
| KV carrier conversion | 0.048465 (0.217%) | 0.048433 (0.212%) |
| KV append DMA | 0.092019 (0.413%) | 0.092259 (0.403%) |
| Block orchestration | 0.018658 (0.084%) | 0.018201 (0.080%) |
| Layer bookkeeping | 0.010033 (0.045%) | 0.010020 (0.044%) |
| Stage-boundary bookkeeping | 0.001334 (0.006%) | 0.001345 (0.006%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.052401 (0.235%) | 0.052557 (0.230%) |
| Embedding | 0.001745 (0.008%) | 0.001699 (0.007%) |
| Final model RMSNorm | 0.055376 (0.248%) | 0.055307 (0.242%) |
| LM head＋greedy（不含 final norm） | 3.322290 (14.894%) | 3.351959 (14.650%) |
| Host-DSP boundary | 1.448338 (6.493%) | 1.308190 (5.717%) |
| Host wall | 22.306882 | 22.880634 |
| E2E token/s | 44.8292 | 43.7051 |

Paired split/fused Host wall: {'wall_ratio': 1.0257208759675231, 'ci95': [1.019606166308473, 1.0317809285029877]}

## warmup scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 48477500 | 59819531 | 23.3965% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 256 | 256 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 1048576 | 1048576 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 2229 | 2231 | 0.0897% |
| scan_cache_pack_ticks | 502 | 502 | 0.0000% |
| block_orchestration_ticks | 473 | 489 | 3.3827% |
| layer_bookkeeping_ticks | 263 | 246 | -6.4639% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 881705 | 1096611 | 24.3739% |
| invocation_ticks | 882491 | 1097388 | 24.3512% |
| runtime_setup_ticks | 786 | 777 | -1.1450% |
| runtime_teardown_ticks | 813 | 772 | -5.0431% |
| stage_boundary_ticks | 162 | 155 | -4.3210% |
| ledger_named_ticks | 882491 | 1097388 | 24.3512% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 3 | 6 | 100.0000% |
| metadata_stage_ticks | 2611 | 2645 | 1.3022% |
| input_norm_ticks | 25721 | 25771 | 0.1944% |
| qkv_projection_ticks | 28714 | 28545 | -0.5886% |
| qk_norm_rope_ticks | 45526 | 45492 | -0.0747% |
| attention_ticks | 180558 | 188249 | 4.2596% |
| o_projection_ticks | 23228 | 23193 | -0.1507% |
| post_attention_residual_ticks | 25147 | 25133 | -0.0557% |
| post_attention_norm_ticks | 9 | 11 | 22.2222% |
| gate_up_ticks | 95354 | 96124 | 0.8075% |
| activation_ticks | 327231 | 531779 | 62.5087% |
| down_ticks | 58541 | 58552 | 0.0188% |
| final_residual_ticks | 35 | 41 | 17.1429% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1236 | 1232 | -0.3236% |
| generation_final_norm_ticks | 1137 | 1130 | -0.6157% |
| generation_lm_head_ticks | 63349 | 65443 | 3.3055% |
| generation_lm_head_weight_dma_ticks | 48614 | 48363 | -0.5163% |
| generation_lm_head_scale_dma_ticks | 369 | 340 | -7.8591% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 52592 | 54412 | 3.4606% |
| generation_lm_head_argmax_ticks | 8362 | 8639 | 3.3126% |
| generation_lm_head_weight_dma_wait_ticks | 44755 | 44141 | -1.3719% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 4562 | 6636 | 45.4625% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 221897 | 222673 | 0.3497% |
| hmx_compute_ticks | 163275 | 162171 | -0.6762% |
| projection_pack_ticks | 172 | 168 | -2.3256% |
| projection_hmx_wait_ticks | 29751 | 29422 | -1.1058% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 60593 | 66420 | 9.6166% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 24 | 18 | -25.0000% |
| u8_attention_k_pack_ticks | 24103 | 24078 | -0.1037% |
| u8_attention_v_pack_ticks | 79129 | 79220 | 0.1150% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 40408 | 41485 | 2.6653% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 15499 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 40921 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10000 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 95472 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 309888 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 125045 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 235683 | 247861 | 5.1671% |
| u8_attention_av_hmx_ticks | 40619 | 41051 | 1.0635% |
| u8_attention_av_requant_ticks | 17437 | 17515 | 0.4473% |
| u8_attention_pipeline_wait_ticks | 175808 | 182414 | 3.7575% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 180883 | 181374 | 0.2714% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 3381 | 3381 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1473024 | 1473024 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 193 | 193 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2764544 | 2764544 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 23026823 | 23814792 | 3.4220% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 128 | 128 | 0.0000% |
| w4u8_av_requant_vector_count | 1024 | 1024 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 28 | 34 | 21.4286% |
| w4u8_o_gate_prefetch_lifetime_ticks | 16994 | 16971 | -0.1353% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3145728 | 3145728 | 0.0000% |
| scan_attention_overlay_required_bytes | 75008 | 75008 | 0.0000% |
| scan_cache_dma_descriptor_count | 512 | 512 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 1064960 | 1064960 | 0.0000% |
| scan_cache_ddr_write_bytes | 16384 | 16384 | 0.0000% |
| scan_cache_stage_ticks | 4318 | 4381 | 1.4590% |
| scan_cache_append_ticks | 1754 | 1790 | 2.0525% |
| scan_cache_pack_ticks | 943 | 928 | -1.5907% |
| block_orchestration_ticks | 362 | 347 | -4.1436% |
| layer_bookkeeping_ticks | 194 | 208 | 7.2165% |
| scan_dynamic_attention_ticks | 90145 | 90417 | 0.3017% |
| total_ticks | 397727 | 413267 | 3.9072% |
| invocation_ticks | 398257 | 413809 | 3.9050% |
| runtime_setup_ticks | 530 | 542 | 2.2642% |
| runtime_teardown_ticks | 483 | 479 | -0.8282% |
| stage_boundary_ticks | 30 | 25 | -16.6667% |
| ledger_named_ticks | 398257 | 413809 | 3.9050% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 2 | 1 | -50.0000% |
| metadata_stage_ticks | 2260 | 2293 | 1.4602% |
| input_norm_ticks | 17127 | 17118 | -0.0525% |
| qkv_projection_ticks | 28393 | 28307 | -0.3029% |
| qk_norm_rope_ticks | 3444 | 3441 | -0.0871% |
| attention_ticks | 90218 | 90490 | 0.3015% |
| o_projection_ticks | 17541 | 17015 | -2.9987% |
| post_attention_residual_ticks | 16833 | 16799 | -0.2020% |
| post_attention_norm_ticks | 8 | 7 | -12.5000% |
| gate_up_ticks | 87195 | 88095 | 1.0322% |
| activation_ticks | 18575 | 31756 | 70.9610% |
| down_ticks | 49200 | 49260 | 0.1220% |
| final_residual_ticks | 23 | 24 | 4.3478% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 30 | 34 | 13.3333% |
| generation_final_norm_ticks | 1068 | 1062 | -0.5618% |
| generation_lm_head_ticks | 63112 | 64850 | 2.7538% |
| generation_lm_head_weight_dma_ticks | 48484 | 48495 | 0.0227% |
| generation_lm_head_scale_dma_ticks | 365 | 336 | -7.9452% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 52617 | 54117 | 2.8508% |
| generation_lm_head_argmax_ticks | 8240 | 8503 | 3.1917% |
| generation_lm_head_weight_dma_wait_ticks | 44665 | 44315 | -0.7836% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 4695 | 6132 | 30.6070% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 231233 | 232379 | 0.4956% |
| hmx_compute_ticks | 120819 | 123302 | 2.0551% |
| projection_pack_ticks | 162 | 162 | 0.0000% |
| projection_hmx_wait_ticks | 13381 | 12563 | -6.1131% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 8505 | 8616 | 1.3051% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 22381 | 22404 | 0.1028% |
| u8_attention_v_pack_ticks | 39027 | 38963 | -0.1640% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 256 | 256 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 6080 | 6199 | 1.9572% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 6040 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 15200 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 8997 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6170 | 6150 | -0.3241% |
| u8_attention_av_hmx_ticks | 5939 | 6050 | 1.8690% |
| u8_attention_av_requant_ticks | 2116 | 2108 | -0.3781% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 168953 | 169401 | 0.2652% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 1589 | 1589 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1212928 | 1212928 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 130 | 130 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2506496 | 2506496 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
## short scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 47348953 | 58516864 | 23.5864% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 256 | 256 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 1048576 | 1048576 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 2146.4 | 2148.4 | 0.0932% |
| scan_cache_pack_ticks | 511.7 | 510.2 | -0.2931% |
| block_orchestration_ticks | 500.7 | 490.1 | -2.1170% |
| layer_bookkeeping_ticks | 257 | 248 | -3.5019% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 882994.8 | 1097470.9 | 24.2896% |
| invocation_ticks | 883949.3 | 1098477.7 | 24.2693% |
| runtime_setup_ticks | 955.5 | 929.5 | -2.7211% |
| runtime_teardown_ticks | 687.4 | 684.7 | -0.3928% |
| stage_boundary_ticks | 109 | 104.8 | -3.8532% |
| ledger_named_ticks | 883949.3 | 1098477.7 | 24.2693% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 4.3 | 4.2 | -2.3256% |
| metadata_stage_ticks | 2266.2 | 2292.2 | 1.1473% |
| input_norm_ticks | 25760.8 | 25763.9 | 0.0120% |
| qkv_projection_ticks | 28362.5 | 28288.2 | -0.2620% |
| qk_norm_rope_ticks | 45506.5 | 45516.3 | 0.0215% |
| attention_ticks | 181913 | 187668.9 | 3.1641% |
| o_projection_ticks | 23122.8 | 23110.5 | -0.0532% |
| post_attention_residual_ticks | 25155.9 | 25142.5 | -0.0533% |
| post_attention_norm_ticks | 11.8 | 11.6 | -1.6949% |
| gate_up_ticks | 95463.1 | 95683.5 | 0.2309% |
| activation_ticks | 327230.8 | 532683.1 | 62.7851% |
| down_ticks | 58293.5 | 58441.2 | 0.2534% |
| final_residual_ticks | 36.2 | 38.5 | 6.3536% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1071.4 | 1074.9 | 0.3267% |
| generation_final_norm_ticks | 1120.5 | 1123.4 | 0.2588% |
| generation_lm_head_ticks | 64156.9 | 65627.6 | 2.2923% |
| generation_lm_head_weight_dma_ticks | 48753.2 | 48539.8 | -0.4377% |
| generation_lm_head_scale_dma_ticks | 351.4 | 346.8 | -1.3090% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 53372.5 | 54666.1 | 2.4237% |
| generation_lm_head_argmax_ticks | 8441.5 | 8599.4 | 1.8705% |
| generation_lm_head_weight_dma_wait_ticks | 44790.7 | 44274.8 | -1.1518% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 5112.5 | 6613.2 | 29.3535% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 221569.1 | 221572.4 | 0.0015% |
| hmx_compute_ticks | 164568.4 | 166898.4 | 1.4158% |
| projection_pack_ticks | 170 | 173.8 | 2.2353% |
| projection_hmx_wait_ticks | 29423.7 | 29413.4 | -0.0350% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 60538.9 | 66380.9 | 9.6500% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 21.6 | 23.9 | 10.6481% |
| u8_attention_k_pack_ticks | 24033 | 23999.9 | -0.1377% |
| u8_attention_v_pack_ticks | 79188.8 | 79189 | 0.0003% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 41182.3 | 41124.8 | -0.1396% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 15486 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 40942.4 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 9959.2 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 95791.7 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 310170.6 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 125288 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 235795.4 | 247899.5 | 5.1333% |
| u8_attention_av_hmx_ticks | 41001.3 | 40938.2 | -0.1539% |
| u8_attention_av_requant_ticks | 17439.8 | 17413.6 | -0.1502% |
| u8_attention_pipeline_wait_ticks | 178750.3 | 179010.6 | 0.1456% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 180772.8 | 181075.7 | 0.1676% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 3381 | 3381 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1473024 | 1473024 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 193 | 193 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2764544 | 2764544 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 22075302 | 22975114 | 4.0761% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 128 | 128 | 0.0000% |
| w4u8_av_requant_vector_count | 1024 | 1024 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 30.6 | 31.8 | 3.9216% |
| w4u8_o_gate_prefetch_lifetime_ticks | 16995.2 | 16995.8 | 0.0035% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3145728 | 3145728 | 0.0000% |
| scan_attention_overlay_required_bytes | 75008 | 75008 | 0.0000% |
| scan_cache_dma_descriptor_count | 512 | 512 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 1064960 | 1064960 | 0.0000% |
| scan_cache_ddr_write_bytes | 16384 | 16384 | 0.0000% |
| scan_cache_stage_ticks | 4332.7 | 4380.4 | 1.1009% |
| scan_cache_append_ticks | 1771 | 1775.9 | 0.2767% |
| scan_cache_pack_ticks | 930.1 | 926.9 | -0.3440% |
| block_orchestration_ticks | 358.1 | 351.4 | -1.8710% |
| layer_bookkeeping_ticks | 191.8 | 197.2 | 2.8154% |
| scan_dynamic_attention_ticks | 90187.3 | 90178.5 | -0.0098% |
| total_ticks | 398799.8 | 413744.6 | 3.7474% |
| invocation_ticks | 399330.9 | 414281.5 | 3.7439% |
| runtime_setup_ticks | 530.4 | 532.1 | 0.3205% |
| runtime_teardown_ticks | 473.8 | 478.2 | 0.9287% |
| stage_boundary_ticks | 25.6 | 25.7 | 0.3906% |
| ledger_named_ticks | 399330.9 | 414281.5 | 3.7439% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 2.3 | 1.9 | -17.3913% |
| metadata_stage_ticks | 2237.7 | 2241.5 | 0.1698% |
| input_norm_ticks | 17129.3 | 17126 | -0.0193% |
| qkv_projection_ticks | 28246.6 | 28240.2 | -0.0227% |
| qk_norm_rope_ticks | 3457.3 | 3469.2 | 0.3442% |
| attention_ticks | 90261.5 | 90251.2 | -0.0114% |
| o_projection_ticks | 17105.2 | 16996.6 | -0.6349% |
| post_attention_residual_ticks | 16827.1 | 16828.7 | 0.0095% |
| post_attention_norm_ticks | 9 | 8.7 | -3.3333% |
| gate_up_ticks | 87423.6 | 87760.6 | 0.3855% |
| activation_ticks | 18538.1 | 31671.3 | 70.8444% |
| down_ticks | 49387.5 | 49430.2 | 0.0865% |
| final_residual_ticks | 24 | 23.9 | -0.4167% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 33.6 | 31.9 | -5.0595% |
| generation_final_norm_ticks | 1062.7 | 1061.5 | -0.1129% |
| generation_lm_head_ticks | 64284.4 | 65509.4 | 1.9056% |
| generation_lm_head_weight_dma_ticks | 48682.8 | 48395 | -0.5912% |
| generation_lm_head_scale_dma_ticks | 349.2 | 344.2 | -1.4318% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 53683.9 | 54708.6 | 1.9088% |
| generation_lm_head_argmax_ticks | 8394.2 | 8566.1 | 2.0478% |
| generation_lm_head_weight_dma_wait_ticks | 44697.8 | 44307.8 | -0.8725% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 5634.1 | 6643.2 | 17.9106% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 231891.7 | 231782.6 | -0.0470% |
| hmx_compute_ticks | 124762.1 | 127935.8 | 2.5438% |
| projection_pack_ticks | 161 | 161 | 0.0000% |
| projection_hmx_wait_ticks | 12859.2 | 12734.9 | -0.9666% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 8560 | 8619.1 | 0.6904% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 22362.3 | 22377.1 | 0.0662% |
| u8_attention_v_pack_ticks | 38955.4 | 38941.1 | -0.0367% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 256 | 256 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 6172.2 | 6161.8 | -0.1685% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 6095 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 15161 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 8938 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6124.3 | 6143.8 | 0.3184% |
| u8_attention_av_hmx_ticks | 5978.2 | 5953.4 | -0.4148% |
| u8_attention_av_requant_ticks | 2093.1 | 2090.6 | -0.1194% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 169157.8 | 169205.4 | 0.0281% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 1589 | 1589 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1212928 | 1212928 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 130 | 130 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2506496 | 2506496 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
## formal scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 47506740 | 58499810 | 23.1400% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 256 | 256 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 1048576 | 1048576 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 2151.55 | 2151.25 | -0.0139% |
| scan_cache_pack_ticks | 513.05 | 510.85 | -0.4288% |
| block_orchestration_ticks | 504.9 | 500.4 | -0.8913% |
| layer_bookkeeping_ticks | 260.45 | 253.4 | -2.7069% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 883004.2 | 1094807.2 | 23.9866% |
| invocation_ticks | 884022.2 | 1095833.8 | 23.9600% |
| runtime_setup_ticks | 978.6 | 951.7 | -2.7488% |
| runtime_teardown_ticks | 703.95 | 684.65 | -2.7417% |
| stage_boundary_ticks | 112.1 | 107.8 | -3.8359% |
| ledger_named_ticks | 884022.2 | 1095833.8 | 23.9600% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 4.35 | 3.9 | -10.3448% |
| metadata_stage_ticks | 2261.35 | 2297.9 | 1.6163% |
| input_norm_ticks | 25770.6 | 25765.95 | -0.0180% |
| qkv_projection_ticks | 28467.8 | 28471.5 | 0.0130% |
| qk_norm_rope_ticks | 45524.05 | 45514.2 | -0.0216% |
| attention_ticks | 181188.95 | 187285.45 | 3.3647% |
| o_projection_ticks | 23186.9 | 23079.8 | -0.4619% |
| post_attention_residual_ticks | 25151 | 25144.95 | -0.0241% |
| post_attention_norm_ticks | 12.25 | 11.8 | -3.6735% |
| gate_up_ticks | 95798.25 | 95815 | 0.0175% |
| activation_ticks | 327228.55 | 532686.4 | 62.7873% |
| down_ticks | 58222.4 | 58288.65 | 0.1138% |
| final_residual_ticks | 38.6 | 37.85 | -1.9430% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1078.3 | 1078.8 | 0.0464% |
| generation_final_norm_ticks | 1123.5 | 1122.15 | -0.1202% |
| generation_lm_head_ticks | 64254.6 | 64459.5 | 0.3189% |
| generation_lm_head_weight_dma_ticks | 48755 | 48663.25 | -0.1882% |
| generation_lm_head_scale_dma_ticks | 351.9 | 349.35 | -0.7246% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 53463.3 | 53634.85 | 0.3209% |
| generation_lm_head_argmax_ticks | 8408.15 | 8447.2 | 0.4644% |
| generation_lm_head_weight_dma_wait_ticks | 44749 | 44640.9 | -0.2416% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 5202.35 | 5445.45 | 4.6729% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 222027.4 | 221912.2 | -0.0519% |
| hmx_compute_ticks | 164608.25 | 165632.7 | 0.6224% |
| projection_pack_ticks | 173.35 | 173.1 | -0.1442% |
| projection_hmx_wait_ticks | 29077.95 | 29175.15 | 0.3343% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 60544.6 | 66390.1 | 9.6549% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 24.85 | 22.9 | -7.8471% |
| u8_attention_k_pack_ticks | 24012.15 | 24031.7 | 0.0814% |
| u8_attention_v_pack_ticks | 79192.75 | 79159.4 | -0.0421% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 40862.15 | 40880 | 0.0437% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 15484.9 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 40946.45 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 9965.25 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 95738.8 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 310204.4 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 125322.2 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 235778.4 | 247879.65 | 5.1325% |
| u8_attention_av_hmx_ticks | 40679.85 | 40761.7 | 0.2012% |
| u8_attention_av_requant_ticks | 17442.95 | 17415.4 | -0.1579% |
| u8_attention_pipeline_wait_ticks | 178157.2 | 179141.55 | 0.5525% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 181027.3 | 180861.8 | -0.0914% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 3381 | 3381 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1473024 | 1473024 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 193 | 193 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2764544 | 2764544 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 22302099 | 22872271 | 2.5566% |
| output_mismatches | 1 | 1 | 0.0000% |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 0 | 0 | N/A (zero control) |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 64 | 64 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 128 | 128 | 0.0000% |
| w4u8_av_requant_vector_count | 1024 | 1024 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 16 | 16 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 32.2 | 33.3 | 3.4161% |
| w4u8_o_gate_prefetch_lifetime_ticks | 16999.75 | 16996.4 | -0.0197% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 4096 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3145728 | 3145728 | 0.0000% |
| scan_attention_overlay_required_bytes | 75008 | 75008 | 0.0000% |
| scan_cache_dma_descriptor_count | 512 | 512 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 1064960 | 1064960 | 0.0000% |
| scan_cache_ddr_write_bytes | 16384 | 16384 | 0.0000% |
| scan_cache_stage_ticks | 4340.65 | 4326.45 | -0.3271% |
| scan_cache_append_ticks | 1768.05 | 1771.2 | 0.1782% |
| scan_cache_pack_ticks | 930.35 | 929.3 | -0.1129% |
| block_orchestration_ticks | 357.7 | 350.35 | -2.0548% |
| layer_bookkeeping_ticks | 191.65 | 192.2 | 0.2870% |
| scan_dynamic_attention_ticks | 90241.75 | 90068.75 | -0.1917% |
| total_ticks | 399236.75 | 412593.55 | 3.3456% |
| invocation_ticks | 399769.75 | 413126.3 | 3.3411% |
| runtime_setup_ticks | 530.9 | 532.8 | 0.3579% |
| runtime_teardown_ticks | 475.15 | 476.2 | 0.2210% |
| stage_boundary_ticks | 25.7 | 25.6 | -0.3891% |
| ledger_named_ticks | 399769.75 | 413126.3 | 3.3411% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 1.7 | 2 | 17.6471% |
| metadata_stage_ticks | 2223.6 | 2239.7 | 0.7241% |
| input_norm_ticks | 17125.85 | 17125.7 | -0.0009% |
| qkv_projection_ticks | 28347 | 28361.5 | 0.0512% |
| qk_norm_rope_ticks | 3465.55 | 3468.65 | 0.0895% |
| attention_ticks | 90315.4 | 90144.6 | -0.1891% |
| o_projection_ticks | 17093.3 | 16955.3 | -0.8073% |
| post_attention_residual_ticks | 16826.05 | 16826.95 | 0.0053% |
| post_attention_norm_ticks | 8.25 | 8.9 | 7.8788% |
| gate_up_ticks | 88186.05 | 87900.35 | -0.3240% |
| activation_ticks | 18523.65 | 31711.6 | 71.1952% |
| down_ticks | 49443.65 | 49558 | 0.2313% |
| final_residual_ticks | 23.25 | 23.9 | 2.7957% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 33.65 | 33 | -1.9316% |
| generation_final_norm_ticks | 1062.65 | 1061.4 | -0.1176% |
| generation_lm_head_ticks | 63909.4 | 64135.4 | 0.3536% |
| generation_lm_head_weight_dma_ticks | 48842.4 | 48639.65 | -0.4151% |
| generation_lm_head_scale_dma_ticks | 346.6 | 346.9 | 0.0866% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 53358.95 | 53515.8 | 0.2940% |
| generation_lm_head_argmax_ticks | 8323.85 | 8388.95 | 0.7821% |
| generation_lm_head_weight_dma_wait_ticks | 44918.5 | 44667 | -0.5599% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 4972.4 | 5306.25 | 6.7141% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 501 | 501 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 500 | 500 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.3236019e+08 | 1.3236019e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1333 | 1333 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 128 | 128 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 6.1787341e+08 | 6.1787341e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.2357468e+09 | 1.2357468e+09 | 0.0000% |
| weight_dma_ticks | 232816 | 232491.1 | -0.1396% |
| hmx_compute_ticks | 124837.05 | 126709 | 1.4995% |
| projection_pack_ticks | 159.9 | 161.35 | 0.9068% |
| projection_hmx_wait_ticks | 12799.6 | 12714.45 | -0.6653% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 8578.9 | 8558.5 | -0.2378% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 22375.7 | 22369.9 | -0.0259% |
| u8_attention_v_pack_ticks | 38956.15 | 38949.7 | -0.0166% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 256 | 256 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 6180.35 | 6082.9 | -1.5768% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 6095 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 15191.6 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 8931.2 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 499 | 499 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6118.35 | 6143.6 | 0.4127% |
| u8_attention_av_hmx_ticks | 5966.35 | 5911.6 | -0.9176% |
| u8_attention_av_requant_ticks | 2092.65 | 2092.85 | 0.0096% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 169656.85 | 169544.65 | -0.0661% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8098272 | 8098272 | 0.0000% |
| block_invocation_count | 16 | 16 | 0.0000% |
| hmx_command_count | 1589 | 1589 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1212928 | 1212928 | 0.0000% |
| weight_dma_descriptor_count | 2166 | 2166 | 0.0000% |
| boundary_dma_descriptor_count | 130 | 130 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 6.2191821e+08 | 6.2191821e+08 | 0.0000% |
| boundary_ddr_read_bytes | 2506496 | 2506496 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |

## Provenance and gates

{
  "audits": [
    {
      "file": "audit-full-fused/hash-checks.json",
      "checks": 32,
      "normalized_numeric_match": true
    },
    {
      "file": "audit-full-split/hash-checks.json",
      "checks": 32,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-fused/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-fused-a02/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-split-a02/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit3-split-a02/hash-checks.json",
      "checks": 6,
      "normalized_numeric_match": true
    }
  ],
  "exact_cross_arm_payload_files": 36,
  "formal_token_invocations": 400,
  "short_token_invocations": 200,
  "requested_and_granted_vtcm": 8388608,
  "no_intermediate_ddr": true,
  "no_spill": true,
  "quality_claim": false,
  "baseline_promoted": false
}

Sealed binaries:

[
  {
    "path": "binaries-1-a01/seal.json",
    "model_size": "1B",
    "layer_count": "1",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "3003c06587169d33472d48e8b30b9f47e9e3e389",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "412054d4b7c13215237b97e38564d497ee1973116d4baed76bc3e99983fd830c",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "25e3e52a0379e71745dea4035d44451f1239379ecb55cdf081a8525c3f5f9035"
    }
  },
  {
    "path": "binaries-1-a02/seal.json",
    "model_size": "1B",
    "layer_count": "1",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "412054d4b7c13215237b97e38564d497ee1973116d4baed76bc3e99983fd830c",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "25e3e52a0379e71745dea4035d44451f1239379ecb55cdf081a8525c3f5f9035"
    }
  },
  {
    "path": "binaries-16-a02/seal.json",
    "model_size": "1B",
    "layer_count": "16",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "fea04f5c3fbba94017785df42629a442e1ee2513ef2d510cc1424e61fae981d4",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "95908c591eb17f12a71bfb6ef4ca9c70db94fb6a37ea64fd5778e159c5082c07"
    }
  },
  {
    "path": "binaries-3-a02/seal.json",
    "model_size": "1B",
    "layer_count": "3",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "62bd37bdf72929d807304449c5e1864a0b46ea316584273d2d1b212db0509896",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "04a5804e29b7dc668d7dfa6fd69b5d430683a5b178d8e958867d7b6ddba46ac6"
    }
  }
]

Historical G fused measurement is preserved. No W16A16/W4A16 comparison measured: outside this diagnostic request.


# L32-0073 Llama3.2-3B explicit QDQ profiling

Full-model M64 prefill plus one fixed decode. Same binary, weights and token trajectory; alternating AB/BA. Five short paired repeat10 rounds and ten formal paired repeat10 rounds. Two warmup repeat1 measurements are auxiliary only. No model quality or speed acceptance gate. Original G data remains unchanged.

Fused control and explicit DQ/FP/Q preserve vector arithmetic, fixed scales, INT16 Down and FP32 residual. Prefill Softmax and SwiGLU are separated; decode Softmax remains fused, so no complete decode FP/QDQ split is claimed. DQ includes stable integer max/centering before float scaling. Q includes scaling, rounding and consumer-layout publication. Stage timers include materialization and dispatch/join. SwiGLU uses 2048-element tiles, three 16KiB phase scratch regions in dead VTCM. No timed DDR materialization. This is a new diagnostic schedule, not a retrospective decomposition of the historical fused cost.

## prefill

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.240117 (0.288%) | 0.227060 (0.219%) |
| Input RMSNorm | 3.718384 (4.453%) | 3.713299 (3.580%) |
| QKV + Q/K preparation | 9.025057 (10.808%) | 8.996974 (8.674%) |
| QK–Softmax–AV | 7.727572 (9.254%) | 8.200545 (7.906%) |
| O projection | 3.155718 (3.779%) | 3.177189 (3.063%) |
| Post-attention residual＋RMSNorm | 3.576589 (4.283%) | 3.576248 (3.448%) |
| Gate/Up＋SwiGLU | 42.602843 (51.019%) | 62.260766 (60.024%) |
| Down | 7.737771 (9.266%) | 7.782916 (7.503%) |
| Final residual | 0.002922 (0.003%) | 0.002893 (0.003%) |
| KV carrier conversion | 0.081179 (0.097%) | 0.080047 (0.077%) |
| KV append DMA | 0.233892 (0.280%) | 0.235174 (0.227%) |
| Block orchestration | 0.040448 (0.048%) | 0.040072 (0.039%) |
| Layer bookkeeping | 0.022966 (0.028%) | 0.023301 (0.022%) |
| Stage-boundary bookkeeping | 0.006724 (0.008%) | 0.006656 (0.006%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.089258 (0.107%) | 0.087528 (0.084%) |
| Embedding | 0.074075 (0.089%) | 0.071157 (0.069%) |
| Final model RMSNorm | 0.082665 (0.099%) | 0.082294 (0.079%) |
| LM head＋greedy（不含 final norm） | 3.793292 (4.543%) | 3.772881 (3.637%) |
| Host-DSP boundary | 1.292294 (1.548%) | 1.388715 (1.339%) |
| Host wall | 83.503767 | 103.725715 |
| E2E token/s | 766.4325 | 617.0119 |

Paired split/fused Host wall: {'wall_ratio': 1.242168092454805, 'ci95': [1.240304669522667, 1.2439916722857387]}

## decode_one_step

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.239221 (0.564%) | 0.224853 (0.516%) |
| Input RMSNorm | 2.313830 (5.457%) | 2.314245 (5.314%) |
| QKV + Q/K preparation | 3.761207 (8.870%) | 3.743301 (8.596%) |
| QK–Softmax–AV | 5.712610 (13.473%) | 5.713725 (13.120%) |
| O projection | 2.871854 (6.773%) | 2.880388 (6.614%) |
| Post-attention residual＋RMSNorm | 2.326726 (5.487%) | 2.326519 (5.342%) |
| Gate/Up＋SwiGLU | 13.634009 (32.154%) | 14.803309 (33.993%) |
| Down | 6.130485 (14.458%) | 6.123124 (14.061%) |
| Final residual | 0.002084 (0.005%) | 0.002106 (0.005%) |
| KV carrier conversion | 0.013047 (0.031%) | 0.012761 (0.029%) |
| KV append DMA | 0.178201 (0.420%) | 0.167626 (0.385%) |
| Block orchestration | 0.031241 (0.074%) | 0.030769 (0.071%) |
| Layer bookkeeping | 0.017883 (0.042%) | 0.017658 (0.041%) |
| Stage-boundary bookkeeping | 0.001933 (0.005%) | 0.001920 (0.004%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.053256 (0.126%) | 0.053222 (0.122%) |
| Embedding | 0.002096 (0.005%) | 0.002009 (0.005%) |
| Final model RMSNorm | 0.081756 (0.193%) | 0.081565 (0.187%) |
| LM head＋greedy（不含 final norm） | 3.803385 (8.970%) | 3.771607 (8.661%) |
| Host-DSP boundary | 1.226988 (2.894%) | 1.277661 (2.934%) |
| Host wall | 42.401813 | 43.548370 |
| E2E token/s | 23.5839 | 22.9630 |

Paired split/fused Host wall: {'wall_ratio': 1.0270402918983865, 'ci95': [1.0231395008314972, 1.0308169643552145]}

## warmup scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 84829427 | 1.0484484e+08 | 23.5949% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 448 | 448 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 3670016 | 3670016 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4431 | 4439 | 0.1805% |
| scan_cache_pack_ticks | 1560 | 1542 | -1.1538% |
| block_orchestration_ticks | 750 | 747 | -0.4000% |
| layer_bookkeeping_ticks | 443 | 474 | 6.9977% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1578677 | 1969547 | 24.7593% |
| invocation_ticks | 1579474 | 1970355 | 24.7475% |
| runtime_setup_ticks | 797 | 808 | 1.3802% |
| runtime_teardown_ticks | 769 | 792 | 2.9909% |
| stage_boundary_ticks | 139 | 145 | 4.3165% |
| ledger_named_ticks | 1579474 | 1970355 | 24.7475% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 3 | 5 | 66.6667% |
| metadata_stage_ticks | 4840 | 4769 | -1.4669% |
| input_norm_ticks | 71732 | 71817 | 0.1185% |
| qkv_projection_ticks | 65625 | 65873 | 0.3779% |
| qk_norm_rope_ticks | 108136 | 108146 | 0.0092% |
| attention_ticks | 148450 | 157332 | 5.9832% |
| o_projection_ticks | 59272 | 59457 | 0.3121% |
| post_attention_residual_ticks | 68672 | 68597 | -0.1092% |
| post_attention_norm_ticks | 21 | 21 | 0.0000% |
| gate_up_ticks | 248207 | 248255 | 0.0193% |
| activation_ticks | 572440 | 954205 | 66.6908% |
| down_ticks | 148023 | 148205 | 0.1230% |
| final_residual_ticks | 51 | 62 | 21.5686% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1535 | 1521 | -0.9121% |
| generation_final_norm_ticks | 1597 | 1593 | -0.2505% |
| generation_lm_head_ticks | 73578 | 73143 | -0.5912% |
| generation_lm_head_weight_dma_ticks | 63682 | 63273 | -0.6423% |
| generation_lm_head_scale_dma_ticks | 369 | 411 | 11.3821% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63291 | 62832 | -0.7252% |
| generation_lm_head_argmax_ticks | 7476 | 7497 | 0.2809% |
| generation_lm_head_weight_dma_wait_ticks | 62300 | 61902 | -0.6388% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 318 | 242 | -23.8994% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 393472 | 393472 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 599105 | 599077 | -0.0047% |
| hmx_compute_ticks | 325834 | 327246 | 0.4333% |
| projection_pack_ticks | 268 | 266 | -0.7463% |
| projection_hmx_wait_ticks | 53908 | 54083 | 0.3246% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 80488 | 89321 | 10.9743% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 47 | 44 | -6.3830% |
| u8_attention_k_pack_ticks | 36533 | 36599 | 0.1807% |
| u8_attention_v_pack_ticks | 66968 | 66953 | -0.0224% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 11994 | 12095 | 0.8421% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 20727 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 54778 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 13816 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 173284 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 551781 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 226388 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 310921 | 328422 | 5.6288% |
| u8_attention_av_hmx_ticks | 11922 | 11652 | -2.2647% |
| u8_attention_av_requant_ticks | 47364 | 47380 | 0.0338% |
| u8_attention_pipeline_wait_ticks | 51014 | 51665 | 1.2761% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 471390 | 472466 | 0.2283% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3836160 | 3836160 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 5352832 | 5352832 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 43964063 | 44867292 | 2.0545% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 2688 | 2688 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 48 | 40 | -16.6667% |
| w4u8_o_gate_prefetch_lifetime_ticks | 44997 | 44884 | -0.2511% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3295232 | 3295232 | 0.0000% |
| scan_attention_overlay_required_bytes | 57088 | 57088 | 0.0000% |
| scan_cache_dma_descriptor_count | 896 | 896 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 3727360 | 3727360 | 0.0000% |
| scan_cache_ddr_write_bytes | 57344 | 57344 | 0.0000% |
| scan_cache_stage_ticks | 8151 | 8267 | 1.4231% |
| scan_cache_append_ticks | 3811 | 3826 | 0.3936% |
| scan_cache_pack_ticks | 254 | 241 | -5.1181% |
| block_orchestration_ticks | 595 | 569 | -4.3697% |
| layer_bookkeeping_ticks | 331 | 334 | 0.9063% |
| scan_dynamic_attention_ticks | 109077 | 109714 | 0.5840% |
| total_ticks | 793796 | 815223 | 2.6993% |
| invocation_ticks | 794440 | 815764 | 2.6842% |
| runtime_setup_ticks | 644 | 541 | -15.9938% |
| runtime_teardown_ticks | 490 | 473 | -3.4694% |
| stage_boundary_ticks | 39 | 36 | -7.6923% |
| ledger_named_ticks | 794440 | 815764 | 2.6842% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 2 | 4 | 100.0000% |
| metadata_stage_ticks | 4923 | 4845 | -1.5844% |
| input_norm_ticks | 44408 | 44422 | 0.0315% |
| qkv_projection_ticks | 65713 | 65566 | -0.2237% |
| qk_norm_rope_ticks | 7471 | 7234 | -3.1723% |
| attention_ticks | 109184 | 109820 | 0.5825% |
| o_projection_ticks | 54451 | 54391 | -0.1102% |
| post_attention_residual_ticks | 44737 | 44630 | -0.2392% |
| post_attention_norm_ticks | 16 | 16 | 0.0000% |
| gate_up_ticks | 233135 | 231721 | -0.6065% |
| activation_ticks | 32223 | 55647 | 72.6934% |
| down_ticks | 118223 | 118319 | 0.0812% |
| final_residual_ticks | 51 | 45 | -11.7647% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 40 | 41 | 2.5000% |
| generation_final_norm_ticks | 1580 | 1584 | 0.2532% |
| generation_lm_head_ticks | 73699 | 73043 | -0.8901% |
| generation_lm_head_weight_dma_ticks | 63736 | 62970 | -1.2018% |
| generation_lm_head_scale_dma_ticks | 379 | 415 | 9.4987% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63367 | 62668 | -1.1031% |
| generation_lm_head_argmax_ticks | 7638 | 7636 | -0.0262% |
| generation_lm_head_weight_dma_wait_ticks | 62359 | 61644 | -1.1466% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 276 | 344 | 24.6377% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 6400 | 6400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 602259 | 598744 | -0.5836% |
| hmx_compute_ticks | 260178 | 266115 | 2.2819% |
| projection_pack_ticks | 248 | 253 | 2.0161% |
| projection_hmx_wait_ticks | 19641 | 20730 | 5.5445% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 14103 | 14223 | 0.8509% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 31612 | 31609 | -0.0095% |
| u8_attention_v_pack_ticks | 28182 | 28266 | 0.2981% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 448 | 448 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 12111 | 12368 | 2.1220% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 10778 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 26263 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 16012 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 7448 | 7464 | 0.2148% |
| u8_attention_av_hmx_ticks | 12563 | 12715 | 1.2099% |
| u8_attention_av_requant_ticks | 3165 | 3175 | 0.3160% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 461852 | 460425 | -0.3090% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3142656 | 3142656 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 4965760 | 4965760 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
## short scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 83330714 | 1.0375698e+08 | 24.5123% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 448 | 448 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 3670016 | 3670016 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4508.8 | 4496.4 | -0.2750% |
| scan_cache_pack_ticks | 1553.1 | 1539.3 | -0.8885% |
| block_orchestration_ticks | 756.9 | 775.7 | 2.4838% |
| layer_bookkeeping_ticks | 425.1 | 445.6 | 4.8224% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1573902.6 | 1963275.1 | 24.7393% |
| invocation_ticks | 1574828.6 | 1964200.5 | 24.7247% |
| runtime_setup_ticks | 915.7 | 991.3 | 8.2560% |
| runtime_teardown_ticks | 683.4 | 714.8 | 4.5947% |
| stage_boundary_ticks | 120 | 132.7 | 10.5833% |
| ledger_named_ticks | 1574828.6 | 1964200.5 | 24.7247% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 5.1 | 5.3 | 3.9216% |
| metadata_stage_ticks | 4279.4 | 4311.1 | 0.7408% |
| input_norm_ticks | 71374.3 | 71305.8 | -0.0960% |
| qkv_projection_ticks | 64410.4 | 64577.3 | 0.2591% |
| qk_norm_rope_ticks | 108179.7 | 108110.3 | -0.0642% |
| attention_ticks | 148280.9 | 157574.5 | 6.2676% |
| o_projection_ticks | 60928.3 | 60657.7 | -0.4441% |
| post_attention_residual_ticks | 68640.9 | 68631.9 | -0.0131% |
| post_attention_norm_ticks | 22.2 | 23.5 | 5.8559% |
| gate_up_ticks | 243068 | 243462.7 | 0.1624% |
| activation_ticks | 572430.2 | 951853.5 | 66.2829% |
| down_ticks | 148997.5 | 149251.8 | 0.1707% |
| final_residual_ticks | 55.6 | 56.4 | 1.4388% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1339.9 | 1354.9 | 1.1195% |
| generation_final_norm_ticks | 1571.6 | 1582.2 | 0.6745% |
| generation_lm_head_ticks | 73750.5 | 73826.4 | 0.1029% |
| generation_lm_head_weight_dma_ticks | 63983.7 | 64011.3 | 0.0431% |
| generation_lm_head_scale_dma_ticks | 347.5 | 362.7 | 4.3741% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63556.6 | 63578.4 | 0.0343% |
| generation_lm_head_argmax_ticks | 7489.2 | 7495.8 | 0.0881% |
| generation_lm_head_weight_dma_wait_ticks | 62600.3 | 62625.5 | 0.0403% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 323 | 259.9 | -19.5356% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 393472 | 393472 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 588162 | 587815.5 | -0.0589% |
| hmx_compute_ticks | 339462.8 | 341878.2 | 0.7115% |
| projection_pack_ticks | 269.2 | 269.6 | 0.1486% |
| projection_hmx_wait_ticks | 59771 | 59492.1 | -0.4666% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 80554.6 | 89381.9 | 10.9582% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 42.3 | 40.5 | -4.2553% |
| u8_attention_k_pack_ticks | 36552.1 | 36606.5 | 0.1488% |
| u8_attention_v_pack_ticks | 66942 | 67027.8 | 0.1282% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 11976.3 | 12029.1 | 0.4409% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 20739.6 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 54775.4 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 13840.2 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 173230.8 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 549926.4 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 226226.1 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 310932.6 | 328400.7 | 5.6180% |
| u8_attention_av_hmx_ticks | 11714.2 | 11845.5 | 1.1209% |
| u8_attention_av_requant_ticks | 47385 | 47357.5 | -0.0580% |
| u8_attention_pipeline_wait_ticks | 50796.6 | 51636.2 | 1.6529% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 469036.6 | 469572.5 | 0.1143% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3836160 | 3836160 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 5352832 | 5352832 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 42322292 | 43542740 | 2.8837% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 2688 | 2688 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 44.1 | 44.2 | 0.2268% |
| w4u8_o_gate_prefetch_lifetime_ticks | 44910.2 | 44912 | 0.0040% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3295232 | 3295232 | 0.0000% |
| scan_attention_overlay_required_bytes | 57088 | 57088 | 0.0000% |
| scan_cache_dma_descriptor_count | 896 | 896 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 3727360 | 3727360 | 0.0000% |
| scan_cache_ddr_write_bytes | 57344 | 57344 | 0.0000% |
| scan_cache_stage_ticks | 8440.7 | 8424.6 | -0.1907% |
| scan_cache_append_ticks | 3161.4 | 3160.8 | -0.0190% |
| scan_cache_pack_ticks | 244.4 | 248.5 | 1.6776% |
| block_orchestration_ticks | 598.7 | 590.8 | -1.3195% |
| layer_bookkeeping_ticks | 340.7 | 341.2 | 0.1468% |
| scan_dynamic_attention_ticks | 109580.4 | 109666.6 | 0.0787% |
| total_ticks | 785133.6 | 810554 | 3.2377% |
| invocation_ticks | 785674.3 | 811096 | 3.2357% |
| runtime_setup_ticks | 540 | 544.7 | 0.8704% |
| runtime_teardown_ticks | 475.9 | 477.6 | 0.3572% |
| stage_boundary_ticks | 36.1 | 36.9 | 2.2161% |
| ledger_named_ticks | 785674.3 | 811096 | 3.2357% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 3.6 | 3 | -16.6667% |
| metadata_stage_ticks | 4248.1 | 4239.4 | -0.2048% |
| input_norm_ticks | 44432.9 | 44433.2 | 0.0007% |
| qkv_projection_ticks | 64225.2 | 64409.7 | 0.2873% |
| qk_norm_rope_ticks | 7262.7 | 7273.1 | 0.1432% |
| attention_ticks | 109693.5 | 109778.1 | 0.0771% |
| o_projection_ticks | 55106.2 | 54986.6 | -0.2170% |
| post_attention_residual_ticks | 44647.9 | 44650.9 | 0.0067% |
| post_attention_norm_ticks | 16.2 | 16.9 | 4.3210% |
| gate_up_ticks | 226778.3 | 227936.3 | 0.5106% |
| activation_ticks | 32424.4 | 56110.6 | 73.0505% |
| down_ticks | 117814.8 | 117971.2 | 0.1328% |
| final_residual_ticks | 40.5 | 40.4 | -0.2469% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 38.2 | 40.3 | 5.4974% |
| generation_final_norm_ticks | 1562 | 1566.2 | 0.2689% |
| generation_lm_head_ticks | 73578.9 | 74156.2 | 0.7846% |
| generation_lm_head_weight_dma_ticks | 63612 | 64263.3 | 1.0239% |
| generation_lm_head_scale_dma_ticks | 350.9 | 370 | 5.4431% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63290.3 | 63879.1 | 0.9303% |
| generation_lm_head_argmax_ticks | 7639.3 | 7638.1 | -0.0157% |
| generation_lm_head_weight_dma_wait_ticks | 62286.2 | 62941.7 | 1.0524% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 317.7 | 278.1 | -12.4646% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 6400 | 6400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 589685.4 | 590434.3 | 0.1270% |
| hmx_compute_ticks | 284147.2 | 282765.6 | -0.4862% |
| projection_pack_ticks | 250.5 | 247 | -1.3972% |
| projection_hmx_wait_ticks | 23418.2 | 22496.9 | -3.9341% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 14372.8 | 14380.8 | 0.0557% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 31683.6 | 31674.6 | -0.0284% |
| u8_attention_v_pack_ticks | 28203.4 | 28222.2 | 0.0667% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 448 | 448 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 12188.3 | 12306.3 | 0.9681% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 10841.9 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 26681.5 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 15973.8 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 7464.1 | 7485.6 | 0.2880% |
| u8_attention_av_hmx_ticks | 12740.1 | 12647.6 | -0.7261% |
| u8_attention_av_requant_ticks | 3163.1 | 3162.5 | -0.0190% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 454430.3 | 455919.6 | 0.3277% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3142656 | 3142656 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 4965760 | 4965760 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
## formal scalar diagnostics

All values use their field units; *_ticks are 19.2MHz timer ticks. Worker/engine/wait counters overlap and MUST NOT be added to module wall. Medians are over per-run means. No missing counters silently replaced by zero.

### prefill

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 0 | 0 | N/A (zero control) |
| logical_m | 64 | 64 | 0.0000% |
| first_position | 0 | 0 | N/A (zero control) |
| valid_length | 64 | 64 | 0.0000% |
| host_wall_ns | 83467404 | 1.0372457e+08 | 24.2696% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_call_count | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 64 | 64 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_vector_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 64 | 64 | 0.0000% |
| scan_total_kv_length | 64 | 64 | 0.0000% |
| scan_padded_kv_length | 64 | 64 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 0 | 0 | N/A (zero control) |
| scan_attention_overlay_required_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_dma_descriptor_count | 448 | 448 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_write_bytes | 3670016 | 3670016 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4501 | 4519.05 | 0.4010% |
| scan_cache_pack_ticks | 1558.3 | 1538 | -1.3027% |
| block_orchestration_ticks | 773.85 | 776.35 | 0.3231% |
| layer_bookkeeping_ticks | 435.45 | 446.85 | 2.6180% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1576584.3 | 1963720.9 | 24.5554% |
| invocation_ticks | 1577562.9 | 1964668.2 | 24.5382% |
| runtime_setup_ticks | 987.9 | 988.9 | 0.1012% |
| runtime_teardown_ticks | 721.15 | 711.1 | -1.3936% |
| stage_boundary_ticks | 129.25 | 128 | -0.9671% |
| ledger_named_ticks | 1577562.9 | 1964668.2 | 24.5382% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 5.3 | 5.85 | 10.3774% |
| metadata_stage_ticks | 4344.3 | 4280.5 | -1.4686% |
| input_norm_ticks | 71389.35 | 71287.2 | -0.1431% |
| qkv_projection_ticks | 64773.9 | 64595.65 | -0.2752% |
| qk_norm_rope_ticks | 108197.7 | 108108.9 | -0.0821% |
| attention_ticks | 148354.85 | 157473.45 | 6.1465% |
| o_projection_ticks | 60764.25 | 61073.6 | 0.5091% |
| post_attention_residual_ticks | 68648.2 | 68640.55 | -0.0111% |
| post_attention_norm_ticks | 24.15 | 23.45 | -2.8986% |
| gate_up_ticks | 244128.25 | 243571.15 | -0.2282% |
| activation_ticks | 572457.7 | 951766.55 | 66.2597% |
| down_ticks | 148736.55 | 149483.45 | 0.5022% |
| final_residual_ticks | 56.6 | 55.5 | -1.9435% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1368.75 | 1351.75 | -1.2420% |
| generation_final_norm_ticks | 1579.45 | 1577.35 | -0.1330% |
| generation_lm_head_ticks | 74228.95 | 74032.3 | -0.2649% |
| generation_lm_head_weight_dma_ticks | 64381.8 | 64214.65 | -0.2596% |
| generation_lm_head_scale_dma_ticks | 364.65 | 355.95 | -2.3858% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63987.85 | 63792.9 | -0.3047% |
| generation_lm_head_argmax_ticks | 7490.05 | 7492.7 | 0.0354% |
| generation_lm_head_weight_dma_wait_ticks | 62980.85 | 62835.3 | -0.2311% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 287.65 | 282.75 | -1.7035% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 393472 | 393472 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 590948.05 | 589594.7 | -0.2290% |
| hmx_compute_ticks | 337928.2 | 339723.15 | 0.5312% |
| projection_pack_ticks | 271.5 | 270.9 | -0.2210% |
| projection_hmx_wait_ticks | 58807.1 | 59933.2 | 1.9149% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 80575.15 | 89347.6 | 10.8873% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 41.1 | 40.8 | -0.7299% |
| u8_attention_k_pack_ticks | 36555.15 | 36608 | 0.1446% |
| u8_attention_v_pack_ticks | 66985.85 | 66990.55 | 0.0070% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 11966.05 | 11991.05 | 0.2089% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 20735.35 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 54786 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 13827.7 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 173004.1 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 549954.35 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 226142.7 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 310954.3 | 328422.3 | 5.6175% |
| u8_attention_av_hmx_ticks | 11694.3 | 11804.2 | 0.9398% |
| u8_attention_av_requant_ticks | 47385.25 | 47358.85 | -0.0557% |
| u8_attention_pipeline_wait_ticks | 50777 | 51356.2 | 1.1407% |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 470610.35 | 470115.65 | -0.1051% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3836160 | 3836160 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 5352832 | 5352832 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 42381508 | 43514953 | 2.6744% |
| output_mismatches | 0 | 0 | N/A (zero control) |
| output_max_abs | 0 | 0 | N/A (zero control) |
| output_cosine | 1 | 1 | 0.0000% |
| output_nrmse | 0 | 0 | N/A (zero control) |
| output_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| output_nonfinite_count | 0 | 0 | N/A (zero control) |
| output_max_required_rtol_after_atol | 0 | 0 | N/A (zero control) |
| output_fp16_atol | 0.0625 | 0.0625 | 0.0000% |
| output_fp16_rtol | 0.002 | 0.002 | 0.0000% |
| output_fp16_max_composed_nrmse | 0.003 | 0.003 | 0.0000% |
| output_max_lsb | 0 | 0 | N/A (zero control) |
| cache_prefix_mismatches | 0 | 0 | N/A (zero control) |
| cache_mismatches | 0 | 0 | N/A (zero control) |
| cache_structure_mismatches | 0 | 0 | N/A (zero control) |
| cache_min_cosine | 1 | 1 | 0.0000% |
| cache_max_mixed_tolerance_violation_fraction | 0 | 0 | N/A (zero control) |
| cache_max_nrmse | 0 | 0 | N/A (zero control) |
| cache_compared_elements | 0 | 0 | N/A (zero control) |
| cache_mixed_tolerance_violations | 0 | 0 | N/A (zero control) |
| cache_nonfinite_count | 0 | 0 | N/A (zero control) |
| cache_tensor_count | 0 | 0 | N/A (zero control) |
| cache_composed_cosine_diagnostic_failure_count | 0 | 0 | N/A (zero control) |
| cache_legacy_mixed_bound_failure_count | 0 | 0 | N/A (zero control) |
| cache_fp16_min_cosine | 0.99999 | 0.99999 | 0.0000% |
| cache_fp16_max_violation_fraction | 0.01 | 0.01 | 0.0000% |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 1 | 1 | 0.0000% |
| kv_cache_v_format | 1 | 1 | 0.0000% |
| w4u8_prefill_cache_mode | 0 | 0 | N/A (zero control) |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 168 | 168 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 2688 | 2688 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 64 | 64 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 0 | 0 | N/A (zero control) |
| w4u8_decode_direct_n_down_batch_n_tiles | 8 | 8 | 0.0000% |
| w4u8_decode_direct_n_down_single_dma | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_single_dma | 1 | 1 | 0.0000% |
| w4u8_qkv_ring_slot_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_prep_worker_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_batch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_task_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_dispatch_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_head_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_dma_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkv_ring_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_o_gate_prefetch_start_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_consume_count | 28 | 28 | 0.0000% |
| w4u8_o_gate_prefetch_wait_ticks | 43.6 | 44 | 0.9174% |
| w4u8_o_gate_prefetch_lifetime_ticks | 44923.45 | 44918.65 | -0.0107% |
| w4u8_gate_up_swiglu_publish_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_consume_count | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_overlap_observed | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_worker_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_gate_up_swiglu_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_rows | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_qk_norm_rope_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_pair_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_rows_processed | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_temp_carrier_skipped_count | 0 | 0 | N/A (zero control) |
| w4u8_qk_padding_poison_pair_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_q_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_decode_k_valid_row_hash | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_task_count | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_residual_active_contexts | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_task_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_main_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_worker_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_hvx_tile4_mismatch_count | 0 | 0 | N/A (zero control) |
| w4u8_swiglu_rows_observed | 4 | 4 | 0.0000% |
| w4u8_decode_swiglu_row4_call_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 7168 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 3295232 | 3295232 | 0.0000% |
| scan_attention_overlay_required_bytes | 57088 | 57088 | 0.0000% |
| scan_cache_dma_descriptor_count | 896 | 896 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 3727360 | 3727360 | 0.0000% |
| scan_cache_ddr_write_bytes | 57344 | 57344 | 0.0000% |
| scan_cache_stage_ticks | 8412.65 | 8407.25 | -0.0642% |
| scan_cache_append_ticks | 3159.6 | 3154.35 | -0.1662% |
| scan_cache_pack_ticks | 250.85 | 247.75 | -1.2358% |
| block_orchestration_ticks | 599.4 | 591.2 | -1.3680% |
| layer_bookkeeping_ticks | 338.5 | 337.4 | -0.3250% |
| scan_dynamic_attention_ticks | 109541.6 | 109474.9 | -0.0609% |
| total_ticks | 787757.4 | 810602.2 | 2.9000% |
| invocation_ticks | 788296.4 | 811143.45 | 2.8983% |
| runtime_setup_ticks | 543 | 540.6 | -0.4420% |
| runtime_teardown_ticks | 476.55 | 479.35 | 0.5876% |
| stage_boundary_ticks | 36.9 | 37.15 | 0.6775% |
| ledger_named_ticks | 788296.4 | 811143.45 | 2.8983% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 3.1 | 3 | -3.2258% |
| metadata_stage_ticks | 4273.75 | 4236.05 | -0.8821% |
| input_norm_ticks | 44421.95 | 44431.9 | 0.0224% |
| qkv_projection_ticks | 64672.4 | 64546.05 | -0.1954% |
| qk_norm_rope_ticks | 7268.25 | 7262.75 | -0.0757% |
| attention_ticks | 109649.9 | 109582.55 | -0.0614% |
| o_projection_ticks | 55114.95 | 55274.75 | 0.2899% |
| post_attention_residual_ticks | 44658.1 | 44653.9 | -0.0094% |
| post_attention_norm_ticks | 16.2 | 16.1 | -0.6173% |
| gate_up_ticks | 228207.1 | 227769.05 | -0.1920% |
| activation_ticks | 32364.5 | 56039.75 | 73.1519% |
| down_ticks | 117597.45 | 117552.25 | -0.0384% |
| final_residual_ticks | 40.35 | 40.5 | 0.3717% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 38.9 | 38.35 | -1.4139% |
| generation_final_norm_ticks | 1565.7 | 1563 | -0.1724% |
| generation_lm_head_ticks | 74066.35 | 74031 | -0.0477% |
| generation_lm_head_weight_dma_ticks | 64160.35 | 64107 | -0.0832% |
| generation_lm_head_scale_dma_ticks | 370.95 | 362.85 | -2.1836% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 63793.4 | 63748.65 | -0.0701% |
| generation_lm_head_argmax_ticks | 7639.7 | 7638.4 | -0.0170% |
| generation_lm_head_weight_dma_wait_ticks | 62845.55 | 62794.35 | -0.0815% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 285.55 | 281.65 | -1.3658% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 126 | 126 | 0.0000% |
| generation_lm_head_n_tiles | 4008 | 4008 | 0.0000% |
| generation_lm_head_prefetch_count | 125 | 125 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1026048 | 1026048 | 0.0000% |
| generation_embedding_ddr_read_bytes | 6400 | 6400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.9802726e+08 | 1.9802726e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 197 | 197 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 1218 | 1218 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 336 | 336 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 1.6062874e+09 | 1.6062874e+09 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 3.2125747e+09 | 3.2125747e+09 | 0.0000% |
| weight_dma_ticks | 592524.65 | 591673.05 | -0.1437% |
| hmx_compute_ticks | 279749.3 | 281679.85 | 0.6901% |
| projection_pack_ticks | 248.5 | 249.05 | 0.2213% |
| projection_hmx_wait_ticks | 22432.3 | 23097.7 | 2.9663% |
| projection_unpack_ticks | 0 | 0 | N/A (zero control) |
| hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_cross_prefetch_lifetime_ticks | 0 | 0 | N/A (zero control) |
| attention_setup_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_qk_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_softmax_ticks | 0 | 0 | N/A (zero control) |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 14372.75 | 14352.4 | -0.1416% |
| u8_attention_qk_norm_rope_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_k_pack_ticks | 31685.75 | 31683.8 | -0.0062% |
| u8_attention_v_pack_ticks | 28192.15 | 28197 | 0.0172% |
| u8_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_full_prefix_pack_count | 448 | 448 | 0.0000% |
| u8_cache_segment_tail_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 12021.8 | 12021.55 | -0.0021% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 10834.45 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 26624.65 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 15985.2 | N/A (zero control) |
| llama_fp32_residual | 1 | 1 | 0.0000% |
| dense_r4_mode | 0 | 0 | N/A (zero control) |
| dense_r4_optimization | 0 | 0 | N/A (zero control) |
| dense_r4_calls | 0 | 0 | N/A (zero control) |
| dense_r4_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r4_rows | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_batches | 0 | 0 | N/A (zero control) |
| dense_r4_pipeline_hvx_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_join_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_worker_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_dispatches | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_prepare_tiles | 0 | 0 | N/A (zero control) |
| dense_r4_parallel_finish_groups | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_publish_count | 0 | 0 | N/A (zero control) |
| dense_r4_prefill_consume_count | 0 | 0 | N/A (zero control) |
| dense_r4_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_layout_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_finish_ticks | 0 | 0 | N/A (zero control) |
| dense_r4_audit_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_mode | 0 | 0 | N/A (zero control) |
| dense_r3_optimization | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt | 0 | 0 | N/A (zero control) |
| w4f16_decode_audit | 0 | 0 | N/A (zero control) |
| w4f16_decode_opt_calls | 0 | 0 | N/A (zero control) |
| w4f16_decode_conversion_audit_mismatches | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_heads | 0 | 0 | N/A (zero control) |
| dense_r3_constant_read_bytes | 0 | 0 | N/A (zero control) |
| dense_r3_total_parallel_work_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_direct_slot_join_count | 124 | 124 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 0 | 0 | N/A (zero control) |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 7468.25 | 7481.25 | 0.1741% |
| u8_attention_av_hmx_ticks | 12722.7 | 12717 | -0.0448% |
| u8_attention_av_requant_ticks | 3159 | 3160.8 | 0.0570% |
| u8_attention_pipeline_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_prefetch_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_qkvo_hmx_lifetime_ticks | 456482 | 455994.15 | -0.1069% |
| w4f16_gate_up_weight_dma_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_expand_pool_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_hmx_tail_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_work_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4f16_gate_up_stream_join_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_gate_up_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_down_pipeline_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_activation_work_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_stage_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_weight_expand_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_compute_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_hmx_ready_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_producer_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| w4u8_mlp_expanded_slot_wait_ticks | 0 | 0 | N/A (zero control) |
| vtcm_requested_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_acquired_bytes | 8388608 | 8388608 | 0.0000% |
| vtcm_peak_plan_bytes | 8360416 | 8360416 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1666 | 1666 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 3142656 | 3142656 | 0.0000% |
| weight_dma_descriptor_count | 2311 | 2311 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 1.6135127e+09 | 1.6135127e+09 | 0.0000% |
| boundary_ddr_read_bytes | 4965760 | 4965760 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
| weight_segment_count | 0 | 0 | N/A (zero control) |
| weight_segment_map_count | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_count | 0 | 0 | N/A (zero control) |
| weight_segment_error | 0 | 0 | N/A (zero control) |
| weight_segment_map_ticks | 0 | 0 | N/A (zero control) |
| weight_segment_unmap_ticks | 0 | 0 | N/A (zero control) |

## Provenance and gates

{
  "audits": [
    {
      "file": "audit-full-fused/hash-checks.json",
      "checks": 56,
      "normalized_numeric_match": true
    },
    {
      "file": "audit-full-split/hash-checks.json",
      "checks": 56,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-fused/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-split/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit3-split/hash-checks.json",
      "checks": 6,
      "normalized_numeric_match": true
    }
  ],
  "exact_cross_arm_payload_files": 60,
  "formal_token_invocations": 400,
  "short_token_invocations": 200,
  "requested_and_granted_vtcm": 8388608,
  "no_intermediate_ddr": true,
  "no_spill": true,
  "quality_claim": false,
  "baseline_promoted": false
}

Sealed binaries:

[
  {
    "path": "binaries-1-a01/seal.json",
    "model_size": "3B",
    "layer_count": "1",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "ba8aca2c36350163fc901aea384927a19e16b41453fd949e931a32efbdfef781",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "59f7f5b6ad453d669ac9896d8ecaa9d73e93170a4ee9fd52fc534ec22a85807a"
    }
  },
  {
    "path": "binaries-28-a01/seal.json",
    "model_size": "3B",
    "layer_count": "28",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "56320323351601849f2c8f018f4f0fe9175f8d9a432cdfab507ca74c5cf3bba5",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "bca24db677059283ae13b2f6e3c7cfb63a6399bf26c690e3a1cf1d679c413c4f"
    }
  },
  {
    "path": "binaries-3-a01/seal.json",
    "model_size": "3B",
    "layer_count": "3",
    "fp_islands": true,
    "paper_trace": false,
    "source_head": "272298e3958e65bc9e9d89c0a7c9800fb8aad465",
    "files": {
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "2cec928df76b79dbea2e2268025c144b603af5bbc6818f491bd312acc93564aa",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "01caa0344b37d9967de4aa9bcfe24b3ab460c0d453ea4075355084e646bd2827",
      "/home/daniuniu/work/llama32-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "350da4e9d86ff277979e21e03eb6dd81694081b3da1e1cc24a2d5fcc45ee9da0",
      "/home/daniuniu/work/llama32-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "3fc6241c66092b17290f558c18ba492cc85fcc834633bf936b7e12de31fa6f7b"
    }
  }
]

Historical G fused measurement is preserved. No W16A16/W4A16 comparison measured: outside this diagnostic request.
