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



# EXP-0312 Qwen3-0.6B explicit QDQ profiling

Full-model M64 prefill plus one fixed decode. Same binary, weights and token trajectory; alternating AB/BA. Five short paired repeat10 rounds and ten formal paired repeat10 rounds. Two warmup repeat1 measurements are auxiliary only. No model quality or speed acceptance gate. Original G data remains unchanged.

Fused control and explicit DQ/FP/Q preserve vector arithmetic, fixed scales, INT16 Down and FP32 residual. Prefill Softmax and SwiGLU are separated; decode Softmax remains fused, so no complete decode FP/QDQ split is claimed. DQ includes stable integer max/centering before float scaling. Q includes scaling, rounding and consumer-layout publication. Stage timers include materialization and dispatch/join. SwiGLU uses 2048-element tiles, three 16KiB phase scratch regions in dead VTCM. No timed DDR materialization. This is a new diagnostic schedule, not a retrospective decomposition of the historical fused cost.

## prefill

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.206739 (0.418%) | 0.205534 (0.360%) |
| Input RMSNorm | 1.020364 (2.064%) | 1.019664 (1.787%) |
| QKV + Q/K preparation | 21.421155 (43.336%) | 21.430162 (37.562%) |
| QK–Softmax–AV | 5.682350 (11.496%) | 6.121864 (10.730%) |
| O projection | 1.234589 (2.498%) | 1.232564 (2.160%) |
| Post-attention residual＋RMSNorm | 1.022585 (2.069%) | 1.023095 (1.793%) |
| Gate/Up＋SwiGLU | 13.096145 (26.494%) | 20.183719 (35.378%) |
| Down | 1.613333 (3.264%) | 1.603844 (2.811%) |
| Final residual | 0.003423 (0.007%) | 0.003341 (0.006%) |
| KV carrier conversion | 0.136585 (0.276%) | 0.137154 (0.240%) |
| KV append DMA | 0.237987 (0.481%) | 0.238074 (0.417%) |
| Block orchestration | 0.038800 (0.078%) | 0.038834 (0.068%) |
| Layer bookkeeping | 0.023163 (0.047%) | 0.023195 (0.041%) |
| Stage-boundary bookkeeping | 0.019828 (0.040%) | 0.019752 (0.035%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.088391 (0.179%) | 0.087788 (0.154%) |
| Embedding | 0.041180 (0.083%) | 0.041269 (0.072%) |
| Final model RMSNorm | 0.007650 (0.015%) | 0.007669 (0.013%) |
| LM head＋greedy（不含 final norm） | 2.011318 (4.069%) | 2.041039 (3.577%) |
| Host-DSP boundary | 1.524980 (3.085%) | 1.593721 (2.793%) |
| Host wall | 49.430566 | 57.052282 |
| E2E token/s | 1294.7454 | 1121.7781 |

Paired split/fused Host wall: {'wall_ratio': 1.154190347386252, 'ci95': [1.1511471786239305, 1.1575979036902706]}

## decode_one_step

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.202645 (0.639%) | 0.201648 (0.627%) |
| Input RMSNorm | 0.195534 (0.616%) | 0.195271 (0.607%) |
| QKV + Q/K preparation | 21.358445 (67.327%) | 21.363801 (66.416%) |
| QK–Softmax–AV | 1.817743 (5.730%) | 1.817094 (5.649%) |
| O projection | 0.774672 (2.442%) | 0.772402 (2.401%) |
| Post-attention residual＋RMSNorm | 0.184854 (0.583%) | 0.184715 (0.574%) |
| Gate/Up＋SwiGLU | 2.270452 (7.157%) | 2.731120 (8.491%) |
| Down | 1.138396 (3.589%) | 1.127448 (3.505%) |
| Final residual | 0.002236 (0.007%) | 0.002256 (0.007%) |
| KV carrier conversion | 0.150249 (0.474%) | 0.149945 (0.466%) |
| KV append DMA | 0.074074 (0.234%) | 0.074022 (0.230%) |
| Block orchestration | 0.031727 (0.100%) | 0.030573 (0.095%) |
| Layer bookkeeping | 0.017712 (0.056%) | 0.017558 (0.055%) |
| Stage-boundary bookkeeping | 0.001900 (0.006%) | 0.001872 (0.006%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.051840 (0.163%) | 0.051935 (0.161%) |
| Embedding | 0.001491 (0.005%) | 0.001369 (0.004%) |
| Final model RMSNorm | 0.007216 (0.023%) | 0.007144 (0.022%) |
| LM head＋greedy（不含 final norm） | 2.018292 (6.362%) | 2.046566 (6.362%) |
| Host-DSP boundary | 1.423836 (4.488%) | 1.390031 (4.321%) |
| Host wall | 31.723317 | 32.166771 |
| E2E token/s | 31.5226 | 31.0880 |

Paired split/fused Host wall: {'wall_ratio': 1.0139788108508065, 'ci95': [1.0079501470684962, 1.0198721762422887]}

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
| host_wall_ns | 50195782 | 58175417 | 15.8970% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4741 | 4702 | -0.8226% |
| scan_cache_pack_ticks | 2548 | 2676 | 5.0235% |
| block_orchestration_ticks | 727 | 730 | 0.4127% |
| layer_bookkeeping_ticks | 431 | 421 | -2.3202% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 918727 | 1066455 | 16.0796% |
| invocation_ticks | 919516 | 1067243 | 16.0657% |
| runtime_setup_ticks | 789 | 788 | -0.1267% |
| runtime_teardown_ticks | 744 | 775 | 4.1667% |
| stage_boundary_ticks | 419 | 410 | -2.1480% |
| ledger_named_ticks | 919516 | 1067243 | 16.0657% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 8 | 13 | 62.5000% |
| metadata_stage_ticks | 4072 | 4070 | -0.0491% |
| input_norm_ticks | 19664 | 19586 | -0.3967% |
| qkv_projection_ticks | 411650 | 411996 | 0.0841% |
| qk_norm_rope_ticks | 8 | 8 | 0.0000% |
| attention_ticks | 108882 | 117233 | 7.6698% |
| o_projection_ticks | 23884 | 23994 | 0.4606% |
| post_attention_residual_ticks | 19617 | 19623 | 0.0306% |
| post_attention_norm_ticks | 37 | 24 | -35.1351% |
| gate_up_ticks | 37289 | 37383 | 0.2521% |
| activation_ticks | 214021 | 349293 | 63.2050% |
| down_ticks | 30580 | 30477 | -0.3368% |
| final_residual_ticks | 60 | 56 | -6.6667% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 872 | 883 | 1.2615% |
| generation_final_norm_ticks | 143 | 143 | 0.0000% |
| generation_lm_head_ticks | 38473 | 42102 | 9.4326% |
| generation_lm_head_weight_dma_ticks | 26977 | 26353 | -2.3131% |
| generation_lm_head_scale_dma_ticks | 392 | 381 | -2.8061% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28352 | 31754 | 11.9992% |
| generation_lm_head_argmax_ticks | 9064 | 9305 | 2.6589% |
| generation_lm_head_weight_dma_wait_ticks | 25348 | 23824 | -6.0123% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 1735 | 5760 | 231.9885% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 131328 | 131328 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 131789 | 130595 | -0.9060% |
| hmx_compute_ticks | 103863 | 111697 | 7.5426% |
| projection_pack_ticks | 280 | 278 | -0.7143% |
| projection_hmx_wait_ticks | 34119 | 34791 | 1.9696% |
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
| attention_softmax_ticks | 54842 | 62962 | 14.8062% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1058923 | 1058432 | -0.0464% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 72724 | 72535 | -0.2599% |
| u8_cache_native_append_update_ticks | 7257 | 7340 | 1.1437% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 10926 | 11167 | 2.2057% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14719 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37861 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10382 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 62789 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 202913 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 82493 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 208621 | 222035 | 6.4298% |
| u8_attention_av_hmx_ticks | 11071 | 11041 | -0.2710% |
| u8_attention_av_requant_ticks | 32094 | 31984 | -0.3427% |
| u8_attention_pipeline_wait_ticks | 43773 | 46135 | 5.3960% |
| w4u8_qkvo_weight_expand_ticks | 61253 | 61376 | 0.2008% |
| w4u8_qkvo_prefetch_wait_ticks | 39183 | 39121 | -0.1582% |
| w4u8_qkvo_hmx_lifetime_ticks | 173541 | 174072 | 0.3060% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 675200 | 675200 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4861312 | 4861312 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 32849011 | 33286302 | 1.3312% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 1376 | 1429 | 3.8517% |
| w4u8_o_gate_prefetch_lifetime_ticks | 5132 | 5190 | 1.1302% |
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
| w4u8_decode_swiglu_row4_call_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 1507328 | 1507328 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 9930 | 9807 | -1.2387% |
| scan_cache_append_ticks | 1433 | 1439 | 0.4187% |
| scan_cache_pack_ticks | 2885 | 2841 | -1.5251% |
| block_orchestration_ticks | 606 | 595 | -1.8152% |
| layer_bookkeeping_ticks | 344 | 336 | -2.3256% |
| scan_dynamic_attention_ticks | 34579 | 34592 | 0.0376% |
| total_ticks | 579617 | 594281 | 2.5299% |
| invocation_ticks | 580149 | 594821 | 2.5290% |
| runtime_setup_ticks | 532 | 540 | 1.5038% |
| runtime_teardown_ticks | 476 | 447 | -6.0924% |
| stage_boundary_ticks | 35 | 35 | 0.0000% |
| ledger_named_ticks | 580149 | 594821 | 2.5290% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 3 | 5 | 66.6667% |
| metadata_stage_ticks | 3513 | 3509 | -0.1139% |
| input_norm_ticks | 3760 | 3774 | 0.3723% |
| qkv_projection_ticks | 410093 | 410273 | 0.0439% |
| qk_norm_rope_ticks | 4 | 10 | 150.0000% |
| attention_ticks | 34707 | 34717 | 0.0288% |
| o_projection_ticks | 14765 | 15110 | 2.3366% |
| post_attention_residual_ticks | 3517 | 3516 | -0.0284% |
| post_attention_norm_ticks | 20 | 18 | -10.0000% |
| gate_up_ticks | 30927 | 31233 | 0.9894% |
| activation_ticks | 12286 | 21377 | 73.9948% |
| down_ticks | 21732 | 21880 | 0.6810% |
| final_residual_ticks | 45 | 42 | -6.6667% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 25 | 27 | 8.0000% |
| generation_final_norm_ticks | 144 | 143 | -0.6944% |
| generation_lm_head_ticks | 38441 | 43097 | 12.1121% |
| generation_lm_head_weight_dma_ticks | 26692 | 26002 | -2.5850% |
| generation_lm_head_scale_dma_ticks | 387 | 391 | 1.0336% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28528 | 32891 | 15.2937% |
| generation_lm_head_argmax_ticks | 8989 | 9290 | 3.3485% |
| generation_lm_head_weight_dma_wait_ticks | 25149 | 23193 | -7.7776% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 2121 | 7151 | 237.1523% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 2304 | 2304 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 128264 | 127326 | -0.7313% |
| hmx_compute_ticks | 85902 | 95102 | 10.7099% |
| projection_pack_ticks | 265 | 270 | 1.8868% |
| projection_hmx_wait_ticks | 15620 | 16788 | 7.4776% |
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
| attention_unattributed_ticks | 16187 | 15989 | -1.2232% |
| u8_attention_qk_norm_rope_ticks | 1057987 | 1058742 | 0.0714% |
| u8_attention_k_pack_ticks | 127 | 134 | 5.5118% |
| u8_attention_v_pack_ticks | 2696 | 2679 | -0.6306% |
| u8_cache_native_append_update_ticks | 4260 | 4224 | -0.8451% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 749 | 748 | -0.1335% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2520 | 2583 | 2.5000% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 4305 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 10115 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 5941 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 6019 | 6162 | 2.3758% |
| u8_attention_av_hmx_ticks | 3165 | 3160 | -0.1580% |
| u8_attention_av_requant_ticks | 2788 | 2777 | -0.3945% |
| u8_attention_pipeline_wait_ticks | 1205 | 1233 | 2.3237% |
| w4u8_qkvo_weight_expand_ticks | 60693 | 61010 | 0.5223% |
| w4u8_qkvo_prefetch_wait_ticks | 39059 | 39053 | -0.0154% |
| w4u8_qkvo_hmx_lifetime_ticks | 157839 | 159100 | 0.7989% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 592768 | 592768 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4732288 | 4732288 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
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
| host_wall_ns | 49619187 | 56758062 | 14.3873% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4572.7 | 4571 | -0.0372% |
| scan_cache_pack_ticks | 2618.5 | 2622.9 | 0.1680% |
| block_orchestration_ticks | 748.4 | 742.4 | -0.8017% |
| layer_bookkeeping_ticks | 444.9 | 447.4 | 0.5619% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 917516.2 | 1062105.4 | 15.7588% |
| invocation_ticks | 918516.2 | 1063031.9 | 15.7336% |
| runtime_setup_ticks | 998.5 | 967.1 | -3.1447% |
| runtime_teardown_ticks | 723.6 | 699.5 | -3.3306% |
| stage_boundary_ticks | 383.7 | 375.7 | -2.0850% |
| ledger_named_ticks | 918516.2 | 1063031.9 | 15.7336% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 9.8 | 9.4 | -4.0816% |
| metadata_stage_ticks | 3891.6 | 3985.6 | 2.4155% |
| input_norm_ticks | 19586.2 | 19561.5 | -0.1261% |
| qkv_projection_ticks | 411254.2 | 411442.3 | 0.0457% |
| qk_norm_rope_ticks | 10.8 | 11.5 | 6.4815% |
| attention_ticks | 108983.3 | 117287.2 | 7.6194% |
| o_projection_ticks | 23625.5 | 23705.7 | 0.3395% |
| post_attention_residual_ticks | 19609.8 | 19604.7 | -0.0260% |
| post_attention_norm_ticks | 27.4 | 27.4 | 0.0000% |
| gate_up_ticks | 37282 | 37238.4 | -0.1169% |
| activation_ticks | 214066.1 | 349393 | 63.2173% |
| down_ticks | 30728.4 | 30760.1 | 0.1032% |
| final_residual_ticks | 64.5 | 64.6 | 0.1550% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 792 | 782.9 | -1.1490% |
| generation_final_norm_ticks | 146.2 | 146.4 | 0.1368% |
| generation_lm_head_ticks | 38503.2 | 38558.6 | 0.1439% |
| generation_lm_head_weight_dma_ticks | 26942.5 | 26893.5 | -0.1819% |
| generation_lm_head_scale_dma_ticks | 386.8 | 387.9 | 0.2844% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28440.1 | 28481.2 | 0.1445% |
| generation_lm_head_argmax_ticks | 9036.5 | 9050.6 | 0.1560% |
| generation_lm_head_weight_dma_wait_ticks | 25392.7 | 25321.8 | -0.2792% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 1821.4 | 1911.4 | 4.9413% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 131328 | 131328 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 130088.8 | 129878 | -0.1620% |
| hmx_compute_ticks | 104031.3 | 103696.8 | -0.3215% |
| projection_pack_ticks | 277.1 | 283.3 | 2.2375% |
| projection_hmx_wait_ticks | 34743.5 | 35187.8 | 1.2788% |
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
| attention_softmax_ticks | 54832.7 | 62988.5 | 14.8740% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1057923.9 | 1057463.6 | -0.0435% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 72619.4 | 72485.8 | -0.1840% |
| u8_cache_native_append_update_ticks | 7156.3 | 7157 | 0.0098% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 11183.1 | 11059.1 | -1.1088% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14778.8 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37772.7 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10436.9 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 62810.9 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 203121.5 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 82413.1 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 208835.9 | 221786.8 | 6.2015% |
| u8_attention_av_hmx_ticks | 10992.2 | 11014.7 | 0.2047% |
| u8_attention_av_requant_ticks | 32004.2 | 32060.8 | 0.1769% |
| u8_attention_pipeline_wait_ticks | 44202.4 | 46236.9 | 4.6027% |
| w4u8_qkvo_weight_expand_ticks | 61225.3 | 61259.6 | 0.0560% |
| w4u8_qkvo_prefetch_wait_ticks | 38678.1 | 38614.7 | -0.1639% |
| w4u8_qkvo_hmx_lifetime_ticks | 173392.5 | 173821.2 | 0.2472% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 675200 | 675200 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4861312 | 4861312 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 31860214 | 32089651 | 0.7201% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 1389.4 | 1376.3 | -0.9429% |
| w4u8_o_gate_prefetch_lifetime_ticks | 5167.3 | 5158.7 | -0.1664% |
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
| w4u8_decode_swiglu_row4_call_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 1507328 | 1507328 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 9925.9 | 9943.1 | 0.1733% |
| scan_cache_append_ticks | 1423 | 1420.7 | -0.1616% |
| scan_cache_pack_ticks | 2892.5 | 2871.8 | -0.7156% |
| block_orchestration_ticks | 610.9 | 586.4 | -4.0105% |
| layer_bookkeeping_ticks | 341.8 | 337.8 | -1.1703% |
| scan_dynamic_attention_ticks | 34833.1 | 34737.8 | -0.2736% |
| total_ticks | 579340.6 | 589632 | 1.7764% |
| invocation_ticks | 579882.7 | 590178.3 | 1.7755% |
| runtime_setup_ticks | 541.1 | 541.3 | 0.0370% |
| runtime_teardown_ticks | 454.2 | 454.2 | 0.0000% |
| stage_boundary_ticks | 36.8 | 36.1 | -1.9022% |
| ledger_named_ticks | 579882.7 | 590178.3 | 1.7755% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 5.5 | 5.8 | 5.4545% |
| metadata_stage_ticks | 3802 | 3938.6 | 3.5928% |
| input_norm_ticks | 3749.6 | 3748.7 | -0.0240% |
| qkv_projection_ticks | 409989 | 410276.3 | 0.0701% |
| qk_norm_rope_ticks | 7.4 | 7.1 | -4.0541% |
| attention_ticks | 34962.5 | 34865.8 | -0.2766% |
| o_projection_ticks | 14829.5 | 14863.5 | 0.2293% |
| post_attention_residual_ticks | 3528.8 | 3531.5 | 0.0765% |
| post_attention_norm_ticks | 20.2 | 19.6 | -2.9703% |
| gate_up_ticks | 31191.4 | 31177 | -0.0462% |
| activation_ticks | 12295.1 | 21285.9 | 73.1251% |
| down_ticks | 21523.7 | 21759.9 | 1.0974% |
| final_residual_ticks | 42.2 | 42.6 | 0.9479% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 24.6 | 26.7 | 8.5366% |
| generation_final_norm_ticks | 138 | 138 | 0.0000% |
| generation_lm_head_ticks | 38459.9 | 38380.3 | -0.2070% |
| generation_lm_head_weight_dma_ticks | 26805.9 | 26828.7 | 0.0851% |
| generation_lm_head_scale_dma_ticks | 390.7 | 390 | -0.1792% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28576.6 | 28499 | -0.2716% |
| generation_lm_head_argmax_ticks | 8964 | 8961.1 | -0.0324% |
| generation_lm_head_weight_dma_wait_ticks | 25264.2 | 25302.1 | 0.1500% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 2029.7 | 1894.5 | -6.6611% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 2304 | 2304 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 128551.8 | 128775.2 | 0.1738% |
| hmx_compute_ticks | 84165.5 | 85231.9 | 1.2670% |
| projection_pack_ticks | 269.1 | 268.9 | -0.0743% |
| projection_hmx_wait_ticks | 15429 | 16104.9 | 4.3807% |
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
| attention_unattributed_ticks | 16109.5 | 16094.8 | -0.0913% |
| u8_attention_qk_norm_rope_ticks | 1057500.1 | 1058216.1 | 0.0677% |
| u8_attention_k_pack_ticks | 131.7 | 135.2 | 2.6576% |
| u8_attention_v_pack_ticks | 2700.6 | 2689.1 | -0.4258% |
| u8_cache_native_append_update_ticks | 4257.8 | 4239.6 | -0.4275% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 766.4 | 765.2 | -0.1566% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2629.8 | 2620.3 | -0.3612% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 4311.5 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 10032.1 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 5931.8 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 6060.7 | 6039.5 | -0.3498% |
| u8_attention_av_hmx_ticks | 3238.2 | 3252.1 | 0.4293% |
| u8_attention_av_requant_ticks | 2818.3 | 2808.7 | -0.3406% |
| u8_attention_pipeline_wait_ticks | 1258.8 | 1247.3 | -0.9136% |
| w4u8_qkvo_weight_expand_ticks | 60826 | 60912.1 | 0.1416% |
| w4u8_qkvo_prefetch_wait_ticks | 38941.8 | 38886.8 | -0.1412% |
| w4u8_qkvo_hmx_lifetime_ticks | 157759.5 | 158372.8 | 0.3888% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 592768 | 592768 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4732288 | 4732288 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
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
| host_wall_ns | 49450734 | 56952576 | 15.1703% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 4568.55 | 4569.15 | 0.0131% |
| scan_cache_pack_ticks | 2614.05 | 2634.1 | 0.7670% |
| block_orchestration_ticks | 748.95 | 746.2 | -0.3672% |
| layer_bookkeeping_ticks | 446.25 | 445.1 | -0.2577% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 918478.6 | 1062578.5 | 15.6890% |
| invocation_ticks | 919479.4 | 1063593.4 | 15.6734% |
| runtime_setup_ticks | 986.45 | 971 | -1.5662% |
| runtime_teardown_ticks | 721.4 | 707 | -1.9961% |
| stage_boundary_ticks | 380.7 | 377.95 | -0.7224% |
| ledger_named_ticks | 919479.4 | 1063593.4 | 15.6734% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 9.75 | 9.4 | -3.5897% |
| metadata_stage_ticks | 3973.1 | 3957.05 | -0.4040% |
| input_norm_ticks | 19591.65 | 19583.1 | -0.0436% |
| qkv_projection_ticks | 411334.6 | 411466.15 | 0.0320% |
| qk_norm_rope_ticks | 11.1 | 11.3 | 1.8018% |
| attention_ticks | 109060.1 | 117501.45 | 7.7401% |
| o_projection_ticks | 23674.6 | 23710.5 | 0.1516% |
| post_attention_residual_ticks | 19607.5 | 19612.65 | 0.0263% |
| post_attention_norm_ticks | 27.9 | 28.45 | 1.9713% |
| gate_up_ticks | 37341.45 | 37359.35 | 0.0479% |
| activation_ticks | 214078.35 | 350253.15 | 63.6098% |
| down_ticks | 30963.25 | 30867.95 | -0.3078% |
| final_residual_ticks | 65.5 | 64.15 | -2.0611% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 791.85 | 792.1 | 0.0316% |
| generation_final_norm_ticks | 146.65 | 147.45 | 0.5455% |
| generation_lm_head_ticks | 38485.8 | 38769.15 | 0.7362% |
| generation_lm_head_weight_dma_ticks | 26942.75 | 26806.95 | -0.5040% |
| generation_lm_head_scale_dma_ticks | 388.6 | 388.4 | -0.0515% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28406.4 | 28672.5 | 0.9368% |
| generation_lm_head_argmax_ticks | 9048.55 | 9077.4 | 0.3188% |
| generation_lm_head_weight_dma_wait_ticks | 25398.85 | 25148.8 | -0.9845% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 1811.2 | 2217.45 | 22.4299% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 131328 | 131328 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 130050.55 | 129636.2 | -0.3186% |
| hmx_compute_ticks | 104233.25 | 103488.45 | -0.7146% |
| projection_pack_ticks | 279.05 | 282.3 | 1.1647% |
| projection_hmx_wait_ticks | 35517.5 | 35401 | -0.3280% |
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
| attention_softmax_ticks | 54858.25 | 63025.75 | 14.8884% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1057853.9 | 1057603.4 | -0.0237% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 72620.05 | 72564.35 | -0.0767% |
| u8_cache_native_append_update_ticks | 7143.1 | 7167.4 | 0.3402% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 11235.75 | 11235.55 | -0.0018% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14784.9 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37801.05 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10444.65 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 63086.75 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 203514.1 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 82614.55 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 208880.1 | 221923.15 | 6.2443% |
| u8_attention_av_hmx_ticks | 11019.6 | 11144 | 1.1289% |
| u8_attention_av_requant_ticks | 32007.1 | 32019.55 | 0.0389% |
| u8_attention_pipeline_wait_ticks | 44348.3 | 46690.05 | 5.2804% |
| w4u8_qkvo_weight_expand_ticks | 61209.95 | 61273.9 | 0.1045% |
| w4u8_qkvo_prefetch_wait_ticks | 38681.05 | 38564.9 | -0.3003% |
| w4u8_qkvo_hmx_lifetime_ticks | 174062.35 | 173940.7 | -0.0699% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 675200 | 675200 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4861312 | 4861312 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 31733714 | 32081401 | 1.0956% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 56 | 56 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 1393.65 | 1415.9 | 1.5965% |
| w4u8_o_gate_prefetch_lifetime_ticks | 5168.8 | 5185.55 | 0.3241% |
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
| w4u8_decode_swiglu_row4_call_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 2688 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 1507328 | 1507328 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 9940.05 | 9927.35 | -0.1278% |
| scan_cache_append_ticks | 1422.45 | 1420.85 | -0.1125% |
| scan_cache_pack_ticks | 2885.65 | 2880.05 | -0.1941% |
| block_orchestration_ticks | 608.65 | 587.25 | -3.5160% |
| layer_bookkeeping_ticks | 340.6 | 335.35 | -1.5414% |
| scan_dynamic_attention_ticks | 34752.2 | 34764.25 | 0.0347% |
| total_ticks | 580580.6 | 589423.15 | 1.5231% |
| invocation_ticks | 581124.15 | 589967.85 | 1.5218% |
| runtime_setup_ticks | 542.4 | 542.45 | 0.0092% |
| runtime_teardown_ticks | 451.8 | 453.1 | 0.2877% |
| stage_boundary_ticks | 36.2 | 36 | -0.5525% |
| ledger_named_ticks | 581124.15 | 589967.85 | 1.5218% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 5.45 | 5.95 | 9.1743% |
| metadata_stage_ticks | 3876.85 | 3880.35 | 0.0903% |
| input_norm_ticks | 3755.25 | 3749.3 | -0.1584% |
| qkv_projection_ticks | 410044.2 | 410166.95 | 0.0299% |
| qk_norm_rope_ticks | 7.05 | 6.9 | -2.1277% |
| attention_ticks | 34883.75 | 34892.4 | 0.0248% |
| o_projection_ticks | 14878.45 | 14821.6 | -0.3821% |
| post_attention_residual_ticks | 3529.3 | 3524.95 | -0.1233% |
| post_attention_norm_ticks | 20.3 | 20.05 | -1.2315% |
| gate_up_ticks | 31279.7 | 31269.25 | -0.0334% |
| activation_ticks | 12315.05 | 21250.4 | 72.5563% |
| down_ticks | 21780.5 | 21732.7 | -0.2195% |
| final_residual_ticks | 42.7 | 43.15 | 1.0539% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 29.85 | 26.35 | -11.7253% |
| generation_final_norm_ticks | 137.9 | 137 | -0.6526% |
| generation_lm_head_ticks | 38564.05 | 38509.55 | -0.1413% |
| generation_lm_head_weight_dma_ticks | 26795.7 | 26779.45 | -0.0606% |
| generation_lm_head_scale_dma_ticks | 390.15 | 391.55 | 0.3588% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 28661.5 | 28606.25 | -0.1928% |
| generation_lm_head_argmax_ticks | 8980.2 | 8980.5 | 0.0033% |
| generation_lm_head_weight_dma_wait_ticks | 25241.25 | 25205.4 | -0.1420% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 2127.3 | 2109.3 | -0.8461% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 2304 | 2304 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 79006720 | 79006720 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 485 | 485 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 2.3927194e+08 | 2.3927194e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 4.7854387e+08 | 4.7854387e+08 | 0.0000% |
| weight_dma_ticks | 128617.55 | 128599.7 | -0.0139% |
| hmx_compute_ticks | 85565.85 | 84822.7 | -0.8685% |
| projection_pack_ticks | 267.4 | 268.05 | 0.2431% |
| projection_hmx_wait_ticks | 16322.55 | 16037.4 | -1.7470% |
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
| attention_unattributed_ticks | 16123.4 | 16109.2 | -0.0881% |
| u8_attention_qk_norm_rope_ticks | 1057500.8 | 1057868.9 | 0.0348% |
| u8_attention_k_pack_ticks | 132.85 | 132.75 | -0.0753% |
| u8_attention_v_pack_ticks | 2689.85 | 2686.1 | -0.1394% |
| u8_cache_native_append_update_ticks | 4253.7 | 4245.1 | -0.2022% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 766.95 | 763.95 | -0.3912% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2639.95 | 2618.15 | -0.8258% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 4306.45 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 10004.5 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 5922.65 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
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
| u8_attention_softmax_ticks | 6054 | 6069.55 | 0.2569% |
| u8_attention_av_hmx_ticks | 3217.9 | 3216 | -0.0590% |
| u8_attention_av_requant_ticks | 2815.75 | 2797.65 | -0.6428% |
| u8_attention_pipeline_wait_ticks | 1251.05 | 1256.3 | 0.4196% |
| w4u8_qkvo_weight_expand_ticks | 60835 | 60915.45 | 0.1322% |
| w4u8_qkvo_prefetch_wait_ticks | 38926.05 | 38885.8 | -0.1034% |
| w4u8_qkvo_hmx_lifetime_ticks | 158494.95 | 158375.6 | -0.0753% |
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
| vtcm_peak_plan_bytes | 5744384 | 5744384 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 1829 | 1829 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 592768 | 592768 | 0.0000% |
| weight_dma_descriptor_count | 2614 | 2614 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 3.0196224e+08 | 3.0196224e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4732288 | 4732288 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |

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
      "file": "audit1-split-a01/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit3-split-a01/hash-checks.json",
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
    "fp_islands": true,
    "model_size": "0.6B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "04c9bbfa17849e8ba981ec37186e0fb1b488099e0ad0d4f370d1c1a40de8e079",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "778eea81e61d0e29dc29b7a50f8dcdf966895eb8086bce43365cb6fc7fe425a0"
    }
  },
  {
    "path": "binaries-28-a01/seal.json",
    "fp_islands": true,
    "model_size": "0.6B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "46d2c4e59ec36f365888ceda1cbd82f2a66871e4bfab4e632133c93128279afb",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "ea69f32574356b0d2e84f68c794e0b92f17a50ea9259846aefc240c0247422f7"
    }
  },
  {
    "path": "binaries-3-a01/seal.json",
    "fp_islands": true,
    "model_size": "0.6B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "353e17021d2a41abd0a74aed1e090d0a0ce11e88991cb74f44089212329d94b9",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "f84b1680d039fd67b73ba376b9886e08333f2b0a61d530868fcfe8ff3be45a35"
    }
  }
]

Historical G fused measurement is preserved. No W16A16/W4A16 comparison measured: outside this diagnostic request.


# EXP-0312 Qwen3-1.7B explicit QDQ profiling

Full-model M64 prefill plus one fixed decode. Same binary, weights and token trajectory; alternating AB/BA. Five short paired repeat10 rounds and ten formal paired repeat10 rounds. Two warmup repeat1 measurements are auxiliary only. No model quality or speed acceptance gate. Original G data remains unchanged.

Fused control and explicit DQ/FP/Q preserve vector arithmetic, fixed scales, INT16 Down and FP32 residual. Prefill Softmax and SwiGLU are separated; decode Softmax remains fused, so no complete decode FP/QDQ split is claimed. DQ includes stable integer max/centering before float scaling. Q includes scaling, rounding and consumer-layout publication. Stage timers include materialization and dispatch/join. SwiGLU uses 2048-element tiles, three 16KiB phase scratch regions in dead VTCM. No timed DDR materialization. This is a new diagnostic schedule, not a retrospective decomposition of the historical fused cost.

## prefill

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.230525 (0.303%) | 0.232773 (0.256%) |
| Input RMSNorm | 2.122259 (2.793%) | 2.117017 (2.331%) |
| QKV + Q/K preparation | 23.607398 (31.063%) | 23.529883 (25.904%) |
| QK–Softmax–AV | 5.918136 (7.787%) | 6.341304 (6.981%) |
| O projection | 2.116598 (2.785%) | 2.123890 (2.338%) |
| Post-attention residual＋RMSNorm | 2.047104 (2.694%) | 2.048727 (2.255%) |
| Gate/Up＋SwiGLU | 28.905294 (38.034%) | 43.323948 (47.696%) |
| Down | 3.812674 (5.017%) | 3.842390 (4.230%) |
| Final residual | 0.002585 (0.003%) | 0.002523 (0.003%) |
| KV carrier conversion | 0.138487 (0.182%) | 0.138735 (0.153%) |
| KV append DMA | 0.290216 (0.382%) | 0.290976 (0.320%) |
| Block orchestration | 0.039607 (0.052%) | 0.039372 (0.043%) |
| Layer bookkeeping | 0.023669 (0.031%) | 0.024067 (0.026%) |
| Stage-boundary bookkeeping | 0.019707 (0.026%) | 0.019306 (0.021%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.089658 (0.118%) | 0.087761 (0.097%) |
| Embedding | 0.056753 (0.075%) | 0.056566 (0.062%) |
| Final model RMSNorm | 0.014085 (0.019%) | 0.014083 (0.016%) |
| LM head＋greedy（不含 final norm） | 5.178963 (6.815%) | 5.175212 (5.697%) |
| Host-DSP boundary | 1.384334 (1.822%) | 1.425881 (1.570%) |
| Host wall | 75.998054 | 90.834413 |
| E2E token/s | 842.1268 | 704.5788 |

Paired split/fused Host wall: {'wall_ratio': 1.1952202401994871, 'ci95': [1.1929321036014622, 1.197130719936874]}

## decode_one_step

| Module (ms; Host wall share) | Fused | Explicit |
|---|---:|---:|
| I/O、metadata | 0.229356 (0.526%) | 0.230307 (0.518%) |
| Input RMSNorm | 0.371649 (0.852%) | 0.371584 (0.836%) |
| QKV + Q/K preparation | 23.632437 (54.150%) | 23.617969 (53.109%) |
| QK–Softmax–AV | 1.879733 (4.307%) | 1.881052 (4.230%) |
| O projection | 1.383845 (3.171%) | 1.384876 (3.114%) |
| Post-attention residual＋RMSNorm | 0.353730 (0.811%) | 0.353985 (0.796%) |
| Gate/Up＋SwiGLU | 7.376716 (16.903%) | 8.298115 (18.660%) |
| Down | 3.503829 (8.029%) | 3.503586 (7.878%) |
| Final residual | 0.001912 (0.004%) | 0.001909 (0.004%) |
| KV carrier conversion | 0.150214 (0.344%) | 0.149566 (0.336%) |
| KV append DMA | 0.107199 (0.246%) | 0.107795 (0.242%) |
| Block orchestration | 0.031720 (0.073%) | 0.031177 (0.070%) |
| Layer bookkeeping | 0.017564 (0.040%) | 0.017928 (0.040%) |
| Stage-boundary bookkeeping | 0.001834 (0.004%) | 0.001832 (0.004%) |
| DSP unattributed | 0.000000 (0.000%) | 0.000000 (0.000%) |
| Runtime setup/teardown | 0.055228 (0.127%) | 0.055108 (0.124%) |
| Embedding | 0.001693 (0.004%) | 0.001664 (0.004%) |
| Final model RMSNorm | 0.016324 (0.037%) | 0.016296 (0.037%) |
| LM head＋greedy（不含 final norm） | 3.168736 (7.261%) | 3.186943 (7.166%) |
| Host-DSP boundary | 1.358640 (3.113%) | 1.259285 (2.832%) |
| Host wall | 43.642360 | 44.470978 |
| E2E token/s | 22.9135 | 22.4866 |

Paired split/fused Host wall: {'wall_ratio': 1.0189865466078258, 'ci95': [1.0153048854811553, 1.0222709303491793]}

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
| host_wall_ns | 77355729 | 92394115 | 19.4406% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 5748 | 5713 | -0.6089% |
| scan_cache_pack_ticks | 2701 | 2615 | -3.1840% |
| block_orchestration_ticks | 737 | 742 | 0.6784% |
| layer_bookkeeping_ticks | 412 | 439 | 6.5534% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1433191 | 1722588 | 20.1925% |
| invocation_ticks | 1433948 | 1723372 | 20.1837% |
| runtime_setup_ticks | 757 | 784 | 3.5667% |
| runtime_teardown_ticks | 796 | 846 | 6.2814% |
| stage_boundary_ticks | 393 | 409 | 4.0712% |
| ledger_named_ticks | 1433948 | 1723372 | 20.1837% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 8 | 13 | 62.5000% |
| metadata_stage_ticks | 4771 | 4782 | 0.2306% |
| input_norm_ticks | 40804 | 40705 | -0.2426% |
| qkv_projection_ticks | 453610 | 453400 | -0.0463% |
| qk_norm_rope_ticks | 11 | 5 | -54.5455% |
| attention_ticks | 114148 | 122137 | 6.9988% |
| o_projection_ticks | 40939 | 40319 | -1.5144% |
| post_attention_residual_ticks | 39227 | 39241 | 0.0357% |
| post_attention_norm_ticks | 23 | 27 | 17.3913% |
| gate_up_ticks | 127666 | 127948 | 0.2209% |
| activation_ticks | 426558 | 708935 | 66.1990% |
| down_ticks | 74202 | 73238 | -1.2992% |
| final_residual_ticks | 63 | 61 | -3.1746% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1142 | 1130 | -1.0508% |
| generation_final_norm_ticks | 277 | 303 | 9.3863% |
| generation_lm_head_ticks | 99232 | 99883 | 0.6560% |
| generation_lm_head_weight_dma_ticks | 98163 | 98809 | 0.6581% |
| generation_lm_head_scale_dma_ticks | 444 | 440 | -0.9009% |
| generation_lm_head_expand_ticks | 70361 | 70355 | -0.0085% |
| generation_lm_head_hmx_ticks | 86502 | 87117 | 0.7110% |
| generation_lm_head_argmax_ticks | 11579 | 11570 | -0.0777% |
| generation_lm_head_weight_dma_wait_ticks | 5661 | 5698 | 0.6536% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 2686 | 3338 | 24.2740% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 594 | 594 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 593 | 593 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 672 | 672 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 5.8720256e+08 | 5.8720256e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.1744051e+09 | 1.1744051e+09 | 0.0000% |
| weight_dma_ticks | 361901 | 362906 | 0.2777% |
| hmx_compute_ticks | 239849 | 244666 | 2.0083% |
| projection_pack_ticks | 303 | 298 | -1.6502% |
| projection_hmx_wait_ticks | 42502 | 40845 | -3.8986% |
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
| attention_softmax_ticks | 54868 | 62764 | 14.3909% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1150362 | 1146091 | -0.3713% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 68387 | 68273 | -0.1667% |
| u8_cache_native_append_update_ticks | 8420 | 8305 | -1.3658% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 11284 | 11083 | -1.7813% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14785 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37843 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10136 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 128882 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 410025 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 168045 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A (zero control) |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 224 | 224 | 0.0000% |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | 0.0000% |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 208910 | 222023 | 6.2769% |
| u8_attention_av_hmx_ticks | 11098 | 11208 | 0.9912% |
| u8_attention_av_requant_ticks | 30971 | 30956 | -0.0484% |
| u8_attention_pipeline_wait_ticks | 45570 | 46777 | 2.6487% |
| w4u8_qkvo_weight_expand_ticks | 119909 | 119795 | -0.0951% |
| w4u8_qkvo_prefetch_wait_ticks | 56941 | 56724 | -0.3811% |
| w4u8_qkvo_hmx_lifetime_ticks | 380075 | 378647 | -0.3757% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2610 | 2610 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | 0.0000% |
| weight_dma_descriptor_count | 3731 | 3731 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 5107072 | 5107072 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 44510781 | 45755521 | 2.7965% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 2860 | 2873 | 0.4545% |
| w4u8_o_gate_prefetch_lifetime_ticks | 9874 | 9883 | 0.0911% |
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
| w4u8_decode_swiglu_row4_call_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 2359296 | 2359296 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 10968 | 11067 | 0.9026% |
| scan_cache_append_ticks | 2434 | 2452 | 0.7395% |
| scan_cache_pack_ticks | 2931 | 2936 | 0.1706% |
| block_orchestration_ticks | 609 | 593 | -2.6273% |
| layer_bookkeeping_ticks | 333 | 314 | -5.7057% |
| scan_dynamic_attention_ticks | 35720 | 35837 | 0.3275% |
| total_ticks | 809378 | 830577 | 2.6192% |
| invocation_ticks | 809950 | 831122 | 2.6140% |
| runtime_setup_ticks | 572 | 545 | -4.7203% |
| runtime_teardown_ticks | 505 | 502 | -0.5941% |
| stage_boundary_ticks | 38 | 37 | -2.6316% |
| ledger_named_ticks | 809950 | 831122 | 2.6140% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 8 | 4 | -50.0000% |
| metadata_stage_ticks | 4124 | 4271 | 3.5645% |
| input_norm_ticks | 7128 | 7127 | -0.0140% |
| qkv_projection_ticks | 452595 | 456060 | 0.7656% |
| qk_norm_rope_ticks | 8 | 6 | -25.0000% |
| attention_ticks | 35849 | 35967 | 0.3292% |
| o_projection_ticks | 26733 | 26551 | -0.6808% |
| post_attention_residual_ticks | 6767 | 6763 | -0.0591% |
| post_attention_norm_ticks | 20 | 20 | 0.0000% |
| gate_up_ticks | 116724 | 116843 | 0.1019% |
| activation_ticks | 24537 | 41976 | 71.0723% |
| down_ticks | 67141 | 67254 | 0.1683% |
| final_residual_ticks | 44 | 38 | -13.6364% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 27 | 28 | 3.7037% |
| generation_final_norm_ticks | 313 | 321 | 2.5559% |
| generation_lm_head_ticks | 60823 | 60835 | 0.0197% |
| generation_lm_head_weight_dma_ticks | 50585 | 50759 | 0.3440% |
| generation_lm_head_scale_dma_ticks | 403 | 393 | -2.4814% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 50598 | 50633 | 0.0692% |
| generation_lm_head_argmax_ticks | 8894 | 8889 | -0.0562% |
| generation_lm_head_weight_dma_wait_ticks | 49228 | 49445 | 0.4408% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 529 | 357 | -32.5142% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 821 | 821 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.4278502e+08 | 7.4278502e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.48557e+09 | 1.48557e+09 | 0.0000% |
| weight_dma_ticks | 311957 | 312637 | 0.2180% |
| hmx_compute_ticks | 170983 | 170057 | -0.5416% |
| projection_pack_ticks | 304 | 304 | 0.0000% |
| projection_hmx_wait_ticks | 18799 | 18582 | -1.1543% |
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
| attention_unattributed_ticks | 16910 | 17116 | 1.2182% |
| u8_attention_qk_norm_rope_ticks | 1147181 | 1156492 | 0.8116% |
| u8_attention_k_pack_ticks | 132 | 134 | 1.5152% |
| u8_attention_v_pack_ticks | 2634 | 2629 | -0.1898% |
| u8_cache_native_append_update_ticks | 5311 | 5325 | 0.2636% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 815 | 793 | -2.6994% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2667 | 2621 | -1.7248% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 8097 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 19494 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 12394 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6027 | 6008 | -0.3152% |
| u8_attention_av_hmx_ticks | 3364 | 3348 | -0.4756% |
| u8_attention_av_requant_ticks | 2829 | 2860 | 1.0958% |
| u8_attention_pipeline_wait_ticks | 1286 | 1251 | -2.7216% |
| w4u8_qkvo_weight_expand_ticks | 119722 | 119717 | -0.0042% |
| w4u8_qkvo_prefetch_wait_ticks | 56700 | 56808 | 0.1905% |
| w4u8_qkvo_hmx_lifetime_ticks | 364823 | 364899 | 0.0208% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2165 | 2165 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | 0.0000% |
| weight_dma_descriptor_count | 3286 | 3286 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4849024 | 4849024 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
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
| host_wall_ns | 76186542 | 90938636 | 19.3631% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 5572.6 | 5588.7 | 0.2889% |
| scan_cache_pack_ticks | 2644 | 2674.5 | 1.1536% |
| block_orchestration_ticks | 759.4 | 762.6 | 0.4214% |
| layer_bookkeeping_ticks | 453.4 | 454.1 | 0.1544% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1432146.8 | 1717119.8 | 19.8983% |
| invocation_ticks | 1433115.5 | 1718116.4 | 19.8868% |
| runtime_setup_ticks | 968.7 | 985.7 | 1.7549% |
| runtime_teardown_ticks | 738.2 | 745.2 | 0.9483% |
| stage_boundary_ticks | 376.8 | 377.9 | 0.2919% |
| ledger_named_ticks | 1433115.5 | 1718116.4 | 19.8868% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 10 | 10.2 | 2.0000% |
| metadata_stage_ticks | 4398.5 | 4425.8 | 0.6207% |
| input_norm_ticks | 40741.7 | 40653.4 | -0.2167% |
| qkv_projection_ticks | 453983.5 | 451699.8 | -0.5030% |
| qk_norm_rope_ticks | 10.3 | 9.4 | -8.7379% |
| attention_ticks | 113751.6 | 122044.7 | 7.2905% |
| o_projection_ticks | 40670.5 | 40744.2 | 0.1812% |
| post_attention_residual_ticks | 39290.8 | 39309.9 | 0.0486% |
| post_attention_norm_ticks | 26.3 | 27.9 | 6.0837% |
| gate_up_ticks | 127829.1 | 127715 | -0.0893% |
| activation_ticks | 426584.6 | 705492.2 | 65.3815% |
| down_ticks | 73120 | 73567.1 | 0.6115% |
| final_residual_ticks | 50.8 | 48.4 | -4.7244% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1086.2 | 1091.4 | 0.4787% |
| generation_final_norm_ticks | 270.3 | 271.8 | 0.5549% |
| generation_lm_head_ticks | 99862.6 | 99471.2 | -0.3919% |
| generation_lm_head_weight_dma_ticks | 98855.2 | 98467.8 | -0.3919% |
| generation_lm_head_scale_dma_ticks | 409.3 | 413.4 | 1.0017% |
| generation_lm_head_expand_ticks | 70300.5 | 70301.6 | 0.0016% |
| generation_lm_head_hmx_ticks | 87214.3 | 86848.3 | -0.4197% |
| generation_lm_head_argmax_ticks | 11565.6 | 11558.5 | -0.0614% |
| generation_lm_head_weight_dma_wait_ticks | 5659.4 | 5682.9 | 0.4152% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 3572.9 | 3207.8 | -10.2186% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 594 | 594 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 593 | 593 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 672 | 672 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 5.8720256e+08 | 5.8720256e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.1744051e+09 | 1.1744051e+09 | 0.0000% |
| weight_dma_ticks | 362648.4 | 362587.2 | -0.0169% |
| hmx_compute_ticks | 242017.3 | 240349.8 | -0.6890% |
| projection_pack_ticks | 306.5 | 306.4 | -0.0326% |
| projection_hmx_wait_ticks | 40834.5 | 41515.4 | 1.6675% |
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
| attention_softmax_ticks | 54797.3 | 62802.8 | 14.6093% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1150011.5 | 1141251 | -0.7618% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 68652.5 | 68651.8 | -0.0010% |
| u8_cache_native_append_update_ticks | 8202.2 | 8221 | 0.2292% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 11216.9 | 11182.1 | -0.3102% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14791.5 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37829.9 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10181.4 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 128126.3 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 408483.2 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 167020.3 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A (zero control) |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 224 | 224 | 0.0000% |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | 0.0000% |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 208850.6 | 221959 | 6.2764% |
| u8_attention_av_hmx_ticks | 11232.3 | 11311.8 | 0.7078% |
| u8_attention_av_requant_ticks | 30923.3 | 30937.9 | 0.0472% |
| u8_attention_pipeline_wait_ticks | 44575.3 | 46658.1 | 4.6725% |
| w4u8_qkvo_weight_expand_ticks | 119959.8 | 120100.1 | 0.1170% |
| w4u8_qkvo_prefetch_wait_ticks | 56930 | 56883 | -0.0826% |
| w4u8_qkvo_hmx_lifetime_ticks | 379328.8 | 380432.9 | 0.2911% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2610 | 2610 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | 0.0000% |
| weight_dma_descriptor_count | 3731 | 3731 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 5107072 | 5107072 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 43652219 | 44557651 | 2.0742% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 2870.4 | 2821.6 | -1.7001% |
| w4u8_o_gate_prefetch_lifetime_ticks | 9880.5 | 9862.6 | -0.1812% |
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
| w4u8_decode_swiglu_row4_call_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 2359296 | 2359296 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 11053.3 | 11089.7 | 0.3293% |
| scan_cache_append_ticks | 2081.5 | 2089 | 0.3603% |
| scan_cache_pack_ticks | 2884.3 | 2873.6 | -0.3710% |
| block_orchestration_ticks | 608.9 | 598.2 | -1.7573% |
| layer_bookkeeping_ticks | 338.8 | 342.5 | 1.0921% |
| scan_dynamic_attention_ticks | 35904.1 | 35941.8 | 0.1050% |
| total_ticks | 810936.9 | 829457.6 | 2.2839% |
| invocation_ticks | 811509 | 830024.3 | 2.2816% |
| runtime_setup_ticks | 557.8 | 565.3 | 1.3446% |
| runtime_teardown_ticks | 497.8 | 499.5 | 0.3415% |
| stage_boundary_ticks | 35.8 | 35.1 | -1.9553% |
| ledger_named_ticks | 811509 | 830024.3 | 2.2816% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 4.8 | 5.6 | 16.6667% |
| metadata_stage_ticks | 4355.7 | 4404.1 | 1.1112% |
| input_norm_ticks | 7137.6 | 7132.8 | -0.0672% |
| qkv_projection_ticks | 453686.1 | 453491.3 | -0.0429% |
| qk_norm_rope_ticks | 6.5 | 5.9 | -9.2308% |
| attention_ticks | 36036.7 | 36076.1 | 0.1093% |
| o_projection_ticks | 26662.1 | 26696.8 | 0.1301% |
| post_attention_residual_ticks | 6764.4 | 6781.3 | 0.2498% |
| post_attention_norm_ticks | 19.2 | 18.7 | -2.6042% |
| gate_up_ticks | 116986.1 | 116764.1 | -0.1898% |
| activation_ticks | 24469.8 | 42429.1 | 73.3937% |
| down_ticks | 67110.1 | 67247 | 0.2040% |
| final_residual_ticks | 36 | 36.4 | 1.1111% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 32 | 30.8 | -3.7500% |
| generation_final_norm_ticks | 314.3 | 317.9 | 1.1454% |
| generation_lm_head_ticks | 61437 | 61741.6 | 0.4958% |
| generation_lm_head_weight_dma_ticks | 51396.1 | 51740.6 | 0.6703% |
| generation_lm_head_scale_dma_ticks | 411.5 | 417.6 | 1.4824% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 51227.6 | 51539.5 | 0.6089% |
| generation_lm_head_argmax_ticks | 8894.9 | 8891.5 | -0.0382% |
| generation_lm_head_weight_dma_wait_ticks | 50106.3 | 50459.7 | 0.7053% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 310.3 | 269.8 | -13.0519% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 821 | 821 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.4278502e+08 | 7.4278502e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.48557e+09 | 1.48557e+09 | 0.0000% |
| weight_dma_ticks | 314336.2 | 314071.7 | -0.0841% |
| hmx_compute_ticks | 167429 | 166892.4 | -0.3205% |
| projection_pack_ticks | 292.1 | 293.5 | 0.4793% |
| projection_hmx_wait_ticks | 18249.4 | 18566.3 | 1.7365% |
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
| attention_unattributed_ticks | 16969.5 | 16967.2 | -0.0136% |
| u8_attention_qk_norm_rope_ticks | 1149137.2 | 1149476.8 | 0.0296% |
| u8_attention_k_pack_ticks | 133.4 | 131.3 | -1.5742% |
| u8_attention_v_pack_ticks | 2635.6 | 2633 | -0.0986% |
| u8_cache_native_append_update_ticks | 4905.6 | 4913.1 | 0.1529% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 770.9 | 762.4 | -1.1026% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2740.8 | 2773.1 | 1.1785% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 8201.9 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 19885.1 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 12409 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6034.2 | 6047 | 0.2121% |
| u8_attention_av_hmx_ticks | 3383 | 3392.9 | 0.2926% |
| u8_attention_av_requant_ticks | 2831.3 | 2842.9 | 0.4097% |
| u8_attention_pipeline_wait_ticks | 1282.1 | 1265.7 | -1.2792% |
| w4u8_qkvo_weight_expand_ticks | 119797.5 | 119804.3 | 0.0057% |
| w4u8_qkvo_prefetch_wait_ticks | 56964.5 | 56947.4 | -0.0300% |
| w4u8_qkvo_hmx_lifetime_ticks | 365321.1 | 365237.1 | -0.0230% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2165 | 2165 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | 0.0000% |
| weight_dma_descriptor_count | 3286 | 3286 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4849024 | 4849024 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
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
| host_wall_ns | 75983143 | 90978880 | 19.7356% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 1 | 1 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
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
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| scan_cache_ddr_write_bytes | 5906432 | 5906432 | 0.0000% |
| scan_cache_stage_ticks | 0 | 0 | N/A (zero control) |
| scan_cache_append_ticks | 5569.65 | 5581.6 | 0.2146% |
| scan_cache_pack_ticks | 2656.15 | 2659.25 | 0.1167% |
| block_orchestration_ticks | 759.7 | 755.35 | -0.5726% |
| layer_bookkeeping_ticks | 455.2 | 462.95 | 1.7025% |
| scan_dynamic_attention_ticks | 0 | 0 | N/A (zero control) |
| total_ticks | 1431039.6 | 1716028.4 | 19.9148% |
| invocation_ticks | 1431994.9 | 1716969.4 | 19.9005% |
| runtime_setup_ticks | 980.75 | 947.75 | -3.3648% |
| runtime_teardown_ticks | 751.15 | 737.9 | -1.7640% |
| stage_boundary_ticks | 377.8 | 369.5 | -2.1969% |
| ledger_named_ticks | 1431994.9 | 1716969.4 | 19.9005% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 9.35 | 10 | 6.9519% |
| metadata_stage_ticks | 4382.6 | 4475.6 | 2.1220% |
| input_norm_ticks | 40748.95 | 40648.45 | -0.2466% |
| qkv_projection_ticks | 453165.05 | 451750.35 | -0.3122% |
| qk_norm_rope_ticks | 9.8 | 9.55 | -2.5510% |
| attention_ticks | 113671.55 | 121655.75 | 7.0239% |
| o_projection_ticks | 40632.2 | 40666.65 | 0.0848% |
| post_attention_residual_ticks | 39278.3 | 39296.15 | 0.0454% |
| post_attention_norm_ticks | 26.3 | 26.15 | -0.5703% |
| gate_up_ticks | 128470 | 128096.05 | -0.2911% |
| activation_ticks | 426611.25 | 703454.45 | 64.8936% |
| down_ticks | 73203.4 | 73669.1 | 0.6362% |
| final_residual_ticks | 49.6 | 48.25 | -2.7218% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 1089.5 | 1087 | -0.2295% |
| generation_final_norm_ticks | 269.85 | 269.1 | -0.2779% |
| generation_lm_head_ticks | 99719.35 | 99545.2 | -0.1746% |
| generation_lm_head_weight_dma_ticks | 98727.65 | 98521.15 | -0.2092% |
| generation_lm_head_scale_dma_ticks | 410.85 | 419.3 | 2.0567% |
| generation_lm_head_expand_ticks | 70421.9 | 70355.5 | -0.0943% |
| generation_lm_head_hmx_ticks | 87069.8 | 86880.1 | -0.2179% |
| generation_lm_head_argmax_ticks | 11562.3 | 11556.3 | -0.0519% |
| generation_lm_head_weight_dma_wait_ticks | 5679.2 | 5688.45 | 0.1629% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 3103.95 | 3036.25 | -2.1811% |
| generation_lm_head_batch_n_tiles | 8 | 8 | 0.0000% |
| generation_lm_head_command_count | 594 | 594 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 593 | 593 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 262400 | 262400 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 112 | 112 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 672 | 672 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 5.8720256e+08 | 5.8720256e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.1744051e+09 | 1.1744051e+09 | 0.0000% |
| weight_dma_ticks | 362984.7 | 362778.8 | -0.0567% |
| hmx_compute_ticks | 239471.15 | 240544.85 | 0.4484% |
| projection_pack_ticks | 306.7 | 305.95 | -0.2445% |
| projection_hmx_wait_ticks | 40913.05 | 41703.1 | 1.9310% |
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
| attention_softmax_ticks | 54788.05 | 62748.35 | 14.5293% |
| attention_av_pack_ticks | 0 | 0 | N/A (zero control) |
| attention_av_hmx_ticks | 0 | 0 | N/A (zero control) |
| attention_av_unpack_ticks | 0 | 0 | N/A (zero control) |
| attention_gqa_pipeline_ticks | 0 | 0 | N/A (zero control) |
| attention_unattributed_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_norm_rope_ticks | 1147430.8 | 1140786.1 | -0.5791% |
| u8_attention_k_pack_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_v_pack_ticks | 68651.9 | 68611.95 | -0.0582% |
| u8_cache_native_append_update_ticks | 8189.85 | 8224.15 | 0.4188% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 28 | 28 | 0.0000% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080 | 4014080 | 0.0000% |
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
| u8_cache_v_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_v_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_bytes | 917504 | 917504 | 0.0000% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_count | 1 | 1 | 0.0000% |
| u8_cache_k_vtcm_tail_row_update_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_fallback_head_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_init_bytes | 827904 | 827904 | 0.0000% |
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
| u8_attention_qk_hmx_ticks | 11142.6 | 11081.55 | -0.5479% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 14789.5 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 37823.35 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 10155 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 127598.35 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 407821.35 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 166189 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 0 | 0 | N/A (zero control) |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 224 | 224 | 0.0000% |
| prefix_seed_metadata_read_bytes | 57344 | 57344 | 0.0000% |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 208838.9 | 221927.1 | 6.2671% |
| u8_attention_av_hmx_ticks | 11176.65 | 11150.5 | -0.2340% |
| u8_attention_av_requant_ticks | 30923.05 | 30958.6 | 0.1150% |
| u8_attention_pipeline_wait_ticks | 44475.75 | 46637.2 | 4.8598% |
| w4u8_qkvo_weight_expand_ticks | 119967.55 | 120123.75 | 0.1302% |
| w4u8_qkvo_prefetch_wait_ticks | 56909.5 | 56842.75 | -0.1173% |
| w4u8_qkvo_hmx_lifetime_ticks | 379669.85 | 380726.25 | 0.2782% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2610 | 2610 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 2031360 | 2031360 | 0.0000% |
| weight_dma_descriptor_count | 3731 | 3731 | 0.0000% |
| boundary_dma_descriptor_count | 289 | 289 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 5107072 | 5107072 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |
### decode_one_step

| Counter | Fused median | Explicit median | Change |
|---|---:|---:|---:|
| experiment | 218 | 218 | 0.0000% |
| generation_step | 1 | 1 | 0.0000% |
| logical_m | 1 | 1 | 0.0000% |
| first_position | 64 | 64 | 0.0000% |
| valid_length | 65 | 65 | 0.0000% |
| host_wall_ns | 43623216 | 44541529 | 2.1051% |
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
| fp16_norm_contexts | 4 | 4 | 0.0000% |
| fp16_norm_rows_per_task | 4 | 4 | 0.0000% |
| fp16_input_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_input_norm_active_contexts | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_task_count | 0 | 0 | N/A (zero control) |
| fp16_post_residual_norm_active_contexts | 0 | 0 | N/A (zero control) |
| repeat_count | 1 | 1 | 0.0000% |
| prepared_session_run_index | 2 | 2 | 0.0000% |
| numerical_audit_enabled | 0 | 0 | N/A (zero control) |
| projection_failure_result | 0 | 0 | N/A (zero control) |
| projection_failure_index | 0 | 0 | N/A (zero control) |
| projection_failure_n_tile | 0 | 0 | N/A (zero control) |
| projection_failure_step | 0 | 0 | N/A (zero control) |
| kv_cache_k_format | 14 | 14 | 0.0000% |
| kv_cache_v_format | 12 | 12 | 0.0000% |
| w4u8_prefill_cache_mode | 1 | 1 | 0.0000% |
| w4u8_delta_reconstruction_mode | 0 | 0 | N/A (zero control) |
| w4u8_decode_softmax_mode | 1 | 1 | 0.0000% |
| w4u8_decode_lm_head_group_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_o_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_o_batch_n_tiles_observed | 16 | 16 | 0.0000% |
| w4u8_o_batch_count | 112 | 112 | 0.0000% |
| w4u8_decode_av_requant_rows | 4 | 4 | 0.0000% |
| w4u8_decode_av_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_av_requant_rows_observed | 4 | 4 | 0.0000% |
| w4u8_av_requant_call_count | 224 | 224 | 0.0000% |
| w4u8_av_requant_vector_count | 1792 | 1792 | 0.0000% |
| w4u8_av_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_common_op_rows | 4 | 4 | 0.0000% |
| w4u8_decode_common_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_common_op_rows_observed | 0 | 0 | N/A (zero control) |
| w4u8_input_norm_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_post_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_final_residual_direct_row4_call_count | 0 | 0 | N/A (zero control) |
| w4u8_common_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_qk_norm_rope_rows | 4 | 4 | 0.0000% |
| w4u8_decode_qk_padding_poison | 0 | 0 | N/A (zero control) |
| w4u8_decode_projection_mode | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_mask | 63 | 63 | 0.0000% |
| w4u8_decode_direct_n_gate_up_batch_n_tiles | 32 | 32 | 0.0000% |
| w4u8_decode_direct_n_gate_up_continuous | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_o_gate_prefetch | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_gate_up_swiglu_stream | 1 | 1 | 0.0000% |
| w4u8_decode_direct_n_qkv_batch_n_tiles | 16 | 16 | 0.0000% |
| w4u8_decode_direct_n_q_batch_n_tiles | 32 | 32 | 0.0000% |
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
| w4u8_o_gate_prefetch_wait_ticks | 2926.75 | 2901.05 | -0.8781% |
| w4u8_o_gate_prefetch_lifetime_ticks | 9951.05 | 9915.45 | -0.3578% |
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
| w4u8_decode_swiglu_row4_call_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_vector_count | 5376 | 0 | -100.0000% |
| w4u8_decode_swiglu_padding_poison_count | 0 | 0 | N/A (zero control) |
| w4u8_decode_swiglu_valid_row_hash | 0 | 0 | N/A (zero control) |
| dsp_status | 3 | 3 | 0.0000% |
| numerical_status | 1 | 1 | 0.0000% |
| scan_logical_m_observed | 1 | 1 | 0.0000% |
| scan_total_kv_length | 65 | 65 | 0.0000% |
| scan_padded_kv_length | 96 | 96 | 0.0000% |
| scan_attention_overlay_capacity_bytes | 2359296 | 2359296 | 0.0000% |
| scan_attention_overlay_required_bytes | 77824 | 77824 | 0.0000% |
| scan_cache_dma_descriptor_count | 1176 | 1176 | 0.0000% |
| scan_cache_append_mismatch_count | 0 | 0 | N/A (zero control) |
| scan_cache_ddr_read_bytes | 4017664 | 4017664 | 0.0000% |
| scan_cache_ddr_write_bytes | 32256 | 32256 | 0.0000% |
| scan_cache_stage_ticks | 11114.05 | 11122.5 | 0.0760% |
| scan_cache_append_ticks | 2059.4 | 2075.6 | 0.7866% |
| scan_cache_pack_ticks | 2886.35 | 2870.05 | -0.5647% |
| block_orchestration_ticks | 609.75 | 597.25 | -2.0500% |
| layer_bookkeeping_ticks | 335.45 | 343.75 | 2.4743% |
| scan_dynamic_attention_ticks | 35941.05 | 35971.8 | 0.0856% |
| total_ticks | 811008.35 | 829148 | 2.2367% |
| invocation_ticks | 811568.95 | 829709.8 | 2.2353% |
| runtime_setup_ticks | 563.3 | 560.15 | -0.5592% |
| runtime_teardown_ticks | 497.25 | 497.9 | 0.1307% |
| stage_boundary_ticks | 34.95 | 35.15 | 0.5722% |
| ledger_named_ticks | 811568.95 | 829709.8 | 2.2353% |
| ledger_unattributed_ticks | 0 | 0 | N/A (zero control) |
| input_stage_ticks | 5.2 | 5.75 | 10.5769% |
| metadata_stage_ticks | 4400.25 | 4400.65 | 0.0091% |
| input_norm_ticks | 7134.4 | 7133.45 | -0.0133% |
| qkv_projection_ticks | 453774.1 | 453427.3 | -0.0764% |
| qk_norm_rope_ticks | 6.9 | 6.3 | -8.6957% |
| attention_ticks | 36072.45 | 36100.85 | 0.0787% |
| o_projection_ticks | 26550.4 | 26598.65 | 0.1817% |
| post_attention_residual_ticks | 6774 | 6778.75 | 0.0701% |
| post_attention_norm_ticks | 18.9 | 19.25 | 1.8519% |
| gate_up_ticks | 117099.15 | 117017.75 | -0.0695% |
| activation_ticks | 24464.15 | 42390.55 | 73.2762% |
| down_ticks | 67265.55 | 67292 | 0.0393% |
| final_residual_ticks | 36.6 | 36.55 | -0.1366% |
| output_stage_ticks | 0 | 0 | N/A (zero control) |
| generation_embedding_ticks | 32.35 | 30.4 | -6.0278% |
| generation_final_norm_ticks | 313.2 | 312.65 | -0.1756% |
| generation_lm_head_ticks | 61158.95 | 61413.6 | 0.4164% |
| generation_lm_head_weight_dma_ticks | 51082.8 | 51408.55 | 0.6377% |
| generation_lm_head_scale_dma_ticks | 411.35 | 415.8 | 1.0818% |
| generation_lm_head_expand_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_ticks | 50945.7 | 51204.2 | 0.5074% |
| generation_lm_head_argmax_ticks | 8897.15 | 8891.2 | -0.0669% |
| generation_lm_head_weight_dma_wait_ticks | 49776.75 | 50114.15 | 0.6778% |
| generation_lm_head_scale_init_ticks | 0 | 0 | N/A (zero control) |
| generation_lm_head_hmx_tail_wait_ticks | 312.8 | 280.35 | -10.3740% |
| generation_lm_head_batch_n_tiles | 32 | 32 | 0.0000% |
| generation_lm_head_command_count | 149 | 149 | 0.0000% |
| generation_lm_head_n_tiles | 4748 | 4748 | 0.0000% |
| generation_lm_head_prefetch_count | 148 | 148 | 0.0000% |
| generation_lm_head_scale_resident_bytes | 1215488 | 1215488 | 0.0000% |
| generation_embedding_ddr_read_bytes | 4352 | 4352 | 0.0000% |
| generation_lm_head_ddr_read_bytes | 1.5679795e+08 | 1.5679795e+08 | 0.0000% |
| w4u8_decode_direct_n_projection_count | 113 | 113 | 0.0000% |
| w4u8_decode_direct_n_hmx_command_count | 821 | 821 | 0.0000% |
| w4u8_mlp_down_hmx_command_count | 224 | 224 | 0.0000% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 7.4278502e+08 | 7.4278502e+08 | 0.0000% |
| w4u8_decode_direct_n_expand_bytes_avoided | 1.48557e+09 | 1.48557e+09 | 0.0000% |
| weight_dma_ticks | 313680.75 | 314124.35 | 0.1414% |
| hmx_compute_ticks | 168909.3 | 168702.3 | -0.1226% |
| projection_pack_ticks | 292.3 | 290.85 | -0.4961% |
| projection_hmx_wait_ticks | 18436.35 | 18345.3 | -0.4939% |
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
| attention_unattributed_ticks | 17040 | 17067.15 | 0.1593% |
| u8_attention_qk_norm_rope_ticks | 1150219.8 | 1148413.9 | -0.1570% |
| u8_attention_k_pack_ticks | 133.55 | 133.1 | -0.3370% |
| u8_attention_v_pack_ticks | 2638.95 | 2635.55 | -0.1288% |
| u8_cache_native_append_update_ticks | 4895.65 | 4905.15 | 0.1940% |
| u8_cache_native_prefill_build_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| u8_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_native_incremental_append_count | 28 | 28 | 0.0000% |
| u8_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_tail_append_count | 28 | 28 | 0.0000% |
| u8_cache_segment_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_segment_sealed_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_append_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_attention_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_partial_pack_rows | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_full_tile_rmw_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_quartet_native_load_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_row_update_count | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_publish_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_partial_pack_rows | 224 | 224 | 0.0000% |
| u8_cache_v_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_v_vtcm_tail_native_load_bytes | 114688 | 114688 | 0.0000% |
| u8_cache_k_vtcm_tail_init_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_seal_count | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_cached_head_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28 | 28 | 0.0000% |
| u8_cache_k_vtcm_tail_init_bytes | 0 | 0 | N/A (zero control) |
| u8_cache_k_vtcm_tail_native_load_bytes | 802816 | 802816 | 0.0000% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 784 | 784 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 768.3 | 763.95 | -0.5662% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196 | 196 | 0.0000% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088 | 25088 | 0.0000% |
| f16_cache_native_prefill_reuse_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_prefill_reused_carrier_bytes | 0 | 0 | N/A (zero control) |
| f16_cache_native_incremental_append_count | 0 | 0 | N/A (zero control) |
| f16_cache_full_prefix_pack_count | 0 | 0 | N/A (zero control) |
| f16_cache_native_append_update_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_qk_hmx_ticks | 2750.6 | 2741.35 | -0.3363% |
| wide_score_mode | 7 | 7 | 0.0000% |
| paper_format_disable | 0 | 0 | N/A (zero control) |
| paper_pipeline_disable | 3 | 3 | 0.0000% |
| fp_qdq_split | 0 | 1 | N/A (zero control) |
| fp_softmax_dq_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_compute_ticks | 0 | 0 | N/A (zero control) |
| fp_softmax_q_ticks | 0 | 0 | N/A (zero control) |
| fp_swiglu_dq_ticks | 0 | 8172 | N/A (zero control) |
| fp_swiglu_compute_ticks | 0 | 19856.95 | N/A (zero control) |
| fp_swiglu_q_ticks | 0 | 12377.6 | N/A (zero control) |
| fp32_residual | 2 | 2 | 0.0000% |
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
| generation_lm_head_direct_slot_join_count | 147 | 147 | 0.0000% |
| dense_r3_total_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_rows | 0 | 0 | N/A (zero control) |
| dense_r3_total_hmx_calls | 0 | 0 | N/A (zero control) |
| dense_r3_total_refined_values | 0 | 0 | N/A (zero control) |
| dense_r3_total_prepare_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_matmul_ticks | 0 | 0 | N/A (zero control) |
| dense_r3_total_finish_ticks | 0 | 0 | N/A (zero control) |
| prefix_kv_mode | 1 | 1 | 0.0000% |
| prefix_group_patch_count | 0 | 0 | N/A (zero control) |
| prefix_seed_metadata_read_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_qk_requant_ticks | 0 | 0 | N/A (zero control) |
| u8_attention_softmax_ticks | 6035.4 | 6050.75 | 0.2543% |
| u8_attention_av_hmx_ticks | 3372.15 | 3362.7 | -0.2802% |
| u8_attention_av_requant_ticks | 2827.25 | 2827.85 | 0.0212% |
| u8_attention_pipeline_wait_ticks | 1274.95 | 1286 | 0.8667% |
| w4u8_qkvo_weight_expand_ticks | 119808.85 | 119803.35 | -0.0046% |
| w4u8_qkvo_prefetch_wait_ticks | 56965 | 56998.65 | 0.0591% |
| w4u8_qkvo_hmx_lifetime_ticks | 365648.7 | 365325.25 | -0.0885% |
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
| vtcm_peak_plan_bytes | 8365824 | 8365824 | 0.0000% |
| block_invocation_count | 28 | 28 | 0.0000% |
| hmx_command_count | 2165 | 2165 | 0.0000% |
| hmx_fp16_tile_pair_count | 0 | 0 | N/A (zero control) |
| hmx_u8s8_tile_pair_count | 1690880 | 1690880 | 0.0000% |
| weight_dma_descriptor_count | 3286 | 3286 | 0.0000% |
| boundary_dma_descriptor_count | 226 | 226 | 0.0000% |
| intermediate_dma_descriptor_count | 0 | 0 | N/A (zero control) |
| intermediate_spill_fill_count | 0 | 0 | N/A (zero control) |
| weight_ddr_read_bytes | 8.6603264e+08 | 8.6603264e+08 | 0.0000% |
| boundary_ddr_read_bytes | 4849024 | 4849024 | 0.0000% |
| boundary_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_read_bytes | 0 | 0 | N/A (zero control) |
| intermediate_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_audit_ddr_write_bytes | 0 | 0 | N/A (zero control) |
| u8_attention_probability_mask_violation_count | 0 | 0 | N/A (zero control) |
| u8_attention_fused_k_operand_mismatch_count | 0 | 0 | N/A (zero control) |
| w4f16_expand_mismatch_count | 0 | 0 | N/A (zero control) |

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
      "file": "audit1-fused-a01/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit1-split-a01/hash-checks.json",
      "checks": 2,
      "normalized_numeric_match": true
    },
    {
      "file": "audit3-split-a01/hash-checks.json",
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
    "fp_islands": true,
    "model_size": "1.7B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "cd2f12a5c61816645d90ff48c57452d33395fdbbe4b8acb784ce08124a7bfa11",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "d8a64c5294a3ed0b1d184ee3dbef33b0752ccacbf311e2db263b1985c19f943d"
    }
  },
  {
    "path": "binaries-28-a01/seal.json",
    "fp_islands": true,
    "model_size": "1.7B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "e5fda2ab2f65a2b65b81a0472bbcf885d5916f124eb9fc502164d60d47c50dd3",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "3c78f6c9b009660b921048932e39b8df62e9cc5910f1d0f0a72a6a31441cebde"
    }
  },
  {
    "path": "binaries-3-a01/seal.json",
    "fp_islands": true,
    "model_size": "1.7B",
    "paper_trace": false,
    "source_head": "51a00a484e7c0209cdc60441c08a5312c3386c9a",
    "files": {
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/qwen3_block_cli": "824cc88121003cc315f8c11b97070971f827a6cdf4c377d178133fbec31fc071",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/llama_sp2_cli": "024de59265ce82fd5a8fc9726d90b86a24d3566e34f4b9776060a520d751f118",
      "/home/daniuniu/work/qwen3-block-htp/android_ReleaseG_aarch64/ship/libqwen3_probe.so": "971b970add91345bd28587f3e5f46d0d085597e5e9f426b71a21085a67d94793",
      "/home/daniuniu/work/qwen3-block-htp/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so": "cc72179393a05c5565bde70230b4368c584b7cbe95615228ccd2a93b804765da"
    }
  }
]

Historical G fused measurement is preserved. No W16A16/W4A16 comparison measured: outside this diagnostic request.

