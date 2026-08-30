# EXP-0073 — Complete profiling report

## PC-028 current-best three-variant module wall-time

EXP-0072 supplies the W4U8 column. Percentages are shares of complete Host wall; positive speed means W4U8 is faster than W4F16.

| Module | W16A16 | W4A16 | W4A8 current best | A8 vs A16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +80.8% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 19.6 us (0.9%) | +118.0% |
| QKV + Q/K Norm/RoPE | 397.1 us (16.0%) | 439.2 us (19.7%) | 425.4 us (20.6%) | +3.2% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 244.5 us (11.8%) | -42.4% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 170.5 us (8.3%) | +2.0% |
| Post-attn residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 34.1 us (1.7%) | +20.8% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 713.8 us (34.6%) | +36.3% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 358.1 us (17.3%) | -8.6% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 17.5 us (0.8%) | -71.4% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 77.8 us (3.8%) | +2.4% |
| Complete block Host wall | 2488.3 us | 2229.7 us | 2065.3 us | +8.0% |

The control constructs the same fourteen-template SOLE bank inside each of eight GQA tasks. The candidate constructs one bank in an existing VTCM lifetime gap and shares it read-only. All row loops, LUT values, task scheduling and HMX commands are unchanged.

## Repeat 1

### Primary wall-latency targets

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,506,562.000 | 2,499,166.000 | -0.295% | -1.147% |
| `attention_ticks` | 4,692.000 | 4,904.000 | 4.518% | 4.586% |
| `invocation_ticks` | 39,667.000 | 39,979.000 | 0.787% | 0.601% |
| `total_ticks` | 38,948.000 | 39,261.000 | 0.804% | 0.576% |

### Additive Block Timing Ledger

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 51.000 | 51.000 | 0.000% | 0.000% |
| `metadata_stage_ticks` | 119.000 | 116.000 | -2.521% | -2.586% |
| `input_norm_ticks` | 452.000 | 371.000 | -17.920% | -19.089% |
| `qkv_projection_ticks` | 8,195.000 | 8,179.000 | -0.195% | -0.195% |
| `qk_norm_rope_ticks` | 0.000 | 0.000 | n/a | 0.000% |
| `attention_ticks` | 4,692.000 | 4,904.000 | 4.518% | 4.586% |
| `o_projection_ticks` | 3,269.000 | 3,273.000 | 0.122% | -0.031% |
| `post_attention_residual_ticks` | 631.000 | 631.000 | 0.000% | -0.626% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | n/a |
| `gate_up_ticks` | 13,673.000 | 13,598.000 | -0.549% | 0.096% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,832.000 | 6,935.000 | 1.508% | 1.332% |
| `final_residual_ticks` | 341.000 | 336.000 | -1.466% | -2.632% |
| `output_stage_ticks` | 124.000 | 125.000 | 0.806% | 0.000% |
| `stage_boundary_ticks` | 28.000 | 27.000 | -3.571% | -3.571% |

### Overlapping engine work and waits

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 8.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,671.000 | 10,672.000 | 0.009% | 0.402% |
| `hmx_compute_ticks` | 13,065.000 | 13,176.000 | 0.850% | 2.021% |
| `projection_hmx_wait_ticks` | 222.000 | 224.000 | 0.901% | 1.709% |
| `hmx_ready_wait_ticks` | 8,011.000 | 8,047.000 | 0.449% | 1.395% |
| `w4u8_qkvo_weight_expand_ticks` | 6,401.000 | 6,395.000 | -0.094% | -0.016% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,025.000 | 3,044.000 | 0.628% | 1.143% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,292.000 | 9,310.000 | 0.194% | 0.553% |
| `u8_attention_qk_norm_rope_ticks` | 28,642.000 | 28,558.000 | -0.293% | 0.705% |
| `u8_attention_v_pack_ticks` | 3,199.000 | 3,183.000 | -0.500% | 0.189% |
| `u8_attention_qk_hmx_ticks` | 400.000 | 400.000 | 0.000% | -0.255% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,982.000 | 14,979.000 | 7.131% | 6.693% |
| `u8_attention_av_hmx_ticks` | 460.000 | 485.000 | 5.435% | 5.217% |
| `u8_attention_av_requant_ticks` | 1,080.000 | 1,068.000 | -1.111% | -2.198% |
| `u8_attention_pipeline_wait_ticks` | 942.000 | 1,093.000 | 16.030% | 25.346% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,012.000 | 12,943.000 | -0.530% | 0.124% |
| `w4u8_mlp_down_pipeline_ticks` | 5,947.000 | 5,995.000 | 0.807% | 1.181% |
| `w4u8_mlp_activation_work_ticks` | 7,903.000 | 7,886.000 | -0.215% | -0.291% |
| `w4u8_mlp_weight_stage_ticks` | 7,357.000 | 7,383.000 | 0.353% | -0.109% |
| `w4u8_mlp_weight_expand_ticks` | 25,155.000 | 24,966.000 | -0.751% | -0.124% |
| `w4u8_mlp_hmx_compute_ticks` | 7,766.000 | 7,703.000 | -0.811% | -0.811% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,823.000 | 8,863.000 | 0.453% | 1.552% |
| `w4u8_mlp_producer_slot_wait_ticks` | 302.000 | 302.000 | 0.000% | 0.662% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,991.000 | 4,862.000 | -2.585% | 0.573% |
| `attention_qk_norm_pool_wait_ticks` | 1,271.000 | 1,226.000 | -3.541% | -1.574% |
| `w4u8_input_norm_main_work_ticks` | 320.000 | 322.000 | 0.625% | 1.266% |
| `w4u8_input_norm_worker_work_ticks` | 1,674.000 | 1,423.000 | -14.994% | -11.393% |
| `w4u8_input_norm_pool_wait_ticks` | 106.000 | 8.000 | -92.453% | -82.727% |
| `w4u8_post_residual_main_work_ticks` | 395.000 | 588.000 | 48.861% | 4.304% |
| `w4u8_post_residual_worker_work_ticks` | 2,731.000 | 2,715.000 | -0.586% | -0.659% |
| `w4u8_post_residual_pool_wait_ticks` | 203.000 | 186.000 | -8.374% | -8.374% |
| `w4u8_final_residual_main_work_ticks` | 206.000 | 207.000 | 0.485% | 0.000% |
| `w4u8_final_residual_worker_work_ticks` | 1,317.000 | 1,319.000 | 0.152% | 0.076% |
| `w4u8_final_residual_pool_wait_ticks` | 92.000 | 88.000 | -4.348% | -4.348% |

### Traffic, commands, counters and residency

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 176.0 | 176.0 | 0.000% | 0.000% |
| `hmx_u8s8_tile_pair_count` | 49,408.0 | 49,408.0 | 0.000% | 0.000% |
| `weight_dma_descriptor_count` | 512.0 | 512.0 | 0.000% | 0.000% |
| `weight_ddr_read_bytes` | 25,444,352.0 | 25,444,352.0 | 0.000% | 0.000% |
| `boundary_ddr_read_bytes` | 304,096.0 | 304,096.0 | 0.000% | 0.000% |
| `boundary_ddr_write_bytes` | 131,072.0 | 131,072.0 | 0.000% | 0.000% |
| `intermediate_ddr_read_bytes` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_ddr_write_bytes` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_dma_descriptor_count` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_spill_fill_count` | 0.0 | 0.0 | n/a | n/a |
| `u8_attention_direct_o_tile_count` | 64.0 | 64.0 | 0.000% | 0.000% |
| `u8_attention_qkv_unpack_skipped` | 128.0 | 128.0 | 0.000% | 0.000% |
| `w4u8_qkv_batch_count` | 32.0 | 32.0 | 0.000% | 0.000% |
| `w4u8_qkvo_prefetch_count` | 44.0 | 44.0 | 0.000% | 0.000% |
| `w4u8_qkvo_overlap_schedule_count` | 44.0 | 44.0 | 0.000% | 0.000% |
| `w4u8_mlp_input_pack_skipped` | 1.0 | 1.0 | 0.000% | 0.000% |
| `w4u8_mlp_output_unpack_skipped` | 1.0 | 1.0 | 0.000% | 0.000% |
| `w4u8_mlp_pair_publish_count` | 192.0 | 192.0 | 0.000% | 0.000% |
| `w4u8_mlp_pair_consume_count` | 192.0 | 192.0 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hmx_command_count` | 48.0 | 48.0 | 0.000% | 0.000% |
| `w4u8_mlp_down_hmx_command_count` | 64.0 | 64.0 | 0.000% | 0.000% |
| `vtcm_requested_bytes` | 8,388,608.0 | 8,388,608.0 | 0.000% | 0.000% |
| `vtcm_acquired_bytes` | 8,388,608.0 | 8,388,608.0 | 0.000% | 0.000% |
| `vtcm_peak_plan_bytes` | 5,306,080.0 | 5,306,080.0 | 0.000% | 0.000% |
| `attention_hvx_workers_created` | 5.000 | 5.000 | 0.000% | 0.000% |
| `attention_hvx_workers_locked` | 5.000 | 5.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 722.000 | 717.000 | -0.693% | -0.693% |
| `runtime_teardown_ticks` | 633.000 | 630.000 | -0.474% | -0.317% |
| `ledger_named_ticks` | 39,667.000 | 39,979.000 | 0.787% | 0.601% |
| `ledger_unattributed_ticks` | 0.000 | 0.000 | n/a | n/a |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
| `w4u8_input_norm_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_post_residual_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_final_residual_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_residual_active_contexts` | 6.000 | 6.000 | 0.000% | 0.000% |
| `u8_attention_lut_template_build_count` | 8.0 | 1.0 | -87.500% | -87.500% |

Two-target speed gate: **FAIL**; unchanged physical contract: **PASS**; one shared template build: **PASS**.

## Repeat 10

### Primary wall-latency targets

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,065,255.200 | 2,076,588.600 | 0.549% | 0.834% |
| `attention_ticks` | 4,694.000 | 4,898.200 | 4.350% | 4.270% |
| `invocation_ticks` | 38,332.100 | 38,411.600 | 0.207% | 0.333% |
| `total_ticks` | 38,260.700 | 38,329.700 | 0.180% | 0.333% |

### Additive Block Timing Ledger

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 54.400 | 55.300 | 1.654% | -1.113% |
| `metadata_stage_ticks` | 11.800 | 12.000 | 1.695% | 0.000% |
| `input_norm_ticks` | 376.000 | 375.300 | -0.186% | -0.240% |
| `qkv_projection_ticks` | 8,168.200 | 8,156.000 | -0.149% | -0.355% |
| `qk_norm_rope_ticks` | 0.200 | 0.300 | 50.000% | 50.000% |
| `attention_ticks` | 4,694.000 | 4,898.200 | 4.350% | 4.270% |
| `o_projection_ticks` | 3,273.100 | 3,271.900 | -0.037% | -0.092% |
| `post_attention_residual_ticks` | 654.400 | 666.200 | 1.803% | 1.803% |
| `post_attention_norm_ticks` | 0.200 | 0.200 | 0.000% | -33.333% |
| `gate_up_ticks` | 13,704.500 | 13,599.000 | -0.770% | -0.390% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,874.600 | 6,924.200 | 0.721% | 0.448% |
| `final_residual_ticks` | 336.100 | 333.500 | -0.774% | -0.446% |
| `output_stage_ticks` | 12.400 | 12.500 | 0.806% | 0.806% |
| `stage_boundary_ticks` | 25.000 | 25.200 | 0.800% | 1.200% |

### Overlapping engine work and waits

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.300 | 8.100 | -2.410% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,636.200 | 10,521.900 | -1.075% | -1.003% |
| `hmx_compute_ticks` | 13,097.500 | 12,953.600 | -1.099% | -0.944% |
| `projection_hmx_wait_ticks` | 226.900 | 225.100 | -0.793% | 1.683% |
| `hmx_ready_wait_ticks` | 8,034.500 | 7,991.400 | -0.536% | 0.019% |
| `w4u8_qkvo_weight_expand_ticks` | 6,402.700 | 6,413.900 | 0.175% | -0.014% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,077.600 | 3,044.800 | -1.066% | -1.066% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,345.800 | 9,294.600 | -0.548% | -0.233% |
| `u8_attention_qk_norm_rope_ticks` | 28,568.500 | 28,595.200 | 0.093% | -0.209% |
| `u8_attention_v_pack_ticks` | 3,204.900 | 3,201.100 | -0.119% | -0.072% |
| `u8_attention_qk_hmx_ticks` | 411.300 | 406.400 | -1.191% | -1.580% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,960.500 | 15,103.300 | 8.186% | 8.186% |
| `u8_attention_av_hmx_ticks` | 471.700 | 471.500 | -0.042% | -0.485% |
| `u8_attention_av_requant_ticks` | 1,069.500 | 1,066.400 | -0.290% | -0.271% |
| `u8_attention_pipeline_wait_ticks` | 1,043.300 | 1,037.800 | -0.527% | -0.945% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,045.700 | 12,948.900 | -0.742% | -0.438% |
| `w4u8_mlp_down_pipeline_ticks` | 5,989.900 | 6,039.000 | 0.820% | 0.462% |
| `w4u8_mlp_activation_work_ticks` | 7,924.600 | 7,912.000 | -0.159% | -0.198% |
| `w4u8_mlp_weight_stage_ticks` | 7,408.300 | 7,308.300 | -1.350% | -0.566% |
| `w4u8_mlp_weight_expand_ticks` | 25,071.400 | 24,896.200 | -0.699% | -0.629% |
| `w4u8_mlp_hmx_compute_ticks` | 7,796.500 | 7,782.500 | -0.180% | 0.553% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,850.300 | 8,809.400 | -0.462% | 0.108% |
| `w4u8_mlp_producer_slot_wait_ticks` | 306.300 | 301.600 | -1.534% | -1.730% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,927.100 | 4,920.700 | -0.130% | 0.098% |
| `attention_qk_norm_pool_wait_ticks` | 1,202.700 | 1,245.300 | 3.542% | 2.849% |
| `w4u8_input_norm_main_work_ticks` | 267.500 | 271.400 | 1.458% | 1.724% |
| `w4u8_input_norm_worker_work_ticks` | 1,490.400 | 1,481.200 | -0.617% | -0.617% |
| `w4u8_input_norm_pool_wait_ticks` | 73.700 | 66.900 | -9.227% | -12.549% |
| `w4u8_post_residual_main_work_ticks` | 510.300 | 510.500 | 0.039% | 2.923% |
| `w4u8_post_residual_worker_work_ticks` | 2,707.800 | 2,750.200 | 1.566% | 1.363% |
| `w4u8_post_residual_pool_wait_ticks` | 119.100 | 120.600 | 1.259% | -4.840% |
| `w4u8_final_residual_main_work_ticks` | 206.200 | 206.300 | 0.048% | 0.048% |
| `w4u8_final_residual_worker_work_ticks` | 1,314.700 | 1,311.700 | -0.228% | -0.145% |
| `w4u8_final_residual_pool_wait_ticks` | 88.300 | 87.300 | -1.133% | -2.676% |

### Traffic, commands, counters and residency

| Metric | per-group control | shared candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 176.0 | 176.0 | 0.000% | 0.000% |
| `hmx_u8s8_tile_pair_count` | 49,408.0 | 49,408.0 | 0.000% | 0.000% |
| `weight_dma_descriptor_count` | 512.0 | 512.0 | 0.000% | 0.000% |
| `weight_ddr_read_bytes` | 25,444,352.0 | 25,444,352.0 | 0.000% | 0.000% |
| `boundary_ddr_read_bytes` | 148,374.4 | 148,374.4 | 0.000% | 0.000% |
| `boundary_ddr_write_bytes` | 13,107.2 | 13,107.2 | 0.000% | 0.000% |
| `intermediate_ddr_read_bytes` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_ddr_write_bytes` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_dma_descriptor_count` | 0.0 | 0.0 | n/a | n/a |
| `intermediate_spill_fill_count` | 0.0 | 0.0 | n/a | n/a |
| `u8_attention_direct_o_tile_count` | 64.0 | 64.0 | 0.000% | 0.000% |
| `u8_attention_qkv_unpack_skipped` | 128.0 | 128.0 | 0.000% | 0.000% |
| `w4u8_qkv_batch_count` | 32.0 | 32.0 | 0.000% | 0.000% |
| `w4u8_qkvo_prefetch_count` | 44.0 | 44.0 | 0.000% | 0.000% |
| `w4u8_qkvo_overlap_schedule_count` | 44.0 | 44.0 | 0.000% | 0.000% |
| `w4u8_mlp_input_pack_skipped` | 1.0 | 1.0 | 0.000% | 0.000% |
| `w4u8_mlp_output_unpack_skipped` | 1.0 | 1.0 | 0.000% | 0.000% |
| `w4u8_mlp_pair_publish_count` | 192.0 | 192.0 | 0.000% | 0.000% |
| `w4u8_mlp_pair_consume_count` | 192.0 | 192.0 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hmx_command_count` | 48.0 | 48.0 | 0.000% | 0.000% |
| `w4u8_mlp_down_hmx_command_count` | 64.0 | 64.0 | 0.000% | 0.000% |
| `vtcm_requested_bytes` | 8,388,608.0 | 8,388,608.0 | 0.000% | 0.000% |
| `vtcm_acquired_bytes` | 8,388,608.0 | 8,388,608.0 | 0.000% | 0.000% |
| `vtcm_peak_plan_bytes` | 5,306,080.0 | 5,306,080.0 | 0.000% | 0.000% |
| `attention_hvx_workers_created` | 5.000 | 5.000 | 0.000% | 0.000% |
| `attention_hvx_workers_locked` | 5.000 | 5.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 71.500 | 72.100 | 0.839% | 1.259% |
| `runtime_teardown_ticks` | 63.100 | 63.100 | 0.000% | 0.159% |
| `ledger_named_ticks` | 38,332.100 | 38,411.600 | 0.207% | 0.333% |
| `ledger_unattributed_ticks` | 0.000 | 0.000 | n/a | n/a |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
| `w4u8_input_norm_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_post_residual_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_final_residual_task_count` | 16.0 | 16.0 | 0.000% | 0.000% |
| `w4u8_residual_active_contexts` | 0.600 | 0.600 | 0.000% | 0.000% |
| `u8_attention_lut_template_build_count` | 8.0 | 1.0 | -87.500% | -87.500% |

Two-target speed gate: **FAIL**; unchanged physical contract: **PASS**; one shared template build: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final block output | byte-exact to EXP-0072, 0 LSB |
| Input norm / QK / probability / AV boundaries | byte-exact to EXP-0072 |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains primary.

## Decision

EXP-0073 local gate: **FAIL**. Local adoption eligibility: **NO**. Selected Baseline is unchanged without explicit user promotion.


