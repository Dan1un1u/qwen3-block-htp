# EXP-0054 Stage A — Complete profiling report

## Stable three-variant overview (repeat 10)

| Module | F16F16 | W4F16 Selected Baseline | W4U8 EXP-0053 | W4U8 speed vs W4F16 |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +79.0% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 90.9 us (3.3%) | -53.0% |
| QKV + Q/K Norm/RoPE preparation | 397.1 us (16.0%) | 439.2 us (19.7%) | 831.7 us (30.4%) | -47.2% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 421.8 us (15.4%) | -66.7% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 171.5 us (6.3%) | +1.4% |
| Post-Attention residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 45.8 us (1.7%) | -9.9% |
| Gate/Up + SwiGLU | 1116.6 us (44.9%) | 972.7 us (43.6%) | 717.8 us (26.2%) | +35.5% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 354.6 us (12.9%) | -7.7% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 21.8 us (0.8%) | -77.2% |
| Host/RPC and closure gap | 77.5 us (3.1%) | 79.6 us (3.6%) | 80.2 us (2.9%) | -0.7% |
| Complete block Host wall | 2488.3 us (100.0%) | 2229.7 us (100.0%) | 2740.2 us (100.0%) | -18.6% |

F16F16 and W4F16 values come from EXP-0038 formal evidence; the W4U8 column remains EXP-0053 because failed EXP-0054 Stage A is not eligible to replace it.

Stage A publishes each native V head from projection and lets the persistent HVX pool build the AV RHS operand before the Attention callback. Arithmetic, qparams, weights, HMX work, traffic and all other schedules are fixed. Because V-pack work changes attribution, the primary local interval is QKV plus Attention, not either ledger entry alone.

## Repeat 1

### Primary latency gate

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,119,532.000 | 3,184,427.000 | 2.080% | 3.139% |
| `invocation_ticks` | 51,800.000 | 53,313.000 | 2.921% | 2.770% |
| `total_ticks` | 51,305.000 | 52,810.000 | 2.933% | 2.791% |
| `qkv_attention_ticks` | 23,872.000 | 25,268.000 | 5.848% | 5.237% |
| `qkv_projection_ticks` | 15,811.000 | 18,376.000 | 16.223% | 16.210% |
| `attention_ticks` | 8,046.000 | 6,892.000 | -14.343% | -15.573% |
| `attention_qk_norm_pool_wait_ticks` | 8,939.000 | 11,531.000 | 28.997% | 28.418% |

### Additive Block Timing Ledger

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 50.000 | 53.000 | 6.000% | 3.922% |
| `metadata_stage_ticks` | 118.000 | 120.000 | 1.695% | 0.840% |
| `input_norm_ticks` | 1,750.000 | 1,751.000 | 0.057% | 0.000% |
| `qkv_projection_ticks` | 15,811.000 | 18,376.000 | 16.223% | 16.210% |
| `qk_norm_rope_ticks` | 2.000 | 3.000 | 50.000% | -25.000% |
| `attention_ticks` | 8,046.000 | 6,892.000 | -14.343% | -15.573% |
| `o_projection_ticks` | 3,284.000 | 3,287.000 | 0.091% | -0.061% |
| `post_attention_residual_ticks` | 876.000 | 879.000 | 0.342% | 0.567% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,476.000 | 13,628.000 | 1.128% | 0.702% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,901.000 | 6,880.000 | -0.304% | 0.129% |
| `final_residual_ticks` | 430.000 | 425.000 | -1.163% | -0.930% |
| `output_stage_ticks` | 123.000 | 125.000 | 1.626% | 0.813% |

### Overlapping engine work and waits

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 7.000 | -12.500% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,733.000 | 10,739.000 | 0.056% | 0.132% |
| `hmx_compute_ticks` | 13,625.000 | 13,845.000 | 1.615% | 1.383% |
| `projection_hmx_wait_ticks` | 222.000 | 227.000 | 2.252% | -0.926% |
| `hmx_ready_wait_ticks` | 7,853.000 | 7,878.000 | 0.318% | -0.051% |
| `w4u8_qkvo_weight_expand_ticks` | 6,404.000 | 6,363.000 | -0.640% | -0.593% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,078.000 | 3,076.000 | -0.065% | 0.864% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,288.000 | 9,261.000 | -0.291% | 0.021% |
| `u8_attention_qk_norm_rope_ticks` | 44,486.000 | 44,394.000 | -0.207% | -0.240% |
| `u8_attention_v_pack_ticks` | 8,231.000 | 7,910.000 | -3.900% | -4.032% |
| `u8_attention_qk_hmx_ticks` | 1,322.000 | 1,392.000 | 5.295% | 6.095% |
| `u8_attention_qk_requant_ticks` | 389.000 | 391.000 | 0.514% | 2.073% |
| `u8_attention_softmax_ticks` | 13,978.000 | 13,850.000 | -0.916% | -1.241% |
| `u8_attention_av_hmx_ticks` | 2,635.000 | 2,656.000 | 0.797% | 1.214% |
| `u8_attention_av_requant_ticks` | 1,050.000 | 1,066.000 | 1.524% | 0.961% |
| `u8_attention_pipeline_wait_ticks` | 2,966.000 | 4,948.000 | 66.824% | 50.641% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,821.000 | 12,977.000 | 1.217% | 0.698% |
| `w4u8_mlp_down_pipeline_ticks` | 5,964.000 | 5,991.000 | 0.453% | 1.062% |
| `w4u8_mlp_activation_work_ticks` | 7,898.000 | 7,904.000 | 0.076% | -0.114% |
| `w4u8_mlp_weight_stage_ticks` | 7,413.000 | 7,409.000 | -0.054% | -0.054% |
| `w4u8_mlp_weight_expand_ticks` | 24,935.000 | 25,059.000 | 0.497% | 0.416% |
| `w4u8_mlp_hmx_compute_ticks` | 7,857.000 | 7,835.000 | -0.280% | 0.000% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,657.000 | 8,762.000 | 1.213% | 0.069% |
| `w4u8_mlp_producer_slot_wait_ticks` | 294.000 | 298.000 | 1.361% | 0.687% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,800.000 | 4,895.000 | 1.979% | 2.801% |
| `u8_attention_k_pack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `attention_qk_norm_pool_wait_ticks` | 8,939.000 | 11,531.000 | 28.997% | 28.418% |

### Traffic, commands, counters and residency

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 256.0 | 0.000% | 0.000% |
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
| `attention_hvx_workers_created` | 3.000 | 3.000 | 0.000% | 0.000% |
| `attention_hvx_workers_locked` | 3.000 | 3.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 496.000 | 498.000 | 0.403% | 0.605% |
| `runtime_teardown_ticks` | 427.000 | 427.000 | 0.000% | 0.000% |
| `ledger_named_ticks` | 51,773.000 | 53,287.000 | 2.924% | 2.776% |
| `ledger_unattributed_ticks` | 29.000 | 28.000 | -3.448% | 0.000% |

Repeat-1 speed gate: **FAIL**; unchanged math/traffic/resources: **PASS**.

## Repeat 10

### Primary latency gate

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,735,166.700 | 2,784,666.700 | 1.810% | 2.108% |
| `invocation_ticks` | 51,069.200 | 52,009.000 | 1.840% | 1.904% |
| `total_ticks` | 51,018.300 | 51,958.800 | 1.843% | 1.904% |
| `qkv_attention_ticks` | 24,157.200 | 25,151.200 | 4.115% | 4.128% |
| `qkv_projection_ticks` | 15,967.300 | 18,284.100 | 14.510% | 14.408% |
| `attention_ticks` | 8,178.400 | 6,860.000 | -16.121% | -16.192% |
| `attention_qk_norm_pool_wait_ticks` | 9,072.900 | 11,386.700 | 25.502% | 25.158% |

### Additive Block Timing Ledger

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 55.000 | 55.200 | 0.364% | 0.000% |
| `metadata_stage_ticks` | 11.900 | 12.000 | 0.840% | 0.840% |
| `input_norm_ticks` | 1,745.400 | 1,745.600 | 0.011% | 0.006% |
| `qkv_projection_ticks` | 15,967.300 | 18,284.100 | 14.510% | 14.408% |
| `qk_norm_rope_ticks` | 0.200 | 0.700 | 250.000% | 75.000% |
| `attention_ticks` | 8,178.400 | 6,860.000 | -16.121% | -16.192% |
| `o_projection_ticks` | 3,287.200 | 3,279.300 | -0.240% | -0.139% |
| `post_attention_residual_ticks` | 878.100 | 881.200 | 0.353% | 0.272% |
| `post_attention_norm_ticks` | 0.200 | 0.400 | 100.000% | 50.000% |
| `gate_up_ticks` | 13,619.900 | 13,614.600 | -0.039% | -0.609% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,870.000 | 6,837.100 | -0.479% | -0.122% |
| `final_residual_ticks` | 419.800 | 417.400 | -0.572% | 0.121% |
| `output_stage_ticks` | 12.400 | 12.400 | 0.000% | -0.800% |

### Overlapping engine work and waits

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 7.600 | 7.600 | 0.000% | -1.299% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,708.400 | 10,691.500 | -0.158% | -0.710% |
| `hmx_compute_ticks` | 13,765.100 | 13,658.700 | -0.773% | -1.097% |
| `projection_hmx_wait_ticks` | 220.500 | 227.400 | 3.129% | 3.466% |
| `hmx_ready_wait_ticks` | 7,925.900 | 7,910.500 | -0.194% | -0.598% |
| `w4u8_qkvo_weight_expand_ticks` | 6,381.400 | 6,382.700 | 0.020% | -0.103% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,069.800 | 3,068.300 | -0.049% | -0.413% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,298.100 | 9,303.700 | 0.060% | -0.066% |
| `u8_attention_qk_norm_rope_ticks` | 44,796.500 | 44,745.400 | -0.114% | -0.067% |
| `u8_attention_v_pack_ticks` | 8,195.200 | 7,106.400 | -13.286% | -13.180% |
| `u8_attention_qk_hmx_ticks` | 1,325.300 | 1,348.000 | 1.713% | 1.397% |
| `u8_attention_qk_requant_ticks` | 385.000 | 390.500 | 1.429% | 1.749% |
| `u8_attention_softmax_ticks` | 13,944.000 | 13,927.800 | -0.116% | -0.282% |
| `u8_attention_av_hmx_ticks` | 2,619.500 | 2,668.100 | 1.855% | 2.600% |
| `u8_attention_av_requant_ticks` | 1,058.400 | 1,055.700 | -0.255% | -0.368% |
| `u8_attention_pipeline_wait_ticks` | 3,446.600 | 5,239.900 | 52.031% | 57.421% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,968.700 | 12,958.500 | -0.079% | -0.670% |
| `w4u8_mlp_down_pipeline_ticks` | 5,985.800 | 5,950.700 | -0.586% | -0.225% |
| `w4u8_mlp_activation_work_ticks` | 7,921.000 | 7,904.800 | -0.205% | -0.206% |
| `w4u8_mlp_weight_stage_ticks` | 7,458.500 | 7,499.100 | 0.544% | 0.049% |
| `w4u8_mlp_weight_expand_ticks` | 24,978.100 | 24,974.700 | -0.014% | -0.574% |
| `w4u8_mlp_hmx_compute_ticks` | 7,897.600 | 7,798.400 | -1.256% | -0.168% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,742.600 | 8,734.100 | -0.097% | -0.539% |
| `w4u8_mlp_producer_slot_wait_ticks` | 301.600 | 301.000 | -0.199% | -0.066% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,796.100 | 4,768.600 | -0.573% | -0.528% |
| `u8_attention_k_pack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `attention_qk_norm_pool_wait_ticks` | 9,072.900 | 11,386.700 | 25.502% | 25.158% |

### Traffic, commands, counters and residency

| Metric | EXP-0053 control | EXP-0054 Stage A | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 256.0 | 0.000% | 0.000% |
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
| `attention_hvx_workers_created` | 3.000 | 3.000 | 0.000% | 0.000% |
| `attention_hvx_workers_locked` | 3.000 | 3.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 49.700 | 49.600 | -0.201% | -0.201% |
| `runtime_teardown_ticks` | 42.700 | 42.700 | 0.000% | 0.000% |
| `ledger_named_ticks` | 51,045.000 | 51,984.100 | 1.840% | 1.905% |
| `ledger_unattributed_ticks` | 25.800 | 24.900 | -3.488% | -1.938% |

Repeat-10 speed gate: **FAIL**; unchanged math/traffic/resources: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final output vs EXP-0053 and independent reference | byte-exact, 0 LSB |
| QK / probability / AV boundary hashes | unchanged |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping work/wait counters are not summed together. Host wall is the primary end-to-end speed metric.

## Decision

EXP-0054 Stage-A local gate: **FAIL**. Stage B authorization from this gate: **NO**. A failed Stage A remains a negative experiment and does not replace EXP-0053.



