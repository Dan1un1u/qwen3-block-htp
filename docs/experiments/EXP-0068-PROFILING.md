# EXP-0068 — Complete profiling report

## Three-variant repeat-ten overview

F16F16 and W4F16 reuse the latest valid EXP-0038 formal evidence. W4U8 uses
the EXP-0068 six-context candidate. Each cell reports wall time and share of
complete Host wall; the last column is `W4F16 / W4U8 - 1`, so positive means
W4U8 is faster.

| Module | F16F16 (W16A16) | W4F16 | W4U8 EXP-0068 | W4U8 vs W4F16 speed |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us (0.3%) | 7.4 us (0.3%) | 4.1 us (0.2%) | +81.2% |
| Input RMSNorm | 42.7 us (1.7%) | 42.7 us (1.9%) | 80.3 us (3.7%) | -46.8% |
| QKV + Q/K Norm/RoPE preparation | 397.1 us (16.0%) | 439.2 us (19.7%) | 422.3 us (19.7%) | +4.0% |
| QK-Softmax-AV | 140.3 us (5.6%) | 140.7 us (6.3%) | 258.9 us (12.1%) | -45.6% |
| O projection | 201.5 us (8.1%) | 173.9 us (7.8%) | 171.4 us (8.0%) | +1.5% |
| Post-Attention residual + RMSNorm | 41.2 us (1.7%) | 41.2 us (1.8%) | 42.7 us (2.0%) | -3.6% |
| Gate/Up + SwiGLU | 1,116.6 us (44.9%) | 972.7 us (43.6%) | 706.4 us (32.9%) | +37.7% |
| Down projection | 459.9 us (18.5%) | 327.3 us (14.7%) | 354.6 us (16.5%) | -7.7% |
| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 21.8 us (1.0%) | -77.0% |
| Host/RPC and closure | 77.5 us (3.1%) | 79.6 us (3.6%) | 83.1 us (3.9%) | -4.3% |
| **Complete block Host wall** | **2,488.3 us** | **2,229.7 us** | **2,145.5 us** | **+3.9%** |

The control retains EXP-0065's main thread plus three persistent HVX workers. A bounded three-round search selected 6 total contexts (main plus 5 workers). Only Q/K preparation and GQA Attention task capacity changes; all arithmetic, qparams, tensor layouts, HMX commands/tile pairs, DMA traffic, MLP scheduling and physical residency are unchanged.

## Repeat 1

### Primary wall-latency targets

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,689,688.000 | 2,578,177.000 | -4.146% | -4.668% |
| `invocation_ticks` | 43,433.000 | 41,374.000 | -4.741% | -4.929% |
| `total_ticks` | 42,932.000 | 40,659.000 | -5.294% | -5.478% |
| `qkv_projection_ticks` | 10,685.000 | 8,147.000 | -23.753% | -23.504% |
| `attention_ticks` | 5,006.000 | 5,041.000 | 0.699% | 0.699% |
| `qkv_plus_attention_ticks` | 15,688.000 | 13,214.000 | -15.770% | -15.734% |
| `attention_qk_norm_pool_wait_ticks` | 3,810.000 | 1,260.000 | -66.929% | -66.880% |

### Additive Block Timing Ledger

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 47.000 | 48.000 | 2.128% | 6.250% |
| `metadata_stage_ticks` | 117.000 | 119.000 | 1.709% | 3.361% |
| `input_norm_ticks` | 1,537.000 | 1,555.000 | 1.171% | 1.171% |
| `qkv_projection_ticks` | 10,685.000 | 8,147.000 | -23.753% | -23.504% |
| `qk_norm_rope_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `attention_ticks` | 5,006.000 | 5,041.000 | 0.699% | 0.699% |
| `o_projection_ticks` | 3,282.000 | 3,288.000 | 0.183% | 0.092% |
| `post_attention_residual_ticks` | 822.000 | 830.000 | 0.973% | 0.365% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | n/a |
| `gate_up_ticks` | 13,595.000 | 13,615.000 | 0.147% | -0.410% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,831.000 | 6,952.000 | 1.771% | 1.632% |
| `final_residual_ticks` | 417.000 | 412.000 | -1.199% | -0.242% |
| `output_stage_ticks` | 124.000 | 125.000 | 0.806% | 0.000% |

### Overlapping engine work and waits

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 9.000 | 9.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,871.000 | 10,730.000 | -1.297% | -1.242% |
| `hmx_compute_ticks` | 13,198.000 | 13,091.000 | -0.811% | -1.223% |
| `projection_hmx_wait_ticks` | 225.000 | 229.000 | 1.778% | 2.691% |
| `hmx_ready_wait_ticks` | 7,875.000 | 7,898.000 | 0.292% | -0.366% |
| `w4u8_qkvo_weight_expand_ticks` | 6,368.000 | 6,398.000 | 0.471% | 0.504% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,081.000 | 3,018.000 | -2.045% | -2.260% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,276.000 | 9,265.000 | -0.119% | 0.441% |
| `u8_attention_qk_norm_rope_ticks` | 26,923.000 | 28,556.000 | 6.065% | 5.780% |
| `u8_attention_v_pack_ticks` | 2,797.000 | 3,187.000 | 13.944% | 14.066% |
| `u8_attention_qk_hmx_ticks` | 726.000 | 739.000 | 1.791% | 5.372% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 12,926.000 | 13,888.000 | 7.442% | 7.435% |
| `u8_attention_av_hmx_ticks` | 724.000 | 783.000 | 8.149% | 4.006% |
| `u8_attention_av_requant_ticks` | 1,050.000 | 1,076.000 | 2.476% | 2.089% |
| `u8_attention_pipeline_wait_ticks` | 918.000 | 2,318.000 | 152.505% | 120.153% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,940.000 | 12,934.000 | -0.046% | -0.308% |
| `w4u8_mlp_down_pipeline_ticks` | 5,941.000 | 6,066.000 | 2.104% | 1.840% |
| `w4u8_mlp_activation_work_ticks` | 7,882.000 | 7,929.000 | 0.596% | 0.431% |
| `w4u8_mlp_weight_stage_ticks` | 7,499.000 | 7,393.000 | -1.414% | -0.364% |
| `w4u8_mlp_weight_expand_ticks` | 24,917.000 | 24,913.000 | -0.016% | -0.040% |
| `w4u8_mlp_hmx_compute_ticks` | 7,796.000 | 7,946.000 | 1.924% | 1.971% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,685.000 | 8,728.000 | 0.495% | -0.286% |
| `w4u8_mlp_producer_slot_wait_ticks` | 301.000 | 313.000 | 3.987% | 0.965% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,707.000 | 4,885.000 | 3.782% | 2.145% |
| `attention_qk_norm_pool_wait_ticks` | 3,810.000 | 1,260.000 | -66.929% | -66.880% |

### Traffic, commands, counters and residency

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 192.0 | 192.0 | 0.000% | 0.000% |
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
| `attention_hvx_workers_created` | 3.000 | 5.000 | 66.667% | 66.667% |
| `attention_hvx_workers_locked` | 3.000 | 5.000 | 66.667% | 66.667% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 502.000 | 718.000 | 43.028% | 42.116% |
| `runtime_teardown_ticks` | 429.000 | 633.000 | 47.552% | 47.897% |
| `ledger_named_ticks` | 43,404.000 | 41,345.000 | -4.744% | -4.930% |
| `ledger_unattributed_ticks` | 29.000 | 29.000 | 0.000% | -3.333% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |

Two-target speed gate: **PASS**; unchanged math/traffic/commands/resources: **PASS**; worker creation/lock gate: **PASS**.

## Repeat 10

### Primary wall-latency targets

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,277,578.100 | 2,145,526.100 | -5.798% | -5.839% |
| `invocation_ticks` | 42,378.900 | 39,723.100 | -6.267% | -6.254% |
| `total_ticks` | 42,328.500 | 39,651.300 | -6.325% | -6.313% |
| `qkv_projection_ticks` | 10,673.100 | 8,107.300 | -24.040% | -23.913% |
| `attention_ticks` | 4,976.800 | 4,970.200 | -0.133% | -0.074% |
| `qkv_plus_attention_ticks` | 15,658.300 | 13,084.000 | -16.440% | -16.332% |
| `attention_qk_norm_pool_wait_ticks` | 3,758.600 | 1,146.800 | -69.489% | -69.308% |

### Additive Block Timing Ledger

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 54.500 | 54.100 | -0.734% | 0.745% |
| `metadata_stage_ticks` | 11.800 | 11.900 | 0.847% | 0.000% |
| `input_norm_ticks` | 1,539.600 | 1,541.200 | 0.104% | 0.117% |
| `qkv_projection_ticks` | 10,673.100 | 8,107.300 | -24.040% | -23.913% |
| `qk_norm_rope_ticks` | 0.200 | 0.400 | 100.000% | 75.000% |
| `attention_ticks` | 4,976.800 | 4,970.200 | -0.133% | -0.074% |
| `o_projection_ticks` | 3,281.700 | 3,290.900 | 0.280% | 0.158% |
| `post_attention_residual_ticks` | 817.500 | 820.200 | 0.330% | 0.330% |
| `post_attention_norm_ticks` | 0.100 | 0.100 | 0.000% | -50.000% |
| `gate_up_ticks` | 13,666.500 | 13,562.500 | -0.761% | -0.586% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,817.200 | 6,808.600 | -0.126% | 0.086% |
| `final_residual_ticks` | 417.600 | 418.100 | 0.120% | -0.618% |
| `output_stage_ticks` | 12.400 | 12.400 | 0.000% | 0.000% |

### Overlapping engine work and waits

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.900 | 9.100 | 2.247% | 3.371% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,695.900 | 10,662.500 | -0.312% | -0.997% |
| `hmx_compute_ticks` | 13,091.600 | 13,009.300 | -0.629% | -0.040% |
| `projection_hmx_wait_ticks` | 226.000 | 228.800 | 1.239% | 0.992% |
| `hmx_ready_wait_ticks` | 7,967.900 | 7,901.800 | -0.830% | -0.830% |
| `w4u8_qkvo_weight_expand_ticks` | 6,387.000 | 6,401.600 | 0.229% | 0.218% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,053.200 | 3,068.300 | 0.495% | 0.789% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,300.500 | 9,339.400 | 0.418% | 0.587% |
| `u8_attention_qk_norm_rope_ticks` | 26,793.500 | 28,299.400 | 5.620% | 5.680% |
| `u8_attention_v_pack_ticks` | 2,803.500 | 3,179.000 | 13.394% | 13.443% |
| `u8_attention_qk_hmx_ticks` | 725.700 | 732.100 | 0.882% | 2.327% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 12,954.700 | 13,892.800 | 7.241% | 7.016% |
| `u8_attention_av_hmx_ticks` | 709.900 | 780.500 | 9.945% | 9.605% |
| `u8_attention_av_requant_ticks` | 1,049.500 | 1,077.300 | 2.649% | 2.866% |
| `u8_attention_pipeline_wait_ticks` | 937.700 | 2,303.100 | 145.612% | 135.580% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,002.000 | 12,901.700 | -0.771% | -0.671% |
| `w4u8_mlp_down_pipeline_ticks` | 5,933.200 | 5,917.300 | -0.268% | 0.064% |
| `w4u8_mlp_activation_work_ticks` | 7,882.800 | 7,864.600 | -0.231% | -0.029% |
| `w4u8_mlp_weight_stage_ticks` | 7,503.000 | 7,435.300 | -0.902% | -0.560% |
| `w4u8_mlp_weight_expand_ticks` | 25,033.500 | 24,890.100 | -0.573% | -0.556% |
| `w4u8_mlp_hmx_compute_ticks` | 7,852.800 | 7,777.900 | -0.954% | -0.605% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,787.100 | 8,729.000 | -0.661% | -0.671% |
| `w4u8_mlp_producer_slot_wait_ticks` | 306.600 | 305.300 | -0.424% | -0.424% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,790.600 | 4,792.700 | 0.044% | 0.021% |
| `attention_qk_norm_pool_wait_ticks` | 3,758.600 | 1,146.800 | -69.489% | -69.308% |

### Traffic, commands, counters and residency

| Metric | 4-context control | Selected candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 192.0 | 192.0 | 0.000% | 0.000% |
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
| `attention_hvx_workers_created` | 3.000 | 5.000 | 66.667% | 66.667% |
| `attention_hvx_workers_locked` | 3.000 | 5.000 | 66.667% | 66.667% |
| `mlp_hvx_workers_created` | 2.000 | 2.000 | 0.000% | 0.000% |
| `mlp_hvx_workers_locked` | 2.000 | 2.000 | 0.000% | 0.000% |
| `w4u8_qkv_batch_n_tiles` | 4.000 | 4.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_workers` | 3.000 | 3.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_workers` | 6.000 | 6.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_hmx_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_gate_up_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `w4u8_mlp_down_hvx_parallel_overlap` | 1.000 | 1.000 | 0.000% | 0.000% |
| `runtime_setup_ticks` | 50.000 | 72.000 | 44.000% | 43.687% |
| `runtime_teardown_ticks` | 42.500 | 63.600 | 49.647% | 49.412% |
| `ledger_named_ticks` | 42,353.200 | 39,697.300 | -6.271% | -6.258% |
| `ledger_unattributed_ticks` | 25.200 | 25.300 | 0.397% | 0.000% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |

Two-target speed gate: **PASS**; unchanged math/traffic/commands/resources: **PASS**; worker creation/lock gate: **PASS**.

## Physical and correctness gates

| Gate | Result |
|---|---:|
| Final block output | byte-exact to EXP-0065, 0 LSB |
| QK / probability / AV audit boundaries | byte-exact to EXP-0065 |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains primary.

## Decision

EXP-0068 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged without explicit user promotion.

