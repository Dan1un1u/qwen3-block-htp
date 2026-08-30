# EXP-0069 — Complete profiling report

## Three-variant repeat-ten overview

Because EXP-0069 fails its local gate, the latest eligible W4U8 result remains
EXP-0068. F16F16 and W4F16 reuse EXP-0038 formal evidence. Each cell reports
wall time and share of complete Host wall; positive in the final column means
the eligible W4U8 candidate is faster than W4F16.

| Module | F16F16 (W16A16) | W4F16 | Eligible W4U8 EXP-0068 | W4U8 vs W4F16 speed |
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

Both paths create and lock the same main-plus-five persistent HVX domain and use all six contexts for Q/K preparation. The selected candidate allows only 5 total contexts to claim the eight GQA Attention tasks. Arithmetic, qparams, layouts, HMX work, DMA traffic, VTCM and every non-Attention stage are unchanged.

## Repeat 1

### Primary wall-latency targets

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,567,865.000 | 2,567,448.000 | -0.016% | 0.962% |
| `invocation_ticks` | 41,427.000 | 41,073.000 | -0.855% | -0.444% |
| `total_ticks` | 40,708.000 | 40,345.000 | -0.892% | -0.452% |
| `qkv_projection_ticks` | 8,159.000 | 8,158.000 | -0.012% | 0.490% |
| `attention_ticks` | 4,973.000 | 4,889.000 | -1.689% | -1.589% |
| `qkv_plus_attention_ticks` | 13,179.000 | 13,063.000 | -0.880% | -0.812% |
| `attention_qk_norm_pool_wait_ticks` | 1,268.000 | 1,177.000 | -7.177% | -9.227% |

### Additive Block Timing Ledger

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 50.000 | 52.000 | 4.000% | 4.000% |
| `metadata_stage_ticks` | 119.000 | 120.000 | 0.840% | 0.000% |
| `input_norm_ticks` | 1,553.000 | 1,554.000 | 0.064% | 0.000% |
| `qkv_projection_ticks` | 8,159.000 | 8,158.000 | -0.012% | 0.490% |
| `qk_norm_rope_ticks` | 3.000 | 0.000 | -100.000% | -100.000% |
| `attention_ticks` | 4,973.000 | 4,889.000 | -1.689% | -1.589% |
| `o_projection_ticks` | 3,281.000 | 3,280.000 | -0.030% | 0.152% |
| `post_attention_residual_ticks` | 819.000 | 822.000 | 0.366% | 0.000% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,593.000 | 13,566.000 | -0.199% | -0.806% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,820.000 | 6,751.000 | -1.012% | -0.625% |
| `final_residual_ticks` | 416.000 | 415.000 | -0.240% | 0.978% |
| `output_stage_ticks` | 124.000 | 125.000 | 0.806% | 0.806% |

### Overlapping engine work and waits

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 9.000 | 9.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,741.000 | 10,754.000 | 0.121% | 0.037% |
| `hmx_compute_ticks` | 13,234.000 | 13,068.000 | -1.254% | -1.366% |
| `projection_hmx_wait_ticks` | 224.000 | 238.000 | 6.250% | 7.692% |
| `hmx_ready_wait_ticks` | 7,952.000 | 7,880.000 | -0.905% | -1.355% |
| `w4u8_qkvo_weight_expand_ticks` | 6,390.000 | 6,404.000 | 0.219% | 0.031% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,090.000 | 3,092.000 | 0.065% | -0.160% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,353.000 | 9,365.000 | 0.128% | 0.331% |
| `u8_attention_qk_norm_rope_ticks` | 28,650.000 | 28,382.000 | -0.935% | -1.243% |
| `u8_attention_v_pack_ticks` | 3,166.000 | 3,110.000 | -1.769% | -1.770% |
| `u8_attention_qk_hmx_ticks` | 717.000 | 736.000 | 2.650% | 2.204% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,833.000 | 13,091.000 | -5.364% | -5.460% |
| `u8_attention_av_hmx_ticks` | 793.000 | 733.000 | -7.566% | -7.440% |
| `u8_attention_av_requant_ticks` | 1,094.000 | 1,078.000 | -1.463% | -1.726% |
| `u8_attention_pipeline_wait_ticks` | 1,995.000 | 927.000 | -53.534% | -53.387% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,948.000 | 12,906.000 | -0.324% | -0.786% |
| `w4u8_mlp_down_pipeline_ticks` | 5,931.000 | 5,862.000 | -1.163% | -0.635% |
| `w4u8_mlp_activation_work_ticks` | 7,925.000 | 7,893.000 | -0.404% | 0.341% |
| `w4u8_mlp_weight_stage_ticks` | 7,440.000 | 7,404.000 | -0.484% | -1.617% |
| `w4u8_mlp_weight_expand_ticks` | 25,053.000 | 25,008.000 | -0.180% | -0.140% |
| `w4u8_mlp_hmx_compute_ticks` | 7,825.000 | 7,734.000 | -1.163% | -0.746% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,760.000 | 8,706.000 | -0.616% | -0.813% |
| `w4u8_mlp_producer_slot_wait_ticks` | 306.000 | 308.000 | 0.654% | -0.313% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,867.000 | 4,654.000 | -4.376% | -2.345% |
| `attention_qk_norm_pool_wait_ticks` | 1,268.000 | 1,177.000 | -7.177% | -9.227% |

### Traffic, commands, counters and residency

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 720.000 | 724.000 | 0.556% | 0.000% |
| `runtime_teardown_ticks` | 639.000 | 636.000 | -0.469% | -0.470% |
| `ledger_named_ticks` | 41,395.000 | 41,042.000 | -0.853% | -0.447% |
| `ledger_unattributed_ticks` | 30.000 | 30.000 | 0.000% | 0.000% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |

Three-target speed gate: **FAIL**; unchanged math/traffic/commands/resources/pool: **PASS**.

## Repeat 10

### Primary wall-latency targets

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,140,140.600 | 2,137,234.400 | -0.136% | -0.136% |
| `invocation_ticks` | 39,805.700 | 39,698.700 | -0.269% | -0.185% |
| `total_ticks` | 39,733.600 | 39,626.900 | -0.269% | -0.186% |
| `qkv_projection_ticks` | 8,086.800 | 8,108.800 | 0.272% | 0.198% |
| `attention_ticks` | 4,961.900 | 4,871.900 | -1.814% | -1.724% |
| `qkv_plus_attention_ticks` | 13,054.700 | 12,991.500 | -0.484% | -0.532% |
| `attention_qk_norm_pool_wait_ticks` | 1,162.400 | 1,126.400 | -3.097% | -3.097% |

### Additive Block Timing Ledger

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 53.900 | 55.600 | 3.154% | 1.636% |
| `metadata_stage_ticks` | 11.900 | 12.300 | 3.361% | 3.361% |
| `input_norm_ticks` | 1,540.800 | 1,538.100 | -0.175% | -0.110% |
| `qkv_projection_ticks` | 8,086.800 | 8,108.800 | 0.272% | 0.198% |
| `qk_norm_rope_ticks` | 0.400 | 0.500 | 25.000% | 25.000% |
| `attention_ticks` | 4,961.900 | 4,871.900 | -1.814% | -1.724% |
| `o_projection_ticks` | 3,285.800 | 3,292.200 | 0.195% | 0.127% |
| `post_attention_residual_ticks` | 816.900 | 816.900 | 0.000% | -0.073% |
| `post_attention_norm_ticks` | 0.100 | 0.100 | 0.000% | 0.000% |
| `gate_up_ticks` | 13,612.800 | 13,542.300 | -0.518% | -0.095% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,851.200 | 6,820.100 | -0.454% | 0.132% |
| `final_residual_ticks` | 416.200 | 417.300 | 0.264% | 0.528% |
| `output_stage_ticks` | 12.400 | 12.400 | 0.000% | 0.000% |

### Overlapping engine work and waits

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 9.000 | 8.400 | -6.667% | -5.682% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,640.300 | 10,678.900 | 0.363% | -0.083% |
| `hmx_compute_ticks` | 13,113.100 | 13,006.800 | -0.811% | -0.811% |
| `projection_hmx_wait_ticks` | 225.400 | 233.200 | 3.461% | 4.156% |
| `hmx_ready_wait_ticks` | 7,927.400 | 7,938.000 | 0.134% | -0.263% |
| `w4u8_qkvo_weight_expand_ticks` | 6,404.800 | 6,405.500 | 0.011% | 0.005% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,029.000 | 3,077.100 | 1.588% | 1.991% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,301.500 | 9,381.100 | 0.856% | 0.710% |
| `u8_attention_qk_norm_rope_ticks` | 28,281.300 | 28,351.500 | 0.248% | -0.052% |
| `u8_attention_v_pack_ticks` | 3,195.100 | 3,099.200 | -3.001% | -2.745% |
| `u8_attention_qk_hmx_ticks` | 724.400 | 724.800 | 0.055% | -0.326% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,869.900 | 13,091.600 | -5.611% | -5.370% |
| `u8_attention_av_hmx_ticks` | 773.700 | 734.100 | -5.118% | -5.959% |
| `u8_attention_av_requant_ticks` | 1,088.600 | 1,074.900 | -1.258% | -1.194% |
| `u8_attention_pipeline_wait_ticks` | 2,131.900 | 947.500 | -55.556% | -55.002% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,952.400 | 12,886.900 | -0.506% | -0.041% |
| `w4u8_mlp_down_pipeline_ticks` | 5,964.300 | 5,934.300 | -0.503% | 0.200% |
| `w4u8_mlp_activation_work_ticks` | 7,894.900 | 7,894.300 | -0.008% | -0.158% |
| `w4u8_mlp_weight_stage_ticks` | 7,496.900 | 7,404.900 | -1.227% | 0.167% |
| `w4u8_mlp_weight_expand_ticks` | 25,030.500 | 24,924.800 | -0.422% | 0.168% |
| `w4u8_mlp_hmx_compute_ticks` | 7,847.900 | 7,801.000 | -0.598% | -0.334% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,749.100 | 8,762.500 | 0.153% | -0.279% |
| `w4u8_mlp_producer_slot_wait_ticks` | 301.500 | 306.600 | 1.692% | 1.990% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,771.000 | 4,785.900 | 0.312% | 0.715% |
| `attention_qk_norm_pool_wait_ticks` | 1,162.400 | 1,126.400 | -3.097% | -3.097% |

### Traffic, commands, counters and residency

| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 72.100 | 72.200 | 0.139% | 0.278% |
| `runtime_teardown_ticks` | 63.700 | 64.000 | 0.471% | 0.628% |
| `ledger_named_ticks` | 39,780.400 | 39,673.900 | -0.268% | -0.185% |
| `ledger_unattributed_ticks` | 25.300 | 25.200 | -0.395% | -0.395% |
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |

Three-target speed gate: **PASS**; unchanged math/traffic/commands/resources/pool: **PASS**.

## Physical and correctness gates

| Gate | Result |
|---|---:|
| Final block output | byte-exact to EXP-0068, 0 LSB |
| QK / probability / AV audit boundaries | byte-exact to EXP-0068 |
| Persistent HVX pool | six contexts created and locked in both paths |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains primary.

## Decision

EXP-0069 local gate: **FAIL**. Local adoption eligibility: **NO**. Selected Baseline is unchanged without explicit user promotion.

