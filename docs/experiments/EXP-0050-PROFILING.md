# EXP-0050 — Complete profiling report

The candidate changes only the W4U8 Down HMX consumer cadence. One worker command begins after chunk zero is ready, accumulates that chunk, waits inside the command for chunk one while HVX continues producing it, accumulates chunk one, and stores the unchanged U8 output. W4 values/scales, U8 qparams, HMX tile arithmetic, DMA bytes/descriptors, QKV, Attention, O, Gate/Up, tensor boundaries and final output are unchanged.

## Repeat 1

### Primary latency and Down target

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,524,010.000 | 3,386,406.000 | -3.905% | -3.109% |
| `invocation_ticks` | 59,409.000 | 56,657.000 | -4.632% | -4.808% |
| `total_ticks` | 58,908.000 | 56,156.000 | -4.672% | -4.850% |
| `down_ticks` | 9,471.000 | 6,743.000 | -28.804% | -28.235% |
| `w4u8_mlp_down_pipeline_ticks` | 8,575.000 | 5,855.000 | -31.720% | -31.207% |

### Additive Block Timing Ledger

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 50.000 | 48.000 | -4.000% | -4.000% |
| `metadata_stage_ticks` | 121.000 | 119.000 | -1.653% | -1.653% |
| `input_norm_ticks` | 1,748.000 | 1,747.000 | -0.057% | -0.114% |
| `qkv_projection_ticks` | 17,587.000 | 17,638.000 | 0.290% | 0.240% |
| `qk_norm_rope_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `attention_ticks` | 7,979.000 | 7,968.000 | -0.138% | 0.701% |
| `o_projection_ticks` | 3,270.000 | 3,279.000 | 0.275% | 0.521% |
| `post_attention_residual_ticks` | 3,127.000 | 3,134.000 | 0.224% | 0.256% |
| `post_attention_norm_ticks` | 9.000 | 0.000 | -100.000% | -100.000% |
| `gate_up_ticks` | 13,601.000 | 13,631.000 | 0.221% | -1.993% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 9,471.000 | 6,743.000 | -28.804% | -28.235% |
| `final_residual_ticks` | 1,383.000 | 1,383.000 | 0.000% | 0.000% |
| `output_stage_ticks` | 129.000 | 130.000 | 0.775% | 0.000% |

### Overlapping engine work and waits

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 8.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,758.000 | 10,777.000 | 0.177% | -0.773% |
| `hmx_compute_ticks` | 12,592.000 | 13,691.000 | 8.728% | 8.839% |
| `projection_hmx_wait_ticks` | 214.000 | 221.000 | 3.271% | 1.408% |
| `hmx_ready_wait_ticks` | 7,886.000 | 7,878.000 | -0.101% | 0.323% |
| `w4u8_qkvo_weight_expand_ticks` | 6,353.000 | 6,364.000 | 0.173% | 0.173% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,094.000 | 3,088.000 | -0.194% | 0.032% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,243.000 | 9,290.000 | 0.508% | 0.551% |
| `u8_attention_qk_norm_rope_ticks` | 44,507.000 | 44,569.000 | 0.139% | 0.156% |
| `u8_attention_v_pack_ticks` | 8,224.000 | 8,225.000 | 0.012% | -0.435% |
| `u8_attention_qk_hmx_ticks` | 1,242.000 | 1,287.000 | 3.623% | 3.100% |
| `u8_attention_qk_requant_ticks` | 385.000 | 394.000 | 2.338% | 2.067% |
| `u8_attention_softmax_ticks` | 13,957.000 | 13,950.000 | -0.050% | 0.093% |
| `u8_attention_av_hmx_ticks` | 2,515.000 | 2,519.000 | 0.159% | 3.255% |
| `u8_attention_av_requant_ticks` | 1,059.000 | 1,058.000 | -0.094% | -0.283% |
| `u8_attention_pipeline_wait_ticks` | 2,807.000 | 2,991.000 | 6.555% | 6.200% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,957.000 | 12,977.000 | 0.154% | -2.166% |
| `w4u8_mlp_down_pipeline_ticks` | 8,575.000 | 5,855.000 | -31.720% | -31.207% |
| `w4u8_mlp_activation_work_ticks` | 7,922.000 | 7,846.000 | -0.959% | -0.785% |
| `w4u8_mlp_weight_stage_ticks` | 7,402.000 | 7,435.000 | 0.446% | -1.109% |
| `w4u8_mlp_weight_expand_ticks` | 24,496.000 | 24,808.000 | 1.274% | 0.626% |
| `w4u8_mlp_hmx_compute_ticks` | 10,539.000 | 7,849.000 | -25.524% | -25.969% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,742.000 | 8,689.000 | -0.606% | -0.035% |
| `w4u8_mlp_producer_slot_wait_ticks` | 301.000 | 305.000 | 1.329% | -1.356% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 7,168.000 | 4,523.000 | -36.900% | -36.947% |

### Traffic, commands, counters and residency

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 320.0 | 256.0 | -20.000% | -20.000% |
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
| `w4u8_mlp_down_hmx_command_count` | 128.0 | 64.0 | -50.000% | -50.000% |
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
| `runtime_setup_ticks` | 501.000 | 501.000 | 0.000% | 0.000% |
| `runtime_teardown_ticks` | 431.000 | 425.000 | -1.392% | -0.696% |
| `ledger_named_ticks` | 59,381.000 | 56,631.000 | -4.631% | -4.802% |
| `ledger_unattributed_ticks` | 30.000 | 26.000 | -13.333% | -16.667% |

Down HMX commands per block: **128 → 64**; total HMX commands per block: **320 → 256**; HMX tile pairs remain **49,408**.

Repeat-1 speed gate: **PASS**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Repeat 10

### Primary latency and Down target

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,121,328.100 | 2,975,953.100 | -4.657% | -4.497% |
| `invocation_ticks` | 58,644.900 | 55,867.000 | -4.737% | -4.569% |
| `total_ticks` | 58,594.800 | 55,815.600 | -4.743% | -4.572% |
| `down_ticks` | 9,507.600 | 6,835.000 | -28.110% | -28.110% |
| `w4u8_mlp_down_pipeline_ticks` | 8,617.400 | 5,945.400 | -31.007% | -31.007% |

### Additive Block Timing Ledger

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 61.500 | 60.200 | -2.114% | 0.000% |
| `metadata_stage_ticks` | 12.000 | 11.800 | -1.667% | -3.390% |
| `input_norm_ticks` | 1,743.600 | 1,744.000 | 0.023% | -0.006% |
| `qkv_projection_ticks` | 17,712.000 | 17,704.600 | -0.042% | -0.110% |
| `qk_norm_rope_ticks` | 0.700 | 0.200 | -71.429% | -71.429% |
| `attention_ticks` | 8,035.500 | 8,049.900 | 0.179% | 0.194% |
| `o_projection_ticks` | 3,285.800 | 3,270.500 | -0.466% | -0.103% |
| `post_attention_residual_ticks` | 3,123.800 | 3,125.100 | 0.042% | 0.045% |
| `post_attention_norm_ticks` | 0.900 | 0.100 | -88.889% | -90.000% |
| `gate_up_ticks` | 13,680.900 | 13,604.800 | -0.556% | -0.162% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 9,507.600 | 6,835.000 | -28.110% | -28.110% |
| `final_residual_ticks` | 1,382.400 | 1,383.000 | 0.043% | 0.022% |
| `output_stage_ticks` | 12.600 | 13.000 | 3.175% | 0.000% |

### Overlapping engine work and waits

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.600 | 8.200 | -4.651% | -4.706% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,678.000 | 10,682.600 | 0.043% | 0.043% |
| `hmx_compute_ticks` | 12,635.400 | 13,447.400 | 6.426% | 6.426% |
| `projection_hmx_wait_ticks` | 221.500 | 218.200 | -1.490% | -1.933% |
| `hmx_ready_wait_ticks` | 7,918.000 | 7,914.100 | -0.049% | 0.718% |
| `w4u8_qkvo_weight_expand_ticks` | 6,381.700 | 6,383.500 | 0.028% | 0.020% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,056.100 | 2,993.700 | -2.042% | -1.670% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,282.500 | 9,237.000 | -0.490% | -0.221% |
| `u8_attention_qk_norm_rope_ticks` | 44,670.800 | 44,642.300 | -0.064% | -0.125% |
| `u8_attention_v_pack_ticks` | 8,168.200 | 8,155.400 | -0.157% | -0.053% |
| `u8_attention_qk_hmx_ticks` | 1,297.100 | 1,270.800 | -2.028% | -1.295% |
| `u8_attention_qk_requant_ticks` | 384.300 | 385.300 | 0.260% | 0.420% |
| `u8_attention_softmax_ticks` | 13,958.700 | 13,972.100 | 0.096% | 0.096% |
| `u8_attention_av_hmx_ticks` | 2,559.500 | 2,548.600 | -0.426% | -0.192% |
| `u8_attention_av_requant_ticks` | 1,060.100 | 1,054.000 | -0.575% | -0.435% |
| `u8_attention_pipeline_wait_ticks` | 3,238.200 | 3,197.400 | -1.260% | -2.514% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,025.700 | 12,952.800 | -0.560% | -0.103% |
| `w4u8_mlp_down_pipeline_ticks` | 8,617.400 | 5,945.400 | -31.007% | -31.007% |
| `w4u8_mlp_activation_work_ticks` | 7,868.900 | 7,905.300 | 0.463% | 0.468% |
| `w4u8_mlp_weight_stage_ticks` | 7,499.100 | 7,428.000 | -0.948% | 0.339% |
| `w4u8_mlp_weight_expand_ticks` | 24,910.300 | 25,011.900 | 0.408% | 0.587% |
| `w4u8_mlp_hmx_compute_ticks` | 10,543.400 | 7,852.700 | -25.520% | -25.757% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,777.100 | 8,729.200 | -0.546% | 0.302% |
| `w4u8_mlp_producer_slot_wait_ticks` | 304.000 | 303.400 | -0.197% | 0.097% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 7,194.600 | 4,532.300 | -37.004% | -36.894% |

### Traffic, commands, counters and residency

| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 320.0 | 256.0 | -20.000% | -20.000% |
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
| `w4u8_mlp_down_hmx_command_count` | 128.0 | 64.0 | -50.000% | -50.000% |
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
| `runtime_setup_ticks` | 49.700 | 49.600 | -0.201% | 0.402% |
| `runtime_teardown_ticks` | 42.400 | 42.500 | 0.236% | 0.235% |
| `ledger_named_ticks` | 58,619.000 | 55,841.400 | -4.738% | -4.569% |
| `ledger_unattributed_ticks` | 25.600 | 25.600 | 0.000% | 0.392% |

Down HMX commands per block: **128 → 64**; total HMX commands per block: **320 → 256**; HMX tile pairs remain **49,408**.

Repeat-10 speed gate: **PASS**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final block output vs EXP-0049 | byte-exact, 0 LSB |
| Independent block implementation reference | 0 mismatches, 0 LSB |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping work/wait counters are not summed together. Host wall is the primary speed metric.

## Decision

EXP-0050 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged unless the user explicitly promotes it.

