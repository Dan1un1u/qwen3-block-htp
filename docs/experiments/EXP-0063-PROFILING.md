# EXP-0063 — Complete profiling report

The control writes each 8 KiB requantized QK score group and then loads it again in Softmax. The candidate packs the corresponding 64-element rows from both Q heads into one 128-byte HVX vector and applies the identical requantization at full SIMD occupancy. Audit mode writes the converted row back solely to preserve the authoritative QK hash; performance mode performs no score writeback.

## Repeat 1

### Primary latency and fusion targets

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,822,708.000 | 2,777,917.000 | -1.587% | -0.352% |
| `invocation_ticks` | 46,003.000 | 45,675.000 | -0.713% | -0.663% |
| `total_ticks` | 45,502.000 | 45,174.000 | -0.721% | -0.659% |
| `u8_attention_qk_requant_softmax_ticks` | 14,257.000 | 13,129.000 | -7.912% | -7.821% |
| `u8_attention_qk_requant_ticks` | 395.000 | 0.000 | -100.000% | -100.000% |
| `u8_attention_softmax_ticks` | 13,864.000 | 13,129.000 | -5.302% | -5.117% |
| `attention_ticks` | 7,298.000 | 6,898.000 | -5.481% | -5.607% |

### Additive Block Timing Ledger

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 47.000 | 50.000 | 6.383% | 6.383% |
| `metadata_stage_ticks` | 118.000 | 120.000 | 1.695% | 0.840% |
| `input_norm_ticks` | 1,557.000 | 1,552.000 | -0.321% | -0.447% |
| `qkv_projection_ticks` | 10,663.000 | 10,710.000 | 0.441% | 0.320% |
| `qk_norm_rope_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `attention_ticks` | 7,298.000 | 6,898.000 | -5.481% | -5.607% |
| `o_projection_ticks` | 3,268.000 | 3,260.000 | -0.245% | -0.245% |
| `post_attention_residual_ticks` | 819.000 | 822.000 | 0.366% | 0.000% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,880.000 | 13,776.000 | -0.749% | -0.720% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,821.000 | 6,848.000 | 0.396% | 0.721% |
| `final_residual_ticks` | 418.000 | 419.000 | 0.239% | -0.240% |
| `output_stage_ticks` | 125.000 | 125.000 | 0.000% | 0.000% |

### Overlapping engine work and waits

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 8.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,858.000 | 10,800.000 | -0.534% | 0.634% |
| `hmx_compute_ticks` | 14,030.000 | 13,928.000 | -0.727% | -0.706% |
| `projection_hmx_wait_ticks` | 226.000 | 222.000 | -1.770% | -1.327% |
| `hmx_ready_wait_ticks` | 8,004.000 | 7,868.000 | -1.699% | -1.287% |
| `w4u8_qkvo_weight_expand_ticks` | 6,354.000 | 6,354.000 | 0.000% | 0.000% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,072.000 | 3,076.000 | 0.130% | 0.130% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,232.000 | 9,230.000 | -0.022% | -0.032% |
| `u8_attention_qk_norm_rope_ticks` | 26,893.000 | 26,998.000 | 0.390% | 0.251% |
| `u8_attention_v_pack_ticks` | 2,810.000 | 2,813.000 | 0.107% | 0.142% |
| `u8_attention_qk_hmx_ticks` | 1,345.000 | 1,339.000 | -0.446% | -0.900% |
| `u8_attention_qk_requant_ticks` | 395.000 | 0.000 | -100.000% | -100.000% |
| `u8_attention_softmax_ticks` | 13,864.000 | 13,129.000 | -5.302% | -5.117% |
| `u8_attention_av_hmx_ticks` | 2,664.000 | 2,625.000 | -1.464% | -1.444% |
| `u8_attention_av_requant_ticks` | 1,052.000 | 1,050.000 | -0.190% | -0.190% |
| `u8_attention_pipeline_wait_ticks` | 4,999.000 | 4,882.000 | -2.340% | -7.988% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,225.000 | 13,121.000 | -0.786% | -0.779% |
| `w4u8_mlp_down_pipeline_ticks` | 5,948.000 | 5,963.000 | 0.252% | -0.017% |
| `w4u8_mlp_activation_work_ticks` | 7,910.000 | 7,897.000 | -0.164% | -0.140% |
| `w4u8_mlp_weight_stage_ticks` | 7,522.000 | 7,456.000 | -0.877% | 0.134% |
| `w4u8_mlp_weight_expand_ticks` | 25,207.000 | 24,935.000 | -1.079% | -0.044% |
| `w4u8_mlp_hmx_compute_ticks` | 8,032.000 | 8,032.000 | 0.000% | 0.037% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,812.000 | 8,696.000 | -1.316% | -1.292% |
| `w4u8_mlp_producer_slot_wait_ticks` | 296.000 | 304.000 | 2.703% | 1.031% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,556.000 | 4,601.000 | 0.988% | 2.870% |
| `u8_attention_qk_requant_softmax_ticks` | 14,257.000 | 13,129.000 | -7.912% | -7.821% |

### Traffic, commands, counters and residency

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 500.000 | 499.000 | -0.200% | -0.200% |
| `runtime_teardown_ticks` | 424.000 | 423.000 | -0.236% | 0.000% |
| `ledger_named_ticks` | 45,973.000 | 45,645.000 | -0.713% | -0.666% |
| `ledger_unattributed_ticks` | 28.000 | 29.000 | 3.571% | 7.143% |

Three-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**.

## Repeat 10

### Primary latency and fusion targets

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,408,963.500 | 2,396,359.400 | -0.523% | -0.611% |
| `invocation_ticks` | 44,895.700 | 44,616.100 | -0.623% | -0.791% |
| `total_ticks` | 44,845.400 | 44,566.200 | -0.623% | -0.795% |
| `u8_attention_qk_requant_softmax_ticks` | 14,398.800 | 13,121.700 | -8.869% | -8.726% |
| `u8_attention_qk_requant_ticks` | 389.600 | 0.000 | -100.000% | -100.000% |
| `u8_attention_softmax_ticks` | 14,008.300 | 13,121.700 | -6.329% | -6.214% |
| `attention_ticks` | 7,344.200 | 7,027.000 | -4.319% | -4.381% |

### Additive Block Timing Ledger

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 55.100 | 57.200 | 3.811% | 0.896% |
| `metadata_stage_ticks` | 11.700 | 11.800 | 0.855% | 1.695% |
| `input_norm_ticks` | 1,542.700 | 1,543.600 | 0.058% | 0.194% |
| `qkv_projection_ticks` | 10,662.000 | 10,652.300 | -0.091% | -0.052% |
| `qk_norm_rope_ticks` | 0.000 | 0.100 | n/a | 0.000% |
| `attention_ticks` | 7,344.200 | 7,027.000 | -4.319% | -4.381% |
| `o_projection_ticks` | 3,279.700 | 3,282.000 | 0.070% | 0.107% |
| `post_attention_residual_ticks` | 813.800 | 815.200 | 0.172% | -0.244% |
| `post_attention_norm_ticks` | 0.100 | 0.000 | -100.000% | -100.000% |
| `gate_up_ticks` | 13,770.100 | 13,803.100 | 0.240% | 0.040% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,854.900 | 6,824.700 | -0.441% | -0.506% |
| `final_residual_ticks` | 418.300 | 418.800 | 0.120% | 0.096% |
| `output_stage_ticks` | 12.500 | 12.400 | -0.800% | 0.000% |

### Overlapping engine work and waits

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.500 | 8.400 | -1.176% | -3.409% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,688.400 | 10,775.900 | 0.819% | 0.633% |
| `hmx_compute_ticks` | 13,907.400 | 13,920.100 | 0.091% | 0.091% |
| `projection_hmx_wait_ticks` | 229.200 | 229.700 | 0.218% | -0.770% |
| `hmx_ready_wait_ticks` | 7,957.400 | 7,993.800 | 0.457% | 0.504% |
| `w4u8_qkvo_weight_expand_ticks` | 6,382.900 | 6,388.800 | 0.092% | -0.020% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,044.000 | 3,063.500 | 0.641% | 0.318% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,294.300 | 9,298.900 | 0.049% | 0.088% |
| `u8_attention_qk_norm_rope_ticks` | 26,848.300 | 26,833.700 | -0.054% | -0.051% |
| `u8_attention_v_pack_ticks` | 2,810.300 | 2,809.300 | -0.036% | -0.046% |
| `u8_attention_qk_hmx_ticks` | 1,345.900 | 1,338.900 | -0.520% | -0.172% |
| `u8_attention_qk_requant_ticks` | 389.600 | 0.000 | -100.000% | -100.000% |
| `u8_attention_softmax_ticks` | 14,008.300 | 13,121.700 | -6.329% | -6.214% |
| `u8_attention_av_hmx_ticks` | 2,652.400 | 2,648.900 | -0.132% | -0.742% |
| `u8_attention_av_requant_ticks` | 1,040.200 | 1,051.500 | 1.086% | 1.115% |
| `u8_attention_pipeline_wait_ticks` | 4,953.800 | 4,889.800 | -1.292% | -1.476% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,111.900 | 13,150.000 | 0.291% | 0.093% |
| `w4u8_mlp_down_pipeline_ticks` | 5,974.200 | 5,942.800 | -0.526% | -0.517% |
| `w4u8_mlp_activation_work_ticks` | 7,894.200 | 7,904.600 | 0.132% | -0.034% |
| `w4u8_mlp_weight_stage_ticks` | 7,488.200 | 7,491.200 | 0.040% | 0.461% |
| `w4u8_mlp_weight_expand_ticks` | 25,150.100 | 25,001.900 | -0.589% | -0.289% |
| `w4u8_mlp_hmx_compute_ticks` | 7,982.500 | 7,937.800 | -0.560% | -0.738% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,772.500 | 8,809.200 | 0.418% | 0.552% |
| `w4u8_mlp_producer_slot_wait_ticks` | 297.800 | 300.200 | 0.806% | 1.478% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,587.200 | 4,531.300 | -1.219% | -0.495% |
| `u8_attention_qk_requant_softmax_ticks` | 14,398.800 | 13,121.700 | -8.869% | -8.726% |

### Traffic, commands, counters and residency

| Metric | Separate control | Paired fused candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 50.100 | 49.900 | -0.399% | -0.200% |
| `runtime_teardown_ticks` | 42.200 | 42.400 | 0.474% | 0.236% |
| `ledger_named_ticks` | 44,870.200 | 44,590.400 | -0.624% | -0.791% |
| `ledger_unattributed_ticks` | 25.500 | 25.800 | 1.176% | 1.176% |

Three-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**.

## Physical and correctness gates

| Gate | Result |
|---|---:|
| Final block output | byte-exact, 0 LSB |
| QK / probability / AV audit boundaries | byte-exact |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains the primary speed metric.

## Decision

EXP-0063 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged unless the user explicitly promotes it.

