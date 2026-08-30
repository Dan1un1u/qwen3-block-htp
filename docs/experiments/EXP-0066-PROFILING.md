# EXP-0066 — Complete profiling report

The control reduces both probability halves and updates row-sum minima and maxima on every paired row even when telemetry is null. The candidate executes those diagnostic reductions only in numerical audit mode. Probability bytes, audit telemetry, HMX work and all other block arithmetic are unchanged.

## Repeat 1

### Primary latency and Softmax targets

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,687,865.000 | 2,686,146.000 | -0.064% | 0.147% |
| `invocation_ticks` | 43,606.000 | 43,476.000 | -0.298% | -0.289% |
| `total_ticks` | 43,087.000 | 42,967.000 | -0.279% | -0.279% |
| `u8_attention_softmax_ticks` | 12,989.000 | 12,533.000 | -3.511% | -3.577% |
| `attention_ticks` | 5,041.000 | 4,890.000 | -2.995% | -3.154% |

### Additive Block Timing Ledger

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 48.000 | 47.000 | -2.083% | 0.000% |
| `metadata_stage_ticks` | 118.000 | 120.000 | 1.695% | 0.855% |
| `input_norm_ticks` | 1,541.000 | 1,554.000 | 0.844% | 0.065% |
| `qkv_projection_ticks` | 10,659.000 | 10,630.000 | -0.272% | -0.075% |
| `qk_norm_rope_ticks` | 2.000 | 3.000 | 50.000% | 0.000% |
| `attention_ticks` | 5,041.000 | 4,890.000 | -2.995% | -3.154% |
| `o_projection_ticks` | 3,278.000 | 3,259.000 | -0.580% | -0.641% |
| `post_attention_residual_ticks` | 809.000 | 818.000 | 1.112% | 1.496% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,811.000 | 13,779.000 | -0.232% | 0.619% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,784.000 | 6,773.000 | -0.162% | -0.825% |
| `final_residual_ticks` | 420.000 | 415.000 | -1.190% | -1.429% |
| `output_stage_ticks` | 124.000 | 124.000 | 0.000% | 0.000% |

### Overlapping engine work and waits

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 8.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,824.000 | 10,790.000 | -0.314% | 0.372% |
| `hmx_compute_ticks` | 13,309.000 | 13,315.000 | 0.045% | 0.030% |
| `projection_hmx_wait_ticks` | 215.000 | 220.000 | 2.326% | -0.485% |
| `hmx_ready_wait_ticks` | 8,128.000 | 8,095.000 | -0.406% | -0.731% |
| `w4u8_qkvo_weight_expand_ticks` | 6,383.000 | 6,358.000 | -0.392% | 0.031% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,084.000 | 3,082.000 | -0.065% | 0.529% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,264.000 | 9,236.000 | -0.302% | 0.262% |
| `u8_attention_qk_norm_rope_ticks` | 26,881.000 | 26,867.000 | -0.052% | -0.052% |
| `u8_attention_v_pack_ticks` | 2,830.000 | 2,837.000 | 0.247% | 0.318% |
| `u8_attention_qk_hmx_ticks` | 764.000 | 753.000 | -1.440% | 0.390% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 12,989.000 | 12,533.000 | -3.511% | -3.577% |
| `u8_attention_av_hmx_ticks` | 729.000 | 729.000 | 0.000% | -0.803% |
| `u8_attention_av_requant_ticks` | 1,048.000 | 1,044.000 | -0.382% | -0.191% |
| `u8_attention_pipeline_wait_ticks` | 988.000 | 964.000 | -2.429% | -9.717% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,142.000 | 13,114.000 | -0.213% | 0.641% |
| `w4u8_mlp_down_pipeline_ticks` | 5,909.000 | 5,896.000 | -0.220% | -0.931% |
| `w4u8_mlp_activation_work_ticks` | 7,807.000 | 7,832.000 | 0.320% | -0.253% |
| `w4u8_mlp_weight_stage_ticks` | 7,466.000 | 7,493.000 | 0.362% | 0.415% |
| `w4u8_mlp_weight_expand_ticks` | 25,225.000 | 25,174.000 | -0.202% | -0.119% |
| `w4u8_mlp_hmx_compute_ticks` | 7,726.000 | 7,747.000 | 0.272% | -0.539% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,950.000 | 8,908.000 | -0.469% | -0.863% |
| `w4u8_mlp_producer_slot_wait_ticks` | 309.000 | 312.000 | 0.971% | 4.207% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,504.000 | 4,552.000 | 1.066% | -2.145% |
| `u8_attention_qk_av_hmx_ticks` | 1,480.000 | 1,490.000 | 0.676% | -1.622% |
| `u8_attention_qk_requant_softmax_ticks` | 12,989.000 | 12,533.000 | -3.511% | -3.577% |

### Traffic, commands, counters and residency

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 495.000 | 500.000 | 1.010% | -0.202% |
| `runtime_teardown_ticks` | 429.000 | 427.000 | -0.466% | 0.000% |
| `ledger_named_ticks` | 43,578.000 | 43,454.000 | -0.285% | -0.275% |
| `ledger_unattributed_ticks` | 28.000 | 26.000 | -7.143% | -7.143% |

Three-target speed gate: **FAIL**; unchanged math/traffic/commands/resources: **PASS**.

## Repeat 10

### Primary latency and Softmax targets

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,288,062.500 | 2,280,166.700 | -0.345% | -0.239% |
| `invocation_ticks` | 42,532.000 | 42,474.400 | -0.135% | -0.398% |
| `total_ticks` | 42,482.400 | 42,425.000 | -0.135% | -0.397% |
| `u8_attention_softmax_ticks` | 12,965.400 | 12,522.200 | -3.418% | -3.592% |
| `attention_ticks` | 5,020.900 | 4,916.700 | -2.075% | -2.400% |

### Additive Block Timing Ledger

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 54.500 | 56.100 | 2.936% | 0.550% |
| `metadata_stage_ticks` | 11.800 | 11.900 | 0.847% | 0.840% |
| `input_norm_ticks` | 1,541.000 | 1,542.700 | 0.110% | 0.143% |
| `qkv_projection_ticks` | 10,649.500 | 10,647.100 | -0.023% | 0.054% |
| `qk_norm_rope_ticks` | 0.400 | 0.500 | 25.000% | 0.000% |
| `attention_ticks` | 5,020.900 | 4,916.700 | -2.075% | -2.400% |
| `o_projection_ticks` | 3,280.900 | 3,280.300 | -0.018% | -0.055% |
| `post_attention_residual_ticks` | 813.400 | 815.500 | 0.258% | 0.197% |
| `post_attention_norm_ticks` | 0.100 | 0.200 | 100.000% | -8.333% |
| `gate_up_ticks` | 13,847.500 | 13,856.100 | 0.062% | -0.225% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,815.000 | 6,737.700 | -1.134% | -0.318% |
| `final_residual_ticks` | 417.500 | 417.700 | 0.048% | 0.000% |
| `output_stage_ticks` | 12.400 | 12.500 | 0.806% | 0.806% |

### Overlapping engine work and waits

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.100 | 8.300 | 2.469% | 1.235% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,746.000 | 10,748.800 | 0.026% | -0.282% |
| `hmx_compute_ticks` | 13,448.500 | 13,213.100 | -1.750% | -0.576% |
| `projection_hmx_wait_ticks` | 230.400 | 220.900 | -4.123% | -3.284% |
| `hmx_ready_wait_ticks` | 8,126.000 | 8,114.100 | -0.146% | -0.425% |
| `w4u8_qkvo_weight_expand_ticks` | 6,387.300 | 6,385.200 | -0.033% | -0.034% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,049.200 | 3,058.200 | 0.295% | 0.399% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,293.100 | 9,290.800 | -0.025% | -0.009% |
| `u8_attention_qk_norm_rope_ticks` | 26,776.200 | 26,785.000 | 0.033% | 0.084% |
| `u8_attention_v_pack_ticks` | 2,815.900 | 2,816.500 | 0.021% | 0.014% |
| `u8_attention_qk_hmx_ticks` | 743.300 | 732.500 | -1.453% | -0.549% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 12,965.400 | 12,522.200 | -3.418% | -3.592% |
| `u8_attention_av_hmx_ticks` | 741.800 | 740.400 | -0.189% | -0.054% |
| `u8_attention_av_requant_ticks` | 1,047.100 | 1,049.400 | 0.220% | -0.058% |
| `u8_attention_pipeline_wait_ticks` | 957.400 | 1,025.600 | 7.123% | 1.540% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,192.700 | 13,203.500 | 0.082% | -0.270% |
| `w4u8_mlp_down_pipeline_ticks` | 5,933.600 | 5,842.800 | -1.530% | -0.377% |
| `w4u8_mlp_activation_work_ticks` | 7,850.400 | 7,854.100 | 0.047% | 0.042% |
| `w4u8_mlp_weight_stage_ticks` | 7,529.800 | 7,468.400 | -0.815% | -0.815% |
| `w4u8_mlp_weight_expand_ticks` | 25,217.100 | 25,057.700 | -0.632% | -0.594% |
| `w4u8_mlp_hmx_compute_ticks` | 7,798.500 | 7,735.900 | -0.803% | -0.394% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,939.900 | 8,951.900 | 0.134% | -0.329% |
| `w4u8_mlp_producer_slot_wait_ticks` | 309.100 | 306.400 | -0.874% | -0.919% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,540.000 | 4,506.000 | -0.749% | -0.429% |
| `u8_attention_qk_av_hmx_ticks` | 1,486.200 | 1,478.000 | -0.552% | -0.806% |
| `u8_attention_qk_requant_softmax_ticks` | 12,965.400 | 12,522.200 | -3.418% | -3.592% |

### Traffic, commands, counters and residency

| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 49.700 | 49.900 | 0.402% | -0.200% |
| `runtime_teardown_ticks` | 42.600 | 42.600 | 0.000% | -0.233% |
| `ledger_named_ticks` | 42,506.600 | 42,449.600 | -0.134% | -0.397% |
| `ledger_unattributed_ticks` | 25.400 | 25.400 | 0.000% | -1.181% |

Three-target speed gate: **PASS**; unchanged math/traffic/commands/resources: **PASS**.

## Physical and correctness gates

| Gate | Result |
|---|---:|
| Final block output | byte-exact, 0 LSB |
| QK / probability / AV audit boundaries | byte-exact |
| Probability row-sum audit telemetry | unchanged |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains primary.

## Decision

EXP-0066 local gate: **FAIL**. Local adoption eligibility: **NO**. Selected Baseline is unchanged without explicit user promotion.

