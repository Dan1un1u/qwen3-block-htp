# EXP-0065 — Complete profiling report

The control rebuilds two sixteen-entry SOLE probability LUT banks on every paired-head row. The candidate builds the fourteen reachable `(leading, next)` banks once in each GQA group's reused VTCM scratch and selects them per row. QK requantization, log2 codes, sums, probability bytes, HMX commands and all other block work are unchanged.

## Repeat 1

### Primary latency and Softmax targets

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,677,969.000 | 2,667,552.000 | -0.389% | -0.815% |
| `invocation_ticks` | 43,529.000 | 43,636.000 | 0.246% | 0.085% |
| `total_ticks` | 43,035.000 | 43,125.000 | 0.209% | 0.093% |
| `u8_attention_softmax_ticks` | 13,153.000 | 12,958.000 | -1.483% | -1.961% |
| `attention_ticks` | 5,047.000 | 4,994.000 | -1.050% | -1.113% |

### Additive Block Timing Ledger

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 48.000 | 50.000 | 4.167% | 4.167% |
| `metadata_stage_ticks` | 119.000 | 118.000 | -0.840% | -0.847% |
| `input_norm_ticks` | 1,557.000 | 1,557.000 | 0.000% | 0.000% |
| `qkv_projection_ticks` | 10,660.000 | 10,692.000 | 0.300% | -0.019% |
| `qk_norm_rope_ticks` | 3.000 | 3.000 | 0.000% | 0.000% |
| `attention_ticks` | 5,047.000 | 4,994.000 | -1.050% | -1.113% |
| `o_projection_ticks` | 3,278.000 | 3,291.000 | 0.397% | 0.213% |
| `post_attention_residual_ticks` | 820.000 | 822.000 | 0.244% | 0.000% |
| `post_attention_norm_ticks` | 0.000 | 1.000 | n/a | n/a |
| `gate_up_ticks` | 13,671.000 | 13,684.000 | 0.095% | 0.065% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,802.000 | 6,839.000 | 0.544% | 0.530% |
| `final_residual_ticks` | 417.000 | 421.000 | 0.959% | 0.719% |
| `output_stage_ticks` | 124.000 | 125.000 | 0.806% | 0.000% |

### Overlapping engine work and waits

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 9.000 | 8.000 | -11.111% | -11.111% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,711.000 | 10,788.000 | 0.719% | 0.660% |
| `hmx_compute_ticks` | 13,298.000 | 13,390.000 | 0.692% | 0.304% |
| `projection_hmx_wait_ticks` | 222.000 | 226.000 | 1.802% | 2.262% |
| `hmx_ready_wait_ticks` | 8,040.000 | 7,959.000 | -1.007% | -0.025% |
| `w4u8_qkvo_weight_expand_ticks` | 6,358.000 | 6,368.000 | 0.157% | 0.157% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,090.000 | 3,080.000 | -0.324% | -0.233% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,245.000 | 9,275.000 | 0.324% | 0.575% |
| `u8_attention_qk_norm_rope_ticks` | 26,929.000 | 26,965.000 | 0.134% | -0.056% |
| `u8_attention_v_pack_ticks` | 2,786.000 | 2,795.000 | 0.323% | 0.323% |
| `u8_attention_qk_hmx_ticks` | 733.000 | 738.000 | 0.682% | 0.545% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,153.000 | 12,958.000 | -1.483% | -1.961% |
| `u8_attention_av_hmx_ticks` | 729.000 | 736.000 | 0.960% | 2.469% |
| `u8_attention_av_requant_ticks` | 1,034.000 | 1,033.000 | -0.097% | -0.097% |
| `u8_attention_pipeline_wait_ticks` | 1,017.000 | 996.000 | -2.065% | -6.330% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,017.000 | 13,036.000 | 0.146% | -0.053% |
| `w4u8_mlp_down_pipeline_ticks` | 5,922.000 | 5,943.000 | 0.355% | 0.220% |
| `w4u8_mlp_activation_work_ticks` | 7,925.000 | 7,898.000 | -0.341% | -0.291% |
| `w4u8_mlp_weight_stage_ticks` | 7,523.000 | 7,488.000 | -0.465% | 0.456% |
| `w4u8_mlp_weight_expand_ticks` | 24,992.000 | 25,018.000 | 0.104% | 0.380% |
| `w4u8_mlp_hmx_compute_ticks` | 7,853.000 | 7,890.000 | 0.471% | 1.445% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,852.000 | 8,769.000 | -0.938% | 0.123% |
| `w4u8_mlp_producer_slot_wait_ticks` | 313.000 | 302.000 | -3.514% | -5.648% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,541.000 | 4,564.000 | 0.506% | -0.396% |
| `u8_attention_qk_av_hmx_ticks` | 1,450.000 | 1,474.000 | 1.655% | 2.190% |
| `u8_attention_qk_requant_softmax_ticks` | 13,153.000 | 12,958.000 | -1.483% | -1.961% |

### Traffic, commands, counters and residency

| Metric | Per-row control | Template candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 499.000 | 501.000 | 0.401% | 1.215% |
| `runtime_teardown_ticks` | 425.000 | 430.000 | 1.176% | 1.425% |
| `ledger_named_ticks` | 43,501.000 | 43,612.000 | 0.255% | 0.085% |
| `ledger_unattributed_ticks` | 27.000 | 26.000 | -3.704% | -7.692% |

Three-target speed gate: **PASS**; unchanged math/traffic/commands/resources: **PASS**.

## Repeat 10

### Primary latency and Softmax targets

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,291,161.500 | 2,288,052.100 | -0.136% | -0.002% |
| `invocation_ticks` | 42,624.900 | 42,498.700 | -0.296% | 0.004% |
| `total_ticks` | 42,574.700 | 42,449.000 | -0.295% | 0.005% |
| `u8_attention_softmax_ticks` | 13,179.900 | 12,926.300 | -1.924% | -1.958% |
| `attention_ticks` | 5,069.700 | 4,987.900 | -1.614% | -1.570% |

### Additive Block Timing Ledger

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 54.900 | 55.300 | 0.729% | 0.000% |
| `metadata_stage_ticks` | 12.000 | 12.000 | 0.000% | 1.667% |
| `input_norm_ticks` | 1,542.900 | 1,541.300 | -0.104% | -0.129% |
| `qkv_projection_ticks` | 10,670.100 | 10,655.700 | -0.135% | -0.088% |
| `qk_norm_rope_ticks` | 0.400 | 0.400 | 0.000% | 0.000% |
| `attention_ticks` | 5,069.700 | 4,987.900 | -1.614% | -1.570% |
| `o_projection_ticks` | 3,291.900 | 3,288.600 | -0.100% | 0.055% |
| `post_attention_residual_ticks` | 816.100 | 818.000 | 0.233% | 0.232% |
| `post_attention_norm_ticks` | 0.200 | 0.200 | 0.000% | 25.000% |
| `gate_up_ticks` | 13,801.100 | 13,788.900 | -0.088% | 0.745% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,825.300 | 6,836.700 | 0.167% | -0.203% |
| `final_residual_ticks` | 419.000 | 419.300 | 0.072% | 0.048% |
| `output_stage_ticks` | 12.500 | 12.400 | -0.800% | 0.000% |

### Overlapping engine work and waits

| Metric | Per-row control | Template candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.300 | 8.300 | 0.000% | 1.205% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,764.000 | 10,771.100 | 0.066% | 0.056% |
| `hmx_compute_ticks` | 13,432.700 | 13,336.700 | -0.715% | -0.715% |
| `projection_hmx_wait_ticks` | 228.300 | 226.000 | -1.007% | -0.991% |
| `hmx_ready_wait_ticks` | 8,024.300 | 7,968.600 | -0.694% | 0.136% |
| `w4u8_qkvo_weight_expand_ticks` | 6,387.100 | 6,388.400 | 0.020% | -0.041% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,056.900 | 3,074.800 | 0.586% | -0.371% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,305.200 | 9,317.200 | 0.129% | 0.070% |
| `u8_attention_qk_norm_rope_ticks` | 26,883.200 | 26,837.200 | -0.171% | -0.074% |
| `u8_attention_v_pack_ticks` | 2,797.000 | 2,798.300 | 0.046% | 0.150% |
| `u8_attention_qk_hmx_ticks` | 741.300 | 734.600 | -0.904% | -0.924% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,179.900 | 12,926.300 | -1.924% | -1.958% |
| `u8_attention_av_hmx_ticks` | 745.800 | 731.200 | -1.958% | -1.958% |
| `u8_attention_av_requant_ticks` | 1,043.300 | 1,045.400 | 0.201% | 0.096% |
| `u8_attention_pipeline_wait_ticks` | 998.600 | 969.300 | -2.934% | -5.608% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,142.500 | 13,131.700 | -0.082% | 0.778% |
| `w4u8_mlp_down_pipeline_ticks` | 5,941.200 | 5,950.100 | 0.150% | -0.298% |
| `w4u8_mlp_activation_work_ticks` | 7,901.300 | 7,890.400 | -0.138% | -0.173% |
| `w4u8_mlp_weight_stage_ticks` | 7,578.900 | 7,507.900 | -0.937% | -0.057% |
| `w4u8_mlp_weight_expand_ticks` | 25,283.500 | 25,130.300 | -0.606% | -0.343% |
| `w4u8_mlp_hmx_compute_ticks` | 7,921.300 | 7,918.500 | -0.035% | -0.035% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,840.300 | 8,786.000 | -0.614% | 0.016% |
| `w4u8_mlp_producer_slot_wait_ticks` | 303.400 | 303.100 | -0.099% | -0.719% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,562.500 | 4,541.400 | -0.462% | -0.389% |
| `u8_attention_qk_av_hmx_ticks` | 1,485.900 | 1,467.000 | -1.272% | -0.881% |
| `u8_attention_qk_requant_softmax_ticks` | 13,179.900 | 12,926.300 | -1.924% | -1.958% |

### Traffic, commands, counters and residency

| Metric | Per-row control | Template candidate | Delta | Paired delta |
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
| `runtime_setup_ticks` | 50.200 | 50.100 | -0.199% | -0.595% |
| `runtime_teardown_ticks` | 42.500 | 42.400 | -0.235% | -0.235% |
| `ledger_named_ticks` | 42,599.600 | 42,473.600 | -0.296% | 0.004% |
| `ledger_unattributed_ticks` | 25.200 | 25.100 | -0.397% | -0.397% |

Three-target speed gate: **PASS**; unchanged math/traffic/commands/resources: **PASS**.

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

The additive ledger and overlapping engine-work counters are not summed together. Complete Host wall remains primary.

## Decision

EXP-0065 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged without explicit user promotion.

