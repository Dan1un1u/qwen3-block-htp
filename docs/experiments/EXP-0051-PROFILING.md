# EXP-0051 — Complete profiling report

Stage A changes only the Q projection worker cadence. One command computes two consecutive 128-channel Q heads, but publishes the first head before waiting for the second. K/V/O, all arithmetic, weights, DMA traffic, qparams and downstream Attention are fixed.

## Repeat 1

### Primary latency and QKV target

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,380,156.000 | 3,392,292.000 | 0.359% | 0.359% |
| `invocation_ticks` | 56,870.000 | 56,908.000 | 0.067% | -0.005% |
| `total_ticks` | 56,370.000 | 56,412.000 | 0.075% | 0.004% |
| `qkv_projection_ticks` | 17,659.000 | 17,542.000 | -0.663% | -0.711% |

### Additive Block Timing Ledger

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 48.000 | 47.000 | -2.083% | -2.083% |
| `metadata_stage_ticks` | 121.000 | 118.000 | -2.479% | -2.542% |
| `input_norm_ticks` | 1,758.000 | 1,759.000 | 0.057% | -0.057% |
| `qkv_projection_ticks` | 17,659.000 | 17,542.000 | -0.663% | -0.711% |
| `qk_norm_rope_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `attention_ticks` | 7,928.000 | 7,978.000 | 0.631% | 0.025% |
| `o_projection_ticks` | 3,273.000 | 3,280.000 | 0.214% | 0.427% |
| `post_attention_residual_ticks` | 3,128.000 | 3,125.000 | -0.096% | -0.096% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,691.000 | 13,683.000 | -0.058% | -0.383% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,802.000 | 6,863.000 | 0.897% | 0.146% |
| `final_residual_ticks` | 1,383.000 | 1,390.000 | 0.506% | 0.072% |
| `output_stage_ticks` | 130.000 | 124.000 | -4.615% | -3.077% |

### Overlapping engine work and waits

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 6.000 | -25.000% | -25.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,797.000 | 10,741.000 | -0.519% | -0.102% |
| `hmx_compute_ticks` | 13,838.000 | 15,342.000 | 10.869% | 10.583% |
| `projection_hmx_wait_ticks` | 214.000 | 206.000 | -3.738% | -2.415% |
| `hmx_ready_wait_ticks` | 8,049.000 | 9,226.000 | 14.623% | 16.898% |
| `w4u8_qkvo_weight_expand_ticks` | 6,368.000 | 6,366.000 | -0.031% | 0.220% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,052.000 | 3,081.000 | 0.950% | 0.838% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,210.000 | 9,265.000 | 0.597% | -0.408% |
| `u8_attention_qk_norm_rope_ticks` | 44,637.000 | 44,666.000 | 0.065% | 0.174% |
| `u8_attention_v_pack_ticks` | 8,219.000 | 8,188.000 | -0.377% | 0.791% |
| `u8_attention_qk_hmx_ticks` | 1,235.000 | 1,219.000 | -1.296% | -0.164% |
| `u8_attention_qk_requant_ticks` | 387.000 | 393.000 | 1.550% | 4.523% |
| `u8_attention_softmax_ticks` | 13,970.000 | 13,956.000 | -0.100% | -0.029% |
| `u8_attention_av_hmx_ticks` | 2,428.000 | 2,492.000 | 2.636% | 1.649% |
| `u8_attention_av_requant_ticks` | 1,072.000 | 1,059.000 | -1.213% | -1.467% |
| `u8_attention_pipeline_wait_ticks` | 3,068.000 | 2,981.000 | -2.836% | 4.497% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,048.000 | 13,029.000 | -0.146% | 0.069% |
| `w4u8_mlp_down_pipeline_ticks` | 5,911.000 | 5,975.000 | 1.083% | 0.135% |
| `w4u8_mlp_activation_work_ticks` | 7,920.000 | 7,899.000 | -0.265% | -0.393% |
| `w4u8_mlp_weight_stage_ticks` | 7,530.000 | 7,426.000 | -1.381% | -1.418% |
| `w4u8_mlp_weight_expand_ticks` | 25,224.000 | 25,112.000 | -0.444% | -0.444% |
| `w4u8_mlp_hmx_compute_ticks` | 7,877.000 | 7,929.000 | 0.660% | 0.507% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,893.000 | 8,720.000 | -1.945% | -0.707% |
| `w4u8_mlp_producer_slot_wait_ticks` | 299.000 | 298.000 | -0.334% | -1.338% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,618.000 | 4,491.000 | -2.750% | -2.750% |

### Traffic, commands, counters and residency

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 248.0 | -3.125% | -3.125% |
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
| `w4u8_qkv_batch_count` | 32.0 | 24.0 | -25.000% | -25.000% |
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
| `vtcm_peak_plan_bytes` | 6,093,536.0 | 6,093,536.0 | 0.000% | 0.000% |
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
| `runtime_setup_ticks` | 497.000 | 501.000 | 0.805% | 0.200% |
| `runtime_teardown_ticks` | 425.000 | 425.000 | 0.000% | -0.471% |
| `ledger_named_ticks` | 56,846.000 | 56,883.000 | 0.065% | -0.005% |
| `ledger_unattributed_ticks` | 26.000 | 26.000 | 0.000% | 0.000% |

Q commands per block: **16 → 8**; total QKV commands per block: **32 → 24**; total HMX commands: **256 → 248**; HMX tile pairs remain **49,408**. Head readiness remains one head.

Repeat-1 speed gate: **FAIL**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Repeat 10

### Primary latency and QKV target

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,989,135.400 | 2,971,614.500 | -0.586% | -0.562% |
| `invocation_ticks` | 55,865.000 | 55,698.300 | -0.298% | -0.364% |
| `total_ticks` | 55,815.200 | 55,648.400 | -0.299% | -0.365% |
| `qkv_projection_ticks` | 17,688.700 | 17,570.200 | -0.670% | -0.593% |

### Additive Block Timing Ledger

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 60.800 | 59.500 | -2.138% | -2.619% |
| `metadata_stage_ticks` | 12.300 | 11.900 | -3.252% | -1.667% |
| `input_norm_ticks` | 1,744.300 | 1,744.700 | 0.023% | 0.011% |
| `qkv_projection_ticks` | 17,688.700 | 17,570.200 | -0.670% | -0.593% |
| `qk_norm_rope_ticks` | 0.200 | 0.200 | 0.000% | -25.000% |
| `attention_ticks` | 7,970.600 | 7,993.900 | 0.292% | -0.503% |
| `o_projection_ticks` | 3,281.600 | 3,279.800 | -0.055% | -0.006% |
| `post_attention_residual_ticks` | 3,123.500 | 3,124.100 | 0.019% | 0.022% |
| `post_attention_norm_ticks` | 0.100 | 0.000 | -100.000% | -100.000% |
| `gate_up_ticks` | 13,665.800 | 13,620.400 | -0.332% | -0.332% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,796.400 | 6,808.700 | 0.181% | -0.098% |
| `final_residual_ticks` | 1,382.500 | 1,382.800 | 0.022% | 0.014% |
| `output_stage_ticks` | 12.900 | 13.000 | 0.775% | 0.000% |

### Overlapping engine work and waits

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 7.400 | 7.300 | -1.351% | -2.703% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,749.200 | 10,713.800 | -0.329% | -0.475% |
| `hmx_compute_ticks` | 13,714.700 | 15,017.200 | 9.497% | 9.811% |
| `projection_hmx_wait_ticks` | 216.300 | 209.700 | -3.051% | -2.492% |
| `hmx_ready_wait_ticks` | 7,966.700 | 9,394.100 | 17.917% | 17.117% |
| `w4u8_qkvo_weight_expand_ticks` | 6,382.600 | 6,392.200 | 0.150% | 0.147% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,039.200 | 3,044.900 | 0.188% | -0.749% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,277.300 | 9,275.300 | -0.022% | -0.252% |
| `u8_attention_qk_norm_rope_ticks` | 44,607.300 | 44,739.100 | 0.295% | 0.295% |
| `u8_attention_v_pack_ticks` | 8,185.700 | 8,170.400 | -0.187% | 0.060% |
| `u8_attention_qk_hmx_ticks` | 1,240.200 | 1,247.600 | 0.597% | 1.324% |
| `u8_attention_qk_requant_ticks` | 383.900 | 381.900 | -0.521% | -0.156% |
| `u8_attention_softmax_ticks` | 13,963.900 | 13,972.600 | 0.062% | 0.130% |
| `u8_attention_av_hmx_ticks` | 2,461.300 | 2,469.800 | 0.345% | 0.407% |
| `u8_attention_av_requant_ticks` | 1,058.300 | 1,061.900 | 0.340% | 0.198% |
| `u8_attention_pipeline_wait_ticks` | 3,040.100 | 3,166.600 | 4.161% | -0.376% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,012.000 | 12,968.200 | -0.337% | -0.337% |
| `w4u8_mlp_down_pipeline_ticks` | 5,905.900 | 5,921.600 | 0.266% | -0.237% |
| `w4u8_mlp_activation_work_ticks` | 7,898.900 | 7,889.900 | -0.114% | 0.213% |
| `w4u8_mlp_weight_stage_ticks` | 7,500.600 | 7,502.200 | 0.021% | 0.094% |
| `w4u8_mlp_weight_expand_ticks` | 25,145.400 | 25,147.100 | 0.007% | 0.165% |
| `w4u8_mlp_hmx_compute_ticks` | 7,835.300 | 7,796.200 | -0.499% | 0.162% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,798.500 | 8,773.100 | -0.289% | -0.523% |
| `w4u8_mlp_producer_slot_wait_ticks` | 306.700 | 308.300 | 0.522% | -1.026% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,504.300 | 4,497.700 | -0.147% | -0.243% |

### Traffic, commands, counters and residency

| Metric | Control | Q head-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 248.0 | -3.125% | -3.125% |
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
| `w4u8_qkv_batch_count` | 32.0 | 24.0 | -25.000% | -25.000% |
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
| `vtcm_peak_plan_bytes` | 6,093,536.0 | 6,093,536.0 | 0.000% | 0.000% |
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
| `runtime_setup_ticks` | 49.900 | 49.900 | 0.000% | -0.400% |
| `runtime_teardown_ticks` | 42.500 | 42.500 | 0.000% | 0.000% |
| `ledger_named_ticks` | 55,839.700 | 55,672.100 | -0.300% | -0.364% |
| `ledger_unattributed_ticks` | 25.600 | 25.300 | -1.172% | -0.791% |

Q commands per block: **16 → 8**; total QKV commands per block: **32 → 24**; total HMX commands: **256 → 248**; HMX tile pairs remain **49,408**. Head readiness remains one head.

Repeat-10 speed gate: **PASS**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final block output vs control | byte-exact, 0 LSB |
| Independent block implementation reference | 0 mismatches, 0 LSB |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping work/wait counters are not summed together. Host wall is the primary speed metric.

## Decision

EXP-0051 Stage-A local gate: **FAIL**. Stage B eligibility: **NO**. Selected Baseline is unchanged unless the user explicitly promotes it.


