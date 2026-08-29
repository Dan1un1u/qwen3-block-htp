# EXP-0049 — Complete profiling report

The candidate changes only the W4U8 Gate/Up HMX consumer cadence. It fuses the final U8 output store into the streaming accumulate command and consumes eight consecutive output tiles per worker wake-up. W4 values/scales, U8 qparams, HMX tile arithmetic, DMA bytes/descriptors, Attention, Down, tensor boundaries and final output are unchanged.

## Repeat 1

### Primary latency and Gate/Up target

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 5,141,198.000 | 3,523,593.000 | -31.464% | -31.336% |
| `invocation_ticks` | 90,532.000 | 59,728.000 | -34.026% | -34.054% |
| `total_ticks` | 90,031.000 | 59,169.000 | -34.279% | -34.262% |
| `gate_up_ticks` | 44,697.000 | 13,590.000 | -69.595% | -69.596% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 44,029.000 | 12,930.000 | -70.633% | -70.638% |

### Additive Block Timing Ledger

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 50.000 | 50.000 | 0.000% | 0.000% |
| `metadata_stage_ticks` | 123.000 | 118.000 | -4.065% | -3.419% |
| `input_norm_ticks` | 1,752.000 | 1,748.000 | -0.228% | -0.285% |
| `qkv_projection_ticks` | 17,664.000 | 17,702.000 | 0.215% | 0.242% |
| `qk_norm_rope_ticks` | 2.000 | 1.000 | -50.000% | -50.000% |
| `attention_ticks` | 8,012.000 | 8,004.000 | -0.100% | -0.100% |
| `o_projection_ticks` | 3,281.000 | 3,283.000 | 0.061% | 0.091% |
| `post_attention_residual_ticks` | 3,126.000 | 3,128.000 | 0.064% | 0.096% |
| `post_attention_norm_ticks` | 0.000 | 9.000 | n/a | n/a |
| `gate_up_ticks` | 44,697.000 | 13,590.000 | -69.595% | -69.596% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 9,359.000 | 9,420.000 | 0.652% | 0.892% |
| `final_residual_ticks` | 1,383.000 | 1,383.000 | 0.000% | 0.000% |
| `output_stage_ticks` | 130.000 | 129.000 | -0.769% | 0.000% |

### Overlapping engine work and waits

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 9.000 | 12.500% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 11,534.000 | 10,798.000 | -6.381% | -6.996% |
| `hmx_compute_ticks` | 13,547.000 | 12,749.000 | -5.891% | -5.942% |
| `projection_hmx_wait_ticks` | 220.000 | 216.000 | -1.818% | -3.571% |
| `hmx_ready_wait_ticks` | 6,021.000 | 7,920.000 | 31.540% | 29.679% |
| `w4u8_qkvo_weight_expand_ticks` | 6,356.000 | 6,358.000 | 0.031% | 0.016% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,062.000 | 3,096.000 | 1.110% | 0.288% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,237.000 | 9,246.000 | 0.097% | -0.065% |
| `u8_attention_qk_norm_rope_ticks` | 44,698.000 | 44,738.000 | 0.089% | 0.089% |
| `u8_attention_v_pack_ticks` | 8,231.000 | 8,199.000 | -0.389% | 0.145% |
| `u8_attention_qk_hmx_ticks` | 1,284.000 | 1,317.000 | 2.570% | 2.711% |
| `u8_attention_qk_requant_ticks` | 384.000 | 391.000 | 1.823% | 0.000% |
| `u8_attention_softmax_ticks` | 13,975.000 | 13,988.000 | 0.093% | 0.114% |
| `u8_attention_av_hmx_ticks` | 2,529.000 | 2,549.000 | 0.791% | 1.769% |
| `u8_attention_av_requant_ticks` | 1,061.000 | 1,056.000 | -0.471% | 0.000% |
| `u8_attention_pipeline_wait_ticks` | 3,230.000 | 3,061.000 | -5.232% | -4.678% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 44,029.000 | 12,930.000 | -70.633% | -70.638% |
| `w4u8_mlp_down_pipeline_ticks` | 8,471.000 | 8,537.000 | 0.779% | 0.906% |
| `w4u8_mlp_activation_work_ticks` | 7,998.000 | 7,915.000 | -1.038% | -1.081% |
| `w4u8_mlp_weight_stage_ticks` | 8,221.000 | 7,419.000 | -9.756% | -9.976% |
| `w4u8_mlp_weight_expand_ticks` | 34,908.000 | 24,604.000 | -29.518% | -29.094% |
| `w4u8_mlp_hmx_compute_ticks` | 26,002.000 | 10,397.000 | -60.015% | -59.901% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 6,877.000 | 8,782.000 | 27.701% | 26.174% |
| `w4u8_mlp_producer_slot_wait_ticks` | 320.000 | 306.000 | -4.375% | -4.658% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 31,722.000 | 7,126.000 | -77.536% | -77.536% |

### Traffic, commands, counters and residency

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 1,040.0 | 320.0 | -69.231% | -69.231% |
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
| `runtime_setup_ticks` | 499.000 | 499.000 | 0.000% | 0.000% |
| `runtime_teardown_ticks` | 429.000 | 428.000 | -0.233% | -0.235% |
| `ledger_named_ticks` | 90,506.000 | 59,699.000 | -34.039% | -34.068% |
| `ledger_unattributed_ticks` | 26.000 | 30.000 | 15.385% | 15.385% |

Gate/Up HMX commands per block: **768 → 48**; total HMX commands per block: **1040 → 320**; HMX tile pairs remain **49,408**.

Repeat-1 speed gate: **PASS**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Repeat 10

### Primary latency and Gate/Up target

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 4,726,427.100 | 3,125,567.700 | -33.870% | -34.128% |
| `invocation_ticks` | 89,636.000 | 58,639.600 | -34.580% | -34.532% |
| `total_ticks` | 89,586.400 | 58,589.900 | -34.600% | -34.551% |
| `gate_up_ticks` | 44,727.000 | 13,738.200 | -69.284% | -69.349% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 44,077.200 | 13,084.000 | -70.316% | -70.383% |

### Additive Block Timing Ledger

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 61.100 | 61.900 | 1.309% | -0.818% |
| `metadata_stage_ticks` | 12.000 | 11.900 | -0.833% | 3.306% |
| `input_norm_ticks` | 1,744.700 | 1,743.600 | -0.063% | -0.103% |
| `qkv_projection_ticks` | 17,730.100 | 17,688.400 | -0.235% | -0.289% |
| `qk_norm_rope_ticks` | 0.500 | 0.300 | -40.000% | -50.000% |
| `attention_ticks` | 8,094.100 | 8,040.500 | -0.662% | -0.557% |
| `o_projection_ticks` | 3,283.500 | 3,275.500 | -0.244% | -0.158% |
| `post_attention_residual_ticks` | 3,124.700 | 3,123.900 | -0.026% | -0.019% |
| `post_attention_norm_ticks` | 0.100 | 1.000 | 900.000% | 575.000% |
| `gate_up_ticks` | 44,727.000 | 13,738.200 | -69.284% | -69.349% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 9,364.700 | 9,442.800 | 0.834% | 1.202% |
| `final_residual_ticks` | 1,383.100 | 1,382.800 | -0.022% | -0.014% |
| `output_stage_ticks` | 13.000 | 13.000 | 0.000% | 0.769% |

### Overlapping engine work and waits

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.400 | 8.500 | 1.190% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 11,469.000 | 10,697.900 | -6.723% | -7.430% |
| `hmx_compute_ticks` | 13,468.400 | 12,663.300 | -5.978% | -5.978% |
| `projection_hmx_wait_ticks` | 226.900 | 220.800 | -2.688% | -2.292% |
| `hmx_ready_wait_ticks` | 6,082.400 | 7,924.800 | 30.291% | 29.542% |
| `w4u8_qkvo_weight_expand_ticks` | 6,381.100 | 6,375.700 | -0.085% | -0.049% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,073.300 | 3,060.000 | -0.433% | -1.197% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,308.800 | 9,265.200 | -0.468% | -0.346% |
| `u8_attention_qk_norm_rope_ticks` | 44,681.700 | 44,568.200 | -0.254% | -0.330% |
| `u8_attention_v_pack_ticks` | 8,165.800 | 8,147.000 | -0.230% | -0.147% |
| `u8_attention_qk_hmx_ticks` | 1,275.000 | 1,291.100 | 1.263% | 0.243% |
| `u8_attention_qk_requant_ticks` | 389.100 | 386.800 | -0.591% | -0.129% |
| `u8_attention_softmax_ticks` | 13,964.200 | 13,941.100 | -0.165% | -0.144% |
| `u8_attention_av_hmx_ticks` | 2,551.500 | 2,594.500 | 1.685% | 1.138% |
| `u8_attention_av_requant_ticks` | 1,063.100 | 1,057.500 | -0.527% | -0.329% |
| `u8_attention_pipeline_wait_ticks` | 3,294.100 | 3,154.200 | -4.247% | -4.191% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 44,077.200 | 13,084.000 | -70.316% | -70.383% |
| `w4u8_mlp_down_pipeline_ticks` | 8,459.800 | 8,558.200 | 1.163% | 1.333% |
| `w4u8_mlp_activation_work_ticks` | 7,989.600 | 7,891.000 | -1.234% | -1.435% |
| `w4u8_mlp_weight_stage_ticks` | 8,255.600 | 7,479.000 | -9.407% | -9.407% |
| `w4u8_mlp_weight_expand_ticks` | 35,058.800 | 24,854.500 | -29.106% | -29.106% |
| `w4u8_mlp_hmx_compute_ticks` | 25,952.900 | 10,505.000 | -59.523% | -59.360% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 6,936.600 | 8,793.000 | 26.762% | 25.972% |
| `w4u8_mlp_producer_slot_wait_ticks` | 324.900 | 307.200 | -5.448% | -4.911% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 31,762.600 | 7,155.500 | -77.472% | -77.413% |

### Traffic, commands, counters and residency

| Metric | EXP-0048 control | EXP-0049 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 1,040.0 | 320.0 | -69.231% | -69.231% |
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
| `runtime_setup_ticks` | 49.700 | 49.800 | 0.201% | 0.606% |
| `runtime_teardown_ticks` | 42.600 | 42.400 | -0.469% | -0.926% |
| `ledger_named_ticks` | 89,610.600 | 58,613.200 | -34.591% | -34.542% |
| `ledger_unattributed_ticks` | 25.400 | 25.600 | 0.787% | 2.024% |

Gate/Up HMX commands per block: **768 → 48**; total HMX commands per block: **1040 → 320**; HMX tile pairs remain **49,408**.

Repeat-10 speed gate: **PASS**; unchanged math/traffic: **PASS**; command coarsening: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final block output vs EXP-0048 | byte-exact, 0 LSB |
| Independent block implementation reference | 0 mismatches, 0 LSB |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping work/wait counters are not summed together. Host wall is the primary speed metric.

## Decision

EXP-0049 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged unless the user explicitly promotes it.

