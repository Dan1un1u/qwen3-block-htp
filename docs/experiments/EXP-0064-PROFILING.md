# EXP-0064 — Complete profiling report

The control submits one integer-HMX command per QK or AV output tile. The candidate submits both QK tiles and all four AV tiles of each Q head through the worker's existing multi-output loop. Tile arithmetic, ordering, weights, biases, qparams, score and probability paths are unchanged.

## Repeat 1

### Primary latency and Attention-HMX targets

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,795,781.000 | 2,719,323.000 | -2.735% | -1.228% |
| `invocation_ticks` | 45,712.000 | 43,864.000 | -4.043% | -4.040% |
| `total_ticks` | 45,214.000 | 43,369.000 | -4.081% | -4.065% |
| `u8_attention_qk_av_hmx_ticks` | 3,993.000 | 1,502.000 | -62.384% | -62.375% |
| `u8_attention_qk_hmx_ticks` | 1,350.000 | 738.000 | -45.333% | -43.319% |
| `u8_attention_av_hmx_ticks` | 2,663.000 | 747.000 | -71.949% | -71.842% |
| `attention_ticks` | 7,046.000 | 5,090.000 | -27.760% | -27.789% |

### Additive Block Timing Ledger

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 50.000 | 50.000 | 0.000% | 8.696% |
| `metadata_stage_ticks` | 118.000 | 119.000 | 0.847% | 0.000% |
| `input_norm_ticks` | 1,558.000 | 1,552.000 | -0.385% | 0.000% |
| `qkv_projection_ticks` | 10,654.000 | 10,661.000 | 0.066% | 0.103% |
| `qk_norm_rope_ticks` | 3.000 | 2.000 | -33.333% | -16.667% |
| `attention_ticks` | 7,046.000 | 5,090.000 | -27.760% | -27.789% |
| `o_projection_ticks` | 3,275.000 | 3,267.000 | -0.244% | 0.431% |
| `post_attention_residual_ticks` | 821.000 | 830.000 | 1.096% | 0.607% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | -100.000% |
| `gate_up_ticks` | 13,855.000 | 13,906.000 | 0.368% | -0.380% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,765.000 | 6,848.000 | 1.227% | 0.805% |
| `final_residual_ticks` | 416.000 | 419.000 | 0.721% | 0.000% |
| `output_stage_ticks` | 124.000 | 125.000 | 0.806% | 0.806% |

### Overlapping engine work and waits

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 9.000 | 9.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,889.000 | 10,842.000 | -0.432% | -0.111% |
| `hmx_compute_ticks` | 13,945.000 | 13,547.000 | -2.854% | -1.972% |
| `projection_hmx_wait_ticks` | 219.000 | 223.000 | 1.826% | 0.000% |
| `hmx_ready_wait_ticks` | 8,048.000 | 8,011.000 | -0.460% | 0.235% |
| `w4u8_qkvo_weight_expand_ticks` | 6,356.000 | 6,358.000 | 0.031% | 0.031% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,060.000 | 3,080.000 | 0.654% | 1.275% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,227.000 | 9,236.000 | 0.098% | 0.650% |
| `u8_attention_qk_norm_rope_ticks` | 26,942.000 | 26,912.000 | -0.111% | -0.263% |
| `u8_attention_v_pack_ticks` | 2,813.000 | 2,808.000 | -0.178% | 0.569% |
| `u8_attention_qk_hmx_ticks` | 1,350.000 | 738.000 | -45.333% | -43.319% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,137.000 | 13,157.000 | 0.152% | 0.341% |
| `u8_attention_av_hmx_ticks` | 2,663.000 | 747.000 | -71.949% | -71.842% |
| `u8_attention_av_requant_ticks` | 1,041.000 | 1,043.000 | 0.192% | 0.576% |
| `u8_attention_pipeline_wait_ticks` | 5,366.000 | 915.000 | -82.948% | -82.947% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,190.000 | 13,244.000 | 0.409% | -0.406% |
| `w4u8_mlp_down_pipeline_ticks` | 5,873.000 | 5,967.000 | 1.601% | 0.800% |
| `w4u8_mlp_activation_work_ticks` | 7,867.000 | 7,931.000 | 0.814% | 0.822% |
| `w4u8_mlp_weight_stage_ticks` | 7,541.000 | 7,547.000 | 0.080% | -0.039% |
| `w4u8_mlp_weight_expand_ticks` | 25,023.000 | 25,082.000 | 0.236% | 0.519% |
| `w4u8_mlp_hmx_compute_ticks` | 7,857.000 | 8,022.000 | 2.100% | 0.802% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,859.000 | 8,848.000 | -0.124% | 0.606% |
| `w4u8_mlp_producer_slot_wait_ticks` | 303.000 | 298.000 | -1.650% | -1.672% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,568.000 | 4,706.000 | 3.021% | 3.568% |
| `u8_attention_qk_av_hmx_ticks` | 3,993.000 | 1,502.000 | -62.384% | -62.375% |
| `u8_attention_qk_requant_softmax_ticks` | 13,137.000 | 13,157.000 | 0.152% | 0.341% |

### Traffic, commands, counters and residency

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 192.0 | -25.000% | -25.000% |
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
| `runtime_setup_ticks` | 501.000 | 499.000 | -0.399% | -0.398% |
| `runtime_teardown_ticks` | 428.000 | 424.000 | -0.935% | 0.000% |
| `ledger_named_ticks` | 45,687.000 | 43,838.000 | -4.047% | -4.049% |
| `ledger_unattributed_ticks` | 30.000 | 29.000 | -3.333% | -3.333% |

Three-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**; command batching: **PASS**.

## Repeat 10

### Primary latency and Attention-HMX targets

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,401,099.000 | 2,290,593.800 | -4.602% | -4.803% |
| `invocation_ticks` | 44,474.200 | 42,704.300 | -3.980% | -4.238% |
| `total_ticks` | 44,423.800 | 42,654.100 | -3.984% | -4.243% |
| `u8_attention_qk_av_hmx_ticks` | 4,045.100 | 1,488.300 | -63.207% | -62.722% |
| `u8_attention_qk_hmx_ticks` | 1,359.500 | 747.300 | -45.031% | -44.782% |
| `u8_attention_av_hmx_ticks` | 2,683.700 | 742.900 | -72.318% | -71.998% |
| `attention_ticks` | 7,058.200 | 5,111.100 | -27.586% | -27.639% |

### Additive Block Timing Ledger

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 57.000 | 55.000 | -3.509% | -0.874% |
| `metadata_stage_ticks` | 11.900 | 11.800 | -0.840% | -0.840% |
| `input_norm_ticks` | 1,543.100 | 1,542.900 | -0.013% | 0.104% |
| `qkv_projection_ticks` | 10,689.300 | 10,675.700 | -0.127% | 0.103% |
| `qk_norm_rope_ticks` | 0.600 | 0.500 | -16.667% | -25.000% |
| `attention_ticks` | 7,058.200 | 5,111.100 | -27.586% | -27.639% |
| `o_projection_ticks` | 3,282.900 | 3,282.700 | -0.006% | 0.088% |
| `post_attention_residual_ticks` | 814.100 | 816.800 | 0.332% | 0.148% |
| `post_attention_norm_ticks` | 0.000 | 0.200 | n/a | -33.333% |
| `gate_up_ticks` | 13,809.800 | 13,869.200 | 0.430% | -0.048% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,776.500 | 6,794.800 | 0.270% | 0.742% |
| `final_residual_ticks` | 418.100 | 418.400 | 0.072% | 0.072% |
| `output_stage_ticks` | 12.500 | 12.300 | -1.600% | -0.806% |

### Overlapping engine work and waits

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.800 | 9.000 | 2.273% | 1.136% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,739.900 | 10,793.900 | 0.503% | 0.559% |
| `hmx_compute_ticks` | 13,740.800 | 13,314.600 | -3.102% | -3.102% |
| `projection_hmx_wait_ticks` | 234.400 | 226.200 | -3.498% | -2.332% |
| `hmx_ready_wait_ticks` | 8,010.600 | 8,006.900 | -0.046% | 0.047% |
| `w4u8_qkvo_weight_expand_ticks` | 6,387.300 | 6,384.900 | -0.038% | -0.038% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,065.700 | 3,065.100 | -0.020% | -0.356% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,309.100 | 9,304.000 | -0.055% | -0.055% |
| `u8_attention_qk_norm_rope_ticks` | 26,885.900 | 26,879.300 | -0.025% | 0.101% |
| `u8_attention_v_pack_ticks` | 2,818.700 | 2,817.900 | -0.028% | 0.050% |
| `u8_attention_qk_hmx_ticks` | 1,359.500 | 747.300 | -45.031% | -44.782% |
| `u8_attention_qk_requant_ticks` | 0.000 | 0.000 | n/a | n/a |
| `u8_attention_softmax_ticks` | 13,067.200 | 13,289.200 | 1.699% | 1.845% |
| `u8_attention_av_hmx_ticks` | 2,683.700 | 742.900 | -72.318% | -71.998% |
| `u8_attention_av_requant_ticks` | 1,049.700 | 1,041.100 | -0.819% | -0.831% |
| `u8_attention_pipeline_wait_ticks` | 5,000.400 | 990.400 | -80.194% | -80.245% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 13,148.700 | 13,217.600 | 0.524% | -0.048% |
| `w4u8_mlp_down_pipeline_ticks` | 5,896.300 | 5,903.100 | 0.115% | 0.605% |
| `w4u8_mlp_activation_work_ticks` | 7,880.100 | 7,894.600 | 0.184% | 0.151% |
| `w4u8_mlp_weight_stage_ticks` | 7,491.900 | 7,571.700 | 1.065% | 0.254% |
| `w4u8_mlp_weight_expand_ticks` | 24,959.700 | 25,115.300 | 0.623% | -0.157% |
| `w4u8_mlp_hmx_compute_ticks` | 7,872.000 | 7,933.900 | 0.786% | 0.645% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,829.300 | 8,831.800 | 0.028% | 0.126% |
| `w4u8_mlp_producer_slot_wait_ticks` | 297.400 | 296.300 | -0.370% | -0.370% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,576.000 | 4,540.400 | -0.778% | -0.218% |
| `u8_attention_qk_av_hmx_ticks` | 4,045.100 | 1,488.300 | -63.207% | -62.722% |
| `u8_attention_qk_requant_softmax_ticks` | 13,067.200 | 13,289.200 | 1.699% | 1.845% |

### Traffic, commands, counters and residency

| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `hmx_command_count` | 256.0 | 192.0 | -25.000% | -25.000% |
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
| `runtime_setup_ticks` | 50.000 | 49.900 | -0.200% | 0.000% |
| `runtime_teardown_ticks` | 42.900 | 42.800 | -0.233% | 0.233% |
| `ledger_named_ticks` | 44,448.600 | 42,678.500 | -3.982% | -4.242% |
| `ledger_unattributed_ticks` | 25.600 | 25.700 | 0.391% | 0.392% |

Three-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**; command batching: **PASS**.

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

EXP-0064 local gate: **PASS**. Local adoption eligibility: **YES**. Selected Baseline is unchanged without explicit user promotion.

