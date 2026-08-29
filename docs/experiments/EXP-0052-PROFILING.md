# EXP-0052 — Complete profiling report

The candidate changes only Q/K Norm-RoPE preparation task claim order. The control claims one GQA group and can block a worker on an unavailable K head after its two Q heads. The candidate exposes 16 ordered Q-head tasks followed by eight K-head tasks, allowing workers to consume every already-published Q head before waiting for K. Projection commands, arithmetic, weights, qparams, traffic, Attention, O, Gate/Up and Down are fixed.

## Repeat 1

### Primary latency and Q/K preparation target

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,403,385.000 | 3,280,834.000 | -3.601% | -3.458% |
| `invocation_ticks` | 57,002.000 | 55,141.000 | -3.265% | -3.331% |
| `total_ticks` | 56,500.000 | 54,644.000 | -3.285% | -3.356% |
| `qkv_projection_ticks` | 17,724.000 | 15,831.000 | -10.680% | -10.489% |
| `attention_qk_norm_pool_wait_ticks` | 10,901.000 | 8,984.000 | -17.586% | -16.381% |

### Additive Block Timing Ledger

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 48.000 | 47.000 | -2.083% | -2.083% |
| `metadata_stage_ticks` | 118.000 | 116.000 | -1.695% | -2.542% |
| `input_norm_ticks` | 1,749.000 | 1,750.000 | 0.057% | -0.057% |
| `qkv_projection_ticks` | 17,724.000 | 15,831.000 | -10.680% | -10.489% |
| `qk_norm_rope_ticks` | 3.000 | 3.000 | 0.000% | 0.000% |
| `attention_ticks` | 8,083.000 | 8,043.000 | -0.495% | -0.525% |
| `o_projection_ticks` | 3,290.000 | 3,286.000 | -0.122% | 0.610% |
| `post_attention_residual_ticks` | 3,126.000 | 3,130.000 | 0.128% | 0.032% |
| `post_attention_norm_ticks` | 0.000 | 0.000 | n/a | n/a |
| `gate_up_ticks` | 13,648.000 | 13,691.000 | 0.315% | -0.058% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,783.000 | 6,775.000 | -0.118% | -0.192% |
| `final_residual_ticks` | 1,383.000 | 1,382.000 | -0.072% | -0.072% |
| `output_stage_ticks` | 130.000 | 130.000 | 0.000% | 0.000% |

### Overlapping engine work and waits

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.000 | 8.000 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,827.000 | 10,865.000 | 0.351% | 0.470% |
| `hmx_compute_ticks` | 13,733.000 | 13,792.000 | 0.430% | -0.537% |
| `projection_hmx_wait_ticks` | 223.000 | 212.000 | -4.933% | -5.000% |
| `hmx_ready_wait_ticks` | 7,938.000 | 7,976.000 | 0.479% | 0.907% |
| `w4u8_qkvo_weight_expand_ticks` | 6,366.000 | 6,380.000 | 0.220% | 0.125% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,103.000 | 3,058.000 | -1.450% | -1.063% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,276.000 | 9,273.000 | -0.032% | -0.279% |
| `u8_attention_qk_norm_rope_ticks` | 44,725.000 | 44,505.000 | -0.492% | -0.324% |
| `u8_attention_v_pack_ticks` | 8,162.000 | 8,102.000 | -0.735% | -0.062% |
| `u8_attention_qk_hmx_ticks` | 1,312.000 | 1,291.000 | -1.601% | -2.182% |
| `u8_attention_qk_requant_ticks` | 390.000 | 392.000 | 0.513% | -0.769% |
| `u8_attention_softmax_ticks` | 14,037.000 | 14,039.000 | 0.014% | -0.057% |
| `u8_attention_av_hmx_ticks` | 2,577.000 | 2,585.000 | 0.310% | 0.189% |
| `u8_attention_av_requant_ticks` | 1,064.000 | 1,052.000 | -1.128% | -0.096% |
| `u8_attention_pipeline_wait_ticks` | 3,383.000 | 3,323.000 | -1.774% | -2.029% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,994.000 | 13,019.000 | 0.192% | -0.155% |
| `w4u8_mlp_down_pipeline_ticks` | 5,901.000 | 5,887.000 | -0.237% | -0.239% |
| `w4u8_mlp_activation_work_ticks` | 7,918.000 | 7,849.000 | -0.871% | -0.886% |
| `w4u8_mlp_weight_stage_ticks` | 7,475.000 | 7,560.000 | 1.137% | 1.069% |
| `w4u8_mlp_weight_expand_ticks` | 24,863.000 | 25,071.000 | 0.837% | 0.947% |
| `w4u8_mlp_hmx_compute_ticks` | 7,851.000 | 7,862.000 | 0.140% | 0.140% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,748.000 | 8,800.000 | 0.594% | 0.972% |
| `w4u8_mlp_producer_slot_wait_ticks` | 307.000 | 299.000 | -2.606% | -1.987% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,493.000 | 4,507.000 | 0.312% | 0.330% |
| `attention_qk_norm_pool_wait_ticks` | 10,901.000 | 8,984.000 | -17.586% | -16.381% |

### Traffic, commands, counters and residency

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
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
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
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
| `runtime_setup_ticks` | 500.000 | 500.000 | 0.000% | -0.201% |
| `runtime_teardown_ticks` | 426.000 | 425.000 | -0.235% | -0.468% |
| `ledger_named_ticks` | 56,977.000 | 55,114.000 | -3.270% | -3.336% |
| `ledger_unattributed_ticks` | 27.000 | 27.000 | 0.000% | 7.407% |

Logical prep work remains **16 Q heads + 8 K heads**; QKV HMX commands remain **32**; total HMX commands remain **256**; HMX tile pairs remain **49,408**.

Repeat-1 speed gate: **PASS**; unchanged math/traffic/resources: **PASS**; scheduler contract: **PASS**.

## Repeat 10

### Primary latency and Q/K preparation target

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,991,921.900 | 2,895,281.200 | -3.230% | -3.080% |
| `invocation_ticks` | 55,966.200 | 54,136.100 | -3.270% | -3.073% |
| `total_ticks` | 55,916.200 | 54,086.200 | -3.273% | -3.074% |
| `qkv_projection_ticks` | 17,675.800 | 15,957.900 | -9.719% | -9.766% |
| `attention_qk_norm_pool_wait_ticks` | 10,805.100 | 9,100.000 | -15.781% | -15.740% |

### Additive Block Timing Ledger

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `input_stage_ticks` | 62.000 | 59.500 | -4.032% | -0.645% |
| `metadata_stage_ticks` | 11.700 | 11.800 | 0.855% | 1.802% |
| `input_norm_ticks` | 1,743.100 | 1,743.800 | 0.040% | 0.034% |
| `qkv_projection_ticks` | 17,675.800 | 15,957.900 | -9.719% | -9.766% |
| `qk_norm_rope_ticks` | 0.800 | 0.900 | 12.500% | 25.000% |
| `attention_ticks` | 8,081.600 | 8,099.300 | 0.219% | 0.089% |
| `o_projection_ticks` | 3,276.100 | 3,281.500 | 0.165% | 0.150% |
| `post_attention_residual_ticks` | 3,123.700 | 3,123.600 | -0.003% | 0.006% |
| `post_attention_norm_ticks` | 0.000 | 0.100 | n/a | -100.000% |
| `gate_up_ticks` | 13,607.900 | 13,615.900 | 0.059% | 0.058% |
| `activation_ticks` | 0.000 | 0.000 | n/a | n/a |
| `down_ticks` | 6,833.400 | 6,825.900 | -0.110% | -0.075% |
| `final_residual_ticks` | 1,382.700 | 1,382.500 | -0.014% | 0.000% |
| `output_stage_ticks` | 13.000 | 12.800 | -1.538% | 0.000% |

### Overlapping engine work and waits

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `projection_pack_ticks` | 8.400 | 8.400 | 0.000% | 0.000% |
| `projection_unpack_ticks` | 0.000 | 0.000 | n/a | n/a |
| `weight_dma_ticks` | 10,681.700 | 10,665.200 | -0.154% | 0.211% |
| `hmx_compute_ticks` | 13,604.800 | 13,507.900 | -0.712% | -0.801% |
| `projection_hmx_wait_ticks` | 220.600 | 219.200 | -0.635% | -2.249% |
| `hmx_ready_wait_ticks` | 7,891.500 | 7,901.100 | 0.122% | -0.232% |
| `w4u8_qkvo_weight_expand_ticks` | 6,387.000 | 6,391.200 | 0.066% | 0.066% |
| `w4u8_qkvo_prefetch_wait_ticks` | 3,068.500 | 3,050.300 | -0.593% | -0.212% |
| `w4u8_qkvo_hmx_lifetime_ticks` | 9,305.800 | 9,286.200 | -0.211% | -0.149% |
| `u8_attention_qk_norm_rope_ticks` | 44,585.100 | 44,750.500 | 0.371% | 0.339% |
| `u8_attention_v_pack_ticks` | 8,189.900 | 8,194.500 | 0.056% | 0.020% |
| `u8_attention_qk_hmx_ticks` | 1,280.400 | 1,284.400 | 0.312% | 0.312% |
| `u8_attention_qk_requant_ticks` | 386.400 | 385.200 | -0.311% | 0.259% |
| `u8_attention_softmax_ticks` | 13,963.700 | 13,979.600 | 0.114% | -0.004% |
| `u8_attention_av_hmx_ticks` | 2,554.600 | 2,569.100 | 0.568% | 0.204% |
| `u8_attention_av_requant_ticks` | 1,062.300 | 1,057.100 | -0.490% | -0.076% |
| `u8_attention_pipeline_wait_ticks` | 3,312.700 | 3,329.300 | 0.501% | 3.132% |
| `w4u8_mlp_gate_up_pipeline_ticks` | 12,959.900 | 12,959.300 | -0.005% | 0.075% |
| `w4u8_mlp_down_pipeline_ticks` | 5,949.300 | 5,922.500 | -0.450% | -0.203% |
| `w4u8_mlp_activation_work_ticks` | 7,890.800 | 7,898.600 | 0.099% | -0.019% |
| `w4u8_mlp_weight_stage_ticks` | 7,447.500 | 7,440.500 | -0.094% | 0.325% |
| `w4u8_mlp_weight_expand_ticks` | 25,008.400 | 24,998.000 | -0.042% | 0.331% |
| `w4u8_mlp_hmx_compute_ticks` | 7,879.600 | 7,862.300 | -0.220% | -0.280% |
| `w4u8_mlp_hmx_ready_wait_ticks` | 8,714.500 | 8,716.900 | 0.028% | -0.057% |
| `w4u8_mlp_producer_slot_wait_ticks` | 308.800 | 306.400 | -0.777% | 0.032% |
| `w4u8_mlp_expanded_slot_wait_ticks` | 4,547.700 | 4,530.000 | -0.389% | -1.208% |
| `attention_qk_norm_pool_wait_ticks` | 10,805.100 | 9,100.000 | -15.781% | -15.740% |

### Traffic, commands, counters and residency

| Metric | Group-task control | Head-task candidate | Delta | Paired delta |
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
| `attention_qk_norm_task_count` | 24.0 | 24.0 | 0.000% | 0.000% |
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
| `runtime_setup_ticks` | 50.200 | 50.000 | -0.398% | -0.398% |
| `runtime_teardown_ticks` | 42.400 | 42.500 | 0.236% | 0.473% |
| `ledger_named_ticks` | 55,940.800 | 54,111.300 | -3.270% | -3.074% |
| `ledger_unattributed_ticks` | 25.400 | 25.700 | 1.181% | 1.176% |

Logical prep work remains **16 Q heads + 8 K heads**; QKV HMX commands remain **32**; total HMX commands remain **256**; HMX tile pairs remain **49,408**.

Repeat-10 speed gate: **PASS**; unchanged math/traffic/resources: **PASS**; scheduler contract: **PASS**.

## Correctness and physical gates

| Gate | Result |
|---|---:|
| Final block output vs EXP-0050 | byte-exact, 0 LSB |
| Independent block implementation reference | 0 mismatches, 0 LSB |
| QK / probability / AV audit hashes | candidate equals control |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC / HMX ownership | one execution unit / one owner |
| QNN dependency | none |

The additive ledger and overlapping work/wait counters are not summed together. Host wall is the primary speed metric.

## Decision

EXP-0052 local gate: **PASS**. Local adoption eligibility: **YES**. The Selected Baseline is unchanged unless the user explicitly promotes this candidate.
