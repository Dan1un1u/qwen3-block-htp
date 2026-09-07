# EXP-0241: exact LPBQ32 direct-W4 experiment

Decision: stop_before_full_model_discuss. No baseline promotion. Full model not started.

Quantizer and integer semantics are unchanged from EXP0240. For m in1..16 and signed q in[-8,7], q*m=sum_b(2**b)*(q if bit_b(m) else0). Five packed-W4 masks are produced inside timed DSP VTCM. Each mask is consumed2**b times with weight.n into a single accumulator, then one unchanged bias/scale/U8 conversion. This implementation issues31 HMX passes including zero-mask work; no S8 weight buffer is materialized. It is a bounded exact algorithm, not a lower bound on optimal LPBQ execution.

DDR weights retain the original compressed size plus multiplier metadata. Host only reorders nibbles to direct-n lane order; group products/masks are never pre-expanded in DDR. All seven projections use mode3. Historical expansion counter names measure packed-W4 masking for mode3, not S8 expansion. HMX tile-pair counters include every repeated pass; attention U8xS8 is unchanged. QKV batches4; O/Gate/Up batches4; Down batches2. Mask buffers fit existing1MiB slots (QKV4=655360 bytes; Down2=983040 bytes); zero intermediate DDR/spill,8MiB VTCM and oneRPC/step remain.

Scope: real layer14 M64 prefill followed by eight teacher-input M1 steps with self-computed persistent KV. Repeat10 means ten complete replays within one loaded Prepared Runtime, state/cache reset before each replay; no discarded warmup or outliers. Five short and ten formal rounds, three-way rotated/reversed order. Latencies are medians of ten round means; ratios are medians of paired round ratios. Bootstrap50000 seed241. No token boundary or extrapolated throughput.

Correctness: {'checks': 63, 'elements': 1474560, 'mismatches': 0}. Independent S8 reconstruction, int64 cross-check and SDK native final conversion; complete layer/KV against EXP0240 and unit-multiplier against per-channel verified before collection. Original failed diagnostics, if any, are retained separately.

| Scope | Per-channel us | LPBQ S8 us | LPBQ direct-W4 us | Direct vs per-channel | Ratio95% CI |
|---|---:|---:|---:|---:|---|
|repeat1_prefill|1864.6090|4435.2345|6320.9375|+240.333%|[3.3207601532113316, 3.5864606988986347]|
|repeat1_decode|1037.4186|3740.5306|5772.7216|+459.679%|[5.325740451875142, 5.718062490879376]|
|repeat10_prefill|1645.7291|4227.5833|6230.7498|+276.549%|[3.666567878571326, 3.8360769925467975]|
|repeat10_decode|1016.4336|3689.6279|5786.2855|+469.002%|[5.622724891530963, 5.731104966868891]|

## Complete additive module table: repeat1_prefill

Units us; parentheses percent of complete Host wall. Same median-Host-ranked two rounds for every field within each cell.

| Module | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|I/O、metadata|19.8958 (1.067%)|19.9740 (0.450%)|16.6406 (0.263%)|
|Input RMSNorm|22.2135 (1.191%)|22.1354 (0.499%)|23.2292 (0.367%)|
|QKV＋Q/K Norm-RoPE|261.3021 (14.014%)|369.4792 (8.331%)|870.2604 (13.768%)|
|QK–Softmax–AV|129.7135 (6.957%)|124.2188 (2.801%)|123.3854 (1.952%)|
|O projection|47.6302 (2.554%)|295.2344 (6.657%)|441.5625 (6.986%)|
|Post-attention residual＋RMSNorm|25.2865 (1.356%)|25.4167 (0.573%)|25.8594 (0.409%)|
|Gate/Up＋SwiGLU|631.5365 (33.870%)|2116.0938 (47.711%)|2939.2448 (46.500%)|
|Down|142.3177 (7.633%)|874.7396 (19.723%)|1282.2135 (20.285%)|
|Final residual|7.5781 (0.406%)|7.3177 (0.165%)|7.9427 (0.126%)|
|KV carrier conversion|5.4167 (0.290%)|5.4688 (0.123%)|5.8854 (0.093%)|
|KV append DMA|8.6979 (0.466%)|8.9844 (0.203%)|7.3958 (0.117%)|
|Block orchestration|6.0417 (0.324%)|5.9635 (0.134%)|7.6302 (0.121%)|
|Layer bookkeeping|2.7083 (0.145%)|2.7865 (0.063%)|3.5938 (0.057%)|
|Stage-boundary bookkeeping|6.1719 (0.331%)|6.2240 (0.140%)|6.0417 (0.096%)|
|DSP unattributed|0.0000 (0.000%)|0.0000 (0.000%)|0.0000 (0.000%)|
|Runtime setup/teardown|104.5052 (5.605%)|102.0052 (2.300%)|104.5833 (1.655%)|
|Host-DSP boundary|443.5934 (23.790%)|449.1928 (10.128%)|455.4688 (7.206%)|
|Complete Host wall|1864.6090 (100.000%)|4435.2345 (100.000%)|6320.9375 (100.000%)|

## Complete additive module table: repeat1_decode

Units us; parentheses percent of complete Host wall. Same median-Host-ranked two rounds for every field within each cell.

| Module | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|I/O、metadata|17.8743 (1.723%)|18.1576 (0.485%)|18.1738 (0.315%)|
|Input RMSNorm|5.3125 (0.512%)|5.3288 (0.142%)|5.3158 (0.092%)|
|QKV＋Q/K Norm-RoPE|87.6953 (8.453%)|229.5540 (6.137%)|858.8542 (14.878%)|
|QK–Softmax–AV|123.8704 (11.940%)|124.1960 (3.320%)|124.8958 (2.164%)|
|O projection|43.7109 (4.213%)|289.0430 (7.727%)|434.8633 (7.533%)|
|Post-attention residual＋RMSNorm|7.1191 (0.686%)|6.8848 (0.184%)|6.8945 (0.119%)|
|Gate/Up＋SwiGLU|229.0918 (22.083%)|1752.8548 (46.861%)|2581.0124 (44.710%)|
|Down|117.3145 (11.308%)|872.9915 (23.339%)|1282.2396 (22.212%)|
|Final residual|1.5527 (0.150%)|1.5365 (0.041%)|1.5788 (0.027%)|
|KV carrier conversion|10.6641 (1.028%)|10.9115 (0.292%)|10.7552 (0.186%)|
|KV append DMA|3.0371 (0.293%)|2.9102 (0.078%)|2.8971 (0.050%)|
|Block orchestration|1.0547 (0.102%)|1.0970 (0.029%)|1.1165 (0.019%)|
|Layer bookkeeping|0.5827 (0.056%)|0.5729 (0.015%)|0.6055 (0.010%)|
|Stage-boundary bookkeeping|0.3971 (0.038%)|0.4069 (0.011%)|0.3874 (0.007%)|
|DSP unattributed|0.0000 (0.000%)|0.0000 (0.000%)|0.0000 (0.000%)|
|Runtime setup/teardown|70.6543 (6.811%)|70.8496 (1.894%)|71.3672 (1.236%)|
|Host-DSP boundary|317.4870 (30.604%)|353.2356 (9.443%)|371.7645 (6.440%)|
|Complete Host wall|1037.4186 (100.000%)|3740.5306 (100.000%)|5772.7216 (100.000%)|

## Complete additive module table: repeat10_prefill

Units us; parentheses percent of complete Host wall. Same median-Host-ranked two rounds for every field within each cell.

| Module | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|I/O、metadata|17.8828 (1.087%)|17.9479 (0.425%)|17.5260 (0.281%)|
|Input RMSNorm|19.6953 (1.197%)|19.4766 (0.461%)|19.5391 (0.314%)|
|QKV＋Q/K Norm-RoPE|249.9453 (15.188%)|357.4714 (8.456%)|860.0234 (13.803%)|
|QK–Softmax–AV|124.7500 (7.580%)|121.7865 (2.881%)|122.1719 (1.961%)|
|O projection|43.6068 (2.650%)|289.5599 (6.849%)|435.5313 (6.990%)|
|Post-attention residual＋RMSNorm|23.4740 (1.426%)|23.6042 (0.558%)|23.5521 (0.378%)|
|Gate/Up＋SwiGLU|608.0130 (36.945%)|2107.1771 (49.844%)|2935.6615 (47.116%)|
|Down|118.4531 (7.198%)|872.8464 (20.646%)|1282.2083 (20.579%)|
|Final residual|6.5495 (0.398%)|6.6250 (0.157%)|6.6302 (0.106%)|
|KV carrier conversion|4.4896 (0.273%)|4.4557 (0.105%)|4.4844 (0.072%)|
|KV append DMA|7.1276 (0.433%)|7.3151 (0.173%)|7.2839 (0.117%)|
|Block orchestration|1.6432 (0.100%)|1.5755 (0.037%)|1.7526 (0.028%)|
|Layer bookkeeping|0.7969 (0.048%)|0.7917 (0.019%)|0.8672 (0.014%)|
|Stage-boundary bookkeeping|1.6536 (0.100%)|1.6016 (0.038%)|1.5990 (0.026%)|
|DSP unattributed|0.0000 (0.000%)|0.0000 (0.000%)|0.0000 (0.000%)|
|Runtime setup/teardown|74.8229 (4.546%)|74.8307 (1.770%)|74.6979 (1.199%)|
|Host-DSP boundary|342.8255 (20.831%)|320.5182 (7.582%)|437.2212 (7.017%)|
|Complete Host wall|1645.7291 (100.000%)|4227.5833 (100.000%)|6230.7498 (100.000%)|

## Complete additive module table: repeat10_decode

Units us; parentheses percent of complete Host wall. Same median-Host-ranked two rounds for every field within each cell.

| Module | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|I/O、metadata|17.9538 (1.766%)|18.1735 (0.493%)|18.0778 (0.312%)|
|Input RMSNorm|5.3174 (0.523%)|5.3210 (0.144%)|5.3138 (0.092%)|
|QKV＋Q/K Norm-RoPE|86.8304 (8.543%)|231.3835 (6.271%)|858.8522 (14.843%)|
|QK–Softmax–AV|123.7516 (12.175%)|123.9206 (3.359%)|123.9704 (2.142%)|
|O projection|43.3844 (4.268%)|289.0843 (7.835%)|434.9622 (7.517%)|
|Post-attention residual＋RMSNorm|7.0234 (0.691%)|6.8509 (0.186%)|6.8418 (0.118%)|
|Gate/Up＋SwiGLU|228.4899 (22.480%)|1750.8672 (47.454%)|2581.1598 (44.608%)|
|Down|116.1611 (11.428%)|871.3285 (23.616%)|1282.2773 (22.161%)|
|Final residual|1.4899 (0.147%)|1.4915 (0.040%)|1.4844 (0.026%)|
|KV carrier conversion|10.3561 (1.019%)|10.4242 (0.283%)|10.4189 (0.180%)|
|KV append DMA|2.8942 (0.285%)|2.8770 (0.078%)|2.9020 (0.050%)|
|Block orchestration|1.0218 (0.101%)|1.0667 (0.029%)|1.0625 (0.018%)|
|Layer bookkeeping|0.5690 (0.056%)|0.5690 (0.015%)|0.5713 (0.010%)|
|Stage-boundary bookkeeping|0.3994 (0.039%)|0.4089 (0.011%)|0.4082 (0.007%)|
|DSP unattributed|0.0000 (0.000%)|0.0000 (0.000%)|0.0000 (0.000%)|
|Runtime setup/teardown|70.7526 (6.961%)|70.8747 (1.921%)|70.6859 (1.222%)|
|Host-DSP boundary|300.0384 (29.519%)|304.9867 (8.266%)|387.2969 (6.693%)|
|Complete Host wall|1016.4336 (100.000%)|3689.6279 (100.000%)|5786.2855 (100.000%)|

## Frozen three-recipe scope overview

| Recipe | Measurement in this experiment |
|---|---|
|F16F16|N/A: frozen, no equivalent single-layer measurement|
|W4F16|N/A: frozen, no equivalent single-layer measurement|
|W4U8|The three complete paired single-layer tables above|

Historical full-model evidence is not mixed into this single-layer comparison.

## Physical gates

```json
{
  "control": {
    "vtcm_requested_bytes": [
      8388608
    ],
    "vtcm_acquired_bytes": [
      8388608
    ],
    "vtcm_peak_plan_bytes": [
      6682752
    ],
    "weight_ddr_read_bytes": [
      25329664
    ],
    "hmx_u8s8_tile_pair_count": [
      49408,
      49536
    ],
    "intermediate_ddr_read_bytes": [
      0
    ],
    "intermediate_ddr_write_bytes": [
      0
    ],
    "intermediate_spill_fill_count": [
      0
    ],
    "block_invocation_count": [
      1
    ],
    "ledger_unattributed_ticks": [
      0
    ]
  },
  "lpbq32": {
    "vtcm_requested_bytes": [
      8388608
    ],
    "vtcm_acquired_bytes": [
      8388608
    ],
    "vtcm_peak_plan_bytes": [
      6682752
    ],
    "weight_ddr_read_bytes": [
      26116096
    ],
    "hmx_u8s8_tile_pair_count": [
      49408,
      49536
    ],
    "intermediate_ddr_read_bytes": [
      0
    ],
    "intermediate_ddr_write_bytes": [
      0
    ],
    "intermediate_spill_fill_count": [
      0
    ],
    "block_invocation_count": [
      1
    ],
    "ledger_unattributed_ticks": [
      0
    ]
  },
  "direct": {
    "vtcm_requested_bytes": [
      8388608
    ],
    "vtcm_acquired_bytes": [
      8388608
    ],
    "vtcm_peak_plan_bytes": [
      6682752
    ],
    "weight_ddr_read_bytes": [
      26116096
    ],
    "hmx_u8s8_tile_pair_count": [
      1523968,
      1524096
    ],
    "intermediate_ddr_read_bytes": [
      0
    ],
    "intermediate_ddr_write_bytes": [
      0
    ],
    "intermediate_spill_fill_count": [
      0
    ],
    "block_invocation_count": [
      1
    ],
    "ledger_unattributed_ticks": [
      0
    ]
  }
}
```

## Exact binary and command provenance

```json
{
  "source_head": "a4f7df85c4449f4ad18bd3b1129d55274437a0d6",
  "binaries": {
    "qwen3_block_cli": "9113e3adfc11a6bc5dd0a6d48b175bccf24082bf00e06cf6f42355a578767d97",
    "libqwen3_probe.so": "f173f1a7e9eee98b9c40f343e9f908ef093a3eaa3425b869a79690c6eaa8a665",
    "libqwen3_probe_skel.so": "7cf936d30093d1a228c32e37dfd3bc6db090bbbb49cd65922c0b4226012ff86e"
  },
  "remote_sha256": "9113e3adfc11a6bc5dd0a6d48b175bccf24082bf00e06cf6f42355a578767d97  qwen3_block_cli\nf173f1a7e9eee98b9c40f343e9f908ef093a3eaa3425b869a79690c6eaa8a665  libqwen3_probe.so\n7cf936d30093d1a228c32e37dfd3bc6db090bbbb49cd65922c0b4226012ff86e  libqwen3_probe_skel.so\n",
  "boot": "77ccafe0-8817-4976-bdae-15af5608d4ff\n",
  "threshold_percent": 10,
  "execution": "one layer14 M64 prefill then8 real teacher-input M1 steps; runtime computes persistent KV; oneRPC/step",
  "repeat10": "ten complete replays in a loaded prepared runtime, state and cache reinitialized before each prefill; mean per replay, not a frozen snapshot"
}
```

## All numeric counters: repeat1_prefill

Overlapping work counters are diagnostic and cannot be summed into the additive ledger. Values below independently median each counter across round means.

| Counter | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|activation_ticks|7164.500000|7414.500000|7389.500000|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|
|attention_setup_ticks|0.000000|0.000000|0.000000|
|attention_softmax_ticks|0.000000|0.000000|0.000000|
|attention_ticks|2430.500000|2391.500000|2361.500000|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|
|block_invocation_count|1.000000|1.000000|1.000000|
|block_orchestration_ticks|115.500000|111.500000|111.000000|
|boundary_ddr_read_bytes|304096.000000|304096.000000|304096.000000|
|boundary_ddr_write_bytes|131072.000000|131072.000000|131072.000000|
|boundary_dma_descriptor_count|10.000000|10.000000|10.000000|
|cache_compared_elements|0.000000|0.000000|0.000000|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|
|cache_fp16_max_violation_fraction|0.010000|0.010000|0.010000|
|cache_fp16_min_cosine|0.999990|0.999990|0.999990|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|0.000000|
|cache_max_nrmse|0.000000|0.000000|0.000000|
|cache_min_cosine|0.000000|0.000000|0.000000|
|cache_mismatches|0.000000|0.000000|0.000000|
|cache_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|cache_nonfinite_count|0.000000|0.000000|0.000000|
|cache_prefix_mismatches|0.000000|0.000000|0.000000|
|cache_structure_mismatches|0.000000|0.000000|0.000000|
|cache_tensor_count|0.000000|0.000000|0.000000|
|down_ticks|2572.500000|16770.500000|24632.000000|
|dsp_status|3.000000|3.000000|3.000000|
|experiment|240.000000|240.000000|240.000000|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|final_residual_ticks|143.000000|141.500000|142.500000|
|first_position|0.000000|0.000000|0.000000|
|gate_up_ticks|4870.000000|33173.500000|49027.500000|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_embedding_ticks|0.000000|0.000000|0.000000|
|generation_final_norm_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_argmax_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_command_count|0.000000|0.000000|0.000000|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_prefetch_count|0.000000|0.000000|0.000000|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|0.000000|
|hmx_command_count|46.000000|136.000000|192.000000|
|hmx_compute_ticks|4266.000000|9873.000000|92931.500000|
|hmx_fp16_tile_pair_count|0.000000|0.000000|0.000000|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|1523968.000000|
|host_wall_ns|1864609.000000|4435234.500000|6320937.500000|
|input_norm_ticks|428.500000|430.500000|424.500000|
|input_stage_ticks|146.000000|146.000000|145.000000|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|
|invocation_ticks|27069.500000|76441.500000|112484.500000|
|kv_cache_k_format|14.000000|14.000000|14.000000|
|kv_cache_v_format|12.000000|12.000000|12.000000|
|layer_bookkeeping_ticks|53.500000|52.500000|53.500000|
|ledger_named_ticks|27069.500000|76441.500000|112484.500000|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|
|logical_m|64.000000|64.000000|64.000000|
|metadata_stage_ticks|180.000000|181.500000|181.000000|
|numerical_audit_enabled|0.000000|0.000000|0.000000|
|numerical_status|1.000000|1.000000|1.000000|
|o_projection_ticks|910.000000|5657.000000|8450.500000|
|output_cosine|0.000000|0.000000|0.000000|
|output_fp16_atol|0.062500|0.062500|0.062500|
|output_fp16_max_composed_nrmse|0.003000|0.003000|0.003000|
|output_fp16_rtol|0.002000|0.002000|0.002000|
|output_max_abs|0.000000|0.000000|0.000000|
|output_max_lsb|0.000000|0.000000|0.000000|
|output_max_required_rtol_after_atol|0.000000|0.000000|0.000000|
|output_mismatches|0.000000|0.000000|0.000000|
|output_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|output_nonfinite_count|0.000000|0.000000|0.000000|
|output_nrmse|0.000000|0.000000|0.000000|
|output_stage_ticks|52.000000|48.000000|52.500000|
|post_attention_norm_ticks|3.000000|3.000000|3.000000|
|post_attention_residual_ticks|486.500000|486.500000|484.500000|
|prepared_session_run_index|1.000000|1.000000|1.000000|
|projection_failure_index|0.000000|0.000000|0.000000|
|projection_failure_n_tile|0.000000|0.000000|0.000000|
|projection_failure_result|0.000000|0.000000|0.000000|
|projection_failure_step|0.000000|0.000000|0.000000|
|projection_hmx_wait_ticks|506.000000|473.000000|15517.000000|
|projection_pack_ticks|18.000000|18.000000|18.000000|
|projection_unpack_ticks|0.000000|0.000000|0.000000|
|qk_norm_rope_ticks|3.000000|3.000000|3.000000|
|qkv_projection_ticks|5054.500000|7061.500000|16658.000000|
|repeat_count|1.000000|1.000000|1.000000|
|replay_step|0.000000|0.000000|0.000000|
|runtime_setup_ticks|1094.500000|1086.500000|1109.000000|
|runtime_teardown_ticks|886.000000|890.500000|861.000000|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0.000000|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0.000000|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|
|scan_cache_append_ticks|161.500000|154.500000|153.500000|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0.000000|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|143360.000000|
|scan_cache_dma_descriptor_count|16.000000|16.000000|16.000000|
|scan_cache_pack_ticks|105.000000|105.500000|104.500000|
|scan_cache_stage_ticks|0.000000|0.000000|0.000000|
|scan_dynamic_attention_ticks|0.000000|0.000000|0.000000|
|scan_logical_m_observed|64.000000|64.000000|64.000000|
|scan_padded_kv_length|64.000000|64.000000|64.000000|
|scan_total_kv_length|64.000000|64.000000|64.000000|
|stage_boundary_ticks|112.000000|117.000000|114.000000|
|total_ticks|25935.000000|75372.500000|111375.500000|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|
|u8_attention_av_hmx_ticks|409.000000|420.500000|398.000000|
|u8_attention_av_requant_ticks|1036.500000|1043.000000|1043.500000|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|
|u8_attention_k_pack_ticks|0.000000|0.000000|0.000000|
|u8_attention_pipeline_wait_ticks|1331.500000|1290.000000|1257.500000|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|
|u8_attention_qk_hmx_ticks|434.000000|431.000000|414.500000|
|u8_attention_qk_norm_rope_ticks|17247.500000|15953.000000|22021.000000|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|
|u8_attention_softmax_ticks|5840.000000|5835.500000|5833.500000|
|u8_attention_v_pack_ticks|3580.000000|3525.500000|3472.500000|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_init_bytes|29568.000000|29568.000000|29568.000000|
|u8_cache_k_vtcm_tail_init_count|1.000000|1.000000|1.000000|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|u8_cache_native_append_update_ticks|257.000000|253.000000|254.500000|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reuse_count|1.000000|1.000000|1.000000|
|u8_cache_native_prefill_reused_carrier_bytes|143360.000000|143360.000000|143360.000000|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_bytes|32768.000000|32768.000000|32768.000000|
|u8_cache_v_vtcm_tail_init_count|1.000000|1.000000|1.000000|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|valid_length|64.000000|64.000000|64.000000|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|6682752.000000|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_work_ticks|0.000000|9617.500000|11008.000000|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_rows_observed|0.000000|0.000000|0.000000|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|
|w4u8_common_op_rows_observed|64.000000|64.000000|64.000000|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_av_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_av_requant_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_op_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|8.000000|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_hmx_command_count|30.000000|0.000000|0.000000|
|w4u8_decode_direct_n_mask|63.000000|63.000000|63.000000|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_projection_count|7.000000|0.000000|0.000000|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|0.000000|0.000000|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_projection_mode|1.000000|1.000000|1.000000|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|4.000000|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_mode|1.000000|1.000000|1.000000|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_rows|4.000000|4.000000|4.000000|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|0.000000|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_final_residual_main_work_ticks|80.500000|75.500000|82.500000|
|w4u8_final_residual_pool_wait_ticks|10.000000|13.000000|10.500000|
|w4u8_final_residual_task_count|16.000000|16.000000|16.000000|
|w4u8_final_residual_worker_work_ticks|368.000000|382.000000|365.000000|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_input_norm_main_work_ticks|360.000000|358.000000|359.000000|
|w4u8_input_norm_pool_wait_ticks|24.000000|24.000000|21.000000|
|w4u8_input_norm_task_count|16.000000|16.000000|16.000000|
|w4u8_input_norm_worker_work_ticks|1626.500000|1618.500000|1629.000000|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_down_hmx_command_count|8.000000|32.000000|32.000000|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_weight_expand_ticks|0.000000|41122.500000|48467.000000|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|
|w4u8_o_batch_count|4.000000|8.000000|16.000000|
|w4u8_o_batch_n_tiles_observed|16.000000|8.000000|4.000000|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_post_residual_main_work_ticks|418.500000|423.000000|417.500000|
|w4u8_post_residual_pool_wait_ticks|16.000000|17.000000|21.000000|
|w4u8_post_residual_task_count|16.000000|16.000000|16.000000|
|w4u8_post_residual_worker_work_ticks|1902.000000|1910.000000|1893.000000|
|w4u8_prefill_cache_mode|1.000000|1.000000|1.000000|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|0.000000|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|
|w4u8_qkv_ring_batch_count|6.000000|32.000000|32.000000|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_dma_wait_ticks|1504.500000|2048.000000|3053.000000|
|w4u8_qkv_ring_expand_task_count|0.000000|128.000000|128.000000|
|w4u8_qkv_ring_expand_ticks|0.000000|9617.500000|11008.000000|
|w4u8_qkv_ring_expand_worker_count|0.000000|3.000000|3.000000|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|24.000000|
|w4u8_qkv_ring_hmx_compute_ticks|399.500000|297.000000|14949.000000|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1026.500000|3585.500000|275.000000|
|w4u8_qkv_ring_pipeline_ticks|5004.500000|7011.000000|16607.500000|
|w4u8_qkv_ring_pool_wait_ticks|3219.500000|2381.000000|3.000000|
|w4u8_qkv_ring_prep_worker_count|5.000000|2.000000|2.000000|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.000000|2094.000000|12347.000000|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|
|w4u8_qkvo_hmx_lifetime_ticks|8680.500000|57328.500000|97052.500000|
|w4u8_qkvo_prefetch_wait_ticks|1504.500000|2052.500000|3056.500000|
|w4u8_qkvo_weight_expand_ticks|0.000000|14190.500000|16183.000000|
|w4u8_residual_active_contexts|6.000000|6.000000|6.000000|
|w4u8_swiglu_rows_observed|64.000000|64.000000|64.000000|
|weight_ddr_read_bytes|25329664.000000|26116096.000000|26116096.000000|
|weight_dma_descriptor_count|60.000000|240.000000|352.000000|
|weight_dma_ticks|9581.500000|11543.500000|16141.000000|

## All numeric counters: repeat1_decode

Overlapping work counters are diagnostic and cannot be summed into the additive ledger. Values below independently median each counter across round means.

| Counter | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|activation_ticks|0.000000|577.062500|577.375000|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|
|attention_setup_ticks|0.000000|0.000000|0.000000|
|attention_softmax_ticks|0.000000|0.000000|0.000000|
|attention_ticks|2384.812500|2381.500000|2395.875000|
|attention_unattributed_ticks|1624.062500|1620.437500|1632.437500|
|block_invocation_count|1.000000|1.000000|1.000000|
|block_orchestration_ticks|20.375000|21.187500|21.187500|
|boundary_ddr_read_bytes|304096.000000|304096.000000|304096.000000|
|boundary_ddr_write_bytes|131072.000000|131072.000000|131072.000000|
|boundary_dma_descriptor_count|10.000000|10.000000|10.000000|
|cache_compared_elements|0.000000|0.000000|0.000000|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|
|cache_fp16_max_violation_fraction|0.010000|0.010000|0.010000|
|cache_fp16_min_cosine|0.999990|0.999990|0.999990|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|0.000000|
|cache_max_nrmse|0.000000|0.000000|0.000000|
|cache_min_cosine|0.000000|0.000000|0.000000|
|cache_mismatches|0.000000|0.000000|0.000000|
|cache_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|cache_nonfinite_count|0.000000|0.000000|0.000000|
|cache_prefix_mismatches|0.000000|0.000000|0.000000|
|cache_structure_mismatches|0.000000|0.000000|0.000000|
|cache_tensor_count|0.000000|0.000000|0.000000|
|down_ticks|2251.500000|16754.750000|24617.750000|
|dsp_status|3.000000|3.000000|3.000000|
|experiment|240.000000|240.000000|240.000000|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|final_residual_ticks|29.687500|29.625000|29.500000|
|first_position|67.500000|67.500000|67.500000|
|gate_up_ticks|4402.062500|33068.562500|48977.812500|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_embedding_ticks|0.000000|0.000000|0.000000|
|generation_final_norm_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_argmax_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_command_count|0.000000|0.000000|0.000000|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_prefetch_count|0.000000|0.000000|0.000000|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|0.000000|
|hmx_command_count|46.000000|136.000000|192.000000|
|hmx_compute_ticks|4374.062500|9714.875000|92641.750000|
|hmx_fp16_tile_pair_count|0.000000|0.000000|0.000000|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|1524096.000000|
|host_wall_ns|1037418.625000|3740530.562500|5772721.562500|
|input_norm_ticks|102.000000|102.125000|102.125000|
|input_stage_ticks|136.250000|135.937500|136.000000|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|
|invocation_ticks|13815.062500|65044.562500|103690.250000|
|kv_cache_k_format|14.000000|14.000000|14.000000|
|kv_cache_v_format|12.000000|12.000000|12.000000|
|layer_bookkeeping_ticks|11.250000|11.312500|11.500000|
|ledger_named_ticks|13815.062500|65044.562500|103690.250000|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|
|logical_m|1.000000|1.000000|1.000000|
|metadata_stage_ticks|144.312500|144.625000|143.875000|
|numerical_audit_enabled|0.000000|0.000000|0.000000|
|numerical_status|1.000000|1.000000|1.000000|
|o_projection_ticks|836.500000|5551.562500|8350.562500|
|output_cosine|0.000000|0.000000|0.000000|
|output_fp16_atol|0.062500|0.062500|0.062500|
|output_fp16_max_composed_nrmse|0.003000|0.003000|0.003000|
|output_fp16_rtol|0.002000|0.002000|0.002000|
|output_max_abs|0.000000|0.000000|0.000000|
|output_max_lsb|0.000000|0.000000|0.000000|
|output_max_required_rtol_after_atol|0.000000|0.000000|0.000000|
|output_mismatches|0.000000|0.000000|0.000000|
|output_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|output_nonfinite_count|0.000000|0.000000|0.000000|
|output_nrmse|0.000000|0.000000|0.000000|
|output_stage_ticks|62.750000|69.500000|69.812500|
|post_attention_norm_ticks|0.375000|0.250000|0.375000|
|post_attention_residual_ticks|136.250000|132.125000|131.625000|
|prepared_session_run_index|5.500000|5.500000|5.500000|
|projection_failure_index|0.000000|0.000000|0.000000|
|projection_failure_n_tile|0.000000|0.000000|0.000000|
|projection_failure_result|0.000000|0.000000|0.000000|
|projection_failure_step|0.000000|0.000000|0.000000|
|projection_hmx_wait_ticks|361.875000|450.562500|16040.187500|
|projection_pack_ticks|3.625000|3.375000|3.125000|
|projection_unpack_ticks|0.000000|0.000000|0.000000|
|qk_norm_rope_ticks|0.187500|0.250000|0.000000|
|qkv_projection_ticks|1677.000000|4416.875000|16489.000000|
|repeat_count|1.000000|1.000000|1.000000|
|replay_step|4.500000|4.500000|4.500000|
|runtime_setup_ticks|720.937500|720.562500|720.187500|
|runtime_teardown_ticks|639.687500|644.875000|643.812500|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|2752512.000000|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|77824.000000|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|
|scan_cache_append_ticks|56.812500|55.937500|56.062500|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|143936.000000|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|1152.000000|
|scan_cache_dma_descriptor_count|42.000000|42.000000|42.000000|
|scan_cache_pack_ticks|204.562500|205.437500|206.125000|
|scan_cache_stage_ticks|359.500000|360.250000|368.250000|
|scan_dynamic_attention_ticks|2380.062500|2376.937500|2391.375000|
|scan_logical_m_observed|1.000000|1.000000|1.000000|
|scan_padded_kv_length|96.000000|96.000000|96.000000|
|scan_total_kv_length|68.500000|68.500000|68.500000|
|stage_boundary_ticks|7.812500|7.750000|7.687500|
|total_ticks|13094.187500|64321.187500|102970.437500|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|
|u8_attention_av_hmx_ticks|149.312500|149.500000|152.500000|
|u8_attention_av_requant_ticks|100.125000|100.437500|101.937500|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|
|u8_attention_k_pack_ticks|13.687500|13.750000|13.625000|
|u8_attention_pipeline_wait_ticks|44.937500|45.687500|46.500000|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|
|u8_attention_qk_hmx_ticks|90.812500|94.812500|92.812500|
|u8_attention_qk_norm_rope_ticks|1479.687500|976.500000|1962.937500|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|
|u8_attention_softmax_ticks|271.562500|270.500000|272.375000|
|u8_attention_v_pack_ticks|87.125000|86.312500|86.250000|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|126.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|896.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|1.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|27.062500|27.187500|27.437500|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|28672.000000|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|u8_cache_native_append_update_ticks|259.000000|257.750000|258.000000|
|u8_cache_native_incremental_append_count|1.000000|1.000000|1.000000|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_tail_append_count|1.000000|1.000000|1.000000|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.000000|6144.000000|6144.000000|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.000000|12.000000|12.000000|
|u8_cache_v_vtcm_tail_publish_count|2.000000|2.000000|2.000000|
|u8_cache_v_vtcm_tail_row_update_count|8.000000|8.000000|8.000000|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|valid_length|68.500000|68.500000|68.500000|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|6682752.000000|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_work_ticks|0.000000|9536.687500|10792.062500|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_call_count|8.000000|8.000000|8.000000|
|w4u8_av_requant_rows_observed|4.000000|4.000000|4.000000|
|w4u8_av_requant_vector_count|64.000000|64.000000|64.000000|
|w4u8_common_op_rows_observed|4.000000|4.000000|4.000000|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_av_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_av_requant_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_op_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|8.000000|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_hmx_command_count|30.000000|0.000000|0.000000|
|w4u8_decode_direct_n_mask|63.000000|63.000000|63.000000|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_projection_count|7.000000|0.000000|0.000000|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|0.000000|0.000000|
|w4u8_decode_k_pair_row4_call_count|4.000000|4.000000|4.000000|
|w4u8_decode_k_temp_carrier_skipped_count|8.000000|8.000000|8.000000|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_projection_mode|1.000000|1.000000|1.000000|
|w4u8_decode_q_pair_row4_call_count|8.000000|8.000000|8.000000|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|4.000000|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_qk_rows_processed|96.000000|96.000000|96.000000|
|w4u8_decode_softmax_hvx_tile4_call_count|8.000000|8.000000|8.000000|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_mode|1.000000|1.000000|1.000000|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_row4_call_count|192.000000|192.000000|192.000000|
|w4u8_decode_swiglu_rows|4.000000|4.000000|4.000000|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_vector_count|192.000000|192.000000|192.000000|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|0.000000|
|w4u8_final_residual_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_consume_count|6.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_join_wait_ticks|130.125000|0.000000|0.000000|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_publish_count|6.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_ready_wait_ticks|3315.062500|0.000000|0.000000|
|w4u8_gate_up_swiglu_worker_ticks|722.062500|0.000000|0.000000|
|w4u8_input_norm_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_down_hmx_command_count|8.000000|32.000000|32.000000|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_weight_expand_ticks|0.000000|41116.000000|48320.187500|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|
|w4u8_o_batch_count|4.000000|8.000000|16.000000|
|w4u8_o_batch_n_tiles_observed|16.000000|8.000000|4.000000|
|w4u8_o_gate_prefetch_consume_count|1.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_lifetime_ticks|358.812500|0.000000|0.000000|
|w4u8_o_gate_prefetch_start_count|1.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_wait_ticks|214.687500|0.000000|0.000000|
|w4u8_post_residual_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_prefill_cache_mode|1.000000|1.000000|1.000000|
|w4u8_qk_norm_rope_rows_observed|4.000000|4.000000|4.000000|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|
|w4u8_qkv_ring_batch_count|6.000000|32.000000|32.000000|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_dma_wait_ticks|1437.500000|1987.812500|3093.125000|
|w4u8_qkv_ring_expand_task_count|0.000000|128.000000|128.000000|
|w4u8_qkv_ring_expand_ticks|0.000000|9536.687500|10792.062500|
|w4u8_qkv_ring_expand_worker_count|0.000000|3.000000|3.000000|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|24.000000|
|w4u8_qkv_ring_hmx_compute_ticks|377.375000|259.000000|14815.625000|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_hmx_ready_wait_ticks|939.375000|3522.562500|226.437500|
|w4u8_qkv_ring_pipeline_ticks|1656.875000|4396.062500|16468.625000|
|w4u8_qkv_ring_pool_wait_ticks|2.687500|2.500000|2.750000|
|w4u8_qkv_ring_prep_worker_count|5.000000|2.000000|2.000000|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.437500|2000.375000|12246.562500|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|
|w4u8_qkvo_hmx_lifetime_ticks|8242.687500|57117.125000|96968.687500|
|w4u8_qkvo_prefetch_wait_ticks|1438.062500|1990.937500|3096.437500|
|w4u8_qkvo_weight_expand_ticks|0.000000|14112.687500|15929.250000|
|w4u8_residual_active_contexts|0.000000|0.000000|0.000000|
|w4u8_swiglu_rows_observed|4.000000|4.000000|4.000000|
|weight_ddr_read_bytes|25329664.000000|26116096.000000|26116096.000000|
|weight_dma_descriptor_count|60.000000|240.000000|352.000000|
|weight_dma_ticks|8756.625000|11304.687500|15832.437500|

## All numeric counters: repeat10_prefill

Overlapping work counters are diagnostic and cannot be summed into the additive ledger. Values below independently median each counter across round means.

| Counter | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|activation_ticks|7144.350000|7388.150000|7389.100000|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|
|attention_setup_ticks|0.000000|0.000000|0.000000|
|attention_softmax_ticks|0.000000|0.000000|0.000000|
|attention_ticks|2400.500000|2341.850000|2342.950000|
|attention_unattributed_ticks|0.000000|0.000000|0.000000|
|block_invocation_count|1.000000|1.000000|1.000000|
|block_orchestration_ticks|30.550000|30.150000|30.200000|
|boundary_ddr_read_bytes|304096.000000|304096.000000|304096.000000|
|boundary_ddr_write_bytes|131072.000000|131072.000000|131072.000000|
|boundary_dma_descriptor_count|10.000000|10.000000|10.000000|
|cache_compared_elements|0.000000|0.000000|0.000000|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|
|cache_fp16_max_violation_fraction|0.010000|0.010000|0.010000|
|cache_fp16_min_cosine|0.999990|0.999990|0.999990|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|0.000000|
|cache_max_nrmse|0.000000|0.000000|0.000000|
|cache_min_cosine|0.000000|0.000000|0.000000|
|cache_mismatches|0.000000|0.000000|0.000000|
|cache_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|cache_nonfinite_count|0.000000|0.000000|0.000000|
|cache_prefix_mismatches|0.000000|0.000000|0.000000|
|cache_structure_mismatches|0.000000|0.000000|0.000000|
|cache_tensor_count|0.000000|0.000000|0.000000|
|down_ticks|2300.850000|16748.850000|24620.150000|
|dsp_status|3.000000|3.000000|3.000000|
|experiment|240.000000|240.000000|240.000000|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|final_residual_ticks|126.200000|126.050000|125.700000|
|first_position|0.000000|0.000000|0.000000|
|gate_up_ticks|4542.950000|33079.550000|48980.800000|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_embedding_ticks|0.000000|0.000000|0.000000|
|generation_final_norm_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_argmax_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_command_count|0.000000|0.000000|0.000000|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_prefetch_count|0.000000|0.000000|0.000000|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|0.000000|
|hmx_command_count|46.000000|136.000000|192.000000|
|hmx_compute_ticks|4459.000000|9895.950000|92766.850000|
|hmx_fp16_tile_pair_count|0.000000|0.000000|0.000000|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|1523968.000000|
|host_wall_ns|1645729.100000|4227583.300000|6230749.850000|
|input_norm_ticks|381.950000|376.100000|381.150000|
|input_stage_ticks|135.150000|136.800000|136.600000|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|
|invocation_ticks|25087.500000|74988.000000|111229.500000|
|kv_cache_k_format|14.000000|14.000000|14.000000|
|kv_cache_v_format|12.000000|12.000000|12.000000|
|layer_bookkeeping_ticks|15.300000|15.200000|15.550000|
|ledger_named_ticks|25087.500000|74988.000000|111229.500000|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|
|logical_m|64.000000|64.000000|64.000000|
|metadata_stage_ticks|149.350000|147.350000|146.650000|
|numerical_audit_enabled|0.000000|0.000000|0.000000|
|numerical_status|1.000000|1.000000|1.000000|
|o_projection_ticks|838.750000|5559.950000|8359.500000|
|output_cosine|0.000000|0.000000|0.000000|
|output_fp16_atol|0.062500|0.062500|0.062500|
|output_fp16_max_composed_nrmse|0.003000|0.003000|0.003000|
|output_fp16_rtol|0.002000|0.002000|0.002000|
|output_max_abs|0.000000|0.000000|0.000000|
|output_max_lsb|0.000000|0.000000|0.000000|
|output_max_required_rtol_after_atol|0.000000|0.000000|0.000000|
|output_mismatches|0.000000|0.000000|0.000000|
|output_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|output_nonfinite_count|0.000000|0.000000|0.000000|
|output_nrmse|0.000000|0.000000|0.000000|
|output_stage_ticks|56.600000|58.850000|59.200000|
|post_attention_norm_ticks|0.600000|0.600000|0.600000|
|post_attention_residual_ticks|450.150000|449.400000|451.150000|
|prepared_session_run_index|41.500000|41.500000|41.500000|
|projection_failure_index|0.000000|0.000000|0.000000|
|projection_failure_n_tile|0.000000|0.000000|0.000000|
|projection_failure_result|0.000000|0.000000|0.000000|
|projection_failure_step|0.000000|0.000000|0.000000|
|projection_hmx_wait_ticks|505.400000|464.600000|15847.450000|
|projection_pack_ticks|4.250000|4.300000|4.400000|
|projection_unpack_ticks|0.000000|0.000000|0.000000|
|qk_norm_rope_ticks|0.550000|0.500000|0.550000|
|qkv_projection_ticks|4802.000000|6857.450000|16503.850000|
|repeat_count|1.000000|1.000000|1.000000|
|replay_step|0.000000|0.000000|0.000000|
|runtime_setup_ticks|758.250000|755.850000|761.600000|
|runtime_teardown_ticks|670.200000|667.550000|667.350000|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|0.000000|
|scan_attention_overlay_required_bytes|0.000000|0.000000|0.000000|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|
|scan_cache_append_ticks|136.500000|139.150000|139.050000|
|scan_cache_ddr_read_bytes|0.000000|0.000000|0.000000|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|143360.000000|
|scan_cache_dma_descriptor_count|16.000000|16.000000|16.000000|
|scan_cache_pack_ticks|86.150000|85.650000|85.150000|
|scan_cache_stage_ticks|0.000000|0.000000|0.000000|
|scan_dynamic_attention_ticks|0.000000|0.000000|0.000000|
|scan_logical_m_observed|64.000000|64.000000|64.000000|
|scan_padded_kv_length|64.000000|64.000000|64.000000|
|scan_total_kv_length|64.000000|64.000000|64.000000|
|stage_boundary_ticks|32.200000|31.100000|31.650000|
|total_ticks|24330.350000|74229.350000|110470.050000|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|
|u8_attention_av_hmx_ticks|413.000000|423.150000|422.600000|
|u8_attention_av_requant_ticks|1042.450000|1045.000000|1042.000000|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|
|u8_attention_k_pack_ticks|0.000000|0.000000|0.000000|
|u8_attention_pipeline_wait_ticks|1468.950000|1206.300000|1245.300000|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|
|u8_attention_qk_hmx_ticks|426.250000|433.550000|433.000000|
|u8_attention_qk_norm_rope_ticks|16792.150000|15767.900000|21910.900000|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|
|u8_attention_softmax_ticks|5814.050000|5784.250000|5793.500000|
|u8_attention_v_pack_ticks|3459.800000|3437.000000|3445.700000|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_init_bytes|29568.000000|29568.000000|29568.000000|
|u8_cache_k_vtcm_tail_init_count|1.000000|1.000000|1.000000|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|u8_cache_native_append_update_ticks|221.000000|223.550000|222.700000|
|u8_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reuse_count|1.000000|1.000000|1.000000|
|u8_cache_native_prefill_reused_carrier_bytes|143360.000000|143360.000000|143360.000000|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_tail_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_bytes|32768.000000|32768.000000|32768.000000|
|u8_cache_v_vtcm_tail_init_count|1.000000|1.000000|1.000000|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|valid_length|64.000000|64.000000|64.000000|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|6682752.000000|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_work_ticks|0.000000|9606.400000|10950.350000|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_call_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_rows_observed|0.000000|0.000000|0.000000|
|w4u8_av_requant_vector_count|0.000000|0.000000|0.000000|
|w4u8_common_op_rows_observed|64.000000|64.000000|64.000000|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_av_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_av_requant_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_op_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|8.000000|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_hmx_command_count|30.000000|0.000000|0.000000|
|w4u8_decode_direct_n_mask|63.000000|63.000000|63.000000|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_projection_count|7.000000|0.000000|0.000000|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|0.000000|0.000000|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|0.000000|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_projection_mode|1.000000|1.000000|1.000000|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|4.000000|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_mode|1.000000|1.000000|1.000000|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_rows|4.000000|4.000000|4.000000|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|0.000000|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|0.000000|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_final_residual_main_work_ticks|77.250000|77.700000|77.500000|
|w4u8_final_residual_pool_wait_ticks|14.400000|14.000000|14.300000|
|w4u8_final_residual_task_count|16.000000|16.000000|16.000000|
|w4u8_final_residual_worker_work_ticks|370.600000|369.300000|372.200000|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_input_norm_main_work_ticks|323.400000|324.600000|323.800000|
|w4u8_input_norm_pool_wait_ticks|21.700000|12.350000|18.650000|
|w4u8_input_norm_task_count|16.000000|16.000000|16.000000|
|w4u8_input_norm_worker_work_ticks|1455.800000|1439.250000|1449.250000|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_down_hmx_command_count|8.000000|32.000000|32.000000|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_weight_expand_ticks|0.000000|41120.000000|48437.350000|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|
|w4u8_o_batch_count|4.000000|8.000000|16.000000|
|w4u8_o_batch_n_tiles_observed|16.000000|8.000000|4.000000|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|0.000000|
|w4u8_post_residual_main_work_ticks|374.100000|374.250000|373.300000|
|w4u8_post_residual_pool_wait_ticks|39.300000|42.550000|44.550000|
|w4u8_post_residual_task_count|16.000000|16.000000|16.000000|
|w4u8_post_residual_worker_work_ticks|1776.850000|1777.750000|1781.550000|
|w4u8_prefill_cache_mode|1.000000|1.000000|1.000000|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|0.000000|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|
|w4u8_qkv_ring_batch_count|6.000000|32.000000|32.000000|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_dma_wait_ticks|1436.400000|1986.500000|2977.300000|
|w4u8_qkv_ring_expand_task_count|0.000000|128.000000|128.000000|
|w4u8_qkv_ring_expand_ticks|0.000000|9606.400000|10950.350000|
|w4u8_qkv_ring_expand_worker_count|0.000000|3.000000|3.000000|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|24.000000|
|w4u8_qkv_ring_hmx_compute_ticks|375.500000|281.750000|14928.050000|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_hmx_ready_wait_ticks|970.400000|3527.900000|231.950000|
|w4u8_qkv_ring_pipeline_ticks|4776.100000|6834.200000|16480.350000|
|w4u8_qkv_ring_pool_wait_ticks|3108.950000|2361.800000|2.800000|
|w4u8_qkv_ring_prep_worker_count|5.000000|2.000000|2.000000|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.900000|2049.050000|12361.900000|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|
|w4u8_qkvo_hmx_lifetime_ticks|8082.400000|57160.700000|96980.000000|
|w4u8_qkvo_prefetch_wait_ticks|1436.850000|1989.350000|2980.550000|
|w4u8_qkvo_weight_expand_ticks|0.000000|14186.850000|16130.950000|
|w4u8_residual_active_contexts|6.000000|6.000000|6.000000|
|w4u8_swiglu_rows_observed|64.000000|64.000000|64.000000|
|weight_ddr_read_bytes|25329664.000000|26116096.000000|26116096.000000|
|weight_dma_descriptor_count|60.000000|240.000000|352.000000|
|weight_dma_ticks|8835.900000|11303.850000|15736.050000|

## All numeric counters: repeat10_decode

Overlapping work counters are diagnostic and cannot be summed into the additive ledger. Values below independently median each counter across round means.

| Counter | Per-channel | LPBQ S8 | LPBQ direct-W4 |
|---|---:|---:|---:|
|activation_ticks|0.000000|578.993750|579.768750|
|attention_av_hmx_ticks|0.000000|0.000000|0.000000|
|attention_av_pack_ticks|0.000000|0.000000|0.000000|
|attention_av_unpack_ticks|0.000000|0.000000|0.000000|
|attention_gqa_pipeline_ticks|0.000000|0.000000|0.000000|
|attention_qk_hmx_ticks|0.000000|0.000000|0.000000|
|attention_qk_pack_ticks|0.000000|0.000000|0.000000|
|attention_qk_unpack_ticks|0.000000|0.000000|0.000000|
|attention_setup_ticks|0.000000|0.000000|0.000000|
|attention_softmax_ticks|0.000000|0.000000|0.000000|
|attention_ticks|2373.181250|2377.237500|2384.343750|
|attention_unattributed_ticks|1615.543750|1613.262500|1614.662500|
|block_invocation_count|1.000000|1.000000|1.000000|
|block_orchestration_ticks|19.543750|20.375000|20.462500|
|boundary_ddr_read_bytes|304096.000000|304096.000000|304096.000000|
|boundary_ddr_write_bytes|131072.000000|131072.000000|131072.000000|
|boundary_dma_descriptor_count|10.000000|10.000000|10.000000|
|cache_compared_elements|0.000000|0.000000|0.000000|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|0.000000|
|cache_fp16_max_violation_fraction|0.010000|0.010000|0.010000|
|cache_fp16_min_cosine|0.999990|0.999990|0.999990|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|0.000000|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|0.000000|
|cache_max_nrmse|0.000000|0.000000|0.000000|
|cache_min_cosine|0.000000|0.000000|0.000000|
|cache_mismatches|0.000000|0.000000|0.000000|
|cache_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|cache_nonfinite_count|0.000000|0.000000|0.000000|
|cache_prefix_mismatches|0.000000|0.000000|0.000000|
|cache_structure_mismatches|0.000000|0.000000|0.000000|
|cache_tensor_count|0.000000|0.000000|0.000000|
|down_ticks|2248.712500|16741.668750|24618.587500|
|dsp_status|3.000000|3.000000|3.000000|
|experiment|240.000000|240.000000|240.000000|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|f16_cache_native_append_update_ticks|0.000000|0.000000|0.000000|
|f16_cache_native_incremental_append_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|final_residual_ticks|28.606250|28.706250|28.431250|
|first_position|67.500000|67.500000|67.500000|
|gate_up_ticks|4403.125000|33050.912500|48978.075000|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_embedding_ticks|0.000000|0.000000|0.000000|
|generation_final_norm_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_argmax_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_command_count|0.000000|0.000000|0.000000|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_expand_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_hmx_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_n_tiles|0.000000|0.000000|0.000000|
|generation_lm_head_prefetch_count|0.000000|0.000000|0.000000|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|0.000000|
|generation_lm_head_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|0.000000|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|0.000000|
|hmx_command_count|46.000000|136.000000|192.000000|
|hmx_compute_ticks|4408.943750|9781.381250|92664.106250|
|hmx_fp16_tile_pair_count|0.000000|0.000000|0.000000|
|hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|1524096.000000|
|host_wall_ns|1016433.581250|3689627.937500|5786285.506250|
|input_norm_ticks|102.143750|102.206250|102.162500|
|input_stage_ticks|136.300000|135.668750|135.787500|
|intermediate_ddr_read_bytes|0.000000|0.000000|0.000000|
|intermediate_ddr_write_bytes|0.000000|0.000000|0.000000|
|intermediate_dma_descriptor_count|0.000000|0.000000|0.000000|
|intermediate_spill_fill_count|0.000000|0.000000|0.000000|
|invocation_ticks|13788.287500|65010.087500|103674.031250|
|kv_cache_k_format|14.000000|14.000000|14.000000|
|kv_cache_v_format|12.000000|12.000000|12.000000|
|layer_bookkeeping_ticks|10.975000|11.056250|11.043750|
|ledger_named_ticks|13788.287500|65010.087500|103674.031250|
|ledger_unattributed_ticks|0.000000|0.000000|0.000000|
|logical_m|1.000000|1.000000|1.000000|
|metadata_stage_ticks|145.662500|144.393750|144.668750|
|numerical_audit_enabled|0.000000|0.000000|0.000000|
|numerical_status|1.000000|1.000000|1.000000|
|o_projection_ticks|833.325000|5550.262500|8351.087500|
|output_cosine|0.000000|0.000000|0.000000|
|output_fp16_atol|0.062500|0.062500|0.062500|
|output_fp16_max_composed_nrmse|0.003000|0.003000|0.003000|
|output_fp16_rtol|0.002000|0.002000|0.002000|
|output_max_abs|0.000000|0.000000|0.000000|
|output_max_lsb|0.000000|0.000000|0.000000|
|output_max_required_rtol_after_atol|0.000000|0.000000|0.000000|
|output_mismatches|0.000000|0.000000|0.000000|
|output_mixed_tolerance_violations|0.000000|0.000000|0.000000|
|output_nonfinite_count|0.000000|0.000000|0.000000|
|output_nrmse|0.000000|0.000000|0.000000|
|output_stage_ticks|63.481250|69.700000|69.731250|
|post_attention_norm_ticks|0.362500|0.300000|0.356250|
|post_attention_residual_ticks|133.793750|131.350000|130.662500|
|prepared_session_run_index|46.000000|46.000000|46.000000|
|projection_failure_index|0.000000|0.000000|0.000000|
|projection_failure_n_tile|0.000000|0.000000|0.000000|
|projection_failure_result|0.000000|0.000000|0.000000|
|projection_failure_step|0.000000|0.000000|0.000000|
|projection_hmx_wait_ticks|365.843750|458.693750|15964.568750|
|projection_pack_ticks|3.150000|3.118750|3.118750|
|projection_unpack_ticks|0.000000|0.000000|0.000000|
|qk_norm_rope_ticks|0.162500|0.231250|0.168750|
|qkv_projection_ticks|1668.562500|4438.037500|16490.968750|
|repeat_count|1.000000|1.000000|1.000000|
|replay_step|4.500000|4.500000|4.500000|
|runtime_setup_ticks|720.106250|720.262500|720.025000|
|runtime_teardown_ticks|639.056250|643.725000|642.487500|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|2752512.000000|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|77824.000000|
|scan_cache_append_mismatch_count|0.000000|0.000000|0.000000|
|scan_cache_append_ticks|55.643750|55.493750|55.781250|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|143936.000000|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|1152.000000|
|scan_cache_dma_descriptor_count|42.000000|42.000000|42.000000|
|scan_cache_pack_ticks|199.187500|200.287500|200.600000|
|scan_cache_stage_ticks|364.325000|366.368750|369.281250|
|scan_dynamic_attention_ticks|2369.900000|2373.981250|2381.100000|
|scan_logical_m_observed|1.000000|1.000000|1.000000|
|scan_padded_kv_length|96.000000|96.000000|96.000000|
|scan_total_kv_length|68.500000|68.500000|68.500000|
|stage_boundary_ticks|7.656250|7.756250|7.725000|
|total_ticks|13068.762500|64289.918750|102954.193750|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|0.000000|
|u8_attention_av_hmx_ticks|151.350000|153.568750|155.762500|
|u8_attention_av_requant_ticks|102.743750|103.143750|103.818750|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|0.000000|
|u8_attention_k_pack_ticks|13.150000|13.181250|13.193750|
|u8_attention_pipeline_wait_ticks|46.475000|46.450000|46.518750|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|0.000000|
|u8_attention_qk_hmx_ticks|93.231250|94.075000|95.206250|
|u8_attention_qk_norm_rope_ticks|1516.912500|977.356250|1963.768750|
|u8_attention_qk_requant_ticks|0.000000|0.000000|0.000000|
|u8_attention_softmax_ticks|264.975000|268.312500|269.287500|
|u8_attention_v_pack_ticks|85.462500|84.887500|84.762500|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|126.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|896.000000|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|1.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.025000|26.731250|26.937500|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|0.000000|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|28672.000000|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|7.000000|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|u8_cache_native_append_update_ticks|253.087500|254.125000|254.925000|
|u8_cache_native_incremental_append_count|1.000000|1.000000|1.000000|
|u8_cache_native_prefill_build_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|0.000000|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_seal_count|0.000000|0.000000|0.000000|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|0.000000|
|u8_cache_segment_tail_append_count|1.000000|1.000000|1.000000|
|u8_cache_v_quartet_append_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|0.000000|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|0.000000|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.000000|6144.000000|6144.000000|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.000000|12.000000|12.000000|
|u8_cache_v_vtcm_tail_publish_count|2.000000|2.000000|2.000000|
|u8_cache_v_vtcm_tail_row_update_count|8.000000|8.000000|8.000000|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|0.000000|
|valid_length|68.500000|68.500000|68.500000|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|8388608.000000|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|6682752.000000|
|vtcm_requested_bytes|8388608.000000|8388608.000000|8388608.000000|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|0.000000|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_mismatch_count|0.000000|0.000000|0.000000|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_expand_work_ticks|0.000000|9554.675000|10784.312500|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|0.000000|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|0.000000|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|0.000000|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_av_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_av_requant_call_count|8.000000|8.000000|8.000000|
|w4u8_av_requant_rows_observed|4.000000|4.000000|4.000000|
|w4u8_av_requant_vector_count|64.000000|64.000000|64.000000|
|w4u8_common_op_rows_observed|4.000000|4.000000|4.000000|
|w4u8_common_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_av_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_av_requant_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_op_rows|4.000000|4.000000|4.000000|
|w4u8_decode_common_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|8.000000|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_hmx_command_count|30.000000|0.000000|0.000000|
|w4u8_decode_direct_n_mask|63.000000|63.000000|63.000000|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|0.000000|0.000000|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|1.000000|
|w4u8_decode_direct_n_projection_count|7.000000|0.000000|0.000000|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|0.000000|0.000000|
|w4u8_decode_k_pair_row4_call_count|4.000000|4.000000|4.000000|
|w4u8_decode_k_temp_carrier_skipped_count|8.000000|8.000000|8.000000|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|32.000000|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|16.000000|
|w4u8_decode_projection_mode|1.000000|1.000000|1.000000|
|w4u8_decode_q_pair_row4_call_count|8.000000|8.000000|8.000000|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|4.000000|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_qk_rows_processed|96.000000|96.000000|96.000000|
|w4u8_decode_softmax_hvx_tile4_call_count|8.000000|8.000000|8.000000|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|0.000000|
|w4u8_decode_softmax_mode|1.000000|1.000000|1.000000|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_row4_call_count|192.000000|192.000000|192.000000|
|w4u8_decode_swiglu_rows|4.000000|4.000000|4.000000|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|0.000000|
|w4u8_decode_swiglu_vector_count|192.000000|192.000000|192.000000|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|0.000000|
|w4u8_final_residual_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_final_residual_task_count|0.000000|0.000000|0.000000|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_consume_count|6.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_join_wait_ticks|130.381250|0.000000|0.000000|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_publish_count|6.000000|0.000000|0.000000|
|w4u8_gate_up_swiglu_ready_wait_ticks|3252.025000|0.000000|0.000000|
|w4u8_gate_up_swiglu_worker_ticks|782.287500|0.000000|0.000000|
|w4u8_input_norm_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_input_norm_task_count|0.000000|0.000000|0.000000|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_down_hmx_command_count|8.000000|32.000000|32.000000|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_mlp_weight_expand_ticks|0.000000|41116.931250|48333.681250|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|0.000000|
|w4u8_o_batch_count|4.000000|8.000000|16.000000|
|w4u8_o_batch_n_tiles_observed|16.000000|8.000000|4.000000|
|w4u8_o_gate_prefetch_consume_count|1.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_lifetime_ticks|358.775000|0.000000|0.000000|
|w4u8_o_gate_prefetch_start_count|1.000000|0.000000|0.000000|
|w4u8_o_gate_prefetch_wait_ticks|218.293750|0.000000|0.000000|
|w4u8_post_residual_direct_row4_call_count|1.000000|1.000000|1.000000|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|0.000000|
|w4u8_post_residual_task_count|0.000000|0.000000|0.000000|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|0.000000|
|w4u8_prefill_cache_mode|1.000000|1.000000|1.000000|
|w4u8_qk_norm_rope_rows_observed|4.000000|4.000000|4.000000|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|0.000000|
|w4u8_qkv_ring_batch_count|6.000000|32.000000|32.000000|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_dma_wait_ticks|1432.125000|1996.106250|3092.631250|
|w4u8_qkv_ring_expand_task_count|0.000000|128.000000|128.000000|
|w4u8_qkv_ring_expand_ticks|0.000000|9554.675000|10784.312500|
|w4u8_qkv_ring_expand_worker_count|0.000000|3.000000|3.000000|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|24.000000|
|w4u8_qkv_ring_hmx_compute_ticks|381.450000|263.181250|14816.312500|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|1.000000|
|w4u8_qkv_ring_hmx_ready_wait_ticks|932.181250|3524.537500|226.143750|
|w4u8_qkv_ring_pipeline_ticks|1648.237500|4417.675000|16470.800000|
|w4u8_qkv_ring_pool_wait_ticks|2.643750|2.581250|2.656250|
|w4u8_qkv_ring_prep_worker_count|5.000000|2.000000|2.000000|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.956250|2010.556250|12246.718750|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|2.000000|
|w4u8_qkvo_hmx_lifetime_ticks|8229.168750|57101.881250|96971.718750|
|w4u8_qkvo_prefetch_wait_ticks|1432.706250|1999.087500|3095.850000|
|w4u8_qkvo_weight_expand_ticks|0.000000|14127.868750|15930.737500|
|w4u8_residual_active_contexts|0.000000|0.000000|0.000000|
|w4u8_swiglu_rows_observed|4.000000|4.000000|4.000000|
|weight_ddr_read_bytes|25329664.000000|26116096.000000|26116096.000000|
|weight_dma_descriptor_count|60.000000|240.000000|352.000000|
|weight_dma_ticks|8761.406250|11280.337500|15846.181250|

E2E token/s: N/A. No full model executed. PPL/quality: N/A, speed-only experiment.
