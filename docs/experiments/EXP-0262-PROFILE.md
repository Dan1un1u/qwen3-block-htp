# EXP0262 full-width R4 single-layer cost

Branch codex/exp-0262-w4u8-r4-native-layout-pipeline; reporting source 016cb33213457caf1fa61f1b44655c07bc136c9f; native runtime de5a612e8e044449c04bbaf63f7840321492a89d ABI125. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0262; candidate artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0261/r4.

Two paired arms: current optimized R3 OPT2 + wideNR64/nativeW4, and the same path plus full6144 R4 before middle A8. Five short and ten alternating formal pairs, repeat1/repeat10; one prefillM64 and eight consecutive teacher-input M1 steps per repetition. KV starts empty and is computed/appended on device. No frozen-snapshot decode, discarded warmups, selection, extra rounds or fullmodel extrapolation.

Primary medians of round means; paired effect is median within-round ratio, 10000bootstrap seed261. Decode latency averages the eight positions. Additive modules use the identical two median-Host-ranked rounds for all fields. All2970RPCs and layer ledgers independently reconstructed.

|Scope|Control us|R4 us|Paired time change|Ratio95% CI|
|---|---:|---:|---:|---|
|r1_prefill_ns|1723.411|2384.167|+39.42%|[1.315197, 1.409484]|
|r1_decode_ns|995.049|1006.960|+2.48%|[0.967263, 1.096917]|
|r10_prefill_ns|1543.771|2223.518|+46.40%|[1.430592, 1.488131]|
|r10_decode_ns|954.708|1006.825|+5.97%|[1.049060, 1.083600]|

Speed eligible: False; stop before fullmodel: True. Apply the unchanged10percent rule to paired complete Host wall; this result is not an intrinsic R4 lower bound.

R4 is H12 tensor H512, full6144 coverage. Matrix entries are dense parity/Paley signs; no FWHT/butterfly. Stage1 normalizedH512 operates on12groups, stage2 normalizedH12 mixes allgroups. Intermediate and final HMX casts use declared FP16 normalization constants and rounding. The full transform preserves unquantized xW; deployed Down is freshly RTN quantized from original W R with oneFP32scale/output, signed[-7,7]. OtherC64weights unchanged, no group scales/W4toS8. Middle static range is unchanged, so this fixture establishes cost and implementation correctness, not calibrated accuracy or PPL improvement.

Prepare includes unquantized FP16 SwiGLU LUT gather/packing and constant sign matrix materialization. Layout transposes stage1 for the second GEMM; finish reorganizes/quantizes into native-U8 Down input. Native layout uses exact HVX deal/scatter; finish uses gather, unchanged quantization and native stores. Decode FP16 SwiGLU is produced into the HMX input on Gate/Up readiness. Prefill uses two 256KiB input/output slots; next layout and previous finish are scheduled during the current HMX command. Matmul_ticks is exposed submit/wait cost, not total engine compute. Pipeline HVX ticks report scheduled overlap work, not fully hidden time. Dense factor GEMMs alone cannot stand in for complete cost.

|R4 phase, us|M64 prefill|M1 decode|
|---|---:|---:|
|dense_r4_prepare_ticks|413.891|9.948|
|dense_r4_matmul_ticks|25.435|4.217|
|dense_r4_layout_ticks|61.937|0.967|
|dense_r4_finish_ticks|408.195|16.211|

Numerical: EXP0261 sealed original-shard/export evidence is inherited, with frozen local and remote packages reverified. The inherited export proves H12/H512 orthogonality and explicit dense-factor equivalence, xW identity max1.34e-14, independentW4unpack. The inherited65536SwiGLU LUT entries correctly rounded with Decimal80 and independently certified using mpmath100. All9realsteps HMX eachstage<=1FP16ULP+minnormal; middlequantization, nativeDown and finalresidual exact versus independent arithmetic. Conditional fullblock versus scalarR4 after sameHMXstate max1LSB,mincosine0.999773 passes2/.999. Known idealR3 failure remains; this is not fullmodel quality.

PreR4 live data exact between arms; decode dead padding excluded explicitly, prefill allrows checked. K/V prefill carrier hashes exact and persistent cache metadata advances correctly every invocation. Intra-arm everytimedoutput matches frozen auditedhash. Otherrecipes and baselines unchanged.

Physical: requested/granted8MiB, existing peak plan6682752bytes; phase-dead HMXactivation768KiB and two1MiB expansion arenas reused, no VTCM allocation growth. No timed intermediateDDR/spill/audit, oneFastRPC perlayerstep and oneHMXowner; nativeW4 Down unchangedformat. On device the fullmatrix72MiB is never allocated or fetched; denseH512 512KiB constant materialization and H12 padded2KiB are included. Prefill9/decode2R4 HMXcommands; per-row arithmetic and total tile pairs unchanged. Host-DSP boundary=Hostwall-DSPinvocation perrecord.

Retained recovery: runner_parse_failure.txt records a Python declaration replacement error before staging. Native vector and pipeline audits exact against sealed EXP0261; samebinary original R4 also exact. No numerical relaxation or weight/hash replacement.

Bounded native-layout and pipeline optimization completed. Remaining prefill overhead includes FP16 LUT preparation and gather/quantization traffic. No fullmodel or baseline promotion. Device PPL and R4 E2E token/s: N/A (single layer only).
## prefill repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.6 (1.14%)|17.3 (0.78%)|N/A|
|Input RMSNorm|N/A|N/A|19.5 (1.26%)|20.5 (0.92%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|252.3 (16.34%)|252.8 (11.37%)|N/A|
|QK–Softmax–AV|N/A|N/A|118.3 (7.66%)|116.9 (5.26%)|N/A|
|O projection|N/A|N/A|43.0 (2.79%)|42.7 (1.92%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|23.6 (1.53%)|23.8 (1.07%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|482.5 (31.25%)|1145.5 (51.52%)|N/A|
|Down|N/A|N/A|116.0 (7.51%)|116.8 (5.25%)|N/A|
|Final residual|N/A|N/A|6.5 (0.42%)|6.7 (0.30%)|N/A|
|KV carrier conversion|N/A|N/A|4.4 (0.29%)|4.4 (0.20%)|N/A|
|KV append DMA|N/A|N/A|7.2 (0.46%)|7.1 (0.32%)|N/A|
|Block orchestration|N/A|N/A|1.6 (0.10%)|1.5 (0.07%)|N/A|
|Layer bookkeeping|N/A|N/A|0.8 (0.05%)|0.9 (0.04%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|1.4 (0.09%)|1.4 (0.06%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|74.0 (4.79%)|73.9 (3.32%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|375.2 (24.30%)|391.5 (17.61%)|N/A|
|完整 Host wall|N/A|N/A|1543.8 (100.00%)|2223.5 (100.00%)|N/A|

## decode repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.9 (1.88%)|17.9 (1.78%)|N/A|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.53%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|91.3 (9.56%)|90.6 (9.00%)|N/A|
|QK–Softmax–AV|N/A|N/A|65.8 (6.89%)|65.5 (6.50%)|N/A|
|O projection|N/A|N/A|42.3 (4.43%)|42.7 (4.24%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|7.0 (0.74%)|7.1 (0.71%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|220.7 (23.11%)|259.7 (25.79%)|N/A|
|Down|N/A|N/A|114.5 (12.00%)|116.2 (11.54%)|N/A|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.15%)|N/A|
|KV carrier conversion|N/A|N/A|10.1 (1.06%)|10.1 (1.01%)|N/A|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.0 (0.29%)|N/A|
|Block orchestration|N/A|N/A|1.0 (0.10%)|1.0 (0.10%)|N/A|
|Layer bookkeeping|N/A|N/A|0.6 (0.07%)|0.6 (0.06%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.04%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|70.5 (7.39%)|70.4 (7.00%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|302.7 (31.70%)|314.8 (31.26%)|N/A|
|完整 Host wall|N/A|N/A|954.7 (100.00%)|1006.8 (100.00%)|N/A|


## Complete normalized numeric counters

Counts and bytes in native units, *_ticks in qtimer ticks (19.2ticks/us), *_ns in ns. All numeric telemetry retained below, including nested per-layer records. Independent per-field medians do not sum; engine work, DMA and waits can overlap. Legacy unused reference fields are telemetry, not correctness authority; numerical_gate.json is authoritative for implementation comparisons.

### repeat1 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4870.000000|17621.000000|+261.828%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2326.500000|2385.000000|+2.515%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|115.000000|109.000000|-5.217%|
|boundary_ddr_read_bytes|304096.000000|304096.000000|+0.000%|
|boundary_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|boundary_dma_descriptor_count|10.000000|10.000000|+0.000%|
|cache_compared_elements|0.000000|0.000000|N/A zero denominator|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_fp16_max_violation_fraction|0.010000|0.010000|+0.000%|
|cache_fp16_min_cosine|0.999990|0.999990|+0.000%|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|N/A zero denominator|
|cache_max_nrmse|0.000000|0.000000|N/A zero denominator|
|cache_min_cosine|0.000000|0.000000|N/A zero denominator|
|cache_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|cache_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|cache_prefix_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_structure_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_tensor_count|0.000000|0.000000|N/A zero denominator|
|dense_r3_constant_read_bytes|32768.000000|32768.000000|+0.000%|
|dense_r3_mode|1.000000|1.000000|+0.000%|
|dense_r3_optimization|2.000000|2.000000|+0.000%|
|dense_r3_total_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_finish_ticks|2300.000000|2300.000000|+0.000%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|376.000000|379.000000|+0.798%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5821.000000|5816.500000|-0.077%|
|dense_r3_total_prepare_ticks|424.500000|432.500000|+1.885%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|7806.000000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1195.500000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|488.000000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|2.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|7891.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|8000.500000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2400.000000|2386.500000|-0.562%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|142.000000|146.000000|+2.817%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4504.500000|4515.500000|+0.244%|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_embedding_ticks|0.000000|0.000000|N/A zero denominator|
|generation_final_norm_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_command_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_command_count|47.000000|56.000000|+19.149%|
|hmx_compute_ticks|4302.000000|5023.500000|+16.771%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1723411.500000|2384167.000000|+38.340%|
|input_norm_ticks|430.000000|424.000000|-1.395%|
|input_stage_ticks|145.000000|145.500000|+0.345%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|24186.000000|36999.000000|+52.977%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|55.500000|54.500000|-1.802%|
|ledger_named_ticks|24186.000000|36999.000000|+52.977%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|182.000000|183.500000|+0.824%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|901.500000|907.000000|+0.610%|
|output_cosine|0.000000|0.000000|N/A zero denominator|
|output_fp16_atol|0.062500|0.062500|+0.000%|
|output_fp16_max_composed_nrmse|0.003000|0.003000|+0.000%|
|output_fp16_rtol|0.002000|0.002000|+0.000%|
|output_max_abs|0.000000|0.000000|N/A zero denominator|
|output_max_lsb|0.000000|0.000000|N/A zero denominator|
|output_max_required_rtol_after_atol|0.000000|0.000000|N/A zero denominator|
|output_mismatches|0.000000|0.000000|N/A zero denominator|
|output_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|output_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|output_nrmse|0.000000|0.000000|N/A zero denominator|
|output_stage_ticks|53.500000|55.000000|+2.804%|
|post_attention_norm_ticks|3.500000|3.000000|-14.286%|
|post_attention_residual_ticks|511.500000|507.500000|-0.782%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|1.000000|1.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|507.000000|515.500000|+1.677%|
|projection_pack_ticks|17.000000|17.000000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|qkv_projection_ticks|5152.000000|5172.000000|+0.388%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|1115.500000|1105.500000|-0.896%|
|runtime_teardown_ticks|925.500000|952.000000|+2.863%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|159.500000|159.000000|-0.313%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|103.500000|103.500000|+0.000%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4870.000000|17621.000000|+261.828%|
|slice_layer_0.attention_ticks|2326.500000|2385.000000|+2.515%|
|slice_layer_0.block_orchestration_ticks|115.000000|109.000000|-5.217%|
|slice_layer_0.cache_append_dma_ticks|159.500000|159.000000|-0.313%|
|slice_layer_0.cache_append_pack_ticks|103.500000|103.500000|+0.000%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2400.000000|2386.500000|-0.562%|
|slice_layer_0.final_residual_ticks|142.000000|146.000000|+2.817%|
|slice_layer_0.gate_up_ticks|4504.500000|4515.500000|+0.244%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|430.000000|424.000000|-1.395%|
|slice_layer_0.input_stage_ticks|145.000000|145.500000|+0.345%|
|slice_layer_0.layer_bookkeeping_ticks|55.500000|54.500000|-1.802%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|22027.000000|34773.500000|+57.868%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|182.000000|183.500000|+0.824%|
|slice_layer_0.o_projection_ticks|901.500000|907.000000|+0.610%|
|slice_layer_0.post_attention_norm_ticks|3.500000|3.000000|-14.286%|
|slice_layer_0.post_attention_residual_ticks|511.500000|507.500000|-0.782%|
|slice_layer_0.qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|slice_layer_0.qkv_projection_ticks|5152.000000|5172.000000|+0.388%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|100.500000|102.500000|+1.990%|
|total_ticks|23073.500000|35902.500000|+55.601%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|420.000000|427.500000|+1.786%|
|u8_attention_av_requant_ticks|1168.500000|1176.500000|+0.685%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1347.000000|1319.500000|-2.042%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|413.500000|430.000000|+3.990%|
|u8_attention_qk_norm_rope_ticks|8932.500000|8935.000000|+0.028%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|6033.000000|6043.000000|+0.166%|
|u8_attention_v_pack_ticks|2950.500000|2993.500000|+1.457%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_bytes|29568.000000|29568.000000|+0.000%|
|u8_cache_k_vtcm_tail_init_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|251.500000|245.500000|-2.386%|
|u8_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_build_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|1.000000|1.000000|+0.000%|
|u8_cache_native_prefill_reused_carrier_bytes|143360.000000|143360.000000|+0.000%|
|u8_cache_segment_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_tail_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|32768.000000|32768.000000|+0.000%|
|u8_cache_v_vtcm_tail_init_count|1.000000|1.000000|+0.000%|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|valid_length|64.000000|64.000000|+0.000%|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_audit|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_conversion_audit_mismatches|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt_calls|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_vector_count|0.000000|0.000000|N/A zero denominator|
|w4u8_common_op_rows_observed|64.000000|64.000000|+0.000%|
|w4u8_common_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_requant_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_op_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|+0.000%|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|50331648.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_hmx_command_count|30.000000|30.000000|+0.000%|
|w4u8_decode_direct_n_mask|63.000000|63.000000|+0.000%|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|7.000000|7.000000|+0.000%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|25165824.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_projection_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|N/A zero denominator|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_main_work_ticks|69.000000|69.000000|+0.000%|
|w4u8_final_residual_pool_wait_ticks|25.500000|21.000000|-17.647%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|392.000000|395.000000|+0.765%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|365.000000|358.000000|-1.918%|
|w4u8_input_norm_pool_wait_ticks|18.000000|17.500000|-2.778%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1650.000000|1616.500000|-2.030%|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_down_hmx_command_count|8.000000|8.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_batch_count|4.000000|4.000000|+0.000%|
|w4u8_o_batch_n_tiles_observed|16.000000|16.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_main_work_ticks|305.500000|316.500000|+3.601%|
|w4u8_post_residual_pool_wait_ticks|146.500000|155.500000|+6.143%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|2021.000000|1995.500000|-1.262%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1470.000000|1469.000000|-0.068%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|393.500000|397.000000|+0.889%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|996.500000|971.000000|-2.559%|
|w4u8_qkv_ring_pipeline_ticks|1986.000000|2009.500000|+1.183%|
|w4u8_qkv_ring_pool_wait_ticks|216.500000|240.000000|+10.855%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.500000|4.000000|+14.286%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8165.500000|8222.000000|+0.692%|
|w4u8_qkvo_prefetch_wait_ticks|1472.500000|1472.000000|-0.034%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8955.000000|8965.500000|+0.117%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat1 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|608.625000|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1272.187500|1274.812500|+0.206%|
|attention_unattributed_ticks|575.062500|581.500000|+1.119%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|22.687500|23.375000|+3.030%|
|boundary_ddr_read_bytes|304096.000000|304096.000000|+0.000%|
|boundary_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|boundary_dma_descriptor_count|10.000000|10.000000|+0.000%|
|cache_compared_elements|0.000000|0.000000|N/A zero denominator|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_fp16_max_violation_fraction|0.010000|0.010000|+0.000%|
|cache_fp16_min_cosine|0.999990|0.999990|+0.000%|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|N/A zero denominator|
|cache_max_nrmse|0.000000|0.000000|N/A zero denominator|
|cache_min_cosine|0.000000|0.000000|N/A zero denominator|
|cache_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|cache_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|cache_prefix_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_structure_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_tensor_count|0.000000|0.000000|N/A zero denominator|
|dense_r3_constant_read_bytes|32768.000000|32768.000000|+0.000%|
|dense_r3_mode|1.000000|1.000000|+0.000%|
|dense_r3_optimization|2.000000|2.000000|+0.000%|
|dense_r3_total_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_finish_ticks|50.625000|50.750000|+0.247%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|49.562500|48.937500|-1.261%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|705.812500|715.000000|+1.302%|
|dense_r3_total_prepare_ticks|26.625000|19.500000|-26.761%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|312.875000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|19.062500|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|81.000000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|2.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|190.312500|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2219.375000|2222.687500|+0.149%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.625000|28.937500|-2.321%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4271.125000|4389.062500|+2.761%|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_embedding_ticks|0.000000|0.000000|N/A zero denominator|
|generation_final_norm_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_command_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_command_count|47.000000|49.000000|+4.255%|
|hmx_compute_ticks|4134.750000|4131.687500|-0.074%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|995048.875000|1006959.500000|+1.197%|
|input_norm_ticks|102.625000|102.687500|+0.061%|
|input_stage_ticks|136.375000|136.375000|+0.000%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12616.312500|13304.437500|+5.454%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.375000|12.687500|+2.525%|
|ledger_named_ticks|12616.312500|13304.437500|+5.454%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|142.250000|143.687500|+1.011%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|826.750000|823.937500|-0.340%|
|output_cosine|0.000000|0.000000|N/A zero denominator|
|output_fp16_atol|0.062500|0.062500|+0.000%|
|output_fp16_max_composed_nrmse|0.003000|0.003000|+0.000%|
|output_fp16_rtol|0.002000|0.002000|+0.000%|
|output_max_abs|0.000000|0.000000|N/A zero denominator|
|output_max_lsb|0.000000|0.000000|N/A zero denominator|
|output_max_required_rtol_after_atol|0.000000|0.000000|N/A zero denominator|
|output_mismatches|0.000000|0.000000|N/A zero denominator|
|output_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|output_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|output_nrmse|0.000000|0.000000|N/A zero denominator|
|output_stage_ticks|63.625000|63.812500|+0.295%|
|post_attention_norm_ticks|0.687500|0.437500|-36.364%|
|post_attention_residual_ticks|137.187500|136.500000|-0.501%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|5.500000|5.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|368.250000|368.062500|-0.051%|
|projection_pack_ticks|3.000000|3.000000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.250000|0.187500|-25.000%|
|qkv_projection_ticks|1766.687500|1757.625000|-0.513%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|718.125000|717.250000|-0.122%|
|runtime_teardown_ticks|634.750000|634.875000|+0.020%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.500000|57.437500|-0.109%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|199.750000|199.812500|+0.031%|
|scan_cache_stage_ticks|340.562500|343.375000|+0.826%|
|scan_dynamic_attention_ticks|1266.625000|1269.500000|+0.227%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|608.625000|N/A zero denominator|
|slice_layer_0.attention_ticks|1272.187500|1274.812500|+0.206%|
|slice_layer_0.block_orchestration_ticks|22.687500|23.375000|+3.030%|
|slice_layer_0.cache_append_dma_ticks|57.500000|57.437500|-0.109%|
|slice_layer_0.cache_append_pack_ticks|199.750000|199.812500|+0.031%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2219.375000|2222.687500|+0.149%|
|slice_layer_0.final_residual_ticks|29.625000|28.937500|-2.321%|
|slice_layer_0.gate_up_ticks|4271.125000|4389.062500|+2.761%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.625000|102.687500|+0.061%|
|slice_layer_0.input_stage_ticks|136.375000|136.375000|+0.000%|
|slice_layer_0.layer_bookkeeping_ticks|12.375000|12.687500|+2.525%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11187.062500|11881.812500|+6.210%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|142.250000|143.687500|+1.011%|
|slice_layer_0.o_projection_ticks|826.750000|823.937500|-0.340%|
|slice_layer_0.post_attention_norm_ticks|0.687500|0.437500|-36.364%|
|slice_layer_0.post_attention_residual_ticks|137.187500|136.500000|-0.501%|
|slice_layer_0.qk_norm_rope_ticks|0.250000|0.187500|-25.000%|
|slice_layer_0.qkv_projection_ticks|1766.687500|1757.625000|-0.513%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|8.125000|7.750000|-4.615%|
|total_ticks|11895.250000|12586.937500|+5.815%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|113.562500|115.875000|+2.036%|
|u8_attention_av_requant_ticks|98.437500|99.125000|+0.698%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.875000|13.812500|-0.450%|
|u8_attention_pipeline_wait_ticks|45.812500|45.687500|-0.273%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|98.687500|93.562500|-5.193%|
|u8_attention_qk_norm_rope_ticks|831.750000|833.062500|+0.158%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|243.250000|244.625000|+0.565%|
|u8_attention_v_pack_ticks|83.750000|83.500000|-0.299%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.375000|26.375000|+0.000%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|253.937500|253.625000|-0.123%|
|u8_cache_native_incremental_append_count|1.000000|1.000000|+0.000%|
|u8_cache_native_prefill_build_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_tail_append_count|1.000000|1.000000|+0.000%|
|u8_cache_v_quartet_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.000000|6144.000000|+0.000%|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.000000|12.000000|+0.000%|
|u8_cache_v_vtcm_tail_publish_count|2.000000|2.000000|+0.000%|
|u8_cache_v_vtcm_tail_row_update_count|8.000000|8.000000|+0.000%|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|valid_length|68.500000|68.500000|+0.000%|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_audit|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_conversion_audit_mismatches|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt_calls|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_call_count|8.000000|8.000000|+0.000%|
|w4u8_av_requant_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_av_requant_vector_count|64.000000|64.000000|+0.000%|
|w4u8_common_op_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_common_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_requant_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_op_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|+0.000%|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|50331648.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_hmx_command_count|30.000000|30.000000|+0.000%|
|w4u8_decode_direct_n_mask|63.000000|63.000000|+0.000%|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|7.000000|7.000000|+0.000%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|25165824.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_projection_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|8.000000|8.000000|+0.000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_row4_call_count|192.000000|0.000000|-100.000%|
|w4u8_decode_swiglu_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|192.000000|0.000000|-100.000%|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_consume_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.562500|163.875000|+89.314%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3476.375000|2821.062500|-18.850%|
|w4u8_gate_up_swiglu_worker_ticks|467.250000|1296.750000|+177.528%|
|w4u8_input_norm_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_down_hmx_command_count|8.000000|8.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_batch_count|4.000000|4.000000|+0.000%|
|w4u8_o_batch_n_tiles_observed|16.000000|16.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|348.937500|349.187500|+0.072%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|197.250000|197.625000|+0.190%|
|w4u8_post_residual_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1405.000000|1402.937500|-0.147%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|350.187500|352.562500|+0.678%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|945.500000|940.812500|-0.496%|
|w4u8_qkv_ring_pipeline_ticks|1616.250000|1613.000000|-0.201%|
|w4u8_qkv_ring_pool_wait_ticks|2.875000|2.750000|-4.348%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.875000|4.125000|+6.452%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8078.250000|8098.875000|+0.255%|
|w4u8_qkvo_prefetch_wait_ticks|1405.687500|1403.250000|-0.173%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8587.250000|8576.500000|-0.125%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4844.850000|17510.100000|+261.417%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2248.550000|2266.400000|+0.794%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|29.850000|29.000000|-2.848%|
|boundary_ddr_read_bytes|304096.000000|304096.000000|+0.000%|
|boundary_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|boundary_dma_descriptor_count|10.000000|10.000000|+0.000%|
|cache_compared_elements|0.000000|0.000000|N/A zero denominator|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_fp16_max_violation_fraction|0.010000|0.010000|+0.000%|
|cache_fp16_min_cosine|0.999990|0.999990|+0.000%|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|N/A zero denominator|
|cache_max_nrmse|0.000000|0.000000|N/A zero denominator|
|cache_min_cosine|0.000000|0.000000|N/A zero denominator|
|cache_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|cache_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|cache_prefix_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_structure_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_tensor_count|0.000000|0.000000|N/A zero denominator|
|dense_r3_constant_read_bytes|32768.000000|32768.000000|+0.000%|
|dense_r3_mode|1.000000|1.000000|+0.000%|
|dense_r3_optimization|2.000000|2.000000|+0.000%|
|dense_r3_total_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_finish_ticks|2268.850000|2268.000000|-0.037%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|359.650000|358.900000|-0.209%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5690.600000|5690.800000|+0.004%|
|dense_r3_total_prepare_ticks|330.250000|330.700000|+0.136%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|7837.350000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1189.200000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|488.350000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|2.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|7915.650000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|7946.700000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2234.850000|2227.700000|-0.320%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|126.150000|127.000000|+0.674%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4428.150000|4426.700000|-0.033%|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_embedding_ticks|0.000000|0.000000|N/A zero denominator|
|generation_final_norm_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_command_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_command_count|47.000000|56.000000|+19.149%|
|hmx_compute_ticks|4314.600000|4993.200000|+15.728%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1543770.850000|2223518.200000|+44.032%|
|input_norm_ticks|378.650000|387.750000|+2.403%|
|input_stage_ticks|135.100000|133.000000|-1.554%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|22475.700000|35168.200000|+56.472%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|16.200000|16.400000|+1.235%|
|ledger_named_ticks|22475.700000|35168.200000|+56.472%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|145.150000|144.500000|-0.448%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|825.000000|829.350000|+0.527%|
|output_cosine|0.000000|0.000000|N/A zero denominator|
|output_fp16_atol|0.062500|0.062500|+0.000%|
|output_fp16_max_composed_nrmse|0.003000|0.003000|+0.000%|
|output_fp16_rtol|0.002000|0.002000|+0.000%|
|output_max_abs|0.000000|0.000000|N/A zero denominator|
|output_max_lsb|0.000000|0.000000|N/A zero denominator|
|output_max_required_rtol_after_atol|0.000000|0.000000|N/A zero denominator|
|output_mismatches|0.000000|0.000000|N/A zero denominator|
|output_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|output_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|output_nrmse|0.000000|0.000000|N/A zero denominator|
|output_stage_ticks|55.500000|59.350000|+6.937%|
|post_attention_norm_ticks|0.750000|0.800000|+6.667%|
|post_attention_residual_ticks|454.700000|456.400000|+0.374%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|41.500000|41.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|503.300000|499.050000|-0.844%|
|projection_pack_ticks|4.050000|4.050000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.500000|0.500000|+0.000%|
|qkv_projection_ticks|4845.450000|4857.050000|+0.239%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|756.650000|752.600000|-0.535%|
|runtime_teardown_ticks|669.250000|670.600000|+0.202%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|138.300000|136.550000|-1.265%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|84.250000|85.250000|+1.187%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4844.850000|17510.100000|+261.417%|
|slice_layer_0.attention_ticks|2248.550000|2266.400000|+0.794%|
|slice_layer_0.block_orchestration_ticks|29.850000|29.000000|-2.848%|
|slice_layer_0.cache_append_dma_ticks|138.300000|136.550000|-1.265%|
|slice_layer_0.cache_append_pack_ticks|84.250000|85.250000|+1.187%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2234.850000|2227.700000|-0.320%|
|slice_layer_0.final_residual_ticks|126.150000|127.000000|+0.674%|
|slice_layer_0.gate_up_ticks|4428.150000|4426.700000|-0.033%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|378.650000|387.750000|+2.403%|
|slice_layer_0.input_stage_ticks|135.100000|133.000000|-1.554%|
|slice_layer_0.layer_bookkeeping_ticks|16.200000|16.400000|+1.235%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|20972.350000|33662.800000|+60.510%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|145.150000|144.500000|-0.448%|
|slice_layer_0.o_projection_ticks|825.000000|829.350000|+0.527%|
|slice_layer_0.post_attention_norm_ticks|0.750000|0.800000|+6.667%|
|slice_layer_0.post_attention_residual_ticks|454.700000|456.400000|+0.374%|
|slice_layer_0.qk_norm_rope_ticks|0.500000|0.500000|+0.000%|
|slice_layer_0.qkv_projection_ticks|4845.450000|4857.050000|+0.239%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|27.100000|27.600000|+1.845%|
|total_ticks|21721.450000|34417.550000|+58.450%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|423.300000|426.500000|+0.756%|
|u8_attention_av_requant_ticks|1174.650000|1175.950000|+0.111%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1147.300000|1185.500000|+3.330%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|422.500000|423.550000|+0.249%|
|u8_attention_qk_norm_rope_ticks|8648.700000|8647.800000|-0.010%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|5976.450000|5983.500000|+0.118%|
|u8_attention_v_pack_ticks|2817.500000|2834.400000|+0.600%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_correction_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_fallback_head_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_bytes|29568.000000|29568.000000|+0.000%|
|u8_cache_k_vtcm_tail_init_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|219.600000|219.600000|+0.000%|
|u8_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_build_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|1.000000|1.000000|+0.000%|
|u8_cache_native_prefill_reused_carrier_bytes|143360.000000|143360.000000|+0.000%|
|u8_cache_segment_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_tail_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|32768.000000|32768.000000|+0.000%|
|u8_cache_v_vtcm_tail_init_count|1.000000|1.000000|+0.000%|
|u8_cache_v_vtcm_tail_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_row_update_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|valid_length|64.000000|64.000000|+0.000%|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_audit|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_conversion_audit_mismatches|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt_calls|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_vector_count|0.000000|0.000000|N/A zero denominator|
|w4u8_common_op_rows_observed|64.000000|64.000000|+0.000%|
|w4u8_common_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_requant_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_op_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|+0.000%|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|50331648.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_hmx_command_count|30.000000|30.000000|+0.000%|
|w4u8_decode_direct_n_mask|63.000000|63.000000|+0.000%|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|7.000000|7.000000|+0.000%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|25165824.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_projection_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|0.000000|0.000000|N/A zero denominator|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_main_work_ticks|76.900000|76.600000|-0.390%|
|w4u8_final_residual_pool_wait_ticks|13.200000|14.850000|+12.500%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|376.400000|376.800000|+0.106%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|322.950000|323.550000|+0.186%|
|w4u8_input_norm_pool_wait_ticks|20.050000|24.300000|+21.197%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1469.450000|1478.650000|+0.626%|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_down_hmx_command_count|8.000000|8.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_batch_count|4.000000|4.000000|+0.000%|
|w4u8_o_batch_n_tiles_observed|16.000000|16.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_start_count|0.000000|0.000000|N/A zero denominator|
|w4u8_o_gate_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_main_work_ticks|360.600000|358.750000|-0.513%|
|w4u8_post_residual_pool_wait_ticks|49.950000|59.450000|+19.019%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1777.250000|1796.450000|+1.080%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1396.800000|1385.350000|-0.820%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|371.250000|371.400000|+0.040%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|940.750000|923.800000|-1.802%|
|w4u8_qkv_ring_pipeline_ticks|1860.050000|1869.800000|+0.524%|
|w4u8_qkv_ring_pool_wait_ticks|242.350000|254.800000|+5.137%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.900000|4.200000|+7.692%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|7874.450000|7861.700000|-0.162%|
|w4u8_qkvo_prefetch_wait_ticks|1397.950000|1386.550000|-0.815%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8587.400000|8590.100000|+0.031%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|606.562500|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1260.362500|1261.931250|+0.124%|
|attention_unattributed_ticks|568.043750|571.606250|+0.627%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|19.343750|19.212500|-0.679%|
|boundary_ddr_read_bytes|304096.000000|304096.000000|+0.000%|
|boundary_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|boundary_dma_descriptor_count|10.000000|10.000000|+0.000%|
|cache_compared_elements|0.000000|0.000000|N/A zero denominator|
|cache_composed_cosine_diagnostic_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_fp16_max_violation_fraction|0.010000|0.010000|+0.000%|
|cache_fp16_min_cosine|0.999990|0.999990|+0.000%|
|cache_legacy_mixed_bound_failure_count|0.000000|0.000000|N/A zero denominator|
|cache_max_mixed_tolerance_violation_fraction|0.000000|0.000000|N/A zero denominator|
|cache_max_nrmse|0.000000|0.000000|N/A zero denominator|
|cache_min_cosine|0.000000|0.000000|N/A zero denominator|
|cache_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|cache_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|cache_prefix_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_structure_mismatches|0.000000|0.000000|N/A zero denominator|
|cache_tensor_count|0.000000|0.000000|N/A zero denominator|
|dense_r3_constant_read_bytes|32768.000000|32768.000000|+0.000%|
|dense_r3_mode|1.000000|1.000000|+0.000%|
|dense_r3_optimization|2.000000|2.000000|+0.000%|
|dense_r3_total_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_finish_ticks|51.562500|50.887500|-1.309%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|49.262500|48.906250|-0.723%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|729.550000|792.731250|+8.660%|
|dense_r3_total_prepare_ticks|20.156250|19.412500|-3.690%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|311.243750|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|18.562500|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|80.968750|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|2.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|191.000000|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2210.412500|2226.062500|+0.708%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.387500|28.637500|-2.552%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4256.487500|4349.656250|+2.189%|
|generation_embedding_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_embedding_ticks|0.000000|0.000000|N/A zero denominator|
|generation_final_norm_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_argmax_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_batch_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_command_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_direct_slot_join_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_expand_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_n_tiles|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_prefetch_count|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_init_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_scale_resident_bytes|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|generation_lm_head_weight_dma_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_command_count|47.000000|49.000000|+4.255%|
|hmx_compute_ticks|4149.762500|4116.525000|-0.801%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|954708.387500|1006824.537500|+5.459%|
|input_norm_ticks|102.556250|102.493750|-0.061%|
|input_stage_ticks|136.375000|136.368750|-0.005%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12575.700000|13259.825000|+5.440%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.075000|12.225000|+1.242%|
|ledger_named_ticks|12575.700000|13259.825000|+5.440%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|142.093750|143.625000|+1.078%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|820.275000|820.556250|+0.034%|
|output_cosine|0.000000|0.000000|N/A zero denominator|
|output_fp16_atol|0.062500|0.062500|+0.000%|
|output_fp16_max_composed_nrmse|0.003000|0.003000|+0.000%|
|output_fp16_rtol|0.002000|0.002000|+0.000%|
|output_max_abs|0.000000|0.000000|N/A zero denominator|
|output_max_lsb|0.000000|0.000000|N/A zero denominator|
|output_max_required_rtol_after_atol|0.000000|0.000000|N/A zero denominator|
|output_mismatches|0.000000|0.000000|N/A zero denominator|
|output_mixed_tolerance_violations|0.000000|0.000000|N/A zero denominator|
|output_nonfinite_count|0.000000|0.000000|N/A zero denominator|
|output_nrmse|0.000000|0.000000|N/A zero denominator|
|output_stage_ticks|64.656250|64.600000|-0.087%|
|post_attention_norm_ticks|0.575000|0.512500|-10.870%|
|post_attention_residual_ticks|135.012500|135.287500|+0.204%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|46.000000|46.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|366.968750|367.237500|+0.073%|
|projection_pack_ticks|2.987500|2.987500|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.243750|0.218750|-10.256%|
|qkv_projection_ticks|1757.537500|1741.112500|-0.935%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|716.775000|716.518750|-0.036%|
|runtime_teardown_ticks|634.731250|635.418750|+0.108%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|56.856250|56.856250|+0.000%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|193.712500|193.975000|+0.136%|
|scan_cache_stage_ticks|345.025000|348.093750|+0.889%|
|scan_dynamic_attention_ticks|1256.618750|1258.162500|+0.123%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|606.562500|N/A zero denominator|
|slice_layer_0.attention_ticks|1260.362500|1261.931250|+0.124%|
|slice_layer_0.block_orchestration_ticks|19.343750|19.212500|-0.679%|
|slice_layer_0.cache_append_dma_ticks|56.856250|56.856250|+0.000%|
|slice_layer_0.cache_append_pack_ticks|193.712500|193.975000|+0.136%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2210.412500|2226.062500|+0.708%|
|slice_layer_0.final_residual_ticks|29.387500|28.637500|-2.552%|
|slice_layer_0.gate_up_ticks|4256.487500|4349.656250|+2.189%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.556250|102.493750|-0.061%|
|slice_layer_0.input_stage_ticks|136.375000|136.368750|-0.005%|
|slice_layer_0.layer_bookkeeping_ticks|12.075000|12.225000|+1.242%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11151.387500|11835.318750|+6.133%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|142.093750|143.625000|+1.078%|
|slice_layer_0.o_projection_ticks|820.275000|820.556250|+0.034%|
|slice_layer_0.post_attention_norm_ticks|0.575000|0.512500|-10.870%|
|slice_layer_0.post_attention_residual_ticks|135.012500|135.287500|+0.204%|
|slice_layer_0.qk_norm_rope_ticks|0.243750|0.218750|-10.256%|
|slice_layer_0.qkv_projection_ticks|1757.537500|1741.112500|-0.935%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.956250|7.918750|-0.471%|
|total_ticks|11859.050000|12543.231250|+5.769%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|116.531250|115.243750|-1.105%|
|u8_attention_av_requant_ticks|99.518750|99.225000|-0.295%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.275000|13.143750|-0.989%|
|u8_attention_pipeline_wait_ticks|44.937500|44.750000|-0.417%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|94.581250|95.562500|+1.037%|
|u8_attention_qk_norm_rope_ticks|851.031250|912.168750|+7.184%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|239.137500|239.043750|-0.039%|
|u8_attention_v_pack_ticks|82.356250|82.493750|+0.167%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|25.418750|25.537500|+0.467%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|249.250000|249.243750|-0.003%|
|u8_cache_native_incremental_append_count|1.000000|1.000000|+0.000%|
|u8_cache_native_prefill_build_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_sealed_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_segment_tail_append_count|1.000000|1.000000|+0.000%|
|u8_cache_v_quartet_append_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_attention_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_full_tile_rmw_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_native_load_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_partial_pack_rows|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_quartet_publish_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_v_vtcm_tail_native_load_bytes|6144.000000|6144.000000|+0.000%|
|u8_cache_v_vtcm_tail_partial_pack_rows|12.000000|12.000000|+0.000%|
|u8_cache_v_vtcm_tail_publish_count|2.000000|2.000000|+0.000%|
|u8_cache_v_vtcm_tail_row_update_count|8.000000|8.000000|+0.000%|
|u8_cache_v_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|valid_length|68.500000|68.500000|+0.000%|
|vtcm_acquired_bytes|8388608.000000|8388608.000000|+0.000%|
|vtcm_peak_plan_bytes|6682752.000000|6682752.000000|+0.000%|
|vtcm_requested_bytes|8388608.000000|8388608.000000|+0.000%|
|w4f16_cross_prefetch_lifetime_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_cross_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_audit|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_conversion_audit_mismatches|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt|0.000000|0.000000|N/A zero denominator|
|w4f16_decode_opt_calls|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_expand_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_hmx_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_stream_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_gate_up_weight_dma_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_hmx_tail_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4f16_prefetch_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_av_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_av_requant_call_count|8.000000|8.000000|+0.000%|
|w4u8_av_requant_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_av_requant_vector_count|64.000000|64.000000|+0.000%|
|w4u8_common_op_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_common_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_av_requant_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_op_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_common_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_direct_n_down_batch_n_tiles|8.000000|8.000000|+0.000%|
|w4u8_decode_direct_n_down_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_expand_bytes_avoided|50331648.000000|50331648.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_continuous|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_hmx_command_count|30.000000|30.000000|+0.000%|
|w4u8_decode_direct_n_mask|63.000000|63.000000|+0.000%|
|w4u8_decode_direct_n_o_gate_prefetch|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_o_single_dma|1.000000|1.000000|+0.000%|
|w4u8_decode_direct_n_projection_count|7.000000|7.000000|+0.000%|
|w4u8_decode_direct_n_q_batch_n_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_direct_n_qkv_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_direct_n_weight_ddr_read_bytes|25165824.000000|25165824.000000|+0.000%|
|w4u8_decode_k_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_temp_carrier_skipped_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_k_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_lm_head_group_tiles|32.000000|32.000000|+0.000%|
|w4u8_decode_o_batch_n_tiles|16.000000|16.000000|+0.000%|
|w4u8_decode_projection_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_q_pair_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_q_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_norm_rope_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_qk_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_qk_rows_processed|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_hvx_tile4_call_count|8.000000|8.000000|+0.000%|
|w4u8_decode_softmax_hvx_tile4_mismatch_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_softmax_mode|1.000000|1.000000|+0.000%|
|w4u8_decode_swiglu_padding_poison|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_padding_poison_count|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_row4_call_count|192.000000|0.000000|-100.000%|
|w4u8_decode_swiglu_rows|4.000000|4.000000|+0.000%|
|w4u8_decode_swiglu_valid_row_hash|0.000000|0.000000|N/A zero denominator|
|w4u8_decode_swiglu_vector_count|192.000000|0.000000|-100.000%|
|w4u8_delta_reconstruction_mode|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_final_residual_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_final_residual_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_consume_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.087500|162.875000|+89.197%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3453.381250|2818.212500|-18.393%|
|w4u8_gate_up_swiglu_worker_ticks|459.381250|1256.837500|+173.594%|
|w4u8_input_norm_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_input_norm_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_activation_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_down_hmx_command_count|8.000000|8.000000|+0.000%|
|w4u8_mlp_down_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_expanded_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_gate_up_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_compute_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_producer_slot_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_mlp_weight_stage_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_o_batch_count|4.000000|4.000000|+0.000%|
|w4u8_o_batch_n_tiles_observed|16.000000|16.000000|+0.000%|
|w4u8_o_gate_prefetch_consume_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_lifetime_ticks|343.493750|348.518750|+1.463%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|201.800000|208.075000|+3.110%|
|w4u8_post_residual_direct_row4_call_count|1.000000|1.000000|+0.000%|
|w4u8_post_residual_main_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_pool_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_post_residual_worker_work_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|4.000000|4.000000|+0.000%|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1395.906250|1376.556250|-1.386%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|355.081250|358.750000|+1.033%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|933.812500|913.243750|-2.203%|
|w4u8_qkv_ring_pipeline_ticks|1615.043750|1599.518750|-0.961%|
|w4u8_qkv_ring_pool_wait_ticks|2.756250|2.743750|-0.454%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.662500|7.850000|+68.365%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8068.937500|8063.000000|-0.074%|
|w4u8_qkvo_prefetch_wait_ticks|1396.443750|1377.231250|-1.376%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8585.400000|8563.868750|-0.251%|
|wide_score_mode|4.000000|4.000000|+0.000%|
