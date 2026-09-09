# EXP0261 full-width R4 single-layer cost

Branch codex/exp-0261-w4u8-dense-r4-cost; reporting source 88d11b02428a106bc2e53e90a0b3713d4ad410bd; native runtime dabe2b6fc9a5511a919828d63dda3e6c8973d85f ABI123. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0261; candidate artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0261/r4.

Two paired arms: current optimized R3 OPT2 + wideNR64/nativeW4, and the same path plus full6144 R4 before middle A8. Five short and ten alternating formal pairs, repeat1/repeat10; one prefillM64 and eight consecutive teacher-input M1 steps per repetition. KV starts empty and is computed/appended on device. No frozen-snapshot decode, discarded warmups, selection, extra rounds or fullmodel extrapolation.

Primary medians of round means; paired effect is median within-round ratio, 10000bootstrap seed261. Decode latency averages the eight positions. Additive modules use the identical two median-Host-ranked rounds for all fields. All2970RPCs and layer ledgers independently reconstructed.

|Scope|Control us|R4 us|Paired time change|Ratio95% CI|
|---|---:|---:|---:|---|
|r1_prefill_ns|1587.760|32107.318|+1920.55%|[18.844596, 22.056942]|
|r1_decode_ns|957.747|1608.561|+66.59%|[1.598990, 1.742311]|
|r10_prefill_ns|1518.570|31899.607|+1999.01%|[20.558788, 21.473788]|
|r10_decode_ns|946.401|1583.942|+66.89%|[1.649037, 1.695214]|

Speed eligible: False; stop before fullmodel: True. The unchanged10percent rule is exceeded. This is a first implementation result, not an intrinsic R4 lower bound.

R4 is H12 tensor H512, full6144 coverage. Matrix entries are dense parity/Paley signs; no FWHT/butterfly. Stage1 normalizedH512 operates on12groups, stage2 normalizedH12 mixes allgroups. Intermediate and final HMX casts use declared FP16 normalization constants and rounding. The full transform preserves unquantized xW; deployed Down is freshly RTN quantized from original W R with oneFP32scale/output, signed[-7,7]. OtherC64weights unchanged, no group scales/W4toS8. Middle static range is unchanged, so this fixture establishes cost and implementation correctness, not calibrated accuracy or PPL improvement.

Prepare includes unquantized FP16 SwiGLU LUT gather/packing and constant sign matrix materialization. Layout transposes stage1 for the second GEMM; finish reorganizes/quantizes into native-U8 Down input. Current layout/finish use scalar indexing and dominate. Decode Gate/Up-to-SwiGLU streaming is disabled for R4; its lost overlap counts. Dense factor GEMMs alone cannot stand in for complete cost.

|R4 phase, us|M64 prefill|M1 decode|
|---|---:|---:|
|dense_r4_prepare_ticks|426.456|50.144|
|dense_r4_matmul_ticks|45.852|4.618|
|dense_r4_layout_ticks|11933.557|186.473|
|dense_r4_finish_ticks|17950.143|333.754|

Numerical: fresh original shards verified, H12/H512 orthogonality and explicit dense-factor equivalence, xW identity max1.34e-14, independentW4unpack. All65536SwiGLU LUT entries correctly rounded with Decimal80 and independently certified using mpmath100. All9realsteps HMX eachstage<=1FP16ULP+minnormal; middlequantization, nativeDown and finalresidual exact versus independent arithmetic. Conditional fullblock versus scalarR4 after sameHMXstate max1LSB,mincosine0.999773 passes2/.999. Known idealR3 failure remains; this is not fullmodel quality.

PreR4 live data exact between arms; decode dead padding excluded explicitly, prefill allrows checked. K/V prefill carrier hashes exact and persistent cache metadata advances correctly every invocation. Intra-arm everytimedoutput matches frozen auditedhash. Otherrecipes and baselines unchanged.

Physical: requested/granted8MiB, existing peak plan6682752bytes; phase-dead HMXactivation768KiB and two1MiB expansion arenas reused, no VTCM allocation growth. No timed intermediateDDR/spill/audit, oneFastRPC perlayerstep and oneHMXowner; nativeW4 Down unchangedformat. On device the fullmatrix72MiB is never allocated or fetched; denseH512 512KiB constant materialization and H12 padded2KiB are included. Prefill5/decode2R4 HMXcommands. Host-DSP boundary=Hostwall-DSPinvocation perrecord.

Retained recovery: export.log records two CPU sigmoid/FP16-midpoint discrepancies, fixed by correctly rounded independent highprecision LUT; original partialexport resumed only after hash checks. audit_a0.log records missing PythonPath import; successfulrawcapture revalidated. numerical_gate.log records checker rowmajor/native layout mixup and comparison of nonlive decode padding; corrected checker reuses unchanged rawcaptures. No native arithmetic bugs, newthresholds, oldhash replacement or discarded failure.

Next discussion: vectorize/fuse the two layout conversions and produce the rotation input on Gate/Up readiness; this experiment does not authorize a new optimization campaign. No baseline promotion. Device PPL and R4 E2E token/s: N/A (single layer only).
## prefill repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.0 (1.12%)|17.2 (0.05%)|N/A|
|Input RMSNorm|N/A|N/A|21.0 (1.39%)|20.2 (0.06%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|249.1 (16.40%)|249.4 (0.78%)|N/A|
|QK–Softmax–AV|N/A|N/A|117.2 (7.72%)|117.7 (0.37%)|N/A|
|O projection|N/A|N/A|43.2 (2.85%)|43.8 (0.14%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|23.6 (1.55%)|23.8 (0.07%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|485.9 (32.00%)|30585.5 (95.88%)|N/A|
|Down|N/A|N/A|116.7 (7.68%)|115.0 (0.36%)|N/A|
|Final residual|N/A|N/A|6.5 (0.43%)|6.6 (0.02%)|N/A|
|KV carrier conversion|N/A|N/A|4.4 (0.29%)|4.8 (0.01%)|N/A|
|KV append DMA|N/A|N/A|7.1 (0.47%)|7.2 (0.02%)|N/A|
|Block orchestration|N/A|N/A|1.6 (0.10%)|1.5 (0.00%)|N/A|
|Layer bookkeeping|N/A|N/A|0.8 (0.05%)|0.8 (0.00%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|1.4 (0.09%)|1.5 (0.00%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|74.0 (4.87%)|74.1 (0.23%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|348.9 (22.98%)|630.6 (1.98%)|N/A|
|完整 Host wall|N/A|N/A|1518.6 (100.00%)|31899.6 (100.00%)|N/A|

## decode repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.8 (1.88%)|17.8 (1.13%)|N/A|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.34%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|90.5 (9.56%)|92.1 (5.81%)|N/A|
|QK–Softmax–AV|N/A|N/A|65.9 (6.97%)|65.6 (4.14%)|N/A|
|O projection|N/A|N/A|42.9 (4.53%)|43.1 (2.72%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|7.1 (0.75%)|7.0 (0.44%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|225.4 (23.82%)|790.8 (49.93%)|N/A|
|Down|N/A|N/A|116.0 (12.26%)|113.9 (7.19%)|N/A|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.09%)|N/A|
|KV carrier conversion|N/A|N/A|10.1 (1.07%)|10.1 (0.64%)|N/A|
|KV append DMA|N/A|N/A|3.0 (0.32%)|3.0 (0.19%)|N/A|
|Block orchestration|N/A|N/A|1.0 (0.11%)|1.0 (0.06%)|N/A|
|Layer bookkeeping|N/A|N/A|0.6 (0.06%)|0.6 (0.04%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.03%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|70.5 (7.45%)|70.5 (4.45%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|288.3 (30.46%)|361.1 (22.80%)|N/A|
|完整 Host wall|N/A|N/A|946.4 (100.00%)|1583.9 (100.00%)|N/A|


## Complete normalized numeric counters

Counts and bytes in native units, *_ticks in qtimer ticks (19.2ticks/us), *_ns in ns. All numeric telemetry retained below, including nested per-layer records. Independent per-field medians do not sum; engine work, DMA and waits can overlap. Legacy unused reference fields are telemetry, not correctness authority; numerical_gate.json is authoritative for implementation comparisons.

### repeat1 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4840.500000|583003.500000|+11944.283%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2415.000000|2402.500000|-0.518%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|122.500000|114.500000|-6.531%|
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
|dense_r3_total_finish_ticks|2293.000000|2291.500000|-0.065%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|372.000000|376.000000|+1.075%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5786.500000|5862.000000|+1.305%|
|dense_r3_total_prepare_ticks|298.000000|307.500000|+3.188%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|344707.500000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|5.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|229127.500000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|879.500000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|8288.500000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2411.000000|2401.500000|-0.394%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|145.000000|142.500000|-1.724%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4522.000000|4482.500000|-0.874%|
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
|hmx_command_count|47.000000|52.000000|+10.638%|
|hmx_compute_ticks|4508.500000|5295.000000|+17.445%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1587760.000000|32107317.500000|+1922.177%|
|input_norm_ticks|428.000000|421.500000|-1.519%|
|input_stage_ticks|143.000000|125.000000|-12.587%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|24132.000000|602221.500000|+2395.531%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|52.500000|50.500000|-3.810%|
|ledger_named_ticks|24132.000000|602221.500000|+2395.531%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|166.500000|163.000000|-2.102%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|921.500000|910.500000|-1.194%|
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
|output_stage_ticks|54.000000|55.000000|+1.852%|
|post_attention_norm_ticks|6.000000|5.000000|-16.667%|
|post_attention_residual_ticks|499.000000|491.500000|-1.503%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|1.000000|1.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|511.000000|510.500000|-0.098%|
|projection_pack_ticks|21.000000|21.000000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|qkv_projection_ticks|5034.500000|5052.500000|+0.358%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|1138.500000|1107.000000|-2.767%|
|runtime_teardown_ticks|892.000000|905.000000|+1.457%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|144.000000|158.500000|+10.069%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|106.000000|103.000000|-2.830%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4840.500000|583003.500000|+11944.283%|
|slice_layer_0.attention_ticks|2415.000000|2402.500000|-0.518%|
|slice_layer_0.block_orchestration_ticks|122.500000|114.500000|-6.531%|
|slice_layer_0.cache_append_dma_ticks|144.000000|158.500000|+10.069%|
|slice_layer_0.cache_append_pack_ticks|106.000000|103.000000|-2.830%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2411.000000|2401.500000|-0.394%|
|slice_layer_0.final_residual_ticks|145.000000|142.500000|-1.724%|
|slice_layer_0.gate_up_ticks|4522.000000|4482.500000|-0.874%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|428.000000|421.500000|-1.519%|
|slice_layer_0.input_stage_ticks|143.000000|125.000000|-12.587%|
|slice_layer_0.layer_bookkeeping_ticks|52.500000|50.500000|-3.810%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|21971.500000|600057.000000|+2631.070%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|166.500000|163.000000|-2.102%|
|slice_layer_0.o_projection_ticks|921.500000|910.500000|-1.194%|
|slice_layer_0.post_attention_norm_ticks|6.000000|5.000000|-16.667%|
|slice_layer_0.post_attention_residual_ticks|499.000000|491.500000|-1.503%|
|slice_layer_0.qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|slice_layer_0.qkv_projection_ticks|5034.500000|5052.500000|+0.358%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|100.500000|102.500000|+1.990%|
|total_ticks|23051.000000|601114.500000|+2507.759%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|430.000000|422.000000|-1.860%|
|u8_attention_av_requant_ticks|1170.500000|1171.500000|+0.085%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1346.000000|1432.000000|+6.389%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|418.000000|423.500000|+1.316%|
|u8_attention_qk_norm_rope_ticks|8748.500000|8841.500000|+1.063%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|6075.500000|6107.000000|+0.518%|
|u8_attention_v_pack_ticks|3058.500000|2997.500000|-1.994%|
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
|u8_cache_native_append_update_ticks|246.500000|247.500000|+0.406%|
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
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|-100.000%|
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
|w4u8_final_residual_main_work_ticks|67.000000|82.000000|+22.388%|
|w4u8_final_residual_pool_wait_ticks|27.500000|10.500000|-61.818%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|382.500000|376.000000|-1.699%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|359.500000|354.500000|-1.391%|
|w4u8_input_norm_pool_wait_ticks|18.500000|17.500000|-5.405%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1639.500000|1612.000000|-1.677%|
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
|w4u8_post_residual_main_work_ticks|429.500000|422.000000|-1.746%|
|w4u8_post_residual_pool_wait_ticks|17.500000|23.000000|+31.429%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1935.500000|1908.000000|-1.421%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1464.000000|1452.500000|-0.786%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|393.000000|395.000000|+0.509%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|982.500000|954.000000|-2.901%|
|w4u8_qkv_ring_pipeline_ticks|2010.000000|2013.500000|+0.174%|
|w4u8_qkv_ring_pool_wait_ticks|216.000000|233.500000|+8.102%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|6.500000|7.000000|+7.692%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8254.000000|8192.000000|-0.751%|
|w4u8_qkvo_prefetch_wait_ticks|1464.000000|1454.000000|-0.683%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8980.500000|8936.000000|-0.496%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat1 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|11044.937500|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1272.812500|1268.000000|-0.378%|
|attention_unattributed_ticks|580.312500|581.437500|+0.194%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|22.437500|21.562500|-3.900%|
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
|dense_r3_total_finish_ticks|50.875000|50.750000|-0.246%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|46.500000|48.062500|+3.360%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|734.062500|727.562500|-0.885%|
|dense_r3_total_prepare_ticks|24.875000|19.812500|-20.352%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|6406.125000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|3580.250000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|86.250000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|958.250000|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2219.812500|2219.812500|+0.000%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.687500|28.937500|-2.526%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4281.687500|4173.000000|-2.538%|
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
|hmx_compute_ticks|4352.062500|4450.437500|+2.260%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|957747.250000|1608561.250000|+67.953%|
|input_norm_ticks|102.375000|102.437500|+0.061%|
|input_stage_ticks|136.250000|136.187500|-0.046%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12583.125000|23561.687500|+87.248%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|11.812500|11.937500|+1.058%|
|ledger_named_ticks|12583.125000|23561.687500|+87.248%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|141.000000|141.000000|+0.000%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|830.375000|828.937500|-0.173%|
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
|output_stage_ticks|63.500000|64.375000|+1.378%|
|post_attention_norm_ticks|0.625000|0.625000|+0.000%|
|post_attention_residual_ticks|137.812500|137.812500|+0.000%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|5.500000|5.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|377.062500|370.437500|-1.757%|
|projection_pack_ticks|3.062500|3.125000|+2.041%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.375000|0.250000|-33.333%|
|qkv_projection_ticks|1761.375000|1761.125000|-0.014%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|716.937500|716.187500|-0.105%|
|runtime_teardown_ticks|638.875000|638.750000|-0.020%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.062500|57.750000|+1.205%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|200.500000|199.812500|-0.343%|
|scan_cache_stage_ticks|340.937500|339.062500|-0.550%|
|scan_dynamic_attention_ticks|1267.562500|1262.812500|-0.375%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|11044.937500|N/A zero denominator|
|slice_layer_0.attention_ticks|1272.812500|1268.000000|-0.378%|
|slice_layer_0.block_orchestration_ticks|22.437500|21.562500|-3.900%|
|slice_layer_0.cache_append_dma_ticks|57.062500|57.750000|+1.205%|
|slice_layer_0.cache_append_pack_ticks|200.500000|199.812500|-0.343%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2219.812500|2219.812500|+0.000%|
|slice_layer_0.final_residual_ticks|29.687500|28.937500|-2.526%|
|slice_layer_0.gate_up_ticks|4281.687500|4173.000000|-2.538%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.375000|102.437500|+0.061%|
|slice_layer_0.input_stage_ticks|136.250000|136.187500|-0.046%|
|slice_layer_0.layer_bookkeeping_ticks|11.812500|11.937500|+1.058%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11160.250000|22131.812500|+98.309%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|141.000000|141.000000|+0.000%|
|slice_layer_0.o_projection_ticks|830.375000|828.937500|-0.173%|
|slice_layer_0.post_attention_norm_ticks|0.625000|0.625000|+0.000%|
|slice_layer_0.post_attention_residual_ticks|137.812500|137.812500|+0.000%|
|slice_layer_0.qk_norm_rope_ticks|0.375000|0.250000|-33.333%|
|slice_layer_0.qkv_projection_ticks|1761.375000|1761.125000|-0.014%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.937500|7.812500|-1.575%|
|total_ticks|11867.187500|22844.687500|+92.503%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|114.437500|114.375000|-0.055%|
|u8_attention_av_requant_ticks|99.750000|101.312500|+1.566%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.875000|13.625000|-1.802%|
|u8_attention_pipeline_wait_ticks|44.812500|43.062500|-3.905%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|94.125000|93.625000|-0.531%|
|u8_attention_qk_norm_rope_ticks|856.000000|847.500000|-0.993%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|243.375000|241.500000|-0.770%|
|u8_attention_v_pack_ticks|84.250000|83.625000|-0.742%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.750000|26.375000|-1.402%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|253.812500|253.625000|-0.074%|
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
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|-100.000%|
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
|w4u8_gate_up_swiglu_consume_count|6.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|84.875000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3440.312500|0.000000|-100.000%|
|w4u8_gate_up_swiglu_worker_ticks|534.562500|0.000000|-100.000%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|347.125000|347.500000|+0.108%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|196.062500|196.562500|+0.255%|
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
|w4u8_qkv_ring_dma_wait_ticks|1403.875000|1394.187500|-0.690%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|351.625000|352.812500|+0.338%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|916.500000|921.875000|+0.586%|
|w4u8_qkv_ring_pipeline_ticks|1618.250000|1619.937500|+0.104%|
|w4u8_qkv_ring_pool_wait_ticks|2.625000|2.687500|+2.381%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.375000|4.187500|-4.286%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8102.062500|8085.875000|-0.200%|
|w4u8_qkvo_prefetch_wait_ticks|1404.500000|1394.750000|-0.694%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8572.500000|8583.937500|+0.133%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4842.850000|582858.200000|+11935.438%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2254.950000|2256.000000|+0.047%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|30.850000|29.600000|-4.052%|
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
|dense_r3_total_finish_ticks|2267.050000|2265.500000|-0.068%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|358.350000|358.900000|+0.153%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5681.650000|5669.700000|-0.210%|
|dense_r3_total_prepare_ticks|254.950000|254.800000|-0.059%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|344642.750000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|5.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|229124.300000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|880.350000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|8187.950000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2245.100000|2234.200000|-0.486%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|126.600000|127.000000|+0.316%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4461.800000|4457.350000|-0.100%|
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
|hmx_command_count|47.000000|52.000000|+10.638%|
|hmx_compute_ticks|4222.250000|5016.100000|+18.802%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1518570.300000|31899606.750000|+2000.634%|
|input_norm_ticks|381.750000|398.000000|+4.257%|
|input_stage_ticks|131.950000|130.950000|-0.758%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|22425.400000|600456.050000|+2577.571%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|15.750000|16.050000|+1.905%|
|ledger_named_ticks|22425.400000|600456.050000|+2577.571%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|143.250000|140.350000|-2.024%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|828.750000|832.000000|+0.392%|
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
|output_stage_ticks|55.250000|59.550000|+7.783%|
|post_attention_norm_ticks|1.050000|1.200000|+14.286%|
|post_attention_residual_ticks|453.100000|452.650000|-0.099%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|41.500000|41.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|497.400000|504.050000|+1.337%|
|projection_pack_ticks|4.700000|4.550000|-3.191%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.650000|0.550000|-15.385%|
|qkv_projection_ticks|4780.800000|4785.000000|+0.088%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|756.600000|757.150000|+0.073%|
|runtime_teardown_ticks|666.100000|673.400000|+1.096%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|137.450000|138.100000|+0.473%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|85.650000|86.750000|+1.284%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4842.850000|582858.200000|+11935.438%|
|slice_layer_0.attention_ticks|2254.950000|2256.000000|+0.047%|
|slice_layer_0.block_orchestration_ticks|30.850000|29.600000|-4.052%|
|slice_layer_0.cache_append_dma_ticks|137.450000|138.100000|+0.473%|
|slice_layer_0.cache_append_pack_ticks|85.650000|86.750000|+1.284%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2245.100000|2234.200000|-0.486%|
|slice_layer_0.final_residual_ticks|126.600000|127.000000|+0.316%|
|slice_layer_0.gate_up_ticks|4461.800000|4457.350000|-0.100%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|381.750000|398.000000|+4.257%|
|slice_layer_0.input_stage_ticks|131.950000|130.950000|-0.758%|
|slice_layer_0.layer_bookkeeping_ticks|15.750000|16.050000|+1.905%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|20919.700000|598936.550000|+2763.026%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|143.250000|140.350000|-2.024%|
|slice_layer_0.o_projection_ticks|828.750000|832.000000|+0.392%|
|slice_layer_0.post_attention_norm_ticks|1.050000|1.200000|+14.286%|
|slice_layer_0.post_attention_residual_ticks|453.100000|452.650000|-0.099%|
|slice_layer_0.qk_norm_rope_ticks|0.650000|0.550000|-15.385%|
|slice_layer_0.qkv_projection_ticks|4780.800000|4785.000000|+0.088%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|27.100000|27.100000|+0.000%|
|total_ticks|21670.150000|599694.700000|+2667.377%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|425.600000|424.450000|-0.270%|
|u8_attention_av_requant_ticks|1174.800000|1176.200000|+0.119%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1186.650000|1175.600000|-0.931%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|421.500000|420.700000|-0.190%|
|u8_attention_qk_norm_rope_ticks|8560.400000|8550.750000|-0.113%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|5963.700000|5966.650000|+0.049%|
|u8_attention_v_pack_ticks|2827.250000|2829.550000|+0.081%|
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
|u8_cache_native_append_update_ticks|218.750000|220.150000|+0.640%|
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
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|-100.000%|
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
|w4u8_final_residual_main_work_ticks|76.400000|77.450000|+1.374%|
|w4u8_final_residual_pool_wait_ticks|15.350000|14.800000|-3.583%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|370.700000|376.950000|+1.686%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|322.850000|322.050000|-0.248%|
|w4u8_input_norm_pool_wait_ticks|23.400000|42.800000|+82.906%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1468.400000|1521.350000|+3.606%|
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
|w4u8_post_residual_main_work_ticks|361.000000|361.600000|+0.166%|
|w4u8_post_residual_pool_wait_ticks|56.050000|53.600000|-4.371%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1794.050000|1784.450000|-0.535%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1403.350000|1419.200000|+1.129%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|366.400000|368.500000|+0.573%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|937.150000|939.350000|+0.235%|
|w4u8_qkv_ring_pipeline_ticks|1871.950000|1876.150000|+0.224%|
|w4u8_qkv_ring_pool_wait_ticks|230.450000|225.100000|-2.322%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.200000|4.200000|+0.000%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|7916.350000|7894.650000|-0.274%|
|w4u8_qkvo_prefetch_wait_ticks|1404.000000|1419.600000|+1.111%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8647.400000|8609.500000|-0.438%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|11044.193750|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1261.368750|1259.625000|-0.138%|
|attention_unattributed_ticks|572.331250|573.181250|+0.149%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|19.325000|19.368750|+0.226%|
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
|dense_r3_total_finish_ticks|50.806250|50.781250|-0.049%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|47.875000|48.500000|+1.305%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|720.206250|707.468750|-1.769%|
|dense_r3_total_prepare_ticks|20.331250|19.806250|-2.582%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|6408.075000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|3580.287500|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|88.662500|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|962.762500|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2221.406250|2229.206250|+0.351%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.312500|28.637500|-2.303%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4298.581250|4179.656250|-2.767%|
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
|hmx_compute_ticks|4034.975000|4167.368750|+3.281%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|946401.337500|1583942.075000|+67.365%|
|input_norm_ticks|102.375000|102.293750|-0.079%|
|input_stage_ticks|134.637500|135.362500|+0.538%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12617.987500|23553.150000|+86.663%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|11.700000|11.737500|+0.321%|
|ledger_named_ticks|12617.987500|23553.150000|+86.663%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|141.475000|139.856250|-1.144%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|826.556250|830.881250|+0.523%|
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
|output_stage_ticks|63.443750|65.412500|+3.103%|
|post_attention_norm_ticks|0.612500|0.606250|-1.020%|
|post_attention_residual_ticks|133.431250|134.331250|+0.675%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|46.000000|46.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|363.818750|365.631250|+0.498%|
|projection_pack_ticks|3.025000|3.062500|+1.240%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.275000|0.262500|-4.545%|
|qkv_projection_ticks|1752.600000|1766.381250|+0.786%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|715.343750|716.487500|+0.160%|
|runtime_teardown_ticks|639.175000|639.912500|+0.115%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.237500|57.056250|-0.317%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|194.231250|194.693750|+0.238%|
|scan_cache_stage_ticks|346.356250|346.443750|+0.025%|
|scan_dynamic_attention_ticks|1257.368750|1255.687500|-0.134%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|11044.193750|N/A zero denominator|
|slice_layer_0.attention_ticks|1261.368750|1259.625000|-0.138%|
|slice_layer_0.block_orchestration_ticks|19.325000|19.368750|+0.226%|
|slice_layer_0.cache_append_dma_ticks|57.237500|57.056250|-0.317%|
|slice_layer_0.cache_append_pack_ticks|194.231250|194.693750|+0.238%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2221.406250|2229.206250|+0.351%|
|slice_layer_0.final_residual_ticks|29.312500|28.637500|-2.303%|
|slice_layer_0.gate_up_ticks|4298.581250|4179.656250|-2.767%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.375000|102.293750|-0.079%|
|slice_layer_0.input_stage_ticks|134.637500|135.362500|+0.538%|
|slice_layer_0.layer_bookkeeping_ticks|11.700000|11.737500|+0.321%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11191.862500|22121.650000|+97.658%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|141.475000|139.856250|-1.144%|
|slice_layer_0.o_projection_ticks|826.556250|830.881250|+0.523%|
|slice_layer_0.post_attention_norm_ticks|0.612500|0.606250|-1.020%|
|slice_layer_0.post_attention_residual_ticks|133.431250|134.331250|+0.675%|
|slice_layer_0.qk_norm_rope_ticks|0.275000|0.262500|-4.545%|
|slice_layer_0.qkv_projection_ticks|1752.600000|1766.381250|+0.786%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.956250|7.956250|+0.000%|
|total_ticks|11903.275000|22835.950000|+91.846%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|114.981250|115.656250|+0.587%|
|u8_attention_av_requant_ticks|99.025000|98.775000|-0.252%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.200000|13.231250|+0.237%|
|u8_attention_pipeline_wait_ticks|43.750000|43.656250|-0.214%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|94.856250|93.812500|-1.100%|
|u8_attention_qk_norm_rope_ticks|839.987500|829.450000|-1.254%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|238.281250|238.725000|+0.186%|
|u8_attention_v_pack_ticks|82.500000|82.556250|+0.068%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|25.650000|25.962500|+1.218%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|249.537500|249.881250|+0.138%|
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
|w4u8_decode_direct_n_gate_up_swiglu_stream|1.000000|0.000000|-100.000%|
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
|w4u8_gate_up_swiglu_consume_count|6.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_join_wait_ticks|86.650000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|0.000000|-100.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3492.318750|0.000000|-100.000%|
|w4u8_gate_up_swiglu_worker_ticks|447.918750|0.000000|-100.000%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|355.956250|348.537500|-2.084%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|215.225000|207.393750|-3.639%|
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
|w4u8_qkv_ring_dma_wait_ticks|1398.681250|1410.343750|+0.834%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|353.137500|354.306250|+0.331%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|930.175000|923.668750|-0.699%|
|w4u8_qkv_ring_pipeline_ticks|1610.731250|1622.862500|+0.753%|
|w4u8_qkv_ring_pool_wait_ticks|2.662500|2.693750|+1.174%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|5.250000|5.243750|-0.119%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8110.256250|8109.856250|-0.005%|
|w4u8_qkvo_prefetch_wait_ticks|1399.368750|1411.087500|+0.837%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8636.062500|8593.962500|-0.487%|
|wide_score_mode|4.000000|4.000000|+0.000%|
