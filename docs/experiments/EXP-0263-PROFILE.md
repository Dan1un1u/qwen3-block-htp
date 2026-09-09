# EXP0263 R4 parallel prefill bottleneck optimization

Branch codex/exp-0263-w4u8-r4-parallel-prefill; reporting source 7c80d9407649e48134dfd4fc8daac39dba3d410a; native runtime 05cbf364085d009c5c5acea6d7f45d3e7008612d ABI126. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0263; candidate artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0261/r4.

Two paired arms: current optimized R3 OPT2 + wideNR64/nativeW4, and the same path plus full6144 R4 before middle A8. Five short and ten alternating formal pairs, repeat1/repeat10; one prefillM64 and eight consecutive teacher-input M1 steps per repetition. KV starts empty and is computed/appended on device. No frozen-snapshot decode, discarded warmups, selection, extra rounds or fullmodel extrapolation.

Primary medians of round means; paired effect is median within-round ratio, 10000bootstrap seed261. Decode latency averages the eight positions. Additive modules use the identical two median-Host-ranked rounds for all fields. All2970RPCs and layer ledgers independently reconstructed.

|Scope|Control us|R4 us|Paired time change|Ratio95% CI|
|---|---:|---:|---:|---|
|r1_prefill_ns|1704.479|1909.297|+12.10%|[1.099493, 1.209549]|
|r1_decode_ns|1006.986|1036.598|+5.28%|[0.963036, 1.082602]|
|r10_prefill_ns|1528.893|1703.398|+11.87%|[1.076436, 1.147318]|
|r10_decode_ns|955.976|992.867|+3.46%|[1.013416, 1.043435]|

Speed eligible: False; stop before fullmodel: True. Eligibility requires CI upper<=1.10. Inconclusive intervals crossing1.10 also block escalation; this does not claim a confirmed stable>10percent slowdown. Fixed rounds are complete, with no optional additional sampling.

R4 is H12 tensor H512, full6144 coverage. Matrix entries are dense parity/Paley signs; no FWHT/butterfly. Stage1 normalizedH512 operates on12groups, stage2 normalizedH12 mixes allgroups. Intermediate and final HMX casts use declared FP16 normalization constants and rounding. The full transform preserves unquantized xW; deployed Down is freshly RTN quantized from original W R with oneFP32scale/output, signed[-7,7]. OtherC64weights unchanged, no group scales/W4toS8. Middle static range is unchanged, so this fixture establishes cost and implementation correctness, not calibrated accuracy or PPL improvement.

Prepare includes unquantized FP16 SwiGLU LUT gather/packing and constant sign matrix materialization. Layout transposes stage1 for the second GEMM; finish reorganizes/quantizes into native-U8 Down input. Native layout uses exact HVX deal/scatter; finish uses gather, unchanged quantization and native stores. Decode FP16 SwiGLU is produced into the HMX input on Gate/Up readiness. Prefill uses two 256KiB input/output slots; next layout and previous finish are scheduled during the current HMX command. Matmul_ticks is exposed submit/wait cost, not total engine compute. Pipeline HVX ticks report scheduled overlap work, not fully hidden time. Dense factor GEMMs alone cannot stand in for complete cost.

|R4 phase, us|M64 prefill|M1 decode|
|---|---:|---:|
|dense_r4_prepare_ticks|153.794|9.723|
|dense_r4_matmul_ticks|25.737|4.320|
|dense_r4_layout_ticks|62.120|0.972|
|dense_r4_finish_ticks|200.690|16.318|

Numerical: EXP0261 sealed original-shard/export evidence is inherited, with frozen local and remote packages reverified. The inherited export proves H12/H512 orthogonality and explicit dense-factor equivalence, xW identity max1.34e-14, independentW4unpack. The inherited65536SwiGLU LUT entries correctly rounded with Decimal80 and independently certified using mpmath100. All9realsteps HMX eachstage<=1FP16ULP+minnormal; middlequantization, nativeDown and finalresidual exact versus independent arithmetic. Conditional fullblock versus scalarR4 after sameHMXstate max1LSB,mincosine0.999773 passes2/.999. Known idealR3 failure remains; this is not fullmodel quality.

PreR4 live data exact between arms; decode dead padding excluded explicitly, prefill allrows checked. K/V prefill carrier hashes exact and persistent cache metadata advances correctly every invocation. Intra-arm everytimedoutput matches frozen auditedhash. Otherrecipes and baselines unchanged.

Physical: requested/granted8MiB, existing peak plan6682752bytes; phase-dead HMXactivation768KiB and two1MiB expansion arenas reused, no VTCM allocation growth. No timed intermediateDDR/spill/audit, oneFastRPC perlayerstep and oneHMXowner; nativeW4 Down unchangedformat. On device the fullmatrix72MiB is never allocated or fetched; denseH512 512KiB constant materialization and H12 padded2KiB are included. Prefill9/decode2R4 HMXcommands; per-row arithmetic and total tile pairs unchanged. Host-DSP boundary=Hostwall-DSPinvocation perrecord.

Retained research branch: OPT4 two-worker prefill Up readiness streaming is correct but its154us join tail cancels overlap. Bounded smoke selected OPT3 before all short/formal rounds, as pinned in route.json. No native mismatch or collection repair occurred in this experiment. All five original/parallel/stream/current captures exact against sealed EXP0262; independently recomputed R4 arithmetic and inherited EXP0261 scalar oracle remain valid. No numerical relaxation or weight/hash replacement.

Bounded native-layout and pipeline optimization completed. OPT3 splits FP16 LUT preparation over192tiles into main+two existing workers and output conversion over12groups into the same three contexts, each with a private256byte scratch from existing768bytes. All workloads joined before buffer reuse. Decode OPT2 schedule unchanged. Full end-to-end cost includes dispatch/join and aggregate work is overlapping telemetry, never added to the wall ledger. No fullmodel or baseline promotion. Device PPL and R4 E2E token/s: N/A (single layer only).

Historical EXP0262 comparison (separate campaigns, not paired inference):

|Mode|Previous R4 us|Optimized R4 us|Observed speedup|
|---|---:|---:|---:|
|prefill|2223.518|1703.398|1.305x|
|decode|1006.825|992.867|1.014x|

Exact native comparisons: 145 live tensor/cache files over original, vector and pipeline captures. Both parent491file and original371file seals independently reverified; all144files/package verified locally and remotely. Exhaustive live-address mapping and double-buffer range nonoverlap independently checked. No current model export or calibration.
## prefill repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.7 (1.16%)|17.8 (1.05%)|N/A|
|Input RMSNorm|N/A|N/A|20.2 (1.32%)|20.1 (1.18%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|249.5 (16.32%)|248.9 (14.61%)|N/A|
|QK–Softmax–AV|N/A|N/A|119.0 (7.78%)|118.1 (6.93%)|N/A|
|O projection|N/A|N/A|43.9 (2.87%)|42.9 (2.52%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|23.5 (1.54%)|23.4 (1.38%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|483.1 (31.60%)|674.9 (39.62%)|N/A|
|Down|N/A|N/A|117.0 (7.65%)|116.5 (6.84%)|N/A|
|Final residual|N/A|N/A|6.5 (0.42%)|6.7 (0.39%)|N/A|
|KV carrier conversion|N/A|N/A|4.6 (0.30%)|4.3 (0.25%)|N/A|
|KV append DMA|N/A|N/A|7.3 (0.48%)|7.4 (0.43%)|N/A|
|Block orchestration|N/A|N/A|1.8 (0.12%)|1.5 (0.09%)|N/A|
|Layer bookkeeping|N/A|N/A|0.9 (0.06%)|0.8 (0.05%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|1.7 (0.11%)|1.7 (0.10%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|74.2 (4.85%)|74.2 (4.36%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|358.1 (23.42%)|344.2 (20.21%)|N/A|
|完整 Host wall|N/A|N/A|1528.9 (100.00%)|1703.4 (100.00%)|N/A|

## decode repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.7 (1.85%)|18.0 (1.82%)|N/A|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.54%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|91.2 (9.54%)|92.2 (9.28%)|N/A|
|QK–Softmax–AV|N/A|N/A|65.7 (6.87%)|65.5 (6.60%)|N/A|
|O projection|N/A|N/A|42.7 (4.46%)|42.8 (4.31%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|7.0 (0.73%)|6.9 (0.70%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|221.8 (23.20%)|260.4 (26.23%)|N/A|
|Down|N/A|N/A|115.0 (12.03%)|116.2 (11.70%)|N/A|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.15%)|N/A|
|KV carrier conversion|N/A|N/A|10.2 (1.06%)|10.2 (1.03%)|N/A|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.0 (0.30%)|N/A|
|Block orchestration|N/A|N/A|1.0 (0.10%)|1.0 (0.10%)|N/A|
|Layer bookkeeping|N/A|N/A|0.6 (0.07%)|0.6 (0.06%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.04%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|70.6 (7.38%)|70.5 (7.11%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|302.4 (31.64%)|298.2 (30.04%)|N/A|
|完整 Host wall|N/A|N/A|956.0 (100.00%)|992.9 (100.00%)|N/A|


## Complete normalized numeric counters

Counts and bytes in native units, *_ticks in qtimer ticks (19.2ticks/us), *_ns in ns. All numeric telemetry retained below, including nested per-layer records. Independent per-field medians do not sum; engine work, DMA and waits can overlap. Legacy unused reference fields are telemetry, not correctness authority; numerical_gate.json is authoritative for implementation comparisons.

### repeat1 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4855.500000|8623.000000|+77.592%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2372.000000|2374.500000|+0.105%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|118.500000|105.500000|-10.970%|
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
|dense_r3_total_finish_ticks|2291.000000|2291.000000|+0.000%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|385.000000|381.000000|-1.039%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5903.000000|5922.000000|+0.322%|
|dense_r3_total_prepare_ticks|296.000000|294.500000|-0.507%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|3823.000000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1201.000000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|500.500000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|3.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|9.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|96.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|301.000000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|192.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|18687.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|4402.500000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|3093.000000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2366.500000|2395.000000|+1.204%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|141.000000|139.500000|-1.064%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4505.500000|4529.500000|+0.533%|
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
|hmx_compute_ticks|4632.000000|5342.500000|+15.339%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1704479.000000|1909297.000000|+12.016%|
|input_norm_ticks|435.500000|434.000000|-0.344%|
|input_stage_ticks|148.000000|144.500000|-2.365%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|24157.000000|27873.000000|+15.383%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|51.500000|53.000000|+2.913%|
|ledger_named_ticks|24157.000000|27873.000000|+15.383%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|183.000000|163.000000|-10.929%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|923.000000|917.500000|-0.596%|
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
|output_stage_ticks|57.500000|57.500000|+0.000%|
|post_attention_norm_ticks|3.500000|3.500000|+0.000%|
|post_attention_residual_ticks|495.500000|502.500000|+1.413%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|1.000000|1.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|524.500000|520.000000|-0.858%|
|projection_pack_ticks|21.000000|21.000000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|qkv_projection_ticks|5048.500000|5038.500000|-0.198%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|1110.500000|1121.000000|+0.946%|
|runtime_teardown_ticks|939.000000|923.000000|-1.704%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|146.500000|161.500000|+10.239%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|102.000000|101.000000|-0.980%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4855.500000|8623.000000|+77.592%|
|slice_layer_0.attention_ticks|2372.000000|2374.500000|+0.105%|
|slice_layer_0.block_orchestration_ticks|118.500000|105.500000|-10.970%|
|slice_layer_0.cache_append_dma_ticks|146.500000|161.500000|+10.239%|
|slice_layer_0.cache_append_pack_ticks|102.000000|101.000000|-0.980%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2366.500000|2395.000000|+1.204%|
|slice_layer_0.final_residual_ticks|141.000000|139.500000|-1.064%|
|slice_layer_0.gate_up_ticks|4505.500000|4529.500000|+0.533%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|435.500000|434.000000|-0.344%|
|slice_layer_0.input_stage_ticks|148.000000|144.500000|-2.365%|
|slice_layer_0.layer_bookkeeping_ticks|51.500000|53.000000|+2.913%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|21911.500000|25677.500000|+17.187%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|183.000000|163.000000|-10.929%|
|slice_layer_0.o_projection_ticks|923.000000|917.500000|-0.596%|
|slice_layer_0.post_attention_norm_ticks|3.500000|3.500000|+0.000%|
|slice_layer_0.post_attention_residual_ticks|495.500000|502.500000|+1.413%|
|slice_layer_0.qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|slice_layer_0.qkv_projection_ticks|5048.500000|5038.500000|-0.198%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|111.500000|102.500000|-8.072%|
|total_ticks|23015.000000|26780.000000|+16.359%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|435.000000|428.500000|-1.494%|
|u8_attention_av_requant_ticks|1180.500000|1181.500000|+0.085%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1354.500000|1519.500000|+12.182%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|432.500000|424.000000|-1.965%|
|u8_attention_qk_norm_rope_ticks|8877.500000|8881.000000|+0.039%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|6002.500000|6041.000000|+0.641%|
|u8_attention_v_pack_ticks|3069.000000|2978.000000|-2.965%|
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
|u8_cache_native_append_update_ticks|248.000000|257.000000|+3.629%|
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
|w4u8_final_residual_main_work_ticks|69.000000|68.500000|-0.725%|
|w4u8_final_residual_pool_wait_ticks|25.000000|23.500000|-6.000%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|392.500000|382.500000|-2.548%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|365.000000|361.500000|-0.959%|
|w4u8_input_norm_pool_wait_ticks|22.500000|23.000000|+2.222%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1645.500000|1660.500000|+0.912%|
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
|w4u8_post_residual_main_work_ticks|417.000000|411.000000|-1.439%|
|w4u8_post_residual_pool_wait_ticks|20.000000|18.000000|-10.000%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1908.500000|1948.500000|+2.096%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1470.000000|1453.000000|-1.156%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|397.000000|393.500000|-0.882%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|1001.500000|971.500000|-2.996%|
|w4u8_qkv_ring_pipeline_ticks|1998.500000|2003.000000|+0.225%|
|w4u8_qkv_ring_pool_wait_ticks|230.000000|243.500000|+5.870%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.000000|4.000000|+0.000%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8169.000000|8199.500000|+0.373%|
|w4u8_qkvo_prefetch_wait_ticks|1470.000000|1453.500000|-1.122%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8890.000000|8920.000000|+0.337%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat1 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|605.812500|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1280.625000|1273.937500|-0.522%|
|attention_unattributed_ticks|584.750000|582.250000|-0.428%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|19.937500|19.625000|-1.567%|
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
|dense_r3_total_matmul_ticks|51.312500|49.062500|-4.385%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|780.312500|840.937500|+7.769%|
|dense_r3_total_prepare_ticks|24.937500|19.437500|-22.055%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|312.687500|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|19.625000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|81.500000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|3.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|186.062500|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2206.437500|2228.937500|+1.020%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.187500|28.687500|-1.713%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4264.937500|4362.000000|+2.276%|
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
|hmx_compute_ticks|4416.562500|4392.812500|-0.538%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|1006985.625000|1036598.250000|+2.941%|
|input_norm_ticks|102.375000|102.375000|+0.000%|
|input_stage_ticks|136.437500|137.000000|+0.412%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12651.187500|13294.937500|+5.088%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.500000|12.500000|+0.000%|
|ledger_named_ticks|12651.187500|13294.937500|+5.088%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|143.062500|143.437500|+0.262%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|831.125000|828.125000|-0.361%|
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
|output_stage_ticks|64.625000|64.125000|-0.774%|
|post_attention_norm_ticks|0.687500|0.500000|-27.273%|
|post_attention_residual_ticks|135.875000|134.875000|-0.736%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|5.500000|5.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|390.812500|371.750000|-4.878%|
|projection_pack_ticks|2.875000|2.750000|-4.348%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.250000|0.125000|-50.000%|
|qkv_projection_ticks|1774.687500|1755.125000|-1.102%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|719.937500|718.000000|-0.269%|
|runtime_teardown_ticks|637.562500|638.250000|+0.108%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.562500|57.437500|-0.217%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|201.312500|199.875000|-0.714%|
|scan_cache_stage_ticks|347.437500|341.500000|-1.709%|
|scan_dynamic_attention_ticks|1274.625000|1267.812500|-0.534%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|605.812500|N/A zero denominator|
|slice_layer_0.attention_ticks|1280.625000|1273.937500|-0.522%|
|slice_layer_0.block_orchestration_ticks|19.937500|19.625000|-1.567%|
|slice_layer_0.cache_append_dma_ticks|57.562500|57.437500|-0.217%|
|slice_layer_0.cache_append_pack_ticks|201.312500|199.875000|-0.714%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2206.437500|2228.937500|+1.020%|
|slice_layer_0.final_residual_ticks|29.187500|28.687500|-1.713%|
|slice_layer_0.gate_up_ticks|4264.937500|4362.000000|+2.276%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.375000|102.375000|+0.000%|
|slice_layer_0.input_stage_ticks|136.437500|137.000000|+0.412%|
|slice_layer_0.layer_bookkeeping_ticks|12.500000|12.500000|+0.000%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11212.875000|11863.625000|+5.804%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|143.062500|143.437500|+0.262%|
|slice_layer_0.o_projection_ticks|831.125000|828.125000|-0.361%|
|slice_layer_0.post_attention_norm_ticks|0.687500|0.500000|-27.273%|
|slice_layer_0.post_attention_residual_ticks|135.875000|134.875000|-0.736%|
|slice_layer_0.qk_norm_rope_ticks|0.250000|0.125000|-50.000%|
|slice_layer_0.qkv_projection_ticks|1774.687500|1755.125000|-1.102%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.625000|7.812500|+2.459%|
|total_ticks|11927.312500|12577.187500|+5.449%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|114.687500|111.687500|-2.616%|
|u8_attention_av_requant_ticks|100.687500|98.687500|-1.986%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.875000|13.875000|+0.000%|
|u8_attention_pipeline_wait_ticks|46.187500|44.875000|-2.842%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|95.937500|91.812500|-4.300%|
|u8_attention_qk_norm_rope_ticks|906.250000|962.812500|+6.241%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|243.687500|244.062500|+0.154%|
|u8_attention_v_pack_ticks|84.250000|83.750000|-0.593%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.312500|26.750000|+1.663%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|255.562500|254.750000|-0.318%|
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
|w4u8_gate_up_swiglu_join_wait_ticks|87.812500|162.625000|+85.196%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3325.812500|2673.875000|-19.602%|
|w4u8_gate_up_swiglu_worker_ticks|571.937500|1446.125000|+152.847%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|344.937500|345.812500|+0.254%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|196.500000|198.812500|+1.177%|
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
|w4u8_qkv_ring_dma_wait_ticks|1399.250000|1382.437500|-1.202%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|373.687500|360.500000|-3.529%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|923.250000|918.187500|-0.548%|
|w4u8_qkv_ring_pipeline_ticks|1626.812500|1613.625000|-0.811%|
|w4u8_qkv_ring_pool_wait_ticks|2.812500|2.812500|+0.000%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|10.437500|4.375000|-58.084%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8081.500000|8082.437500|+0.012%|
|w4u8_qkvo_prefetch_wait_ticks|1399.750000|1383.125000|-1.188%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8547.875000|8560.812500|+0.151%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4840.700000|8546.000000|+76.545%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2278.750000|2271.950000|-0.298%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|30.550000|29.150000|-4.583%|
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
|dense_r3_total_finish_ticks|2268.050000|2265.650000|-0.106%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|359.300000|358.950000|-0.097%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5662.850000|5699.950000|+0.655%|
|dense_r3_total_prepare_ticks|254.250000|254.500000|+0.098%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|3853.250000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1192.700000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|494.150000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|3.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|9.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|96.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|290.400000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|192.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|18614.500000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|4427.750000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|2952.850000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2233.950000|2223.500000|-0.468%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|125.150000|128.350000|+2.557%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4443.350000|4436.050000|-0.164%|
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
|hmx_compute_ticks|4209.750000|4983.200000|+18.373%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1528893.100000|1703398.450000|+11.414%|
|input_norm_ticks|388.300000|382.650000|-1.455%|
|input_stage_ticks|135.650000|137.250000|+1.180%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|22438.850000|26135.000000|+16.472%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|16.150000|16.050000|-0.619%|
|ledger_named_ticks|22438.850000|26135.000000|+16.472%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|147.850000|148.950000|+0.744%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|832.050000|826.350000|-0.685%|
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
|output_stage_ticks|56.900000|57.200000|+0.527%|
|post_attention_norm_ticks|0.800000|0.800000|+0.000%|
|post_attention_residual_ticks|450.450000|451.100000|+0.144%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|41.500000|41.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|509.450000|510.250000|+0.157%|
|projection_pack_ticks|4.350000|4.400000|+1.149%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.500000|0.550000|+10.000%|
|qkv_projection_ticks|4768.650000|4775.450000|+0.143%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|755.950000|753.400000|-0.337%|
|runtime_teardown_ticks|669.000000|666.950000|-0.306%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|140.500000|140.300000|-0.142%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|83.350000|83.400000|+0.060%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4840.700000|8546.000000|+76.545%|
|slice_layer_0.attention_ticks|2278.750000|2271.950000|-0.298%|
|slice_layer_0.block_orchestration_ticks|30.550000|29.150000|-4.583%|
|slice_layer_0.cache_append_dma_ticks|140.500000|140.300000|-0.142%|
|slice_layer_0.cache_append_pack_ticks|83.350000|83.400000|+0.060%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2233.950000|2223.500000|-0.468%|
|slice_layer_0.final_residual_ticks|125.150000|128.350000|+2.557%|
|slice_layer_0.gate_up_ticks|4443.350000|4436.050000|-0.164%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|388.300000|382.650000|-1.455%|
|slice_layer_0.input_stage_ticks|135.650000|137.250000|+1.180%|
|slice_layer_0.layer_bookkeeping_ticks|16.150000|16.050000|-0.619%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|20926.450000|24620.900000|+17.654%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|147.850000|148.950000|+0.744%|
|slice_layer_0.o_projection_ticks|832.050000|826.350000|-0.685%|
|slice_layer_0.post_attention_norm_ticks|0.800000|0.800000|+0.000%|
|slice_layer_0.post_attention_residual_ticks|450.450000|451.100000|+0.144%|
|slice_layer_0.qk_norm_rope_ticks|0.500000|0.550000|+10.000%|
|slice_layer_0.qkv_projection_ticks|4768.650000|4775.450000|+0.143%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|31.650000|31.700000|+0.158%|
|total_ticks|21683.700000|25379.900000|+17.046%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|437.750000|432.150000|-1.279%|
|u8_attention_av_requant_ticks|1178.950000|1175.850000|-0.263%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1251.050000|1230.850000|-1.615%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|433.150000|439.400000|+1.443%|
|u8_attention_qk_norm_rope_ticks|8544.600000|8581.400000|+0.431%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|5989.500000|5977.600000|-0.199%|
|u8_attention_v_pack_ticks|2832.950000|2839.950000|+0.247%|
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
|u8_cache_native_append_update_ticks|223.150000|222.300000|-0.381%|
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
|w4u8_final_residual_main_work_ticks|76.950000|67.250000|-12.606%|
|w4u8_final_residual_pool_wait_ticks|14.950000|26.450000|+76.923%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|373.350000|385.700000|+3.308%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|324.150000|323.700000|-0.139%|
|w4u8_input_norm_pool_wait_ticks|34.800000|21.350000|-38.649%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1494.050000|1469.650000|-1.633%|
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
|w4u8_post_residual_main_work_ticks|368.900000|360.150000|-2.372%|
|w4u8_post_residual_pool_wait_ticks|48.550000|56.050000|+15.448%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1788.300000|1793.900000|+0.313%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1400.850000|1392.650000|-0.585%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|369.250000|368.600000|-0.176%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|953.850000|932.100000|-2.280%|
|w4u8_qkv_ring_pipeline_ticks|1861.150000|1863.300000|+0.116%|
|w4u8_qkv_ring_pool_wait_ticks|227.050000|244.700000|+7.774%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|3.900000|3.900000|+0.000%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|7886.600000|7883.700000|-0.037%|
|w4u8_qkvo_prefetch_wait_ticks|1401.600000|1393.100000|-0.606%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8603.850000|8579.700000|-0.281%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|606.506250|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1260.462500|1261.675000|+0.096%|
|attention_unattributed_ticks|572.568750|573.056250|+0.085%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|19.056250|19.056250|+0.000%|
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
|dense_r3_total_finish_ticks|50.637500|50.950000|+0.617%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|49.306250|49.750000|+0.900%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|699.331250|718.956250|+2.806%|
|dense_r3_total_prepare_ticks|19.981250|19.418750|-2.815%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|313.312500|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|18.668750|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|82.950000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|3.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|0.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|0.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|186.687500|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2215.118750|2216.631250|+0.068%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|28.812500|28.268750|-1.887%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4284.562500|4354.718750|+1.637%|
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
|hmx_compute_ticks|4012.743750|4050.562500|+0.942%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|955976.287500|992867.212500|+3.859%|
|input_norm_ticks|102.431250|102.337500|-0.092%|
|input_stage_ticks|136.800000|136.618750|-0.132%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12616.275000|13283.693750|+5.290%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.331250|12.200000|-1.064%|
|ledger_named_ticks|12616.275000|13283.693750|+5.290%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|143.900000|144.681250|+0.543%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|827.800000|823.318750|-0.541%|
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
|output_stage_ticks|64.650000|64.568750|-0.126%|
|post_attention_norm_ticks|0.556250|0.550000|-1.124%|
|post_attention_residual_ticks|133.481250|132.375000|-0.829%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|46.000000|46.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|371.043750|371.906250|+0.232%|
|projection_pack_ticks|2.787500|2.737500|-1.794%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.237500|0.212500|-10.526%|
|qkv_projection_ticks|1760.668750|1752.193750|-0.481%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|717.106250|717.668750|+0.078%|
|runtime_teardown_ticks|638.281250|637.887500|-0.062%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.087500|57.068750|-0.033%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|195.187500|194.856250|-0.170%|
|scan_cache_stage_ticks|348.443750|347.137500|-0.375%|
|scan_dynamic_attention_ticks|1256.825000|1257.868750|+0.083%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|606.506250|N/A zero denominator|
|slice_layer_0.attention_ticks|1260.462500|1261.675000|+0.096%|
|slice_layer_0.block_orchestration_ticks|19.056250|19.056250|+0.000%|
|slice_layer_0.cache_append_dma_ticks|57.087500|57.068750|-0.033%|
|slice_layer_0.cache_append_pack_ticks|195.187500|194.856250|-0.170%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2215.118750|2216.631250|+0.068%|
|slice_layer_0.final_residual_ticks|28.812500|28.268750|-1.887%|
|slice_layer_0.gate_up_ticks|4284.562500|4354.718750|+1.637%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.431250|102.337500|-0.092%|
|slice_layer_0.input_stage_ticks|136.800000|136.618750|-0.132%|
|slice_layer_0.layer_bookkeeping_ticks|12.331250|12.200000|-1.064%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11188.525000|11853.831250|+5.946%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|143.900000|144.681250|+0.543%|
|slice_layer_0.o_projection_ticks|827.800000|823.318750|-0.541%|
|slice_layer_0.post_attention_norm_ticks|0.556250|0.550000|-1.124%|
|slice_layer_0.post_attention_residual_ticks|133.481250|132.375000|-0.829%|
|slice_layer_0.qk_norm_rope_ticks|0.237500|0.212500|-10.526%|
|slice_layer_0.qkv_projection_ticks|1760.668750|1752.193750|-0.481%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.800000|7.637500|-2.083%|
|total_ticks|11899.556250|12564.525000|+5.588%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|114.543750|113.750000|-0.693%|
|u8_attention_av_requant_ticks|100.187500|100.143750|-0.044%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.131250|13.131250|+0.000%|
|u8_attention_pipeline_wait_ticks|45.243750|45.600000|+0.787%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|95.243750|94.525000|-0.755%|
|u8_attention_qk_norm_rope_ticks|819.325000|839.068750|+2.410%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|237.087500|236.925000|-0.069%|
|u8_attention_v_pack_ticks|82.606250|82.662500|+0.068%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|25.943750|25.787500|-0.602%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|251.000000|250.393750|-0.242%|
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
|w4u8_gate_up_swiglu_join_wait_ticks|88.387500|164.212500|+85.787%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3482.468750|2841.056250|-18.418%|
|w4u8_gate_up_swiglu_worker_ticks|451.493750|1253.206250|+177.569%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|348.118750|345.606250|-0.722%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|208.300000|205.112500|-1.530%|
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
|w4u8_qkv_ring_dma_wait_ticks|1400.368750|1393.543750|-0.487%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|353.812500|353.137500|-0.191%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|932.981250|927.787500|-0.557%|
|w4u8_qkv_ring_pipeline_ticks|1615.800000|1610.431250|-0.332%|
|w4u8_qkv_ring_pool_wait_ticks|2.806250|2.793750|-0.445%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.025000|4.162500|+3.416%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8105.606250|8078.131250|-0.339%|
|w4u8_qkvo_prefetch_wait_ticks|1401.081250|1394.150000|-0.495%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8616.018750|8582.875000|-0.385%|
|wide_score_mode|4.000000|4.000000|+0.000%|
