# EXP0264 R4 exact prepared constants and fused native output

Branch codex/exp-0264-w4u8-r4-fused-quantization; reporting source 44aede359455b1a2a2a82038669e5d2be2c37fe6; native runtime 59cdd72b6ee880d7b44898c20e35b4cbb9072a66 ABI127. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0264; candidate artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0261/r4.

Two paired arms: current optimized R3 OPT2 + wideNR64/nativeW4, and the same path plus full6144 R4 before middle A8. Five short and ten alternating formal pairs, repeat1/repeat10; one prefillM64 and eight consecutive teacher-input M1 steps per repetition. KV starts empty and is computed/appended on device. No frozen-snapshot decode, discarded warmups, selection, extra rounds or fullmodel extrapolation.

Primary medians of round means; paired effect is median within-round ratio, 10000bootstrap seed261. Decode latency averages the eight positions. Additive modules use the identical two median-Host-ranked rounds for all fields. All2970RPCs and layer ledgers independently reconstructed.

|Scope|Control us|R4 us|Paired time change|Ratio95% CI|
|---|---:|---:|---:|---|
|r1_prefill_ns|1699.531|1772.240|+2.40%|[0.954856, 1.114159]|
|r1_decode_ns|976.989|1027.301|+1.02%|[0.975626, 1.130453]|
|r10_prefill_ns|1516.430|1599.661|+5.17%|[1.039363, 1.087151]|
|r10_decode_ns|954.349|970.135|+1.35%|[1.004984, 1.043051]|

Speed eligible: False; stop before fullmodel: True. Eligibility requires CI upper<=1.10. Inconclusive intervals crossing1.10 also block escalation; this does not claim a confirmed stable>10percent slowdown. Fixed rounds are complete, with no optional additional sampling.

R4 is H12 tensor H512, full6144 coverage. Matrix entries are dense parity/Paley signs; no FWHT/butterfly. Stage1 normalizedH512 operates on12groups, stage2 normalizedH12 mixes allgroups. Intermediate and final HMX casts use declared FP16 normalization constants and rounding. The full transform preserves unquantized xW; deployed Down is freshly RTN quantized from original W R with oneFP32scale/output, signed[-7,7]. OtherC64weights unchanged, no group scales/W4toS8. Middle static range is unchanged, so this fixture establishes cost and implementation correctness, not calibrated accuracy or PPL improvement.

Prepare includes unquantized FP16 SwiGLU LUT gather/packing and constant sign matrix materialization. Layout transposes stage1 for the second GEMM; finish reorganizes/quantizes into native-U8 Down input. Native layout uses exact HVX deal/scatter; finish uses gather, unchanged quantization and native stores. Decode FP16 SwiGLU is produced into the HMX input on Gate/Up readiness. Prefill uses two 256KiB input/output slots; next layout and previous finish are scheduled during the current HMX command. Matmul_ticks is exposed submit/wait cost, not total engine compute. Pipeline HVX ticks report scheduled overlap work, not fully hidden time. Dense factor GEMMs alone cannot stand in for complete cost.

|R4 phase, us|M64 prefill|M1 decode|
|---|---:|---:|
|dense_r4_prepare_ticks|154.036|9.777|
|dense_r4_matmul_ticks|25.536|4.344|
|dense_r4_layout_ticks|63.099|0.973|
|dense_r4_finish_ticks|117.242|7.487|

Numerical: EXP0261 sealed original-shard/export evidence is inherited, with frozen local and remote packages reverified. The inherited export proves H12/H512 orthogonality and explicit dense-factor equivalence, xW identity max1.34e-14, independentW4unpack. The inherited65536SwiGLU LUT entries correctly rounded with Decimal80 and independently certified using mpmath100. All9realsteps HMX eachstage<=1FP16ULP+minnormal; middlequantization, nativeDown and finalresidual exact versus independent arithmetic. Conditional fullblock versus scalarR4 after sameHMXstate max1LSB,mincosine0.999773 passes2/.999. Known idealR3 failure remains; this is not fullmodel quality.

PreR4 live data exact between arms; decode dead padding excluded explicitly, prefill allrows checked. K/V prefill carrier hashes exact and persistent cache metadata advances correctly every invocation. Intra-arm everytimedoutput matches frozen auditedhash. Otherrecipes and baselines unchanged.

Physical: requested/granted8MiB, existing peak plan6682752bytes; phase-dead HMXactivation768KiB and two1MiB expansion arenas reused, no VTCM allocation growth. No timed intermediateDDR/spill/audit, oneFastRPC perlayerstep and oneHMXowner; nativeW4 Down unchangedformat. On device the fullmatrix72MiB is never allocated or fetched; denseH512 512KiB constant materialization and H12 padded2KiB are included. Prefill9/decode2R4 HMXcommands; per-row arithmetic and total tile pairs unchanged. Host-DSP boundary=Hostwall-DSPinvocation perrecord.

Bounded OPT5 prepares exact DSP SF32 inverse/zero/rounding once per conversion task while retaining array boundaries. OPT6 also keeps gather, affine quantization and native tile rearrangement in registers. Both candidates passed all87 live tensor/cache comparisons against sealed EXP0263 and all63488 finiteFP16 values on each of9 device audit steps. route.json freezes OPT6 after one repeat10 smoke per candidate and before any short/formal collection. The unchanged generic quantizer is the exhaustive hardware control.

Measured fused assembly has one exact reciprocal before channel/row loops, an8byte scalar frame, no vector stack accesses and no hot-loop calls. The first compiled shared array/fused implementation had an unnecessary0x880byte frame; an inspection-driven implementation cleanup isolated OPT6 before any device audit or smoke. Both original binaries and disassemblies are retained. FP32 multiply/round, add/round, +0.5, signed conversion and saturating pack order unchanged; no approximate reciprocal or FMA substitution.

Inherited OPT3 parallel preparation/finish ownership, stage2 doublebuffer, existing3x256byte scratch and decode producer stream remain. R4 output arithmetic, frozen packages and all other recipes remain unchanged. No model quality claim or promotion. This singlelayer report controls the separately authorized conditional fullmodel continuation; its own E2E and devicePPL are N/A.

Historical EXP0263 comparison (separate campaigns, not paired inference):

|Mode|Previous R4 us|Optimized R4 us|Observed speedup|
|---|---:|---:|---:|
|prefill|1703.398|1599.661|1.065x|
|decode|992.867|970.135|1.023x|

Exact native comparisons: 87 live tensor/cache files over original, vector and pipeline captures. Both parent483file and original371file seals independently reverified; all144files/package verified locally and remotely. Exhaustive live-address mapping and double-buffer range nonoverlap independently checked. No current model export or calibration.
## prefill repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|17.4 (1.15%)|16.7 (1.05%)|N/A|
|Input RMSNorm|N/A|N/A|20.1 (1.32%)|19.4 (1.21%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|250.3 (16.50%)|248.7 (15.55%)|N/A|
|QK–Softmax–AV|N/A|N/A|117.8 (7.77%)|118.1 (7.38%)|N/A|
|O projection|N/A|N/A|43.1 (2.84%)|42.9 (2.68%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|23.8 (1.57%)|23.4 (1.46%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|483.2 (31.87%)|595.1 (37.20%)|N/A|
|Down|N/A|N/A|117.3 (7.73%)|117.8 (7.36%)|N/A|
|Final residual|N/A|N/A|6.5 (0.43%)|6.5 (0.41%)|N/A|
|KV carrier conversion|N/A|N/A|4.6 (0.30%)|4.4 (0.27%)|N/A|
|KV append DMA|N/A|N/A|7.3 (0.48%)|7.2 (0.45%)|N/A|
|Block orchestration|N/A|N/A|1.5 (0.10%)|1.5 (0.09%)|N/A|
|Layer bookkeeping|N/A|N/A|0.8 (0.06%)|0.8 (0.05%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|1.6 (0.11%)|1.7 (0.10%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|73.7 (4.86%)|73.8 (4.61%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|347.4 (22.91%)|321.8 (20.12%)|N/A|
|完整 Host wall|N/A|N/A|1516.4 (100.00%)|1599.7 (100.00%)|N/A|

## decode repeat10 module overview

Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.

|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|
|---|---|---|---:|---:|---|
|I/O、metadata|N/A|N/A|18.0 (1.88%)|17.6 (1.81%)|N/A|
|Input RMSNorm|N/A|N/A|5.3 (0.56%)|5.3 (0.55%)|N/A|
|QKV＋Q/K Norm-RoPE|N/A|N/A|91.5 (9.59%)|91.8 (9.46%)|N/A|
|QK–Softmax–AV|N/A|N/A|65.7 (6.88%)|65.9 (6.79%)|N/A|
|O projection|N/A|N/A|43.5 (4.56%)|43.2 (4.45%)|N/A|
|Post-attention residual＋RMSNorm|N/A|N/A|7.1 (0.74%)|7.0 (0.72%)|N/A|
|Gate/Up＋SwiGLU|N/A|N/A|218.5 (22.89%)|252.5 (26.03%)|N/A|
|Down|N/A|N/A|113.8 (11.93%)|116.2 (11.97%)|N/A|
|Final residual|N/A|N/A|1.5 (0.16%)|1.5 (0.15%)|N/A|
|KV carrier conversion|N/A|N/A|10.1 (1.06%)|10.1 (1.04%)|N/A|
|KV append DMA|N/A|N/A|3.0 (0.31%)|3.0 (0.31%)|N/A|
|Block orchestration|N/A|N/A|1.0 (0.11%)|1.0 (0.10%)|N/A|
|Layer bookkeeping|N/A|N/A|0.6 (0.07%)|0.6 (0.07%)|N/A|
|Stage-boundary bookkeeping|N/A|N/A|0.4 (0.04%)|0.4 (0.04%)|N/A|
|DSP unattributed|N/A|N/A|0.0 (0.00%)|0.0 (0.00%)|N/A|
|Runtime setup/teardown|N/A|N/A|70.6 (7.40%)|70.5 (7.27%)|N/A|
|Embedding|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Final model RMSNorm|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|LM head + greedy|N/A|N/A|N/A: outside layer|N/A: outside layer|N/A|
|Host–DSP 边界|N/A|N/A|303.6 (31.81%)|283.6 (29.23%)|N/A|
|完整 Host wall|N/A|N/A|954.3 (100.00%)|970.1 (100.00%)|N/A|


## Complete normalized numeric counters

Counts and bytes in native units, *_ticks in qtimer ticks (19.2ticks/us), *_ns in ns. All numeric telemetry retained below, including nested per-layer records. Independent per-field medians do not sum; engine work, DMA and waits can overlap. Legacy unused reference fields are telemetry, not correctness authority; numerical_gate.json is authoritative for implementation comparisons.

### repeat1 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4851.500000|7055.000000|+45.419%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2397.000000|2394.500000|-0.104%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|116.000000|103.500000|-10.776%|
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
|dense_r3_total_finish_ticks|2294.500000|2293.000000|-0.065%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|374.500000|373.500000|-0.267%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5800.000000|5837.000000|+0.638%|
|dense_r3_total_prepare_ticks|293.500000|295.500000|+0.681%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|2269.500000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1212.000000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|490.500000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|6.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|9.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|96.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|335.000000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|192.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|13708.500000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|3062.000000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|2984.500000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2373.000000|2393.500000|+0.864%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|141.000000|139.000000|-1.418%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4531.000000|4549.500000|+0.408%|
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
|hmx_compute_ticks|4256.500000|5046.500000|+18.560%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1699531.000000|1772240.000000|+4.278%|
|input_norm_ticks|429.500000|431.000000|+0.349%|
|input_stage_ticks|148.000000|147.500000|-0.338%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|24108.500000|26392.000000|+9.472%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|55.000000|56.000000|+1.818%|
|ledger_named_ticks|24108.500000|26392.000000|+9.472%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|180.000000|184.000000|+2.222%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|910.500000|909.500000|-0.110%|
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
|output_stage_ticks|58.000000|58.000000|+0.000%|
|post_attention_norm_ticks|3.500000|4.000000|+14.286%|
|post_attention_residual_ticks|499.500000|499.500000|+0.000%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|1.000000|1.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|514.500000|528.000000|+2.624%|
|projection_pack_ticks|20.500000|21.000000|+2.439%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|qkv_projection_ticks|5036.500000|5034.500000|-0.040%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|1120.000000|1139.000000|+1.696%|
|runtime_teardown_ticks|915.500000|910.000000|-0.601%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|156.500000|149.000000|-4.792%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|110.500000|110.500000|+0.000%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4851.500000|7055.000000|+45.419%|
|slice_layer_0.attention_ticks|2397.000000|2394.500000|-0.104%|
|slice_layer_0.block_orchestration_ticks|116.000000|103.500000|-10.776%|
|slice_layer_0.cache_append_dma_ticks|156.500000|149.000000|-4.792%|
|slice_layer_0.cache_append_pack_ticks|110.500000|110.500000|+0.000%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2373.000000|2393.500000|+0.864%|
|slice_layer_0.final_residual_ticks|141.000000|139.000000|-1.418%|
|slice_layer_0.gate_up_ticks|4531.000000|4549.500000|+0.408%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|429.500000|431.000000|+0.349%|
|slice_layer_0.input_stage_ticks|148.000000|147.500000|-0.338%|
|slice_layer_0.layer_bookkeeping_ticks|55.000000|56.000000|+1.818%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|21904.000000|24184.500000|+10.411%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|180.000000|184.000000|+2.222%|
|slice_layer_0.o_projection_ticks|910.500000|909.500000|-0.110%|
|slice_layer_0.post_attention_norm_ticks|3.500000|4.000000|+14.286%|
|slice_layer_0.post_attention_residual_ticks|499.500000|499.500000|+0.000%|
|slice_layer_0.qk_norm_rope_ticks|3.000000|3.000000|+0.000%|
|slice_layer_0.qkv_projection_ticks|5036.500000|5034.500000|-0.040%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|105.000000|107.000000|+1.905%|
|total_ticks|22999.500000|25263.000000|+9.842%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|431.000000|434.000000|+0.696%|
|u8_attention_av_requant_ticks|1177.000000|1173.000000|-0.340%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1440.500000|1487.000000|+3.228%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|445.000000|441.000000|-0.899%|
|u8_attention_qk_norm_rope_ticks|8770.000000|8795.000000|+0.285%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|6060.000000|6069.500000|+0.157%|
|u8_attention_v_pack_ticks|3037.500000|3014.500000|-0.757%|
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
|u8_cache_native_append_update_ticks|259.000000|248.000000|-4.247%|
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
|w4u8_final_residual_main_work_ticks|68.000000|83.000000|+22.059%|
|w4u8_final_residual_pool_wait_ticks|26.000000|8.500000|-67.308%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|376.500000|370.500000|-1.594%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|366.000000|357.500000|-2.322%|
|w4u8_input_norm_pool_wait_ticks|20.500000|21.500000|+4.878%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1657.500000|1664.500000|+0.422%|
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
|w4u8_post_residual_main_work_ticks|427.500000|430.500000|+0.702%|
|w4u8_post_residual_pool_wait_ticks|15.000000|16.000000|+6.667%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1951.000000|1935.000000|-0.820%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1471.000000|1468.500000|-0.170%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|398.000000|401.000000|+0.754%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|984.500000|976.000000|-0.863%|
|w4u8_qkv_ring_pipeline_ticks|2007.000000|2011.000000|+0.199%|
|w4u8_qkv_ring_pool_wait_ticks|221.500000|238.000000|+7.449%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8201.000000|8176.000000|-0.305%|
|w4u8_qkvo_prefetch_wait_ticks|1471.500000|1470.500000|-0.068%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8902.000000|8904.000000|+0.022%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat1 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|445.312500|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1277.500000|1279.812500|+0.181%|
|attention_unattributed_ticks|581.125000|577.750000|-0.581%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|20.312500|20.437500|+0.615%|
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
|dense_r3_total_finish_ticks|50.625000|50.687500|+0.123%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|50.062500|49.812500|-0.499%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|687.375000|706.812500|+2.828%|
|dense_r3_total_prepare_ticks|24.562500|19.750000|-19.593%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|144.250000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|19.687500|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|83.187500|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|6.000000|N/A zero denominator|
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
|dense_r4_prepare_ticks|0.000000|187.750000|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2223.500000|2236.062500|+0.565%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.750000|29.187500|-1.891%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4316.750000|4403.062500|+1.999%|
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
|hmx_compute_ticks|4048.250000|4104.000000|+1.377%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|976988.750000|1027301.437500|+5.150%|
|input_norm_ticks|102.437500|102.375000|-0.061%|
|input_stage_ticks|136.437500|136.750000|+0.229%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12659.812500|13217.000000|+4.401%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.125000|12.250000|+1.031%|
|ledger_named_ticks|12659.812500|13217.000000|+4.401%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|142.625000|144.437500|+1.271%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|828.062500|828.437500|+0.045%|
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
|output_stage_ticks|65.250000|65.312500|+0.096%|
|post_attention_norm_ticks|0.500000|0.500000|+0.000%|
|post_attention_residual_ticks|135.687500|135.687500|+0.000%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|5.500000|5.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|376.125000|374.812500|-0.349%|
|projection_pack_ticks|3.000000|3.000000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.250000|0.312500|+25.000%|
|qkv_projection_ticks|1769.625000|1756.375000|-0.749%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|719.062500|720.437500|+0.191%|
|runtime_teardown_ticks|635.562500|638.125000|+0.403%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.500000|57.687500|+0.326%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|200.062500|200.562500|+0.250%|
|scan_cache_stage_ticks|341.187500|343.875000|+0.788%|
|scan_dynamic_attention_ticks|1272.812500|1274.937500|+0.167%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|445.312500|N/A zero denominator|
|slice_layer_0.attention_ticks|1277.500000|1279.812500|+0.181%|
|slice_layer_0.block_orchestration_ticks|20.312500|20.437500|+0.615%|
|slice_layer_0.cache_append_dma_ticks|57.500000|57.687500|+0.326%|
|slice_layer_0.cache_append_pack_ticks|200.062500|200.562500|+0.250%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2223.500000|2236.062500|+0.565%|
|slice_layer_0.final_residual_ticks|29.750000|29.187500|-1.891%|
|slice_layer_0.gate_up_ticks|4316.750000|4403.062500|+1.999%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.437500|102.375000|-0.061%|
|slice_layer_0.input_stage_ticks|136.437500|136.750000|+0.229%|
|slice_layer_0.layer_bookkeeping_ticks|12.125000|12.250000|+1.031%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11230.687500|11786.375000|+4.948%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|142.625000|144.437500|+1.271%|
|slice_layer_0.o_projection_ticks|828.062500|828.437500|+0.045%|
|slice_layer_0.post_attention_norm_ticks|0.500000|0.500000|+0.000%|
|slice_layer_0.post_attention_residual_ticks|135.687500|135.687500|+0.000%|
|slice_layer_0.qk_norm_rope_ticks|0.250000|0.312500|+25.000%|
|slice_layer_0.qkv_projection_ticks|1769.625000|1756.375000|-0.749%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.625000|7.750000|+1.639%|
|total_ticks|11939.000000|12496.812500|+4.672%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|113.375000|114.500000|+0.992%|
|u8_attention_av_requant_ticks|99.562500|100.562500|+1.004%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.750000|13.875000|+0.909%|
|u8_attention_pipeline_wait_ticks|47.187500|46.875000|-0.662%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|95.750000|96.687500|+0.979%|
|u8_attention_qk_norm_rope_ticks|813.625000|828.812500|+1.867%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|243.625000|244.000000|+0.154%|
|u8_attention_v_pack_ticks|83.500000|83.562500|+0.075%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|26.250000|26.375000|+0.476%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|254.750000|255.062500|+0.123%|
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
|w4u8_gate_up_swiglu_join_wait_ticks|87.125000|163.375000|+87.518%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3513.875000|2886.250000|-17.861%|
|w4u8_gate_up_swiglu_worker_ticks|449.000000|1258.500000|+180.290%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|349.625000|345.562500|-1.162%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|202.187500|198.437500|-1.855%|
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
|w4u8_qkv_ring_dma_wait_ticks|1396.937500|1401.312500|+0.313%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|349.937500|350.125000|+0.054%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|947.375000|943.562500|-0.402%|
|w4u8_qkv_ring_pipeline_ticks|1621.375000|1615.812500|-0.343%|
|w4u8_qkv_ring_pool_wait_ticks|2.750000|2.812500|+2.273%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.375000|3.937500|-10.000%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8142.437500|8138.312500|-0.051%|
|w4u8_qkvo_prefetch_wait_ticks|1397.687500|1401.875000|+0.300%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8625.500000|8636.562500|+0.128%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 prefill

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|4841.850000|6962.600000|+43.800%|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|2282.950000|2268.100000|-0.650%|
|attention_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|29.850000|28.300000|-5.193%|
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
|dense_r3_total_finish_ticks|2267.100000|2267.500000|+0.018%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|358.950000|358.350000|-0.167%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|5717.000000|5742.650000|+0.449%|
|dense_r3_total_prepare_ticks|254.200000|254.700000|+0.197%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|1536.000000|1536.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|2251.050000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|9.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|1211.500000|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|490.300000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|6.000000|N/A zero denominator|
|dense_r4_parallel_dispatches|0.000000|9.000000|N/A zero denominator|
|dense_r4_parallel_finish_groups|0.000000|96.000000|N/A zero denominator|
|dense_r4_parallel_join_ticks|0.000000|381.700000|N/A zero denominator|
|dense_r4_parallel_prepare_tiles|0.000000|192.000000|N/A zero denominator|
|dense_r4_parallel_work_ticks|0.000000|13631.700000|N/A zero denominator|
|dense_r4_pipeline_batches|0.000000|8.000000|N/A zero denominator|
|dense_r4_pipeline_hvx_ticks|0.000000|3047.650000|N/A zero denominator|
|dense_r4_prefill_consume_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_join_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_publish_count|0.000000|0.000000|N/A zero denominator|
|dense_r4_prefill_worker_ticks|0.000000|0.000000|N/A zero denominator|
|dense_r4_prepare_ticks|0.000000|2957.500000|N/A zero denominator|
|dense_r4_rows|0.000000|64.000000|N/A zero denominator|
|down_ticks|2246.700000|2215.650000|-1.382%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|125.450000|126.000000|+0.438%|
|first_position|0.000000|0.000000|N/A zero denominator|
|gate_up_ticks|4441.750000|4436.050000|-0.128%|
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
|hmx_compute_ticks|4613.700000|5172.900000|+12.120%|
|hmx_fp16_tile_pair_count|768.000000|7936.000000|+933.333%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49408.000000|49408.000000|+0.000%|
|host_wall_ns|1516429.950000|1599661.400000|+5.489%|
|input_norm_ticks|388.150000|381.000000|-1.842%|
|input_stage_ticks|137.300000|133.100000|-3.059%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|22498.350000|24531.900000|+9.039%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|16.050000|16.300000|+1.558%|
|ledger_named_ticks|22498.350000|24531.900000|+9.039%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|64.000000|64.000000|+0.000%|
|metadata_stage_ticks|146.600000|146.400000|-0.136%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|835.650000|829.600000|-0.724%|
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
|output_stage_ticks|55.000000|56.850000|+3.364%|
|post_attention_norm_ticks|0.750000|0.700000|-6.667%|
|post_attention_residual_ticks|453.900000|453.800000|-0.022%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|41.500000|41.500000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|517.850000|517.550000|-0.058%|
|projection_pack_ticks|4.500000|4.500000|+0.000%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.500000|0.550000|+10.000%|
|qkv_projection_ticks|4798.950000|4785.650000|-0.277%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|0.000000|0.000000|N/A zero denominator|
|runtime_setup_ticks|754.400000|757.000000|+0.345%|
|runtime_teardown_ticks|665.850000|667.200000|+0.203%|
|scan_attention_overlay_capacity_bytes|0.000000|0.000000|N/A zero denominator|
|scan_attention_overlay_required_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|139.550000|137.250000|-1.648%|
|scan_cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|scan_cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|scan_cache_dma_descriptor_count|16.000000|16.000000|+0.000%|
|scan_cache_pack_ticks|89.100000|83.000000|-6.846%|
|scan_cache_stage_ticks|0.000000|0.000000|N/A zero denominator|
|scan_dynamic_attention_ticks|0.000000|0.000000|N/A zero denominator|
|scan_logical_m_observed|64.000000|64.000000|+0.000%|
|scan_padded_kv_length|64.000000|64.000000|+0.000%|
|scan_total_kv_length|64.000000|64.000000|+0.000%|
|slice_layer_0.activation_ticks|4841.850000|6962.600000|+43.800%|
|slice_layer_0.attention_ticks|2282.950000|2268.100000|-0.650%|
|slice_layer_0.block_orchestration_ticks|29.850000|28.300000|-5.193%|
|slice_layer_0.cache_append_dma_ticks|139.550000|137.250000|-1.648%|
|slice_layer_0.cache_append_pack_ticks|89.100000|83.000000|-6.846%|
|slice_layer_0.cache_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.cache_ddr_write_bytes|143360.000000|143360.000000|+0.000%|
|slice_layer_0.cache_valid_after|64.000000|64.000000|+0.000%|
|slice_layer_0.cache_valid_before|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.down_ticks|2246.700000|2215.650000|-1.382%|
|slice_layer_0.final_residual_ticks|125.450000|126.000000|+0.438%|
|slice_layer_0.gate_up_ticks|4441.750000|4436.050000|-0.128%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|388.150000|381.000000|-1.842%|
|slice_layer_0.input_stage_ticks|137.300000|133.100000|-3.059%|
|slice_layer_0.layer_bookkeeping_ticks|16.050000|16.300000|+1.558%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|20997.800000|23022.400000|+9.642%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|146.600000|146.400000|-0.136%|
|slice_layer_0.o_projection_ticks|835.650000|829.600000|-0.724%|
|slice_layer_0.post_attention_norm_ticks|0.750000|0.700000|-6.667%|
|slice_layer_0.post_attention_residual_ticks|453.900000|453.800000|-0.022%|
|slice_layer_0.qk_norm_rope_ticks|0.500000|0.550000|+10.000%|
|slice_layer_0.qkv_projection_ticks|4798.950000|4785.650000|-0.277%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|31.000000|32.000000|+3.226%|
|total_ticks|21747.700000|23778.200000|+9.337%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|435.650000|434.800000|-0.195%|
|u8_attention_av_requant_ticks|1177.100000|1177.850000|+0.064%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_pipeline_wait_ticks|1266.200000|1214.450000|-4.087%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|436.050000|435.200000|-0.195%|
|u8_attention_qk_norm_rope_ticks|8602.500000|8619.600000|+0.199%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|5972.600000|5965.200000|-0.124%|
|u8_attention_v_pack_ticks|2833.500000|2827.300000|-0.219%|
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
|u8_cache_native_append_update_ticks|225.500000|219.450000|-2.683%|
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
|w4u8_final_residual_main_work_ticks|78.250000|77.550000|-0.895%|
|w4u8_final_residual_pool_wait_ticks|13.550000|11.600000|-14.391%|
|w4u8_final_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_final_residual_worker_work_ticks|368.250000|370.800000|+0.692%|
|w4u8_gate_up_swiglu_consume_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_join_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_overlap_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_publish_count|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_gate_up_swiglu_worker_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_direct_row4_call_count|0.000000|0.000000|N/A zero denominator|
|w4u8_input_norm_main_work_ticks|323.800000|324.350000|+0.170%|
|w4u8_input_norm_pool_wait_ticks|27.700000|22.850000|-17.509%|
|w4u8_input_norm_task_count|16.000000|16.000000|+0.000%|
|w4u8_input_norm_worker_work_ticks|1482.250000|1470.350000|-0.803%|
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
|w4u8_post_residual_main_work_ticks|364.700000|361.200000|-0.960%|
|w4u8_post_residual_pool_wait_ticks|51.100000|52.900000|+3.523%|
|w4u8_post_residual_task_count|16.000000|16.000000|+0.000%|
|w4u8_post_residual_worker_work_ticks|1787.750000|1788.350000|+0.034%|
|w4u8_prefill_cache_mode|1.000000|1.000000|+0.000%|
|w4u8_qk_norm_rope_rows_observed|0.000000|0.000000|N/A zero denominator|
|w4u8_qk_padding_poison_pair_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_batch_count|6.000000|6.000000|+0.000%|
|w4u8_qkv_ring_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_dma_wait_ticks|1393.100000|1400.800000|+0.553%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|368.300000|371.650000|+0.910%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|938.600000|925.200000|-1.428%|
|w4u8_qkv_ring_pipeline_ticks|1886.050000|1872.250000|-0.732%|
|w4u8_qkv_ring_pool_wait_ticks|253.650000|238.950000|-5.795%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|4.500000|6.450000|+43.333%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|7904.200000|7890.600000|-0.172%|
|w4u8_qkvo_prefetch_wait_ticks|1393.900000|1401.400000|+0.538%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|6.000000|6.000000|+0.000%|
|w4u8_swiglu_rows_observed|64.000000|64.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8581.600000|8574.450000|-0.083%|
|wide_score_mode|4.000000|4.000000|+0.000%|

### repeat10 decode

|Counter|Control|R4|Change|
|---|---:|---:|---:|
|activation_ticks|0.000000|438.731250|N/A zero denominator|
|attention_av_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_av_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_gqa_pipeline_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_hmx_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_pack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_qk_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|attention_setup_ticks|0.000000|0.000000|N/A zero denominator|
|attention_softmax_ticks|0.000000|0.000000|N/A zero denominator|
|attention_ticks|1259.975000|1263.275000|+0.262%|
|attention_unattributed_ticks|571.225000|571.393750|+0.030%|
|block_invocation_count|1.000000|1.000000|+0.000%|
|block_orchestration_ticks|19.193750|18.737500|-2.377%|
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
|dense_r3_total_finish_ticks|50.562500|50.593750|+0.062%|
|dense_r3_total_hmx_calls|1.000000|1.000000|+0.000%|
|dense_r3_total_matmul_ticks|50.143750|50.500000|+0.710%|
|dense_r3_total_parallel_heads|24.000000|24.000000|+0.000%|
|dense_r3_total_parallel_work_ticks|732.075000|786.256250|+7.401%|
|dense_r3_total_prepare_ticks|19.743750|19.268750|-2.406%|
|dense_r3_total_refined_values|0.000000|0.000000|N/A zero denominator|
|dense_r3_total_rows|24.000000|24.000000|+0.000%|
|dense_r4_audit_bytes|0.000000|0.000000|N/A zero denominator|
|dense_r4_calls|0.000000|1.000000|N/A zero denominator|
|dense_r4_finish_ticks|0.000000|143.750000|N/A zero denominator|
|dense_r4_hmx_calls|0.000000|2.000000|N/A zero denominator|
|dense_r4_layout_ticks|0.000000|18.687500|N/A zero denominator|
|dense_r4_matmul_ticks|0.000000|83.400000|N/A zero denominator|
|dense_r4_mode|0.000000|1.000000|N/A zero denominator|
|dense_r4_optimization|0.000000|6.000000|N/A zero denominator|
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
|dense_r4_prepare_ticks|0.000000|187.712500|N/A zero denominator|
|dense_r4_rows|0.000000|1.000000|N/A zero denominator|
|down_ticks|2222.662500|2207.918750|-0.663%|
|dsp_status|3.000000|3.000000|+0.000%|
|experiment|240.000000|240.000000|+0.000%|
|f16_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_append_update_ticks|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_incremental_append_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reuse_count|0.000000|0.000000|N/A zero denominator|
|f16_cache_native_prefill_reused_carrier_bytes|0.000000|0.000000|N/A zero denominator|
|final_residual_ticks|29.237500|28.718750|-1.774%|
|first_position|67.500000|67.500000|+0.000%|
|gate_up_ticks|4262.856250|4373.181250|+2.588%|
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
|hmx_compute_ticks|4343.637500|4240.618750|-2.372%|
|hmx_fp16_tile_pair_count|16.000000|288.000000|+1700.000%|
|hmx_ready_wait_ticks|0.000000|0.000000|N/A zero denominator|
|hmx_u8s8_tile_pair_count|49536.000000|49536.000000|+0.000%|
|host_wall_ns|954348.600000|970134.793750|+1.654%|
|input_norm_ticks|102.375000|102.393750|+0.018%|
|input_stage_ticks|136.593750|135.343750|-0.915%|
|intermediate_ddr_read_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|intermediate_dma_descriptor_count|0.000000|0.000000|N/A zero denominator|
|intermediate_spill_fill_count|0.000000|0.000000|N/A zero denominator|
|invocation_ticks|12634.237500|13139.581250|+4.000%|
|kv_cache_k_format|14.000000|14.000000|+0.000%|
|kv_cache_v_format|12.000000|12.000000|+0.000%|
|layer_bookkeeping_ticks|12.187500|12.237500|+0.410%|
|ledger_named_ticks|12634.237500|13139.581250|+4.000%|
|ledger_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|logical_m|1.000000|1.000000|+0.000%|
|metadata_stage_ticks|144.006250|142.743750|-0.877%|
|numerical_audit_enabled|0.000000|0.000000|N/A zero denominator|
|numerical_status|1.000000|1.000000|+0.000%|
|o_projection_ticks|831.212500|828.181250|-0.365%|
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
|output_stage_ticks|65.006250|64.368750|-0.981%|
|post_attention_norm_ticks|0.537500|0.531250|-1.163%|
|post_attention_residual_ticks|134.456250|134.131250|-0.242%|
|prefix_group_patch_count|0.000000|0.000000|N/A zero denominator|
|prefix_kv_mode|0.000000|0.000000|N/A zero denominator|
|prefix_seed_metadata_read_bytes|0.000000|0.000000|N/A zero denominator|
|prepared_session_run_index|46.000000|46.000000|+0.000%|
|projection_failure_index|0.000000|0.000000|N/A zero denominator|
|projection_failure_n_tile|0.000000|0.000000|N/A zero denominator|
|projection_failure_result|0.000000|0.000000|N/A zero denominator|
|projection_failure_step|0.000000|0.000000|N/A zero denominator|
|projection_hmx_wait_ticks|381.343750|382.943750|+0.420%|
|projection_pack_ticks|3.037500|3.025000|-0.412%|
|projection_unpack_ticks|0.000000|0.000000|N/A zero denominator|
|qk_norm_rope_ticks|0.212500|0.256250|+20.588%|
|qkv_projection_ticks|1762.375000|1765.868750|+0.198%|
|repeat_count|1.000000|1.000000|+0.000%|
|replay_step|4.500000|4.500000|+0.000%|
|runtime_setup_ticks|718.187500|718.843750|+0.091%|
|runtime_teardown_ticks|637.625000|637.393750|-0.036%|
|scan_attention_overlay_capacity_bytes|2752512.000000|2752512.000000|+0.000%|
|scan_attention_overlay_required_bytes|77824.000000|77824.000000|+0.000%|
|scan_cache_append_mismatch_count|0.000000|0.000000|N/A zero denominator|
|scan_cache_append_ticks|57.256250|57.162500|-0.164%|
|scan_cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|scan_cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|scan_cache_dma_descriptor_count|42.000000|42.000000|+0.000%|
|scan_cache_pack_ticks|194.381250|194.343750|-0.019%|
|scan_cache_stage_ticks|346.137500|347.100000|+0.278%|
|scan_dynamic_attention_ticks|1256.550000|1259.837500|+0.262%|
|scan_logical_m_observed|1.000000|1.000000|+0.000%|
|scan_padded_kv_length|96.000000|96.000000|+0.000%|
|scan_total_kv_length|68.500000|68.500000|+0.000%|
|slice_layer_0.activation_ticks|0.000000|438.731250|N/A zero denominator|
|slice_layer_0.attention_ticks|1259.975000|1263.275000|+0.262%|
|slice_layer_0.block_orchestration_ticks|19.193750|18.737500|-2.377%|
|slice_layer_0.cache_append_dma_ticks|57.256250|57.162500|-0.164%|
|slice_layer_0.cache_append_pack_ticks|194.381250|194.343750|-0.019%|
|slice_layer_0.cache_ddr_read_bytes|143936.000000|143936.000000|+0.000%|
|slice_layer_0.cache_ddr_write_bytes|1152.000000|1152.000000|+0.000%|
|slice_layer_0.cache_valid_after|68.500000|68.500000|+0.000%|
|slice_layer_0.cache_valid_before|67.500000|67.500000|+0.000%|
|slice_layer_0.down_ticks|2222.662500|2207.918750|-0.663%|
|slice_layer_0.final_residual_ticks|29.237500|28.718750|-1.774%|
|slice_layer_0.gate_up_ticks|4262.856250|4373.181250|+2.588%|
|slice_layer_0.hidden_ddr_read_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.hidden_ddr_write_bytes|131072.000000|131072.000000|+0.000%|
|slice_layer_0.input_norm_ticks|102.375000|102.393750|+0.018%|
|slice_layer_0.input_stage_ticks|136.593750|135.343750|-0.915%|
|slice_layer_0.layer_bookkeeping_ticks|12.187500|12.237500|+0.410%|
|slice_layer_0.layer_index|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.layer_ticks|11207.831250|11709.550000|+4.477%|
|slice_layer_0.layer_unattributed_ticks|0.000000|0.000000|N/A zero denominator|
|slice_layer_0.metadata_stage_ticks|144.006250|142.743750|-0.877%|
|slice_layer_0.o_projection_ticks|831.212500|828.181250|-0.365%|
|slice_layer_0.post_attention_norm_ticks|0.537500|0.531250|-1.163%|
|slice_layer_0.post_attention_residual_ticks|134.456250|134.131250|-0.242%|
|slice_layer_0.qk_norm_rope_ticks|0.212500|0.256250|+20.588%|
|slice_layer_0.qkv_projection_ticks|1762.375000|1765.868750|+0.198%|
|slice_layer_0.status|3.000000|3.000000|+0.000%|
|slice_layer_0.weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|stage_boundary_ticks|7.568750|7.793750|+2.973%|
|total_ticks|11916.743750|12419.400000|+4.218%|
|u8_attention_audit_ddr_write_bytes|0.000000|0.000000|N/A zero denominator|
|u8_attention_av_hmx_ticks|116.643750|115.606250|-0.889%|
|u8_attention_av_requant_ticks|98.037500|98.343750|+0.312%|
|u8_attention_fused_k_operand_mismatch_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_k_pack_ticks|13.212500|13.168750|-0.331%|
|u8_attention_pipeline_wait_ticks|45.906250|45.962500|+0.123%|
|u8_attention_probability_mask_violation_count|0.000000|0.000000|N/A zero denominator|
|u8_attention_qk_hmx_ticks|94.662500|95.800000|+1.202%|
|u8_attention_qk_norm_rope_ticks|852.881250|902.637500|+5.834%|
|u8_attention_qk_requant_ticks|0.000000|0.000000|N/A zero denominator|
|u8_attention_softmax_ticks|238.112500|238.731250|+0.260%|
|u8_attention_v_pack_ticks|82.350000|82.443750|+0.114%|
|u8_cache_full_prefix_pack_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_cached_head_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_correction_load_bytes|126.000000|126.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_bytes|896.000000|896.000000|+0.000%|
|u8_cache_k_vtcm_tail_ddr_write_skip_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_fallback_head_count|1.000000|1.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_hvx_row_update_ticks|25.531250|25.506250|-0.098%|
|u8_cache_k_vtcm_tail_init_bytes|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_init_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_k_vtcm_tail_native_load_bytes|28672.000000|28672.000000|+0.000%|
|u8_cache_k_vtcm_tail_row_update_count|7.000000|7.000000|+0.000%|
|u8_cache_k_vtcm_tail_seal_count|0.000000|0.000000|N/A zero denominator|
|u8_cache_native_append_update_ticks|250.025000|249.956250|-0.027%|
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
|w4u8_gate_up_swiglu_join_wait_ticks|87.950000|163.875000|+86.327%|
|w4u8_gate_up_swiglu_overlap_observed|1.000000|1.000000|+0.000%|
|w4u8_gate_up_swiglu_publish_count|6.000000|6.000000|+0.000%|
|w4u8_gate_up_swiglu_ready_wait_ticks|3386.175000|2846.112500|-15.949%|
|w4u8_gate_up_swiglu_worker_ticks|549.412500|1290.543750|+134.895%|
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
|w4u8_o_gate_prefetch_lifetime_ticks|346.356250|348.481250|+0.614%|
|w4u8_o_gate_prefetch_start_count|1.000000|1.000000|+0.000%|
|w4u8_o_gate_prefetch_wait_ticks|206.118750|208.068750|+0.946%|
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
|w4u8_qkv_ring_dma_wait_ticks|1392.418750|1391.537500|-0.063%|
|w4u8_qkv_ring_expand_task_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_expand_worker_count|0.000000|0.000000|N/A zero denominator|
|w4u8_qkv_ring_head_publish_count|24.000000|24.000000|+0.000%|
|w4u8_qkv_ring_hmx_compute_ticks|355.931250|357.087500|+0.325%|
|w4u8_qkv_ring_hmx_dispatch_count|1.000000|1.000000|+0.000%|
|w4u8_qkv_ring_hmx_ready_wait_ticks|923.843750|916.456250|-0.800%|
|w4u8_qkv_ring_pipeline_ticks|1618.662500|1622.531250|+0.239%|
|w4u8_qkv_ring_pool_wait_ticks|2.750000|2.775000|+0.909%|
|w4u8_qkv_ring_prep_worker_count|5.000000|5.000000|+0.000%|
|w4u8_qkv_ring_producer_slot_wait_ticks|6.012500|5.025000|-16.424%|
|w4u8_qkv_ring_slot_count|2.000000|2.000000|+0.000%|
|w4u8_qkvo_hmx_lifetime_ticks|8125.575000|8103.000000|-0.278%|
|w4u8_qkvo_prefetch_wait_ticks|1393.118750|1392.156250|-0.069%|
|w4u8_qkvo_weight_expand_ticks|0.000000|0.000000|N/A zero denominator|
|w4u8_residual_active_contexts|0.000000|0.000000|N/A zero denominator|
|w4u8_swiglu_rows_observed|4.000000|4.000000|+0.000%|
|weight_ddr_read_bytes|25329664.000000|25329664.000000|+0.000%|
|weight_dma_descriptor_count|60.000000|60.000000|+0.000%|
|weight_dma_ticks|8580.062500|8589.587500|+0.111%|
|wide_score_mode|4.000000|4.000000|+0.000%|
