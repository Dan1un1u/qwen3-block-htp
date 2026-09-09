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
