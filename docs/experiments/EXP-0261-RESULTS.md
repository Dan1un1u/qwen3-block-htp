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
