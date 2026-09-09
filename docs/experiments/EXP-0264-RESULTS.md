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
