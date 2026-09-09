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
