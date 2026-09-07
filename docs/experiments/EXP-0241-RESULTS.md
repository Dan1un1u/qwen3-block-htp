# EXP-0241 results and handoff

Completed2026-09-08 under PC056. Evidence valid; speed gate FAIL; adoption pending; no promotion. Both stable single-layer regressions exceed the unchanged10percent threshold. Full model not started. Other recipes remain frozen; next action is discussion only.

## Algorithm and exactness

LPBQ multiplier m in1..16 is decomposed into five bits. For each b, retain the original signed W4 q if bit_b(m) is set, else replace with zero. All five masks stay packed W4. HMX weight.n consumes mask_b 2**b times into one accumulator; unchanged asymmetric-U8 correction and scale are applied once at final conversion. Thus sum(a*q*m) is exactly preserved, including original group32 semantics. No S8 weight buffer is materialized. Current implementation issues31 full HMX passes including zero masks; this is a bounded exact construction, not an optimality result or evidence that every direct-W4 LPBQ algorithm is slow.

Host changes only nibble ordering. Original group codes/scales and metadata remain verified EXP0240 values; no folded mllm weights or quantizer changes. Timed DSP includes metadata DMA, five-mask construction, repeated HMX and waits. Mask buffering reuses existing1MiB slots. QKV/O/Gate/Up batches4, Down2. Historical expansion counters denote packed-W4 masking for mode3. All seven projections use mode3 and actual HMX counters account for31 passes.

Independent integer reconstruction/native conversion passes63 projection-step checks over1,474,560 output values, zero mismatches and zero LSB error. All seven bias tables exact. Whole-layer outputs and KV match EXP0240 LPBQ; unit-multiplier matches per-channel (22 output files checked). Every short/formal output hash matches frozen audited references. These are implementation-correctness results, not a PPL/accuracy improvement claim.

## Formal performance

Five short and ten three-way rotated/reversed formal rounds, repeat1/repeat10; one real layer14 M64 followed by eight M1 positions with self-computed persistent KV. Repeat10 resets state before each of ten complete replays within a loaded runtime. No discarded warmup/outliers. Medians of round means; paired ratios/50000 bootstrap resamples, seed241.

| Repeat10 complete Host wall | Per-channel us | LPBQ S8 expansion us | LPBQ direct-W4 us | Direct vs per-channel | Ratio95% CI |
|---|---:|---:|---:|---:|---|
| M64 prefill | 1645.7291 | 4227.5833 | 6230.74985 | +276.5495% | 3.666568..3.836077 |
| M1 decode mean | 1016.43358125 | 3689.6279375 | 5786.28550625 | +469.0024% | 5.622725..5.731105 |

Direct-W4 is also slower than the paired LPBQ S8 control by47.0580% prefill and56.8901% decode. Repeat1 confirms the primary regression+240.3327%/+459.6791%. Main cost is repeated HMX and mask work in projections, especially MLP; normal/attention kernels remain unchanged. This multi-pass path fails the speed gate despite exact numerical semantics. Do not extrapolate full-model performance or infer a theoretical impossibility from this candidate.

## Physical and provenance

All three variants requested/acquired8,388,608 VTCM bytes, peak plan6,682,752; zero intermediate DDR reads/writes/spill, one FastRPC per step, no QNN. Weight DDR including bias is25,329,664 bytes per-channel and26,116,096 bytes both LPBQ paths. Direct-W4 projection HMX tile pairs31*49152 plus unchanged attention pairs256 prefill/384 decode. Every formal additive ledger closes exactly, zero unattributed ticks. Independent report reduction verifies all12 latency medians and12 additive tables.

Runtime build source3bcb631caf1fe788928ea6b4be8ca6958b2794f7; formal/reporting sourcea4f7df85c4449f4ad18bd3b1129d55274437a0d6 on codex/exp-0241-lpbq32-direct-w4. Later commits modify runners/report only. Exact immutable binaries: /mnt/d/llm_exp/models/qwen3-block-htp/exp0241/artifacts/3bcb631caf1f. Artifact/package hashes in artifact_manifest.json; device/local binaries match final_formal_provenance.json.

Initial smoke_direct and smoke_unit were accidentally routed by the runner to existing scalar mode2; command/telemetry exposed this. They are retained and excluded. Corrected mode3 runs use suffix02, independent projection audit then precedes all formal runs. No numerical mismatch or evidence substitution occurred; recovery_runner_routing.json records the repair.

Results: /mnt/d/llm_exp/results/qwen3-block-htp/exp0241. All451 evidence file hashes/sizes independently reverified at closure. Ledger SHA256:06c90fa925d90adc6b496c19688eef3712f5e43887924eddaef3f6f7fcb498bb. EXP0240 evidence remains immutable. Current ignored build caches still have QBH_EXP0240_SINGLE_LAYER=ON; explicitly disable or use a separate build directory before any future approved full-model build. No further experiment or baseline promotion is implied.

[Complete profile](EXP-0241-PROFILE.md): all three variants, repeat1/repeat10, module/counter ledgers, physical constraints, provenance and frozen-recipe N/A overview.

E2E token/s and PPL: N/A. Full model not run after failed layer gate.
