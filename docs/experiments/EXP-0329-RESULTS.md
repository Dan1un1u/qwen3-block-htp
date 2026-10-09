# EXP-0329 — optimized R4 baselines on Qwen3-1.7B INT16 Down

Completed on actual PJZ110 / Hexagon v79. Both implementations pass their declared numerical and physical contracts. HVX butterfly is faster than the migrated optimized dense factor path. Neither prefill path passes the descriptive 10% overhead threshold; both decode paths pass. No default promotion or model-quality acceptance.

## Matched full-model measurement

| Configuration | Prefill64 Host ms | Prefill token/s | Decode42 Host ms | Decode token/s | Prefill / decode wall overhead |
|---|---|---|---|---|---|
| No R4 | 34.631832 | 1848.01 | 878.719668 | 47.80 | Reference |
| HVX butterfly R4 (OPT9) | 39.400735 | 1624.34 | 904.782562 | 46.42 | +13.77% / +2.97% |
| Dense HMX factor R4 (OPT7) | 41.667766 | 1535.96 | 918.106069 | 45.75 | +20.32% / +4.48% |


64 prefill tokens and 42 fixed teacher-forced decode inputs, all 28 blocks, embedding, final norm, LM head and greedy selection, FastRPC included. External tokenizer/detokenizer, cold model staging, ADB transport, model conversion and audit capture excluded. One RPC per step, repeat_count=1 inside RPC; each round runs ten complete trajectories (the primary repeat10 protocol). Five short rounds precede ten formal rounds. Three-arm order is rotated/reversed; component and throughput tables below use formal arithmetic means. Decode wall is the total for 42 steps; decode profiling below is mean per token. Throughput = token count / measured mean complete Host wall. Paired bootstrap95CI has20,000 samples over ten matched rounds.

| Phase | Configuration | Wall ratio | Paired95CI | 10% overhead criterion |
|---|---|---|---|---|
| prefill | HVX butterfly R4 (OPT9) | 1.137703 | [1.136164, 1.139147] | FAIL |
| prefill | Dense HMX factor R4 (OPT7) | 1.203164 | [1.200895, 1.205528] | FAIL |
| decode | HVX butterfly R4 (OPT9) | 1.029660 | [1.028195, 1.031385] | PASS |
| decode | Dense HMX factor R4 (OPT7) | 1.044822 | [1.042920, 1.047323] | PASS |


## Frozen configurations and provenance

C: parent EXP0308 INT16 package, no rotation. B/D: same normalized full H12 tensor-product H512 logical R4 (6144 coordinates), same new original-derived RTN per-output-channel W4 Down weights. Dense uses two HMX FP16 factors; butterfly uses FP32 fixed arithmetic. R3 is OFF for all arms. The two rotation paths have distinct declared finite-precision contracts and are not asserted bit-equal. Down signed INT16 nearest/ties-away clipping, radix256 low/high physical planes, existing native W4 multiplication, FP32 residual and all other fixed grids/dataflow are retained. R4 consumes raw FP16 SwiGLU LUT values; downstream INT16 uses frozen parent middle scale. This research does not recalibrate quality.

Runtime/source: 6d6f365a91b4043089a6bdafc9c2c185a138f2d2 on codex/exp-0329-r4-down16-baselines; archive binaries-fit28 and runtime-fit28.json. Parent manifest112452e7785f96f5b06c40eacb0d0c328c45268ec2a59456a62dd27bc4896264; rotated manifest2abe3974a237e3f65e17ddc4ce2740f946a3dcca96e1f01fe4d49832a8b9eee0. fold.json pins both original safetensor hashes and all28 mathematical invariance and integer-range checks. Fresh original FP64 folding relative-L2 errors are about1e-15; existing folded quantized weights are not reused. deployment.json records matching local/device payload hashes.

## What was optimized

B: fixed local five-stage HVX permutation/arithmetic;128/256-coordinate register batching, fewer full-vector VTCM round trips; extracted fixed H12 sequence to bound register pressure; single worker dispatch for conversion, transform and quantization; four-token-row native128B stores of both INT16 byte planes. Fixed16-row independent workers retain one output owner per row region. Optimized functions bf329_blocks/bf329_h12 have no activation-vector stack spills in final disassembly; the worker wrapper spills an immutable lane constant, not matrix intermediates.

D: reuse historical batched H512 then two-slot eight-batch H12 pipeline. HMX on one batch overlaps HVX finishing the previous batch and preparing the next. Four-row native full-vector INT16 stores replace masked32B writes in prefill. Decode retains its cheaper original small-row finish. Mode5 transposes H12 coefficient placement to match B logical transform; historical mode contracts remain unchanged. A dead nominal U8 Down carrier is aliased only in this new FP32-residual/INT16 mode, reducing full-model planned VTCM below8MiB without overwriting live raw/low/high buffers.

Within the same new one-layer INT16 base, five-round screening reduced prefill R4 wrapper B6→B9 from 202.623 to 142.412us (29.72%); D6→D7 from 269.575 to 227.129us (15.75%). Screening is auxiliary one-layer evidence, not full-model formal throughput. Finalists were frozen before formal runs; freeze-final.json records the only post-screen change, dead-buffer VTCM reuse.

## Correctness and residency

C and B each independently match all1,204 per-layer outputs across full64+42 (28×43), with no reference injection. B rotation stages and signed INT16 byte encoding are bit-exact to the FP32 reference. D is checked at historical FP16 stage bounds ulp(reference)+2^-14, exact INT16 codes relative to actual rotated FP16 values, and exact conditional integer Down dot-product/scaling/FP32 residual reconstruction. D audit-full43-D7 captures the final layer of all43 steps; one/three/full-layer runs and stable head/logit codes support integration, but this is not a claim that all dense full-layer states are bit-equal to the butterfly/idealFP32 oracle. fold.json proves signed24 partial-store and signed32 reconstruction bounds separately for all28 Down matrices.

All 12,900 formal invocations and 361,200 layer ledgers close exactly; 12,900 head token/logit-code checks match each implementation’s audit trajectory. No timed intermediateDDR read/write, hidden tensor spill/fill or multiple matrix owners. All arms request/acquire8,388,608B VTCM. Per-invocation planned peaks C/B=8,365,824B; D=8,300,288B. The summed decode vtcm_peak_plan_bytes field in raw run summaries is not a peak; use per-record maxima stated here. Battery temperatures during formal rounds span32.4–36.6C. Full arithmetic ledger and outputs remained stable throughout.

## Component diagnosis

| Phase | Implementation | R4 wrapper all28 us | Mean per layer us | Prepare us | Layout / combined us | HMX exposed submit/wait us | Finish us |
|---|---|---|---|---|---|---|---|
| prefill | HVX butterfly R4 (OPT9) | 3858.095 | 137.789 | 0.000 | 3854.561 | 0.000 | 0.000 |
| prefill | Dense HMX factor R4 (OPT7) | 6154.248 | 219.795 | 287.425 | 1770.690 | 717.375 | 3306.128 |
| decode | HVX butterfly R4 (OPT9) | 348.275 | 12.438 | 0.000 | 345.955 | 0.000 | 0.000 |
| decode | Dense HMX factor R4 (OPT7) | 631.875 | 22.567 | 276.966 | 27.435 | 126.614 | 193.984 |


R4 wrapper is a wall-clock interval on the critical operator dependency. B OPT9 layout counter now contains the combined conversion/transform/INT16-output worker interval; prepare/finish fields are zero because they are fused, not because these operations vanished. D HMX field records synchronous submit or exposed wait time after overlapping HVX; it is not total isolated HMX occupation. D phase/worker intervals can overlap and are not an additive resource-utilization ledger. Only the module tables are the additive top-level accounting.

Dense prefill cost is dominated by H12 intermediate layout and final FP16-gather→INT16/native preparation rather than exposed matrix wait. The uniform INT16 boundary is more involved than old U8 output and cannot inherit the old EXP0265 ~8% prefill result. Both new rotation paths also replace direct SwiGLU→INT16 production with raw FP16 production followed by R4, accounting for an additional Gate/Up difference. This establishes realistic current baselines, not a claim that R4 is fully covered in prefill.

Potential subsequent R4 research: reduce intermediate materialization and the full-rotation release barrier, co-design H12/H512 factor layout with the native Down input, and evaluate hybrid vector/matrix factor schedules. Such new algorithms require separate numerical and matched performance verification.

## Retained failed/intermediate attempts

Default ADB5037 had a protocol fault; a process-local5038 server recovered the USB device. New deployment symlink targets were unlinked only within the new experiment destination before pushing changed payloads; parent files were never altered. Initial audit storage was too small for FP32 snapshots and expanded before measurement. A16-vector register candidate generated activation spills and was replaced by8-vector batches. Initial full dense VTCM plan8,431,360B exceeded8MiB; probe-full-D retains the rejected DSPstatus−3. The final dead-carrier reuse is independently audited. Early screening’s disabled layer-hash assertion was repaired using dedicated independent audit gates and deterministic-output comparison; old screen.log remains. These attempts are evidence, not selected performance results.

## Complete formal prefill module profile

| Module (us; Host-wall share) | No R4 | HVX butterfly R4 (OPT9) | Dense HMX factor R4 (OPT7) |
|---|---|---|---|
| I/O and metadata | 243.6 (0.70%) | 243.6 (0.62%) | 245.6 (0.59%) |
| Input RMSNorm | 2055.7 (5.94%) | 2133.6 (5.42%) | 2085.1 (5.00%) |
| QKV + Q/K Norm-RoPE | 7035.6 (20.32%) | 7032.4 (17.85%) | 7035.1 (16.88%) |
| QK-Softmax-AV | 3492.9 (10.09%) | 3491.8 (8.86%) | 3497.3 (8.39%) |
| O projection | 2086.3 (6.02%) | 2086.8 (5.30%) | 2095.6 (5.03%) |
| Post-attention residual + RMSNorm | 2152.1 (6.21%) | 2102.1 (5.34%) | 2108.0 (5.06%) |
| Gate/Up + SwiGLU + R4 | 7019.9 (20.27%) | 11700.1 (29.70%) | 13994.8 (33.59%) |
| Down projection | 3801.2 (10.98%) | 3830.2 (9.72%) | 3814.6 (9.15%) |
| Final residual | 2.1 (0.01%) | 2.2 (0.01%) | 2.1 (0.01%) |
| KV-cache carrier conversion | 138.6 (0.40%) | 137.6 (0.35%) | 137.4 (0.33%) |
| KV-cache append DMA | 290.7 (0.84%) | 291.0 (0.74%) | 291.0 (0.70%) |
| Block internal orchestration | 38.3 (0.11%) | 38.9 (0.10%) | 39.1 (0.09%) |
| Layer bookkeeping | 25.5 (0.07%) | 25.7 (0.07%) | 26.1 (0.06%) |
| Stage-boundary bookkeeping | 20.0 (0.06%) | 20.1 (0.05%) | 20.3 (0.05%) |
| DSP unattributed residual | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| DSP runtime setup/teardown | 116.8 (0.34%) | 117.4 (0.30%) | 119.3 (0.29%) |
| Token embedding | 59.5 (0.17%) | 59.8 (0.15%) | 59.9 (0.14%) |
| Final model RMSNorm | 17.4 (0.05%) | 17.4 (0.04%) | 14.6 (0.04%) |
| LM head + greedy selection (excluding final norm) | 5189.0 (14.98%) | 5191.8 (13.18%) | 5189.2 (12.45%) |
| True Host-DSP boundary | 846.5 (2.44%) | 878.3 (2.23%) | 892.9 (2.14%) |
| Complete Host wall | 34631.8 (100.00%) | 39400.7 (100.00%) | 41667.8 (100.00%) |


## Complete formal decode module profile (mean per token)

| Module (us; Host-wall share) | No R4 | HVX butterfly R4 (OPT9) | Dense HMX factor R4 (OPT7) |
|---|---|---|---|
| I/O and metadata | 240.0 (1.15%) | 241.6 (1.12%) | 243.6 (1.11%) |
| Input RMSNorm | 366.7 (1.75%) | 366.7 (1.70%) | 365.8 (1.67%) |
| QKV + Q/K Norm-RoPE | 2493.0 (11.92%) | 2500.5 (11.61%) | 2502.4 (11.45%) |
| QK-Softmax-AV | 1959.2 (9.36%) | 1967.5 (9.13%) | 1970.5 (9.01%) |
| O projection | 1361.5 (6.51%) | 1363.1 (6.33%) | 1369.2 (6.26%) |
| Post-attention residual + RMSNorm | 373.7 (1.79%) | 369.3 (1.71%) | 373.5 (1.71%) |
| Gate/Up + SwiGLU + R4 | 6284.5 (30.04%) | 6871.0 (31.90%) | 7155.5 (32.73%) |
| Down projection | 3509.6 (16.77%) | 3526.4 (16.37%) | 3527.6 (16.14%) |
| Final residual | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) |
| KV-cache carrier conversion | 165.1 (0.79%) | 165.1 (0.77%) | 165.3 (0.76%) |
| KV-cache append DMA | 111.4 (0.53%) | 112.7 (0.52%) | 112.6 (0.52%) |
| Block internal orchestration | 30.0 (0.14%) | 30.5 (0.14%) | 30.3 (0.14%) |
| Layer bookkeeping | 17.2 (0.08%) | 17.3 (0.08%) | 17.3 (0.08%) |
| Stage-boundary bookkeeping | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) |
| DSP unattributed residual | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| DSP runtime setup/teardown | 73.8 (0.35%) | 73.7 (0.34%) | 73.8 (0.34%) |
| Token embedding | 1.5 (0.01%) | 1.5 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 14.9 (0.07%) | 15.0 (0.07%) | 15.0 (0.07%) |
| LM head + greedy selection (excluding final norm) | 3213.4 (15.36%) | 3218.2 (14.94%) | 3220.7 (14.73%) |
| True Host-DSP boundary | 702.8 (3.36%) | 698.9 (3.24%) | 711.5 (3.25%) |
| Complete Host wall | 20921.9 (100.00%) | 21542.4 (100.00%) | 21859.7 (100.00%) |


References: results.json, profiling-means.json, formal-summary.json, short-summary.json, screen-summary.json, screen-determinism.json, full43-reference-gate.json, audit-*/gate.json, deployment.json, fold.json, freeze-final.json, runtime-fit28.json, final-disassembly.txt, all per-run raw records/stdout/protocols and evidence-ledger.json.
