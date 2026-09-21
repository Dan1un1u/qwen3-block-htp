# L32-0062: full-model AV-to-O scale folding

Llama-3.2-1B-Instruct, all16 layers,64prefill+42decode, batch1/cache128. Both arms use uniform INT16 Down, FP32 residual, no rotation, same packed W4 weights, native HMX two-plane Down mode8. Fixed project-owned token trajectory from0057/0060, not a named-dataset or free-running text-quality test. No baseline promotion.

## Implementation
Native src/include are byte-unchanged from0061. Runtime source5486a5f772641a6717fdc05cba9aae2d40c0837a adds only the experiment tool. AV HMX shift and raw carrier are preserved. Fifteen layers set AV multiplier1/zero128; O input scale absorbs the original integer multiplier and O zero-point compensation changes accordingly. Layer7 already had multiplier1; its original zero/scale remain untouched. Only30 package metadata files differ. All layers' attention groups agree; per-matrix O signed24/S32 bounds and uniform INT16 Down exhaustive LUT/byte-reconstruction/bounds pass.

## Full-model numerical result
Both hardware arms match their own independent complete actual-arithmetic reference:688 layer-output hashes per arm (16x43),43final-norm U8 outputs and43greedy token/code pairs each. Three-layer checks precede fullmodel; each has129exact layer outputs. Direct prefill KV capture matches1,048,576elements per fullmodel arm (plus196,608per3layer arm). Decode mutable VTCM tails are checked indirectly by every subsequent layer output; the stale DDR mirror is not used as a final-tail reference.

Both independently followed trajectories show zero AV post-HMX saturation over6,946,816sampled elements. This is sample evidence, not a universal no-clipping guarantee. No hardware/reference numerical mismatch occurred. Raw runtime output_mismatches against the historical fixed-token labels are not the new-contract reference comparison; the independently recomputed head token/code check is exact on every measured step.

The two arms nevertheless are NOT model-output equivalent. FP32 reassociation differences stay below3.13e-7 through layer10, then one input-norm quantization entry in layer11 crosses a rounding threshold: token position12/channel1201,142.5->142.49998474121094, codes143->142. Subsequent Q/K,attention,MLP and cached history amplify the difference. Final-layer maxabs3.2263207436,relativeL2.2604957955 over106rows;5/43greedy tokens differ,39/43token/code pairs differ. Independent reference and hardware reproduce this effect. It is not AV saturation or a hardware arithmetic bug. No PPL test or quality-equivalence acceptance is claimed. Algebraic scale folding alone does not guarantee finite-precision whole-model equivalence.

## Formal paired timing
Five short paired rounds then ten formal AB/BA paired rounds,repeat10 trajectories per process. Separate untimed captures and warmup processes. Summary is token count divided by the mean full-phase Host wall, with per-round paired bootstrap20000,seed620062. Prefill includes64input tokens and head selection,decode sums42calls. Includes embedding,alllayers,finalnorm,LMhead,greedy,FastRPC; excludes model loading and external tokenizer. Audit-mode hash/capture overhead is excluded.

| Phase | Original E2E token/s | Folded E2E token/s | Original wall ms | Folded wall ms | Folded/original wall (95% CI) |
|---|---:|---:|---:|---:|---|
|prefill|2319.557695|2317.912969|27.591467|27.611045|1.000710 [0.997768, 1.003777]|
|decode|44.658616|46.443725|940.468007|904.320231|0.961564 [0.959600, 0.963832]|

Prefill throughput change -0.0709%, no significant speed improvement. Decode throughput +3.9972%, Host wall -3.8436%, confidence interval excludes equality. Both stay inside the10% slowdown guard.

Prefill summed AV RQ counter falls927.358->5.420us (57.960->.339us/layer), while complete DSP invocation remains26819.038->26821.062us. Therefore accumulated vector-worker time is not equal to critical-path wall time saved. No claimed prefill benefit follows from the disappearing counter.

Decode AV RQ is included inside AV HMX stage, with separate RQ counter0. The combined stage falls50839.694->13632.967us over42tokens/16layers, or75.654->20.287us per layer/token. The original head64 short-decode branch still visits M64 rows; the measured improvement includes deleting that invalid-row work. This is NOT a minimal-row RQ control or proof that HMX MAC got faster. Original first HMX wide-accumulator-to-U8 conversion remains.

## Physical and provenance
12,900measured token boundaries across short/formal campaigns;8MiB VTCM requested/acquired,peak8,098,272bytes,zero timed intermediate DDR read/write/spill. Every invocation's additive ledger reconciles; every layer status and hidden-DDR/unattributed check passes. Start battery85%/33.8C; recorded end85%,approximately38C. AB/BA pairing retained all samples; no fixed frequency or power claim. Native runtime held constant for both arms and all formal runs.

Recovered497files/1,154,466,261bytes from the device-resident0054parent, each matched its retained historical SHA256. New0062 directories only; old cleanup/evidence unchanged. Reproduction tooling, runtime binaries/seals,package manifests,independent references,untimed captures,raw logs and timing summaries retained here. Source native code remains unchanged.

## Workbook
F_AV-O尺度融合 added to desktop HTP hardware workbook. Includes E2E,CI,single-layer average AV/O stage times,ten paired observations,and the numerical/attribution limitations. Existing A-E XML parts,cells,formulas and styles preserved. Artifact Tool authors F; OOXML transplant preserves original content. Spreadsheet operation had namespace/style-index verification issues in temporary exports, repaired before final reimport/render; no experiment data or original A-E changed. No Excel lock prevented writing.

Next discussion: whether to retain the original numerical contract by multiplying the O wide accumulator by the original AV factor before the existing FP32 epilogue, instead of reassociating its floating scale. That may preserve the removed vector pass while avoiding this specific scale-rounding change; it requires a separate bounded overflow/correctness/performance experiment. Not implemented here.
