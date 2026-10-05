# EXP-0326 — Hardware projection-rounding repair

## Result
Opt-in scale-aware nearest-rounding compensation reduces actual Qwen3-1.7B W4A8 uniform INT16 Down + FP32 residual + dense R3 PPL from23.3195920574 to21.5823527415 on the same8192-target WikiText2 raw test subset:7.44969857% lower. All five windows improve. Formal paired throughput remains statistically unchanged. This is a real device-side arithmetic change through HMX bias parameters, not a redefinition of the software target.

This bounded round implements projection repair first. R3 arithmetic has NOT been changed. Remaining strict long-sequence alignment is not declared solved; no quality acceptance or baseline promotion.

## Method and frozen scope
Source branch codex/exp-0326-projection-rounding, commit e3859c3. QBH_PROJECTION_ROUNDING=2 changes only Host load-time backbone Q/K/V/Gate/Up bias construction. Original W4 weights/scales, activation qparams, calibration, LUT, prefix seed, caches, integer attention, FP32 O/Down residual paths and LM head remain fixed. Default0 preserves existing behavior;1 is retained as the declared original-ratio diagnostic, not selected by PPL.

For uncentered integer accumulator A and weight sum S, ideal software quantizes (A-z_i*S)*s_i*s_w/s_o+z_o to nearest. The original HMX setup used offset round(-z_i*S+z_o/r), with r=s_i*s_w/s_o. The corrected offset is

B=round_away(-z_i*S+(z_o+0.5)/r_h), where r_h=half(512*r)/512.

The existing multiplier is unchanged. Bias is checked for finite values and signed32 representability; LPBQ and unsupported recipe/model combinations reject the option. This compensates nearest rounding and the actual encoded multiplier in zero-point placement. It does NOT make the entire HMX converter exactly equal to ideal floating quantization; internal conversion and scale approximation remain.

Both DSP/shared-library binaries and the unrelated llama executable are SHA-identical to EXP0324. No new online operator, HMX invocation, conversion pass, activation buffer or pipeline stage. Additional arithmetic occurs only during model preparation, excluded from warm inference timing.

## Numerical evidence
- Fixed pre-PPL layer0 M64+3 diagnostics: candidate2 has lower ideal-code error than original and candidate1. Choice frozen in selection.json before PPL; no evaluation fitting.
- Broader28-layer Q/K/V fixed-real-input audit,458752 outputs: original mismatch53.212847%; candidate3.577314%; maximum1 code in both. Mean signed errors approximately-0.5321 versus-0.0335. Values are explicitly a sample, not a universal error bound.
- One, three and28 layers:4+12+112 layer-step outputs exact against independent downstream reference conditional on separately validated actual R3 QK codes. Raw Q/K preparation is independently exact. R3 component and quantizer/cache checks retain original bounds. No hidden-state injection.
- Mode0 regression:112 layer-step hashes and184 capture files byte-identical to EXP0324; same selected tokens/codes.
- Independent8192-target candidate simulation uses SDK HMX/HVX arithmetic with no injected outputs; PPL21.6678800295 versus actual21.5823527415. The SDK reference is0.396283% higher;251 target rows differ by>1e-4 NLL, maximum5.078207. This is NOT strict full-sequence bit equality; remaining differences have not been freshly localized in this experiment.
- EXP0324 and EXP0325 parent evidence ledgers and every archived file reverified unchanged.

## Matched actual PPL
8192targets positions64..8255,64-token warmup per2048window,1984targets per fullwindow and256tail. Fixed source corpus/token IDs, local positions, targets, causal masks, prefix replacement and head. This is a subset, not full WikiText2. No optional stopping or checkpoint selection.

|Configuration|PPL|
|---|---:|
|Original BF16 teacher|17.8745976615|
|Frozen ideal R3 fakequant|21.7807545403|
|Actual old hardware|23.3195920574|
|Actual repaired hardware|21.5823527415|
|Independent candidate SDK reference|21.6678800295|

Repaired hardware is20.743153% above teacher. Being slightly below ideal fakequant does not prove arithmetic equality or universal superiority: rounding perturbations have non-monotonic model effects.

|Window|Targets|Old hardware PPL|Repaired hardware PPL|Reduction|
|---|---:|---:|---:|---:|
|0|1984|15.216286|13.237804|13.002%|
|1|1984|23.810750|23.462733|1.462%|
|2|1984|28.739451|27.013482|6.006%|
|3|1984|26.558743|24.008375|9.603%|
|4|256|39.207629|38.378913|2.114%|

## Real-device overhead
Same binary/package/fixed M64+15 trajectory,5short pairedrepeat1 then10 alternating formal pairedrepeat10. All embedding/layers/finalNorm/head/greedy/FastRPC included; Host input staging included, cold load and external tokenizer excluded. No audited/capture timings used. All per-arm output signatures repeat exactly. Every profile reports8MiB requested/granted and zero timed intermediateDDR/spill; ledger residual<=0.1%. See FULL_PROFILING_REPORT.md for complete additive and overlapping counters, repeat1/10 and physical provenance.

|Scope|Old Host ms|New Host ms|Old token/s|New token/s|Wall delta|
|---|---:|---:|---:|---:|---:|
|Prefill64|42.336263|42.293648|1511.706|1513.230|-0.1007%|
|Decode15 total|326.221894|326.216596|45.981|45.982|-0.0016%|

Paired bootstrap95% ratio intervals: prefill[0.9954936391650501, 1.0020980937414647],decode[0.9987430160924126, 1.0014320751507908]. Both contain1 and upper bounds remain below1.10. Conclusion: no measurable warm-inference slowdown, not a claimed speedup. Timings apply to this frozen dense-R3 configuration, not unrelated historical fastest no-rotation baselines.

## Next boundary
The main projection-conversion loss can be repaired without adding an online stage. Preserve this opt-in candidate for review. Do not resume calibration assuming exact SDK/HMX equality; R3 threshold sensitivity and residual ideal-converter differences remain distinct questions. More exact raw-accumulator/HVX conversion and R3 refinement were not needed to demonstrate this low-cost improvement and were not implemented in this round. They require explicit attribution of any further accuracy and performance changes; prior failed/partial alignment evidence stays intact.

## Artifacts
README.md and exact scripts reproduce the run. Runtime seals and binary-regression.json pin binaries. package.json references the unchanged frozen model; no newly quantized weights. quality/ contains actual scores and protocol; software/ contains independent prediction; timing/ retains every raw profile. EVIDENCE_SHA256.json seals all retained outputs.

Evidence ledger SHA256 0634f0639783b824f513e447be4f2c4a8f953688aec510351c1f7131e6d4a755;691 files,1504501048bytes.
