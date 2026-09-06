# EXP0234 static group128 software diagnostic

Independent software acceptance: **fail**. 1024 documents, 16384 targets. Reserve used: True. No baseline promotion.

## Matched development regression

|Variant|PPL|
|---|---:|
|F|25.337473|
|G8|27.478141|
|C64|27.150224|
|G64|26.743973|

F/C64 reproduce stored EXP230 development per-token NLL/top1 exactly; G8 reproduces EXP231 G128 exactly. Development is reused descriptive evidence, never independent final evaluation or a selection input. There is exactly one G64 candidate.

## Independent final PPL (packed FP16 GPU software)

|Stratum|F|G8|C64|G64|G64/F ratio [95% CI]|Limit|
|---|---:|---:|---:|---:|---|---:|
|overall|25.060797|27.478381|27.154144|26.487802|1.056942 [1.045891, 1.069433]|1.05|
|en|20.634639|22.676946|22.083722|21.841406|1.058483 [1.043639, 1.074259]|1.1|
|zh|30.436372|33.296434|33.388735|32.122641|1.055403 [1.037768, 1.074607]|1.1|
|wiki|20.790832|23.283559|22.889492|22.195129|1.067544 [1.049535, 1.086662]|1.1|
|news|30.207717|32.428953|32.213365|31.610703|1.046445 [1.032583, 1.061707]|1.1|
|en_wiki|18.287364|20.295037|19.785471|19.490788|1.065806 [1.042173, 1.091075]|1.1|
|zh_wiki|23.637014|26.712152|26.480484|25.274697|1.069285 [1.040860, 1.099284]|1.1|
|en_news|23.283200|25.338406|24.648936|24.475513|1.051209 [1.032560, 1.070851]|1.1|
|zh_news|39.191614|41.503676|42.099216|40.825970|1.041702 [1.020096, 1.064612]|1.1|

G64/G8 matched-group calibration-budget ratio: {'ppl_ratio': 0.9639505840096796, 'ratio_ci95': [0.9523624634968427, 0.9759454535959614]}. G64/C64 matched64K grouping ratio: {'ppl_ratio': 0.9754607454239, 'ratio_ci95': [0.9633895099148493, 0.9876721006927083]}.

G64 and C64 use identical65536 calibration tokens; G64/G8 compares64K versus8K atidenticalgroup128 format. G8 is frozenEXP231; no recalibration. Allthree use the same frozenhead/norms/embedding. Every final comparison shares original tokenizer, M64 prompt,16targets, masks and FP16 execution. Paired stratified document bootstrap5000seed234; overall5%, every language/domain/cell10%, fixed reserve rule. All36 aggregate PPL values independently reduced from raw token NLL with math.fsum. This is short-context conditional PPL, not published long-context benchmark acceptance.

## Method and correctness

Fresh original transformer weights, unchanged original gamma/embedding/norms and inherited per-channel W4 head. Symmetric[-7,7], FP32 static scales per128 original input columns; no rotation, LPBQ or second-level scale quantization. Global act-order uses original-column group lookup. Three full GPTQ trials, per-group absmax/midpoint/weight-L2.4 ranges, row-wise final projection output-SSE choice. Original CPU Gram/damping0.01 and staged quantized inputs retained. The complete per-layer CPU calibration/quantization AST matches EXP231 exactly; core quantizer hashes match. The inherited exporter docstring mentions C8, but executed input assertions and manifest require the exact C64 65536 positions. stage_execution_provenance.json records actual start sources separately from legacy completion-HEAD labels.

Independent dense Schur-elimination codes match allthree trials; NumPy group clipping and output-SSE/choices match; dead/zero/multiple-group/permutation tests and one-group per-channel equivalence pass. All196 projections pass independent packed-file NumPy FP16 reconstruction;112 staged-vs-HF forwards pass;28retained hidden checkpoints are finiteFP16(512,128,2048). All model tensors, source archives, source-data windows and command logs retain verified SHA256. Repeat, causal-mask and independent CE checks pass for every reported run.

The shared PC052 panel is exactly EXP233 dataset.json, frozen and independently reconstructed before allthree phases, excluding prior roles throughEXP232 by document/text/32grams. This is disclosed paired reuse, not a newly independent dataset perphase. F/C64 controls are byte-identical verified EXP233 scoring evidence; G8/G64 are scored here. Unused inherited AWQ metadata is covered by the referenced EXP233 erratum; actualgroupcalibration is512x128. No scoring-based quantizer changes.

## Profiling boundary and next action

This is a software-only format diagnostic. Grouped DSP correctness, VTCM/DMA/HMX costs,5short/10formal timings and E2E token/s are N/A: no grouped DSP runtime was built or executed. Dequantized PyTorch forward time is not a group-W4 speed result. FP32group metadata is0.25bit/weight,6.25% of raw4bit payload before headers/alignment; future DSP deployment must measure its cost.

Next authorized action: approved_EXP235_C64_sensitivity. F16F16 and W4U8 runtime/packages remain frozen. No automatic baseline promotion. See full_profiling_report.md for all unavailable sections and exact prior EXP230 measured references.
