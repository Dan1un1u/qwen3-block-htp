# EXP0231 static group128 software diagnostic

Independent software acceptance: **fail**. 1024 documents, 16384 targets. Reserve used: True. No baseline promotion.

## Matched development regression

|Variant|PPL|
|---|---:|
|F|25.337473|
|C8|28.030617|
|C64|27.150224|
|G128|27.478141|

F/C8/C64 reproduce all stored EXP230 development per-token NLL and top1 results exactly. Development is reused descriptive evidence, never independent final evaluation or a selection input. There is exactly one G128 candidate.

## Independent final PPL (packed FP16 GPU software)

|Stratum|F|C8|C64|G128|G128/F ratio [95% CI]|Limit|
|---|---:|---:|---:|---:|---|---:|
|overall|23.961270|27.644064|26.085709|26.541898|1.107700 [1.091414, 1.125061]|1.05|
|en|20.439977|23.350578|22.142576|22.422159|1.096976 [1.075576, 1.118597]|1.1|
|zh|28.089192|32.726996|30.731032|31.418578|1.118529 [1.093576, 1.145544]|1.1|
|wiki|20.666208|24.303310|22.719996|23.068796|1.116257 [1.091005, 1.142941]|1.1|
|news|27.781703|31.444042|29.950014|30.537890|1.099209 [1.078861, 1.121508]|1.1|
|en_wiki|19.478982|22.255247|21.127063|21.249847|1.090912 [1.059021, 1.125168]|1.1|
|zh_wiki|21.925795|26.539848|24.433034|25.043443|1.142191 [1.103002, 1.185464]|1.1|
|en_news|21.448383|24.499818|23.206902|23.659145|1.103074 [1.075783, 1.131508]|1.1|
|zh_news|35.985137|40.356535|38.652437|39.416586|1.095357 [1.064848, 1.129091]|1.1|

G128/C8 matched-calibration ratio: {'ppl_ratio': 0.960130090513036, 'ratio_ci95': [0.9446904991335532, 0.9745828103643495]}. G128/C64 descriptive ratio: {'ppl_ratio': 1.0174880860425606, 'ratio_ci95': [1.0027303142790938, 1.0321260209862633]}.

Both G128 and C8 use identical 8192 calibration tokens. C64 uses65536 and is a stronger descriptive control; do not attribute its difference to group size alone. Every final comparison shares original tokenizer, M64 prompt,16targets, masks and FP16 execution. Paired stratified document bootstrap5000seed231; overall5%, every language/domain/cell10%, fixed reserve rule. All36 aggregate PPL values independently reduced from raw token NLL with math.fsum. This is short-context conditional PPL, not published long-context benchmark acceptance.

## Method and correctness

Fresh original transformer weights, unchanged original gamma/embedding/norms and inherited per-channel W4 head. Symmetric[-7,7], FP32 static scales per128 original input columns; no rotation, LPBQ or second-level scale quantization. Global act-order uses original-column group lookup. Three full GPTQ trials, per-group absmax/midpoint/weight-L2.4 ranges, row-wise final projection output-SSE choice. Original CPU Gram/damping0.01 and staged quantized inputs retained.

Independent dense Schur-elimination codes match allthree trials; NumPy group clipping and output-SSE/choices match; dead/zero/multiple-group/permutation tests and one-group per-channel equivalence pass. All196 projections pass independent packed-file NumPy FP16 reconstruction;112 staged-vs-HF forwards pass;28retained hidden checkpoints are finiteFP16(64,128,2048). All model tensors, source archives, source-data windows and command logs retain verified SHA256. Repeat, causal-mask and independent CE checks pass for every reported run.

New primary/reserve sources exclude prior training/calibration/evaluation, including all EXP230 roles, by document identity, text hash and32-token windows;1024source windows independently reconstructed. Used final data is now exposed and cannot become new independent calibration/selection data. No scoring-based quantizer changes.

## Profiling boundary and next action

This is a software-only format diagnostic. Grouped DSP correctness, VTCM/DMA/HMX costs,5short/10formal timings and E2E token/s are N/A: no grouped DSP runtime was built or executed. Dequantized PyTorch forward time is not a group-W4 speed result. FP32group metadata is0.25bit/weight,6.25% of raw4bit payload before headers/alignment; future DSP deployment must measure its cost.

Next authorized action: separately_register_approved_AWQ_input_channel_equalization. F16F16 and W4U8 runtime/packages remain frozen. No automatic baseline promotion. See full_profiling_report.md for all unavailable sections and exact prior EXP230 measured references.
