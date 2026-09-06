# EXP0230 GPTQ calibration coverage/budget

Candidate selected before final testing: C64. Independent DSP acceptance: **fail**. 1024 documents, 16384 scored targets; reserve used: True. No baseline promoted.

## Development (packed FP16 GPU software; not actual DSP)

|Variant|Calibration tokens|PPL|
|---|---:|---:|
|F|N/A|25.337473|
|A0|8192|29.877030|
|C8|8192|28.030617|
|C64|65536|27.150224|

### A0 module sensitivity (development software only)

|FP16-restored family|PPL|Change vs A0|
|---|---:|---:|
|attention|29.260911|-2.062%|
|mlp|26.087285|-12.684%|
|head|29.022526|-2.860%|

These hybrids are diagnostic only, not candidates or deployable mixed-precision claims. Each restores one family from the original checkpoint while every other A0 tensor and buffer remains exact. Effects interact and are not additive. No diagnostic uses final data or changes candidate selection.

C8 and C64 use the unchanged original-coordinate CPU EXP224 GPTQ, act-order, staged quantized inputs and three output-aware row-scale candidates. C8 is a document-balanced subset of C64. Original norms/embedding/head are frozen; only transformer calibration differs. Selection uses development NLL, tie1e-6 favors C8.

## Untouched final (actual DSP, matched token/context/masks)

|Stratum|FP16 PPL|A0 PPL|Selected PPL|A0/F16 ratio [95% CI]|Selected/F16 ratio [95% CI]|Limit|
|---|---:|---:|---:|---|---|---:|
|overall|25.777659|30.558053|27.545607|1.185447 [1.163824, 1.208339]|1.068584 [1.054278, 1.083449]|1.05|
|en|20.762113|24.908112|22.429407|1.199691 [1.170174, 1.231500]|1.080305 [1.060092, 1.101725]|1.10|
|zh|32.004821|37.489579|33.828824|1.171373 [1.138746, 1.204950]|1.056991 [1.037036, 1.077981]|1.10|
|wiki|21.735327|26.728699|23.728434|1.229735 [1.193076, 1.268679]|1.091699 [1.067765, 1.116796]|1.10|
|news|30.571784|34.936030|31.976846|1.142754 [1.117841, 1.169678]|1.045959 [1.028799, 1.062752]|1.10|
|en_wiki|18.639717|23.386910|20.640319|1.254682 [1.203597, 1.309903]|1.107330 [1.073880, 1.143333]|1.10|
|zh_wiki|25.345044|30.548001|27.278579|1.205285 [1.153662, 1.260123]|1.076288 [1.042765, 1.111875]|1.10|
|en_news|23.126174|26.528261|24.373572|1.147110 [1.115946, 1.178701]|1.053939 [1.028892, 1.079208]|1.10|
|zh_news|40.414551|46.008527|41.951940|1.138415 [1.097917, 1.180971]|1.038040 [1.015858, 1.060996]|1.10|

Fixed gates: overall<=1.05, each language/domain/cell<=1.10. Paired stratified document bootstrap5000seed230. Passing requires point and upperCI gates; point failure is fail, passing points with crossing intervals are inconclusive at bounded budget. This is M64+16 conditional short-context PPL, not a published full benchmark or long-context acceptance. All27DSP aggregate PPL values independently reproduced via math.fsum.

## Correctness and provenance

All1664 source windows independently reconstructed; document/text/32gram separation from prior calibration/training/validation/EXP229/qbh includes holdout. Both original qbh controls reproduce exactly; within-session and before/full/after sentinels match. Packed software/device development checks pass. All device targets finite,8MiB VTCM,no intermediateDDR/spill,valid self-computed cache progression. Original model and inherited tensor hashes verified. Export source commits may precede added evaluation/orchestration files; command source archives retain the precise executed export implementation and unchanged quantizer source hashes. See commands and per-layer export records. All3014 retained model/intermediate files have an artifact ledger; all56 hidden checkpoints have verified FP16 shapes and finite values. Package tensors, clipping statistics and source archives match their recorded hashes.

## Profiling

One warmup,5short,10rotated selected/A0 pairs complete;320invocation and8960layer ledgers valid. See full_profiling_report.md and module_table.md for complete measurements and historical nonpaired other-recipe columns.
{
  "A0": {
    "prefill_tokens": 64,
    "prefill_host_us": 63352.917,
    "prefill_tokens_per_second": 1010.2139416879573,
    "decode_tokens": 15,
    "decode_total_host_us": 1387172.994,
    "decode_tokens_per_second": 10.81335930333142
  },
  "C64": {
    "prefill_tokens": 64,
    "prefill_host_us": 63262.994999999995,
    "prefill_tokens_per_second": 1011.6498594478495,
    "decode_tokens": 15,
    "decode_total_host_us": 1389448.9595,
    "decode_tokens_per_second": 10.795646646421487
  }
}

## Next authorized direction

Stop escalation if the fixed acceptance passes. Otherwise PC051 permits the next separately registered group128 software diagnostic, then AWQ input-channel equalization if still needed. W4A8 remains frozen. No automatic promotion or grouped DSP deployment.
