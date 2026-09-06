# EXP0232 AWQ-style per-channel input equalization

Software acceptance: **fail**, 1024 documents / 16384 targets; reserve used: True. Final experiment quality gate: **fail**. No baseline promotion.

## Descriptive development

|Variant|PPL|
|---|---:|
|F|25.337473|
|C8|28.030617|
|C64|27.150224|
|E64|28.467705|

F/C8/C64 reproduce EXP230 per-token development NLL/top1 exactly. Development was never used to select scales or a model; one E64 candidate only.

## Independent final software PPL

|Stratum|F|C8|C64|E64|E64/F ratio [95% CI]|Limit|
|---|---:|---:|---:|---:|---|---:|
|overall|25.132355|28.333026|27.000522|28.058848|1.116443 [1.100105, 1.133179]|1.05|
|en|20.614830|22.852893|22.095820|22.738765|1.103029 [1.080652, 1.126441]|1.1|
|zh|30.639848|35.127297|32.993942|34.623646|1.130020 [1.106573, 1.153729]|1.1|
|wiki|20.739087|24.141292|22.770230|23.800089|1.147596 [1.122050, 1.173340]|1.1|
|news|30.456271|33.252586|32.016725|33.079664|1.086136 [1.065953, 1.105568]|1.1|
|en_wiki|18.429533|20.919484|20.137511|20.811197|1.129231 [1.093736, 1.165113]|1.1|
|zh_wiki|23.338071|27.859290|25.747143|27.218244|1.166259 [1.129136, 1.205222]|1.1|
|en_news|23.059250|24.964991|24.244568|24.844867|1.077436 [1.050298, 1.105450]|1.1|
|zh_news|40.226130|44.291402|42.280427|44.043872|1.094907 [1.066524, 1.125907]|1.1|

Matched E64/C64 ratio: {'ppl_ratio': 1.0391964939265441, 'ratio_ci95': [1.026148646783009, 1.0528653151645242]}. Descriptive E64/C8 ratio: {'ppl_ratio': 0.9903230097509006, 'ratio_ci95': [0.9752641423267877, 1.005411535300361]}.

Both E64 and C64 use65536calibration tokens/context128 and identical CPU GPTQ. C8 uses8192. E64 adds AWQ scale-search on the nested8192calibration subset; no final/development selection. Shared tokenizer/M64+16/masks/canonical FP16 GPU. Paired stratified document bootstrap5000seed232, overall5% and every language/domain/cell10%. All36PPL aggregates independently reproduced from raw token NLL. These are short-context conditional PPL, not long-context benchmark acceptance. Different experiments use different independent final documents; compare paired ratios, not raw PPL across experiments.

## Method and correctness

Official mit-han-lab/llm-awq commit d6e797a42b9ef7778de8ee2352116e0f48a78d61; source/license/provenance retained. Activation-only20ratio scale search on frozen C64 calibration trajectories with symmetric per-row RTN proxy; actual export uses unchanged EXP224 three-range output-aware GPTQ. RTN proxy and final GPTQ objectives differ. Fresh original FP32 folding: input gamma/QKV, post gamma/GateUp, linear Up rows/Down columns. V/O scaling skipped for GQA. No commuting through SiLU, rotation, grouping, LPBQ or upstream folded weights.

FP64 legal compensation and independent NumPy RTN/scale/choice oracles pass. All28real-layer equivalence and whole-model unquantized FP16 hidden/logit checks meet NRMSE<=.003 and cosine>=.99999. Global results: {"hidden": {"nrmse": 0.0025616956436854813, "cosine": 0.9999967199828292, "max_abs": 0.490234375, "finite": true}, "logits": {"nrmse": 0.0022148597666136124, "cosine": 0.9999976135620914, "max_abs": 0.0625, "finite": true}}. Original embedding/final norm/head/QK norms frozen. All196packed roundtrips,112HF forward comparisons and28finiteFP16(512,128,2048)hidden checkpoints verified. Actual export source archive and allfivequantizer files match; three core GPTQ files exactly match C64.

All1024new source windows independently reconstructed; document/text/32gram exclusions include all prior training/calibration/validation/final roles, EXP230/231 and allWikiTexttrain titles. Final data is now exposed and cannot be reused as new independent selection/acceptance data. All commands/source archives/models/scales/checkpoints have retained hash ledgers. Repeat/causal-mask/independent CE gates pass without numerical threshold changes.

## Device and profiling boundary

Software acceptance failed or remained inconclusive, so EXP232 DSP correctness, physical gates,5short/10formal and E64 E2E speed are N/A. No device deployment or runtime change. full_profiling_report.md retains the complete unavailable sections and verified historical EXP230 references.

## Ordered sequence closure

EXP230 calibration coverage/budget, EXP231 symmetric group128 software diagnostic, and EXP232 AWQ input equalization are complete. This ends PC051 authorized escalation. Discuss next direction using these results; do not automatically start another method or promote a baseline. F16F16 and W4U8 runtime/packages remain frozen. Group128 was tested only at8192calibration tokens; its result does not establish performance at65536 or on DSP.
