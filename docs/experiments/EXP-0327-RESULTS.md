# EXP-0327: R3 long-context scheduling and overhead

## Scope and outcome

Qwen3-1.7B full28, W4A8 with uniform INT16 Down, FP32 residual2, projection rounding2, frozen EXP0324 R3 package. User provisionally accepted EXP0326 accuracy for speed work. No recalibration, new weights, arithmetic approximation, PPL rerun or automatic baseline promotion.

Current source54e910c (documentation-only descendant of tested native c163b3e); branch codex/exp-0327-r3-long-pipeline. Exact build seals and package hashes accompany all protocols.

## Implementation

1. Re-enable existing EXP0259 streamed R3 preparation (`QBH_R3_OPT=2`): Q/K-ready HVX workers prepare disjoint raw FP16 rows while QKV runs; HMX remains a single serialized owner. Gate arena owns raw R3, attention-concat owns temporary ring biases.
2. Add opt-in long-option bit21: subsequent complete M64 prefill chunks use the already proven native-W4 QKV ring. Previously only the first prefill chunk qualified, later chunks used expanded-S8 and rejected streamed R3. No numerical semantics changed.
3. Add bounded client-cap diagnostics (bits19:20), test capacity policy,2clients,3clients over64/512/1024/2048,three order-reversed scans each. Keep capacity policy: first chunk retains fixed short path; later attention selects4clients for current validKV<=1664,3clients for1665..2112. No evidence supports a finer heuristic table in this bounded scan; reduced caps hurt long prefill. The unsafe EXP0319 larger reserve is not reused.

Recommended explicit flags: `QBH_DENSE_R3=1 QBH_R3_OPT=2 QBH_LONG_OPT=2555807 QBH_PROJECTION_ROUNDING=2 QBH_SP2=8 QBH_FP32_RESIDUAL=2`, with verified uniformINT16 package and frozen R3 prefix. Defaults unchanged.

## Formal end-to-end results

Five auxiliary short rounds and ten formal pairedrepeat10 rounds, rotating control/candidate/OFF order and reversing length order. Complete Host includes input staging and FastRPC, excludes coldload/tokenization. Fixed token trajectories and persistentKV; one fullmodel RPC/pass. OFF uses correct unrotated prefix/QK metadata and identical other package files; shared native-QKV optimization is provided to both recipes. This is matched deployment-cost comparison, not equal QK codes.

|Shape|Scope|Prior R3 ms / token/s|Optimized R3 ms / token/s|Matched OFF ms / token/s|Candidate/prior wall ratio95%CI|R3/OFF wall ratio95%CI|
|---|---|---:|---:|---:|---|---|
|64+64|prefill_ms|42.158406 / 1518.084|34.758708 / 1841.265|34.860000 / 1835.915|0.825881 [0.8229469463442629, 0.8266973054338985]|0.997382 [0.9960879482194409, 0.9996393656459344]|
|64+64|decode_ms|1396.768625 / 45.820|1387.951220 / 46.111|1376.176418 / 46.506|0.992793 [0.9916865282727119, 0.9971894278327875]|1.008871 [1.007507414257741, 1.0103239583975694]|
|2048+64|prefill_ms|1593.555616 / 1285.176|1240.847130 / 1650.485|1241.634174 / 1649.439|0.778590 [0.7767466323725591, 0.7799156965557621]|0.999242 [0.9985048955059247, 1.0007275159507123]|
|2048+64|decode_ms|1772.033150 / 36.117|1758.102727 / 36.403|1747.435568 / 36.625|0.992307 [0.9903793824717975, 0.9929085046568592]|1.006124 [1.0053011885763792, 1.006642603381341]|

Both R3/OFF performance gates require paired95% upper<=1.10; all measured scopes pass. Small nonzero decode overhead must not be called statistically zero. Result supports historical near-covered R3 deployment cost, not universal free rotation.

## Long/short throughput gate

{"ratio": 0.8958310725426882, "ci95": [0.8953265602483121, 0.8975453591849868], "threshold": 0.9, "pass_gate": false}

This compares optimized2048 prefill throughput to same-round optimized64 reference; it is distinct from R3 overhead. Preserve a failed narrow10percent parity result if present; do not use the slower priorR3 short reference to manufacture a pass. Longer KV still requires scan/packing/attention work.

## Diagnostic client sweep (median milliseconds; decode64 total)

|Prompt|Capacity4/3 prefill/decode|Cap2 prefill/decode|Cap3 prefill/decode|
|---|---:|---:|---:|
|64|36.135/1382.770|36.015/1374.194|36.078/1373.549|
|512|266.607/1425.399|291.900/1483.366|279.042/1454.781|
|1024|549.678/1491.572|629.896/1616.446|585.219/1536.497|
|2048|1236.670/1718.104|1468.596/1862.010|1328.871/1716.957|

## Numerical and physical evidence

Exact capture comparisons: [{"control": "single-opt1", "candidate": "single-opt2", "exact_files": 22}, {"control": "../exp0326/three", "candidate": "three-opt2", "exact_files": 34}, {"control": "../exp0326/full28", "candidate": "full28-opt2", "exact_files": 184}].
Long single/slice/full checks: [{"control": "single-long-opt1", "candidate": "native-single-long-c0", "exact_layer_hashes": 35, "all_nonzero": true}, {"control": "single-long-opt1", "candidate": "native-single-long-c1", "exact_layer_hashes": 35, "all_nonzero": true}, {"control": "single-long-opt1", "candidate": "native-single-long-c2", "exact_layer_hashes": 35, "all_nonzero": true}, {"control": "native-three-control", "candidate": "native-three-c0", "exact_layer_hashes": 105, "all_nonzero": true}, {"control": "native-three-control", "candidate": "native-three-c1", "exact_layer_hashes": 105, "all_nonzero": true}, {"control": "native-three-control", "candidate": "native-three-c2", "exact_layer_hashes": 105, "all_nonzero": true}, {"control": "native-full-control", "candidate": "native-full-c0", "exact_layer_hashes": 2688, "all_nonzero": true}, {"control": "native-full-control", "candidate": "native-full-c1", "exact_layer_hashes": 2688, "all_nonzero": true}, {"control": "native-full-control", "candidate": "native-full-c2", "exact_layer_hashes": 2688, "all_nonzero": true}, {"control": "native-full-control", "candidate": "native-full-poison", "exact_layer_hashes": 2688, "all_nonzero": true}].
Formal/short physical and ledger audit: {"calls": 50715, "layer_ledgers": 1420020}.
No relaxed LSB threshold: scheduling candidates preserve deployed outputs exactly in these tests. All96 long steps include2688 full-model layer hashes; poisoned padding also matches. Cache append guards cover persistent state. Independent EXP0326 component/conditional validation remains the arithmetic authority; no ideal/software8192 bit-equality claim is added. Timed zero hash fields are not numerical evidence.

## Attribution and limits

Complete exclusive stage ledger and overlapping resource counters are in FULL_PROFILING_REPORT.md and complete-profile-counters.json; MODULE_OVERVIEW.md gives the stable recipe layout. Resource work counters cannot be added as critical-path fractions. R3 matrix work remains executed; moving preparation off the main path and fixing later-chunk QKV scheduling reduces exposed latency. Decode gains are much smaller than prefill because long attention work remains. Persistent consumer-native KV with incremental append is a plausible next target, not implemented here.

All failed attempts and recovery notes retained in RECOVERY_NOTES.md. Initial OPT2 second-chunk QKV rejection repaired; no unsafe four-client large-reserve attempt. Early auxiliary short rounds may overlap read-only hash validation; formal rounds start afterward. Battery warmed during sustained work; all fixed samples retained, no fastest-sample selection. Historical EXP0326 evidence and package manifest unchanged. No other model/recipe changes or workbook update.

## Formal R3 work attribution

Across the full2048-token prefill, main-thread R3 preparation258.283ms ->11.757ms; dense matrix work16.794ms ->16.742ms; finish105.697ms ->105.677ms. Candidate parallel preparation work266.373ms overlaps QKV and must not be added to Host wall. QKV ring lifetime132.244ms ->89.883ms. These changes explain major exposed-path savings but are overlapping diagnostics, not an additive causal decomposition. Decode64 preparation12.648ms ->1.982ms, with attention wall about517ms remaining.
