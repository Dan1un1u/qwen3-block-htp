# Qwen3-1.7B research frontier for migration

Keep the recent uniform INT16 Down and FP32 residual family as compatible presets, rather than migrating only one branch tip. Source implementation is inherited from EXP0332 and therefore already contains the EXP0318/0319 context extension, EXP0326 projection correction, EXP0327 streamed R3/native-W4 long QKV, EXP0330 dense R4 and EXP0331 butterfly R4. EXP0333 adds full2048 accuracy validation and a versioned inventory.

## Dimensions that must remain separate

- Physical prefill tile: M64. This does not limit the session to64 contextual tokens.
- Speed development:64 input tokens, normally42 continuous decode steps for recent R4 comparisons. Historical R3 campaign also retains64+64 and2048+64; do not mix their throughput directly.
- Primary quality: same8192 frozen WikiText2 raw-test targets, five windows with64 warmup and1984 scored targets per full2048 window;256-target tail retained. No KV reset inside a window. This remains a subset,not full-corpus PPL.
- KV allocation:128 for consumer-native short session;2112 for long row-major session. Capacity is not an accuracy window or actual valid length.
- Scheduling: compute attention clients from actual current padded KV and available arena. Preserve native-W4 QKV bit21 on every complete M64 chunk,384KiB reserve bit18 only when R4 is off,capacity-clamped diagnostic client caps,and rejected large-reserve evidence. Do not assume the same client thresholds for both R4 backends.

## Presets and compatible packages

|R3|R4|Package family|
|---|---|---|
|off|off|Original Down16, original Q/K/prefix|
|on|off|EXP0324 frozen Q/K/score config and once-transformed prefix|
|off|dense or butterfly|Original-derived full6144 rotated Down W4 and FP16 SwiGLU LUT, original Q/K/prefix|
|on|dense or butterfly|EXP0332 composed rotated Down and matched R3 Q/K/prefix|

Dense R4 uses mode5/OPT10 and FP16 H512/H12 intermediate rounding. Butterfly R4 uses mode4/OPT12 and FP32 stage-order arithmetic. R3 uses production mode1/OPT2. Projection rounding0 preserves historical arithmetic;2 retains the scale-aware HMX-bias repair. These contracts must remain explicit; a shared interface does not make their finite precision identical. Scalar/identity/refined R3 combinations remain unsupported by this release.

The rotated short package omitted long row-major initialization/reference caches. EXP0333 supplements those from the corresponding verified original OFF/ON package and long RoPE/fixed files. The long frontend resets caches before execution. No weights,Q/K parameters or LUT are substituted by this supplement. Keep the rejected first attempt and this compatibility requirement.

## Files and use

`config/qwen17-research-presets.json` owns flags,profile dimensions,package manifests and export provenance. `config/qwen17-migration-frontier.json` owns source/evidence relationships and separately scoped historical speed and new PPL. `scripts/qwen17_research_preset.py` resolves options without touching the device; e.g. `python3 scripts/qwen17_research_preset.py --r3 on --r4 hmx_dense --profile quality2048`. `scripts/run_exp0333_long_rotation.py` and `scripts/check_exp0333_long_rotation.py` retain the actual quality and independent-reference workflows with explicit EXP0333 authorization. A future experiment must use its own ID/output namespace; never overwrite archived evidence.

`config/local-research.example.json` shows per-developer Linux paths relative to the repo root, a shared read-only environment,independent run/build folders,and the single device's fixed identity. It is a configuration example,not an implemented remote job server. Only one central job may own the phone. Existing build scripts are not yet fully parameterized or reorganized into a purified SDK environment; perform that mechanical migration once the new private destination is agreed.

Preserve the quantizer/export chain from original safetensors and frozen calibration,not only runnable model bins. The inventory records original checkpoint hashes,uniform Down16 export,R3 metadata/prefix composition,and fresh original-derived R4 Down transformation/re-RTN/LUT generation. Historical exporters remain in source; the EXP0329 original-derived exporter/evidence is indexed explicitly. SDKs/weights/binaries/raw logs stay outside Git; copy their reproducible configs and sealed provenance.

## Evidence and ranking discipline

Retain EXP0318/0319(long execution/scheduling),EXP0326(rounding),EXP0327(R3 overlap/adaptive lengths),EXP0329/0330/0331(R4 references/optimization) andEXP0332/0333(combination/quality). Retain the no-rotation speed anchor,R3-only accuracy candidate,both R4 backends and both combined configurations. Dense and butterfly trade prefill/decode costs in the same EXP0331 campaign; R3 long timing is a separate shape. The inventory preserves frontier candidates and algorithmic references,not a fabricated global Pareto ranking using unmatched speed or arithmetic contracts. No default or baseline promotion is implied.

Do not use `recipes/w4a8-r3-r4.json` as this release: it is the older EXP0265 SP2/mode1-R4 recipe. The new package selector prevents silently disabling R4 while retaining rotated Down weights,or enabling R3 with unmatched Q/K grids and prefix.

## EXP0333 matched full2048 results

|Arm|R3|R4|Projection rounding|Subset PPL|
|---|---|---|---:|---:|
|C0|off|off|0|28.467048|
|C2|off|off|2|25.229402|
|R2|on|off|2|21.582353|
|D0|off|dense FP16|0|30.253443|
|D2|off|dense FP16|2|27.339091|
|RD2|on|dense FP16|2|23.589335|
|B2|off|butterfly FP32|2|27.819149|
|RB2|on|butterfly FP32|2|23.507789|

Matched original BF16 teacher: 17.874598. Best on this frozen subset: R2; no full-corpus claim or default promotion. All65536 score records pass correspondence,finiteNLL,vocabulary and saturation checks. C0 reproduces EXP0318;R2 reproduces EXP0326 exactly.

Context switches: `--profile long_session --prefill-tokens 2048 --decode-steps 64` retains the existing long inference frontend;512/1024 lengths and capacity-clamped client diagnostics remain selectable. `quality2048` deliberately fixes64warmup and1984 targets per full window; do not silently shorten it for a faster quality run. Long-session combined numerical evidence is2048+3;new combined formal speed remains unmeasured.
