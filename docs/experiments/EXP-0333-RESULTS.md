# EXP-0333: full2048 quality and migration frontier

## Outcome and frozen scope

Qwen3-1.7B full28 W4A8,per-output-channel symmetric native W4[-7,7],uniform INT16 Down,FP32 residual2. This experiment changes the evaluation context policy and checks deployment composition;it does not retrain,recalibrate,re-export weights,change DSP arithmetic or promote a baseline. Speed development remains64tokens.

WikiText2 raw test fixed8192-target subset,positions64..8255,tokenSHA e08c49458e0f03f12781431ad46d6118f636d1129a495307dca2f16a989a39f0. Four full windows each contain64 warmup plus1984 scored targets inside2048 total tokens;the256-target tail is retained. No KV reset within a window,contexts64..2047. Five independent reset windows,not the previous191 short resets. This is complete2048-window subset PPL,not full-corpus PPL. No window clipping or target selection after observing scores.

## Matched PPL results

|Arm|R3|R4|Projection rounding|PPL|PPL increase vs teacher|
|---|---|---|---:|---:|---:|
|Teacher|N/A|N/A|N/A|17.874598|0%|
|C0|0|off|0|28.467048|59.260%|
|C2|0|off|2|25.229402|41.147%|
|R2|1|off|2|21.582353|20.743%|
|D0|0|Dense HMX|0|30.253443|69.254%|
|D2|0|Dense HMX|2|27.339091|52.949%|
|RD2|1|Dense HMX|2|23.589335|31.971%|
|B2|0|HVX butterfly|2|27.819149|55.635%|
|RB2|1|HVX butterfly|2|23.507789|31.515%|

|Comparison numerator/denominator|PPL change|Mean NLL change|
|---|---:|---:|
|C2/C0|-11.3733%|-0.1207372|
|R2/C2|-14.4556%|-0.1561341|
|D2/C2|+8.3620%|+0.0803076|
|RD2/R2|+9.2992%|+0.0889187|
|RD2/D2|-13.7157%|-0.1475229|
|B2/C2|+10.2648%|+0.0977146|
|RB2/R2|+8.9213%|+0.0854558|
|RB2/B2|-15.4978%|-0.1683928|
|B2/D2|+1.7559%|+0.0174070|
|RB2/RD2|-0.3457%|-0.0034629|

Best by this subset PPL: R2. Original no-rotation C0 reproduces EXP0318 all8192 scores exactly;R3 R2 reproduces EXP0326 all8192 scores exactly despite inheriting the OPT2 scheduling improvements. Full result/score correspondence is retained in comparison.json and both long-history regression files.

R3 includes the frozen matched Q/K/score configuration and once-transformed prefix. R4 includes fresh original-derived folded/requantized DownW4 and the FP16 SwiGLU LUT. These are whole-deployment comparisons,not isolated mathematical rotation effects. Dense uses FP16 stage interfaces;butterfly uses FP32 stage arithmetic;finite-precision unification remains deferred. The historical matched original BF16 teacher has no hardware artificial prefix patch;its gap includes all deployed-graph differences,not only W4 error. No new quality-threshold acceptance is claimed.

## Long implementation and physical validation

Both combined backends pass1/3/full28-layer checks with2048prefill plus3decode,including independent captured references at the last prefill chunk and all3decode steps. R3 captures cover every compiled layer;R4 and conditional Down/residual captures cover the last layer at each checked step. Dense is checked against the frozen FP16 bound,butterfly against exact FP32 stage order;INT16 byteplanes and conditional Down+FP32 residual are exact. Sampled R3 quantizer differences are0. This is actual-input implementation correctness,not ideal-FP32 full-model equivalence.

|Gate|Steps|Guarded cache bytes|R4/Down checks|R3 layer checks|Peak planned VTCM bytes|
|---|---:|---:|---:|---:|---:|
|one-RD2|35|147187712|4|4|6554848|
|one-RB2|35|147187712|4|4|6620384|
|three-RD2|35|441563136|4|12|6554848|
|three-RB2|35|441563136|4|12|6620384|
|full-r1-RD2|35|4121255936|4|112|6554848|
|full-r1-RB2|35|4121255936|4|112|6620384|

All score records: finite NLL,target/context/corpus correspondence,vocabulary151936,NLL=logsumexp-targetlogit,and zero histogram saturation. All65536 scoring invocations and1835008 layer additive ledgers close. All request/acquire8MiB VTCM,actual peak within capacity,zero timed intermediate DDR/spill,zero hidden-state DDR transfers. KV storage transfers remain explicit,not mislabelled as intermediate spills. HMX ownership remains serialized by the inherited runtime.

## Preserved failed attempt and recovery

Initial short-D0 failed before DSP execution because the rotated short package omitted the long frontend row-major initialization/reference cache files. Raw RC2 stdout/stderr/protocol are preserved. Supply the missing verified OFF/ON original long buffers and RoPE/fixed inputs through explicit symlinks;no weight,Q/K,LUT,prefix or DSP arithmetic changes. Retried short-D0-r1 matches historical scores. The eight short regressions pass;seven have exact historical comparators and C2 is a new valid repaired-off control. This is a packaging compatibility repair,not a suppressed numerical failure.

## Profiling and historical performance

FULL_PROFILING_REPORT.md plus diagnostic-profiling.json retains every additive module and all numeric work/wait/physical counters by phase. Current timings are scoring-instrumented repeat1 diagnostics. Formalrepeat10,paired speed gate,new F16/W4F16 comparison and new combined long-performance result are N/A:not part of this authorization. Historical EXP0331 matched64+42 C/D10/B12 and EXP0327 matched64+64/2048+64 R3/OFF remain in the migration manifest. Their scopes and arithmetic are not silently assigned to the new PPL arms. Preserve EXP0327 long/short prefill parity failure and EXP0331 butterfly-prefill10percent cost failure.

## Migration inventory and switches

Preserve all8 frozen arms,plus long-session context length/decode length and capacity-derived client controls. No rotation remains the speed reference,R3-only remains an accuracy candidate,both R4 backends and combined presets remain explicit algorithmic references even when not quality winners. Config selector binds rotation to correct DownW4/LUT/QK/prefix manifests,including the required long-cache supplements. No incompatible package/backend combination is silently accepted.

Files:config/qwen17-research-presets.json,config/qwen17-migration-frontier.json,scripts/qwen17_research_preset.py,docs/QWEN17_MIGRATION_FRONTIER.md and globalAGENTS.md entry. All recentEXP0318/0319/0326/0327/0329/0330/0331/0332 source ancestry,evidence ledgers,actual runtime seals and rejected directions are indexed. Original-safetensors quantizer/export provenance is retained,not only binaries;legacySP2/R4mode1 recipe is explicitly distinguished. Per-developer relative-path configuration is an example only;new private repository,SDK purification and remote job server remain future migration work.

## Source and delivery provenance

Branch:codex/exp-0333-long-rotation-quality;parent:d0eec6462f7e3eee158eca153ef8f3b1ba017cdc. Measured full28 runtime seal:49f95a7d8f319dde322fd662523ec66d80882cfd;quality driver/preset source:7317082378051a197ab19d03734b405294d2f8d6. EXP0333 does not modify C/DSP arithmetic.
Delivery source:2971d9db7591ba04170147505ec97dac5457790c;all four delivered binaries byte-identical to measured full-r1 runtime. Local build directories restored to28layers after single/three-layer gates. Source closure seal is retained separately to avoid self-referential commit hashes.

Every accepted window preserves command,model/runtime bindings,input hashes,raw stdout/stderr,exit status and individual target scores. Evidence SHA ledger is sealed after reports/config copies/source closure. No baseline promotion,new performance claim or other-model experiment.
