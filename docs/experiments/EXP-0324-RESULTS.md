# EXP-0324: fixed dense R3 on actual HMX, Qwen3-1.7B Down16

## Result
Same frozen WikiText-2 test subset8192 target tokens, four1984-target windows plus256 tail,64warmup and context64..2047. Not full-corpus PPL.

| Configuration | PPL |
|---|---:|
| Matched original BF16 teacher (EXP0320) |17.8745976615|
| Existing no-R3 actual hardware (EXP0318) |28.4670476589|
| Frozen R3 software simulation (EXP0323) |21.7807545403|
| **Frozen R3 actual HMX (EXP0324)** |**23.3195920574**|

Actual R3 device PPL improves18.0822% against the retained hardware baseline, all5windows improve. R3 device remains7.0651% above ideal R3 simulation and30.4622% above teacher. This establishes real-device benefit for R3 plus frozen Q/K recalibration, not general quality acceptance or exact software/hardware alignment. The old full8192 control is retained evidence, not newly rerun; new same-binary OFF43target control exactly reproduces old pertoken scores. EXP0323 paired float-core OFF/R3 results separately support rotation-specific benefit.

## Frozen implementation
No new quantized weights or calibration. Qwen17 original native W4 per-output-channel and uniform INT16 Down LUT, FP32 residual, V/AV and all non-QK parameters unchanged. Fresh package contains56 changed Q/K qparams/config files and one transformed prefix K/V file (V byte-identical); other969 manifest files unchanged. Independent sealed EXP0323 qparams and score shift/multiplier used verbatim. Prefix K rotated/requantized once from the frozen original decoded seed; body Q/K rotate after Q/K norm+RoPE before A8/cache. No butterfly or learned rotation.

Existing dense R3 FP16 HMX uses +/-1 H128 matrix and rounded FP16 normalization0.08837890625. PostRoPE carrier and rotated output are FP16, unlike EXP0323 FP32 dense reference. Current projection converter and non-R3 normalization differences also remain; do not attribute all residual7.07% gap solely to rotation rounding. R3_OPT1 retains vector preparation/quantization and dense constant matrix; speed optimization was outside this experiment.

Source688a05b73a130be3876213a0f86b8115c72df70a on codex/exp-0324-r3-down16-device extends only supported Qwen17 Down16/FP32 R3 guards and adds opt-in per-layer audit export. Dead Gate/Up and HMX activation scratch reused after QKV join and before MLP; residual storage separate. OFF default preserved; Llama/Qwen06 remain guarded. Build1/3/28 and binary/remote hashes archived.

## Verification
- Single layer: M64 and3M1, actual HMX, identity and scalar dense audit modes. Identity output exactly equals raw carrier. Scalar is audit-only, never production fallback.
- Three-layer and full28: dense component, quantizer and per-layer cache checks pass. Total136 layer-step component checks. Max dense error0.0169069713 is within original1FP16ULP+2^-14 bound; quantizer maxcode error0 against actual rotated carrier.
- Independent preparation and downstream reference across1/3/28 layers matches128 layer-step output hashes exactly. This is explicitly a CONDITIONAL integration oracle: it consumes independently checked actual HMX Q/K codes, not hidden outputs. It is not a hardware-free end-to-end FP16 HMX emulator or a claim that ideal R3 full-layer sensitivity disappeared.
- Repeat full28 M64+3:112 layer hashes and184 audit files byte-identical.
- 2048prefill+3decode:35 cache-guard steps, all28layers' existing history/padding unchanged outside appended ranges. 8MiB VTCM, no unexpected intermediate DDR/spill, one model RPC per step retained.
- All8192 production scoring invocations report28 actual HMX R3 calls,229376total, no refined values or scalar fallback. Audit dumps disabled during PPL. All corpus positions, targets and contexts exactly match both old hardware and software simulation; finite scores, zero LM-head saturation.
- Frozen parent package/hash, EXP0323 ledger and qparam freeze, original unchanged files, device serial/model/boot/build identity and build archives verified.

Original historical ideal R3 full-layer numerical failures remain historical failures. This experiment validates the stated rounded HMX arithmetic plus downstream integration and measures quality without relaxing those old thresholds or promoting a baseline. Direct script execution initially failed execute permission; invoking via bash fixed tooling, with both logs retained.

## Window PPL
|Window|Targets|Old hardware|R3 hardware|R3 software|
|---|---:|---:|---:|---:|
|0|1984|17.955120910|15.216285639|13.147816552|
|1|1984|31.701246499|23.810749736|23.405184039|
|2|1984|33.509616245|28.739451120|27.361646839|
|3|1984|31.795566035|26.558743299|24.786790321|
|4|256|52.753633317|39.207629222|39.083166310|

## Scope and next direction
Actual device benefit confirmed; no automatic baseline promotion, quality acceptance, formal speed claim or other-model change. Next isolate remaining hardware/software gap with the R3 parameters frozen before contemplating learned rotations. Match software FP16 carrier/normalization and actual projection conversion as separate diagnostic factors; do not infer that all remaining loss is hardware rounding, since even ideal R3 PPL21.7808 remains above teacher17.8746.

Retained artifacts: README.md, package.json, binary seals, device.py, check_rotation.py, check_chain.py, long_guard.py, ppl.py, analyze.py, raw device logs, all token scores and summary.json. Profiling comparison sections N/A: this is a quality experiment, not formal repeat10 performance measurement. Existing baselines unchanged.
