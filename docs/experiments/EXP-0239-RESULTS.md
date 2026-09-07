# Per-output-channel W4A16 reference suite: EXP0238 / EXP0239

Storage cleanup EXP0237 deleted 6436 payloads, releasing 162.195 GiB. Original checkpoint, protected baseline packages and all compact scientific evidence retained;24012 protected file records matched their original hashes.

All quantized rows below use signed[-7,7] per-output-channel W4, positive FP32 scales, FP16 reconstruction/activations, identical frozen C64 W4 LM head, original gamma, no grouping and no online transform. Qwen3-1.7B;512x128 calibration tokens;independent PC0541024-document/16384-target balanced English/Chinese Wiki/news panel;M64+16 scoring. No model selection or tuning on this final panel.

|Method|PPL|Change vs F16|95% PPL-ratio CI vs F16|Quality gate|
|---|---:|---:|---|---|
|F16 reference|25.641075|reference|N/A|reference|
|C64|27.310096|+6.509%|1.051286..1.079773|fail|
|AR-P|28.380998|+10.686%|1.091526..1.123229|fail|
|Qronos|29.389287|+14.618%|1.128020..1.166133|fail|
|OmniQuant|32.499020|+26.746%|1.241949..1.294024|fail|

|Cell|F PPL|C64 PPL|AR-P PPL|Qronos PPL|OmniQuant PPL|
|---|---:|---:|---:|---:|---:|
|en_wiki|17.825307|19.103460|19.355038|20.528919|22.492983|
|zh_wiki|24.936815|26.821748|28.196288|29.409699|32.315477|
|en_news|23.386448|24.652038|26.223028|26.285847|29.985297|
|zh_news|41.581739|44.039356|45.335855|47.008624|51.181744|

|Method|Paired PPL ratio vs C64|95% CI|
|---|---:|---|
|AR-P|1.039213|1.023350..1.055438|
|Qronos|1.076133|1.058842..1.094755|
|OmniQuant|1.190000|1.167619..1.213796|

Lowest quantized point estimate on this fixed panel: C64. Acceptance remains overall5% and every language/domain cell10%, with upper paired95% interval required within the limit for a confident pass. The full per-cell intervals and all raw NLLs are retained in closure.json/software files. Short answers are auxiliary and were not used for ranking.

Interpretation is limited to the declared grid, shared head, corpus, and short-context workload. C64 is the existing project-enhanced GPTQ recipe; AR-P is the retained200-iteration official AutoRound adapter; Qronos uses the official fixed-scale mismatch solver; OmniQuant-LWC learns only clipping bounds for20epochs with the predeclared AdamW settings. Different algorithms have different offline optimization budgets. This is not an unconstrained reproduction or ranking of paper-native grouped/asymmetric formats.

Qronos: all196 independent nibble/FP16 roundtrips and28 full-block replays pass; dense mismatch/matched/diagonal solver oracle passes; no numerical fallback. OmniQuant: official quantizer vs NumPy and gradient-flow oracle pass; all28 unquantized native-adapter comparisons and packed block replays are exact; all196 projections pass independent packing; only clipping parameters trained and frozen full-model state is unchanged. All five score files pass finite-value, exact-repeat, causal-mask and independent CE checks; raw-token reductions independently rechecked.

OmniQuant optimizer updates: 35780; gradient-scaler overflow skips: 60. Export/calibration elapsed: 1432.996s, not device throughput.

OmniQuant training source: 0f48739bfc7ecf6fb6e3a74a19ad75d9c4f99d73; reporting/evaluation branch head: ae18bbfbcaac3fceab1a7a3a642c63001ae66f2d; package manifest: 33cc1cf231c70986a47a5f24f9c3abe61f9680b375a13e514fe7645f7553bd1e; shared dataset: 4e5ff780c76c9052203e2cf5363dd0c988bd729418381573ba6b463938e20078.

[Official Qronos implementation](https://github.com/Xilinx/brevitas/blob/f6c8e2e60649249187c0fa94adec6262aeb029cb/src/brevitas/graph/qronos.py). [Pinned official OmniQuant quantizer](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py). Exact source archives and environment freezes are retained.

New DSP/end-to-end token/s: N/A, software-only accuracy references. No runtime changes, no automatic baseline promotion, and no additional experiment authorized.
