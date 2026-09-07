# EXP0238 Qronos per-channel results

Official Brevitas v0.13.0 Qronos on the fixed signed[-7,7] per-output-channel grid;64K calibration tokens; identical frozen C64 W4 LM head. Host-only packed FP16 evaluation;1024 previously unexposed documents,16384 targets, M64+16. C64 is the project-enhanced GPTQ control, not unmodified upstream GPTQ.

|Variant|PPL|Change vs F16|95% PPL-ratio CI vs F16|
|---|---:|---:|---|
|F|25.641075|reference|N/A|
|C64|27.310096|+6.509%|[1.0512861457704874, 1.0797726281427358]|
|AR-P|28.380998|+10.686%|[1.0915264190887108, 1.1232288895024403]|
|Qronos|29.389287|+14.618%|[1.1280195591107232, 1.1661328973296374]|

|Cell|F16 PPL|C64 PPL|AR-P PPL|Qronos PPL|Qronos vs F16|
|---|---:|---:|---:|---:|---:|
|en_wiki|17.825307|19.103460|19.355038|20.528919|+15.167%|
|zh_wiki|24.936815|26.821748|28.196288|29.409699|+17.937%|
|en_news|23.386448|24.652038|26.223028|26.285847|+12.398%|
|zh_news|41.581739|44.039356|45.335855|47.008624|+13.051%|

Qronos fails the fixed5% overall/10% cell acceptance. Its PPL is7.613% above C64 (paired95% ratio interval1.058842..1.094755). This is evidence about this constrained grid/corpus/scale configuration, not a universal ranking of the methods. No parameter sweep or retuning follows this final test.

Independent solver: mismatch G!=H, matched and diagonal cases all match a separate dense conditional-equation oracle. All196 projections pack/unpack exactly, all28 blocks replay exactly, all frozen non-transformer tensors unchanged; no numerical fallback. All four full-model evaluations pass exact repeat, causal mask, finite-value and independent CE gates; raw NLL/PPL reductions independently rechecked. Quantization elapsed 608.506s (calibration/export time, not device throughput).

Source 8980930c4d38fddfe9b72866dba70b015726f620; Qronos manifest 40164c8b2aaddd63d9891e5a249987ff2a4e3b0109f0a1fc6696def381dfdbcb; dataset freeze 4b18d25bd2cef4fe059fa7bc79c09007ed0e9129d51b9cd072f3c9ab4a60ff2d. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0238.

No DSP runtime/package change, no speed measurement, no automatic promotion. EXP239 OmniQuant-LWC proceeds under its already frozen protocol, using the identical shared panel without tuning on EXP238 scores.
