# EXP0274 completed: corrected full-model factorial

Source 338b32137cb962fdec303d0752e5431839361d37, branch codex/exp-0274-fullmodel-format-pipeline, measured native c77f70af1f7c33591c7f2a8fab0ba10c78f52cad, ABI134. Active none,next275. See EXP-0274-RESULTS.md. No baseline promotion/quality acceptance.

Qwen4arms0/14/27,chain3,full28 all bit-exact independent references and88captured fullmodel boundaries/full-vocabularyhead; formal5short10cycles repeat10primary. E2EB00 1603.445/45.0339,B10 1589.073/44.3872,B01 1837.855/47.2596,B11 1846.822/47.1436 token/s. B11/B00 latency .86821835/.95525055. Pipeline effect strong; native format alone is not uniformly beneficial. Full module tables/pairedCI in report. No intermediateDDR/spill,8MiB peak8365824,main9600profilesHMXcommands equal pertoken.

Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0274,2043files,1139347397bytes,ledgerce6873fee5fb0c66d0fdb63043341b7353aaf664875580b6b8fa00e25978f5d6. Eligible timing only consumer-row1-a02. Original campaign preserved but fairness-ineligible: modulardecodeconsumer packed64rows instead of1. Corrected andreran wholefixedprotocol. Initial namespacecollisionbeforedevice preserved. All selected/slice/full/generation gates pass in correctedattempt.

Llama L32-0029 is independently active; selected/chain3/chain16/frontend all exact after fixing null context in decodeSwiGLU. Its5short10formal timing is running; consult Llama memory/resultroot andprocesses before takingdevice. Strict serialized device ownership.

Remaining user-approved A2individualpipeline,A3decodecopack,A5ordinaryA8/SP2 FP32,A6layer scaling,A8supportedshape,A9log2/FPvectorsoftmax are NOT done. Register bounded nextphase under PC086 without asking again. Otherrecipes/rotation/PPL frozen. Source scripts execute_exp0274/report_exp0274/plot_exp0274 archived, point to replacementevidenceroot; do notrerun completedexclusivewrite paths. Future nativebuild requires successfulpreflight/newseal.
