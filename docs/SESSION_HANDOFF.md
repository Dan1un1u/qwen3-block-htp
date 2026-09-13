# L32-0010 closed: full-model SP2 prefill speed gate failed,decode passed

No active experiment or running jobs. Source `327c7a22b895e83d2b624489c50f1bc45162fcf4`,clean and pushed.
Tested3/16-layer build source05383b783dfba81c2985931b8e12782d249226c9.
Current local build16 seal05383b7;closure commit adds report only. Future device
work must rebuild/reseal under a new approved experiment,not relabel the old seal.
Qwen48eb1ea and rotation252aee5 remain frozen. Defaults remain originalU8.

# L32-0010 completed

Full-model prefill fails the retained10% gate: +13.3437% latency,95% CI +11.6943% to +14.5942%. Decode passes:+0.8440%,CI -0.0613% to +1.8670%. Thus small single-layer overhead does not establish small full-model prefill overhead. Main additive prefill increase: Gate/Up+SwiGLU +3546.3us and Down +1082.8us; complete Host +4215.1us. These module timings localize the cost but do not alone establish a DMA/HVX stall mechanism. No further optimization or unchanged formal rerun in this experiment.

Correctness: consecutive3-layer M64+1 outputs/KV exact; both16-step generation arms match their independent integer token/code goldens. Twenty formal processes /320 full-model token boundaries all exact and all ledgers additive. Total23 device processes /354 boundaries including functional checks. SP2 all16 partial-dot/Q31 bounds pass.8MiB acquired,max8212960B VTCM;no intermediate tensorDDR/spill or W4 expansion. Native source unchanged from L32-0009; this experiment adds package/oracle/profiling integration only. Weights,tokenizer,head and non-Down qparams unchanged. Original U8 baseline remains default. PPL not run; SP2 text is repetitive and no quality acceptance is implied.

Tested source05383b783dfba81c2985931b8e12782d249226c9. Three/16-layer binaries,seals,commands,oracle,rawstdout/results and packages are archived under L32-0010. One initial local log-redirection failure occurred before build/device; directory creation repaired it. Rebuilding the3-layer configuration for binary archival matched every tested hash exactly;16-layer final resealed before deployment. No failed native runs or numerical retries.



E2E M64+15,tenfixed AB/BA pairs on same binary. OriginalU8 vsSP2:
prefill2026.0344 vs1787.5134 token/s,latency+13.34373%,CI[1.11694295,1.14594209],FAIL.
decode46.270279 vs45.883036 token/s,latency+0.843978%,CI[.999387032,1.018669812],PASS.
Throughputloss11.7728%/0.8369%. Thus do not claim negligible fullmodelSP2overhead.
Mainprefill additive increases Gate/Up+SwiGLU3546.3us,Down1082.8us;no causal
stall/ownership mechanism established by these module totals alone.

New source tools: prepare_llama32_sp2_e2e.py,run_llama32_sp2_e2e.py,
llama_sp2_reference.py;export_llama32_u8.layer adds default-off SP2 reference.
Existing stackrunner accepts L32-0010. No native C/DSP source changes this experiment.
Models: /mnt/d/llm_exp/models/llama32-htp/l32-0010/{stack3-a01,frontend-a01}.
Results: /mnt/d/llm_exp/results/llama32-htp/l32-0010;PROFILE.md,profiling_summary.json,contract-audit.json,
run_inventory.json,package_manifests.json,independent sp2-oracle.json,
e2e-a01 contains20formal+2functionalruns and tested16binaries/seal;
device-stack3-a01 has3layerresult and binaries/seal. No remaining sessions.
Ledger docs/experiments/L32-0010-evidence-sha256.json,137files,SHA256 eb124f2f2a06ff11d461cb289d3ae0c6a654bf15c8913102dbe4078f3895d9b9.

User requested only E2E slowdown measurement this turn. No PPL run orquality
promotion. SP2 greedy text remains repetitive (token39 repeated16times),even
though exactintegeroracleagrees. Next discuss focusedprefill producer/native
plane staging andDownoptimization;newexperiment requires userdirection.
No repeat unchangedformal to obtainpass. Historical U8 speedbaseline L32-0006
and W16/W4 PPL L32-0007 remain valid independently.
