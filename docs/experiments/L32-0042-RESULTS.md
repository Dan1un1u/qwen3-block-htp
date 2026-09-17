# L32-0042 results: shape-aware Llama 3.2 3B pipeline/layout repair

Completed two exact native optimization directions from L32-0041. Frozen3B per-output-channel W4 GPTQ weights,static A8 scales,241-level SP2mode8,FP32 residual and no rotations. Model package unchanged. PPL/calibration and DRAM-memory optimization remain deferred.

## Formal complete-model result
M64+15 continuous decode,28 layers,cache capacity80. Five short and ten fixed paired AB/BA formal cycles,repeat10 primary. Control is retained L32-0041 measured binary; candidate measured source 76418bbbb504dc30608441b17f49a774c10b72a2.
Complete Host wall includes embedding,all blocks,final RMSNorm,LM head,greedy selection and RPC; excludes cold loading,session preparation and external tokenization.

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| Original3B control,paired | 512.024356 | 18.679376 |
| Optimized3B OPT2 | 1038.737970 | 21.152098 |

Prefill wall ratio 0.49292928,95% CI [0.49240170066314065, 0.49349670227505377]; decode 0.88309800,CI [0.8822698085549525, 0.883800038667266]. Both upper bounds <=1.10. No optional stopping; all20 formal processes included,3200 token-boundary profiles.

## Actual bottleneck and changes
The initial port made head128/GQA3 functional but excluded3B from the1B head64 vector movement specialization. A128-byte head was repeatedly assembled by32-byte memcpy calls. RoPE split it into two artificial64-lane problems with further small copies. K packing and KV row-major export repeated similar gathering. These preparation stages prolonged the critical path,independently of the larger matrix workload.

1. Direct128-lane HVX RoPE with the original SF32 operation ordering and the same division-boundary repair. Four native32-channel tiles are gathered in registers,and masked native stores preserve other rows. No additional rotation or changed quantization.
2. Whole-vector native-to-row-major KV transposes,aligned K rows,exact vector integer K sums and vector gathering for dynamic V row groups. The existing exact V LUT+vdeal consumer is enabled for3B prefill,removing scalar recenter/packing work before HMX. Existing producer/consumer ownership,HVX worker pool,HMX/DMA overlap and matrix work are preserved.

This repairs preparation/layout bottlenecks on the existing pipeline. The improvement is not attributed solely to raw HMX throughput or a speculative hidden hardware implementation.

## Correctness and physical evidence
- OPT1 and OPT2 selected layers0/13/27: independent exact FP32 output and exact KV for M64 and decode after64 historical tokens.
- OPT1 and OPT2 consecutive3 and full28: same exact output/KV gates,no tolerance changes.
- OPT1 and OPT2 full28 true autoregressive greedy: all16 token IDs and selected U8 logit codes unchanged. Correct short Paris answer preserved; general text/PPL quality remains unaccepted.
- Bounded1B layer7 prefill/decode output and KV regression passed. DSP disassembly differs only in one diagnostic __LINE__ immediate,0x5755->0x575f; computation/scheduling instructions unchanged.
- All3200 formal additive ledgers reconcile. HMX commands/tile work and weight DDR bytes identical per token. Peak VTCM 8360416/8388608 bytes,one RPC/token,no intermediate DDR/spill.
-370 prior L32-0041 evidence entries rehashed. Formal runtime binaries and frozen frontend payloads verified locally and on device before paired timing.

## Scope and remaining work
3B has28 layers versus16,hidden3072 versus2048,head128 versus64 and GQA3 versus4. Shape-derived backbone Linear weight count is2.8966x1B; including LM head it is2.5997x. These are workload counts,not a latency prediction. Latest historical1B SP2 fixed-trajectory2335.236/45.740token/s is a non-paired reference.

After optimization,prefill time is mainly Gate/Up+SwiGLU21.81%,attention19.41%,QKV+RoPE17.91%,Down12.94%. Decode is Gate/Up25.73%,attention20.16%,Down14.23%,QKV+RoPE12.33%. Further projection/attention scheduling may be studied separately; no claim that runtime is fully optimal. No grouped weights,mixed-precision change,new quality acceptance or automatic paper-baseline promotion.

## Attempts and reproducibility
Two build-launch/compiler failures retained: direct execution of the non-executable build script,then an unused head64 helper under-Werror. Re-running via bash and guarding the inactive helper fixed these; no failed build was deployed. No hardware numerical failure or crash.
Primary artifacts: SUMMARY.json,MODULES.md,paired-inputs-verified.json,short-paired.json,formal-paired.json,protocol/records/validated files in each immutable run directory. Full module table is MODULES.md.
Source tools: execute_llama32_3b_opt.py,profile_llama32_3b_opt.py,verify_llama32_3b_opt_regression.py,report_llama32_3b_opt.py. Build: QBH_LLAMA_MODEL_SIZE=3B bash scripts/build_llama32.sh 28.1B remains default.
