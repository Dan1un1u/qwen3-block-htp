# Current Llama context

No active experiment or background job. Bootstrap then read four authority files.
L32-0004 speed optimization complete, common code synchronized across both Llama
branches; only config/branch.json differs. Full source heads/ledger in status.
Qwen3 frozen at48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5, original model read-only.

L32-0001 W16A16, L32-0002 W4A16, L32-0003 W4A8 OFF functional chains are complete.
Llama original-derived per-output-channel W4 enhanced C64 GPTQ (signed[-7,7],
one FP32 scale, no groups); A8 fresh65536-token static affine minmax. No Qwen
weights/qparams/prefix/tokenizer reuse. Independent2048-target128-document heldout
is disjoint from calibration and balanced1024EN/1024ZH. No evaluation fitting.

Quality remains UNACCEPTED: W4A16 PPL31.039101 vs original BF16 teacher26.697999
(+16.26%;EN+15.84%,ZH+16.69%), software31.027335; main loss quantization. Retained
5% overall/10% language gate failed. Tested EOS8-token text: The capital of France
is Paris. Forced-after-EOS16-token L32-0002 replay failedstep12 and remains failed.
W4A8 PPL1206603.740108, repeated Sleep; user explicitly sets no A8 model-quality
gate. L32-0004 does not rerun PPL or change weights/qparams/quality acceptance.

L32-0004 removes scalar head64 RoPE bottlenecks using existing SF32 HVX arithmetic,
A8 code packing with sparse exact scalar repair near rounding boundaries, idle
prefill head workers, and valid decode row. K masked scatter respects64-byte
heads; V reuses exact recenter LUT/native delta pack. W4A16 launcher enables
OPT2, with its formerly two-head loops generalized to Llama GQA4. Head64 noQKnorm,
Llama RoPE once before KV publication. No new rotation or HMX W4 weight expansion.

Validation: A8 exact single1/3/16 output/KV; W4 single0 prefill/decode and3/16 pass,
3/16 full outputs and numerical metrics identical baseline. W16 shared-core full16
prefill/decode byte-identical regression. W4 OPT2 conversion/sentinel audit passed.
Full8-step W4 and16-step A8 greedy IDs AND selected FP16/U8 codes match baseline
in every formal run. Exactly8MiB VTCM; W4 peak8330752B, A8 peak7668960B, zero timed
intermediate DDR/spill, one HMX owner. Native W4 all112 transformer projections
plus LM head per A8 token. All480 timed additive Host/DSP ledgers reconcile.

Formal fixed10AB/BA pairs per recipe, same package, confidence95 bootstrap:
W4A16 prefill436.6657 ->1261.0661 tok/s; decode6.35436 ->23.44369 tok/s.
W4A8 OFF prefill215.8940 ->1079.7596 tok/s; decode2.57298 ->16.47835 tok/s.
Candidate/baseline complete Host-wall ratios: W4 prefill.346267 [.344949,.347405],
decode.271048 [.270094,.271933]; A8 prefill.199946 [.199314,.200660],
decode.156143 [.155545,.156634]. All four upper bounds pass1.10 slowdown gate.
M64 prefill; W4 decode7 tokens to EOS, A8 decode15. Includes embedding/16 layers/
finalnorm/head/greedy/FastRPC, excludes loading/frontend tokenizer. Different
continuation lengths are not paired cross-recipe throughput comparisons.

Current scope remains capacity80 and testedM64+7/15; arbitrary-length serving,
segmented native Llama KV and dense R3/R4 unsupported/rejected. No further work
or baseline promotion authorized by closure. User can now choose next direction.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0004/PROFILE.md and
profiling_summary.json;295-file evidence_sha256.json, source/memory copy/hash.
Both baseline-build16 and candidate-build16 archived. Native measured2aa7831;
subsequent common source commits reporting only. Sources docs/LLAMA32_PIPELINE_SPEED.md
and tools/run_llama32_pipeline_profile.py (deploy/gate/formal) reproduce protocol
with a newly approved experiment/immutable destinations. Failed build1-a03 retained.
Original input /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin with six pinned
hashes; generated Llama models/results roots in status. Prior sealed experiments
remain immutable. No active process; next experiment is L32-0005.

Latest user authorization: L32-0005 active no-rotation investigating/fixing A8
relative speed. Prior no-active statements describe L32-0004 closure; see status
and current handoff/protocol. MatchedM64+7 three-arm timing will replace unequal
continuation-length comparisons for this question.
