# EXP0254 running checkpoint

User approves ordered joint attention restoration and conditional attribution. PC068. Branch codex/exp-0254-joint-attention, source7ad7c2dbebb7fc369d3570a5d31b862548cdc850. Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0254. Original per-channel C64 weights/EXP246 R3/static parameters/prefix and EXP253 FP32 TwoSum core frozen. No hardware or model export.

Completed data_exp0254.py prepare/audit, attention_exp0254.py freeze/numerical.256fresh documents split into independent triage128 and confirmation128,4096total targets; freeze080415e6f0cf961fddb52c7ce48fe0917f525180a5a94fe6df1ae9e4311197e4. Retained280 numerical cases plus12 new FP16 high-range causal cases pass.

Sequential runner currently in development_01.log, then triage -> route -> branch_development -> confirmation using /home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python scripts/attention_exp0254.py PHASE. Check running processes/logs before restarting; no duplicates. All commands have exclusive *_01.log outputs; preserve failures if any. Joint J restores QKV projection outputs, postRoPE QK/cache, V tied output/cache and context output. Mixed U8/FP16 cache oracle, hook sentinels/full counts, repeated/causal/CE and model/prefix immutability checked per evaluated arm. F/C64/B development exact parent reproduction mandatory.

After triage, route.json seals the prespecified choice: J/C64<=1.05 overall and<=1.10 eachcell selects attention branch F,C64,B,J,K,QK,V,C; otherwise downstream factorial F,C64,B,J,R,S,RS,JR,JS,JRS. Routing is not acceptance. Confirmation untouched until route frozen, no final-based additional candidate. Missing branch development arms then independent confirmation follow automatically under user approval.

Prepared report_exp0254_candidate.py in results; review and install source after inference, add independent integrity reconstruction, report/nextdirection/profile N/A, source/provenance/evidence seal and memory closure. No deployable A8 candidate yet; restoration quality cannot authorize hardware profiling of FP16 diagnostics. Discuss concrete A8 repair after attribution; native W4/>10percent slowdown/numerical/model gates unchanged.

## Route checkpoint

Triage complete: {"F": 21.34211023294927, "C64": 22.532612616101396, "B": 25.878157651192183, "J": 26.368808132806105}. Joint/C64=1.1702508085530852, all fourcells exceed1.10. Predeclared downstream branch selected, route.json SHA256 678f70521ff577e71e62fbbd7170f81528c32d1d1a435d164aadd764efa8cf06. Confirmation remains untouched at selection. Runner now executes branch_development R,S,RS,JR,JS,JRS then confirmation F,C64,B,J,R,S,RS,JR,JS,JRS. Preserve frozen route; no attention-branch additions from later scores.

## Closed checkpoint

All running instructions above historical. Ordered phases, independent audit and evidence/source/memory closure completed. No inference/device process remains. Active none,next255. Report docs/experiments/EXP-0254-RESULTS.md; direction docs/W4A8_NEXT_DIRECTION_AFTER_EXP0254.md.
