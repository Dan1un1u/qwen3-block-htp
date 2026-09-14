# L32-0018 completed: both FP32 speed gates pass
No active experiment or running job. Next experiment19; no new rotation/PPL run started.
Source e3d065a515221aa1e97c63fa4d58d23e434c45df, codex/llama32-no-rotation, clean/pushed.
Read docs/LLAMA32_FP32_PIPELINE_FOLLOWUP.md and docs/experiments/L32-0018.md.
L32-0012 original SP2 control, fixed10 same-build AB/BA M64+15:
prefill2002.83 ->2069.70tok/s,Host−3.23%,ratioCI[.953911,.977237];
decode45.97 ->42.51tok/s,Host+8.13%,ratioCI[1.071833,1.092762]. Both pass10%.
Norm direct masked HVX native stores and per-row exact repair; O double raw-slot
HMX/HVX epilogue overlap. No arithmetic reassociation, no weights/scales/rotations changes.
Layers0/7/15,3,16 andgenerationexact;29CLI/28success/1Oguardfail retained,
1pre-CLIADBfail;364successful boundaries/320formal.8MiBVTCM,nointermediateDDR/spill.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0018;ledger 5572319f6bb94f623089199ef729a13c3c5d20b82134be14e75028ac707bd734. Complete module tables PROFILE.md.
Current fixed-scale text still unusable; no PPL/quality/default promotion.
Do not infer FP32 intrinsically faster from prefill: existing GateUp trajectory/layout
advantage contributes2.452ms. Decode extra Norm remains1.49ms; speed goal now met.
L32-0015 rotations deferred. Qwen3 and Llama rotation branch remain frozen.
