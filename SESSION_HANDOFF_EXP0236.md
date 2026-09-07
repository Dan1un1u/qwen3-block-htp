# EXP0236 active: W4-only G64 LM head repair

PC053 user-approved, no mixed precision. Read four authorities and EXP-0236.md after bootstrap. Source branch codex/exp-0236-g64-w4-head-repair; sourceHEAD59f2cfb (full via git). Frozen originalG64 transformer; fixedP64 perchannel andH64group128 W4heads. NoDSP/W4A8/MLP changes or promotion.

Fresh1024document16384target dataset frozen SHA256 dd0cc91f86b074efdf0205594bef58f2e78b56ccc39d683847b15d1e8c643920; independent source reconstruction/directtuple32 exclusions passed. Data commands completed session65342. All quantizer oracles and G64 checkpoint/inputnorm/replay checks passed; inputs session89578 completed. Four canonicalCPU replay NRMSE .001771/.002306/.001691/.002372 below.003. All source/command archives retained.

CURRENT live pipeline session14592: run_exp0236_stage.py head P64, then head H64, then evaluate, then summarize. Do not duplicate while process is live. P64 export started atsource045b59c5288133c91a72a0cceb09dacf5b46d178; report-only source additions later do not change quantizer. Progress in exp0236/commands/head_*.log and exp0236/P64/codes.bin. Preserve partial artifacts on interruption, diagnose before any continuation. Model151936x2048 rows;1024rowchunks, all3fullGPTQtrials. No scores yet. Later stages have own clean activepreflight/source archives.

After pipeline completes, run report_exp0236.py CPU directly AFTER stage command JSONs close, then independent allfile/hash and PPL/state verification; copy REPORT/profile into memory, close index/status/handoff and sync. Final new dataset pass/fail/inconclusive and disclosedPC052 regression are separate. No automatic next experiment. Full profiling newdevice sectionsN/A with verifiedEXP230 history.

## Checkpoint: both exports completed, fixed evaluation running

P64 complete551.6s, manifest6386a4d00e5bcad538da1bb8ea52a436e2add479d4467cbbf202acadd52ab1b1; H64complete549.1s, manifest64e9bc23b8c9b0405117cd8b1aa1b32e9720a6810f7930fa126a33dd95729ad2. Both all151936rows independent NumPy/FP16 exact. ActualP64source045b59c5288133c91a72a0cceb09dacf5b46d178; H64source0535c28b1f881524dfa3d01547ab62cbe148be25. Quantizer source hashes identical except declaredperchannel/group method; unchanged old corefiles.

Session14592 now GPU evaluate then summarize, actualsource1d0ac8c810739839d58b345d853b8f64d0a85783. Fresh finalscoring started: no further exports/tuning. F development exact25.337473379675465; newfinal F24.763138630742063. Await G64/P64/H64 fixedscores and G64sentinel. Do not rerun finished exports. After scores/summary and stage JSONs complete, execute report_exp0236.py and verify_exp0236_closure.py; update authoritative closure. Sourcehelper files allcommitted/pushed. No user confirmation needed for routine recovery withinPC037.
