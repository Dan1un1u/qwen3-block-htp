# L32-0019 checkpoint: dense R3/R4 + FP32 residual/SP2 singlelayer passes
Read docs/LLAMA32_FP32_R3_R4.md and docs/experiments/L32-0019.md.
No active experiment; rotation work deferred at the agreed speed gate.
All layers0/7/15 and all three rotation arms pass component bounds, conditional
actual arithmetic and ideal cosine0.999. R4 ideal FP32 comparison not bitexact;
keep CLI exit1 and historical0015failure, no quality promotion.
Fixed10 singlelayer cycles: R4 prefill+35.10%,both+45.39%,both fail; theirdecodepass.
R3 intervals broad (prefill upper1.1933,decode1.3749), no gate pass or stable
large regression claim. No fullmodel rotation/frontend/PPL/E2E. Discuss next scope.
L32-0018 unrotated accepted speed checkpoint remains2069.70/42.51tok/s M64+15.
Default OFF; original BF16, otherrecipes and frozenQwen unchanged.
Source checkpoint b319b63f1eb8823b5bbfbc94ad19cd567d1b73b3
Evidence ledger 48e1658f7a9cd80c8e5d66c8d3733ae272c24f92a247946f51f583e9636d0304
