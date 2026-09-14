# L32-0026 active: H512 fast accumulation and rounding

Read docs/experiments/L32-0026.md. User continues original fast path repair;
PPL/text deferred, numerical/physical/speed gates intact.

# L32-0025 closure: original fast HMX is the main line

User explicitly prioritizes speed and correct implementation, improving the
original fast path. PPL/text quality deferred. Mode3 exact dense reconstruction
is diagnostic only; do not restart production from its expensive scalar sums.
Numerical/physical and10% speed gates, independent goldens remain unchanged.

Source /home/daniuniu/work/llama32-htp, branch codex/llama32-no-rotation, closure
7478516462bf82878838fb3d23a41e220b0f3351, clean/pushed. Active none; next L32-0026.
Read docs/LLAMA32_FAST_HMX_NORMALIZATION.md and docs/experiments/L32-0025.md.

One H16 normalization-placement candidate changed +/-1 with converter.25 into
+/-.25 with converter1. Same HMX/layout/SP2/weights. Layer0/7/15 prefill/decode
all local gates pass, outputs byte-identical to sealed0022; raw/stage1/finalR4
valid payloads identical. H16 conditional on actual stage1 was already exact.
Thus no numerical benefit. Restore original native source via forward commit
bcce0f4c3d6a0b66f19cafe7baecaf774ce79c84; fresh restored layer0 audit agrees.
Native src tree equals pre-experiment5bd8f64. Source tools add result-ID only.

4CLI/8RPC, all retained ideal-exact exit1, no crash/build failure.18prior0022
files and3model manifests/309payload entries verified. No new models/goldens,
chain/full16 rerun, formal profiling, E2E or PPL. Reject no-benefit candidate;
no default/quality promotion. R4 HMX calls9prefill/2decode,8MiB/8229344peak,
ordinary intermediateDDR/spill0; captures explicitly untimed.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0025;
ledger SHA256 29eec5c0f7e4f6ed879d1506336665bab4fe3b3ad1b73a0f1e667dceff295818,57files.
Current N1 build seal bcce0f4 differs from docs closure: rebuild after preflight
before any new device run. No active processes. Qwen/otherLlama branch frozen.

Next bounded work: H512 first-factor fast arithmetic/rounding with preserved
pipeline. In prefill layer0/7/15, stage1 differs31/27/6halfwords and SP2
reconstructions1/1/2. Do not repeat H16 placement test or mode3 speed experiment.
Historical0023 full16 fails (prefill cosine.5051, decode.7883); fast0022
passes local+chain3 and singlelayer speed, not fullmodel. Exact0024 restores
full16 on frozen fixture but costs1482x/37x versusOFF, diagnostic only. These
are implementation-alignment results, not evidence of fast-path PPL usability.
