# L32-0024 closure: exact R4 fixes full16, cost fails

Source `/home/daniuniu/work/llama32-htp`, branch `codex/llama32-no-rotation`,
closure `5bd8f641735291603cd2b595ae713146b4bd0c6e`, clean/pushed. Active experiment none; next ID L32-0025.
Read `docs/LLAMA32_R4_EXACT_REFINEMENT.md` and docs/experiments/L32-0024.md. Earlier0023 failure report
is preserved; no need to repeat its attribution experiments.

Implemented opt-in QBH_DENSE_R4=3 with R4_OPT6/SP2mode8/FP32 residual. Original
HMX calls still run; exact DSP int64 dense sums replace both stages before SP2.
FP16 lattice2^-24, first normalized coefficient181/4096, explicit RNE at lattice
2^-36; second H16 sum/4 at2^-26. No butterfly, new weights or changed goldens.
This is complete expensive recomputation, NOT sparse or deployment-ready repair.
Conservative component-envelope screen marks98.73%-100% of token/channel
columns; it cannot support efficient sparse refinement. Do not fit guards to
validation examples or claim an architectural HMX error-bound proof.

Singlelayer0/7/15 and consecutive3/full16 M64+decode1: output FP32 and KV values
exact, all finite, component/conditional gates pass. Only R4 changed; R3 unchanged.
Full16 tested `66a82019dabdcee29a9a099319e946942bcecb8c`. Mode1 still fails0023 full16;
mode3 restores this frozen-fixture numerical alignment, not text/PPL acceptance.
Neither should be promoted to usable full frontend by inference.

Fixed10 cyclic OFF/HMXboth/exactboth timing,30CLI,one warmup+one measured pair
perprocess (repeat1 not gate),seed24024 bootstrap20000. Formal source
`9d3ad57167d1ce981e7bf3f1084d87502ce266e2` has same N1 native hashes as initial candidate.
Exact/OFF prefill ratio1482.1723 (95%CI1366.4639-1571.9293), decode37.0773
(CI35.5676-38.4606). Exact/HMXboth1446.2484/31.6122. Massive speed gate failure:
retain mode3 as diagnostic only. No frontend/fullmodel E2E/PPL, no extrapolation.
Full stable module tables in report. Original0022 speed evidence remains at its
own measured scope; current HMX control noisy after long scalar calls is not a
replacement baseline.

One post-formal counter-only repair `20091a85a7b5751d81a873b23ce66a6703403a68` excludes
exact recomputation from dense_r4_parallel_prepare_tiles (removed256 overcount
perlayer); latest layer0audit output/component exact and byte-identical to
pre-fix. Full16/performance claims pinned to actual earlier heads. No arithmetic,
layout or scheduling change from that counter fix; original formal counters
preserved with caveat. Source build currently N1 sealed at counter-fix HEAD;
closure docs differs, rebuild with successful preflight before new device work.

Evidence `/mnt/d/llm_exp/results/llama32-htp/l32-0024`; ledger `evidence-ledger-checkpoint-a01.json` SHA256 `fc13eb282b1b7b26253d4a64986988154e4604d32c0d9b7a405fcbe6865fd2c6`.
238result files,six consumed model manifests/969payload entries verified;
previous0023 ledger33files/20manifests reverified.42CLI/144RPC,30exit0/12exit1;
formal120RPC/60measured. No native crash/build failure. Ordinary no intermediate
activation DDR/spill;8MiB allocation/8229344peak. Explicit audit buffers remain
untimed. No models generated; reused0019isolated,0021chain3,0023chain16 packages.

Next discussion: tighter independently supported refinement criterion or HMX-
friendly exact/reproducible accumulation, avoiding scalar dense recomputation.
Training-only robust calibration is a separate changed-quantization experiment.
Keep gates and independent goldens unchanged. Qwen and other Llama branch frozen.
