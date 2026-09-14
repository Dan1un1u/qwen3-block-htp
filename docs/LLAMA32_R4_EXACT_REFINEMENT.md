# L32-0024 exact dense R4 refinement: full16 exact, deployment speed fails

Exact DSP recomputation of the two dense R4 factors eliminates the L32-0023
full16 implementation-reference divergence on the frozen M64+decode1 fixture.
All full16 FP32 outputs and checked integer KV values now match the independent
reference exactly. R3, folded W4 weights, SP2 LUT/scales, residual/norm, attention,
KV and all independent golden files are unchanged. This is not a PPL or model
quality acceptance. The exact control is far too slow for deployment.

## Method and bounds

The FP16 SwiGLU input is an exact integer times2^-24. Decode each finite half
into int64 units. Every output is computed as an explicit dense dot product
against Hadamard signs; no butterfly/FWHT. Stage1 normalization0x29a8 is exactly
181/4096, so sum(units*sign)*181 has lattice2^-36. Stage2 sums the rounded half
values against16 signs and scales by1/4, with lattice2^-26. Explicit integer
round-to-nearest-even yields the frozen FP16 stage boundaries, including
subnormals and ties, without a floating converter's intermediate rounding.
For finite inputs, stage1 integer magnitude is below2^57; int64 is sufficient.
Nonfinite inputs/stage1 overflow are rejected. The exact runtime never reads
reference output or native audit captures to produce its result.

Opt-in `QBH_DENSE_R4=3`, `QBH_R4_OPT=6`, SP2 mode8 and existing FP32 residual.
The normal HMX calls/state transitions still run, after which complete dense
DSP integer recomputation replaces each stage before SP2 consumes it. This is
an explicit expensive numerical control, not a sparse correction algorithm and
not evidence that HMX itself acquired exact accumulation. Default mode1 and
OFF remain available. Scalar dense work explains the large cost.

Stage1 three contexts use disjoint4KiB integer rows in the phase-dead H512 weight
arena after its HMX join; raw input and first output stay in their existing
VTCM arenas. Stage2 refines a completed output slot before its finish workers
consume it. One HMX owner, original8MiB allocation and no intermediate activation
DDR/spill. Exact stage1 branch has no vector stack stores; stage2 scalar frame64B.
The caller's saved vector is a zero constant, not an activation. See disassembly
and `physical-code-review-a01.json` for the bounded code review.

Why not selective refinement first: screening with the existing conservative
component error envelope (relative FP16 ULP plus minnormal, propagated across
both factors) flags98.73%-100% of prefill token/channel columns on the sealed
layer0/7/15 captures, and99.61%-100% on decode. A column is flagged when at least
one of its16 final outputs may cross a SP2 decision. All four observed prefill
SP2 differences are covered, but the bound is insufficiently selective. This
is a feasibility screen conditional on the component envelope, not an
architectural proof of HMX error bounds or a deployed selective algorithm.
Do not shrink the guard to fit the evaluation examples.

## Verification

Portable helper tests: all63488 finite FP16 encodings,453946 rounding cases for
each lattice (including halfway/tie neighbors), and524288 actual stage1 outputs
agree with independent FP64 arithmetic. Native layer0/7/15 each pass raw-input,
R3 component, exact R4-stage1/stage2, exact conditional tail and independent
output gates for prefill/decode. No tolerance was relaxed.

| Device scope | Prefill FP32 mismatches | Decode FP32 mismatches | KV value mismatches | Result |
|---|---:|---:|---:|---|
| Isolated layer0/7/15, each | 0 | 0 | 0 | Pass |
| Consecutive3 | 0 | 0 | 0 | Pass |
| Full16 | 0 | 0 | 0 | Pass |

Full16 minimum row cosine is1 to floating diagnostic roundoff, NRMSE0, all finite.
The previous HMX-only full16 minimum cosines0.5051155954/0.7882501259 and failed
records remain immutable. No R3 repair was needed on this fixture. This paired
intervention strongly supports R4 rounding entering discontinuous SP2/A8
boundaries as the cause of that trajectory divergence; it does not establish
universal R3 exactness or full-model text quality.

Full16 acquires8388608B VTCM, peak8229344B,16 blocks per RPC,zero intermediate
DDR reads/writes/spills and zero additive ledger unattributed ticks. Reference
cache values, prefix/structure and append checks report zero mismatches.
The generic FP16 comparator's zero-element fields are not substituted for
integer cache equality evidence. Frontend/embedding-to-token execution is not
part of these transformer boundary tests.

## Fixed performance result

Ten cyclic OFF/HMXboth/exactboth cycles. Same current binary for all arms.
Each of30 processes executes one warmup and one measured M64+decode1 pair;
120RPC total,60 measured. Complete Host wall; bootstrap20000 seed24024;
paired95%upper<=1.10 both phases. Repeat1 is only each cycle's observation,
never a standalone gate. No optional stopping or discarded outliers.

| Phase | OFF us | Original HMX R3+R4 us | Exact R4 + HMX R3 us | Exact/OFF latency ratio | 95% CI | Gate |
|---|---:|---:|---:|---:|---|---|
| prefill | 2091.568 | 2143.521 | 3100063.614 | 1482.172 | 1366.464-1571.929 | FAIL |
| decode | 1490.563 | 1748.250 | 55266.010 | 37.077 | 35.568-38.461 | FAIL |

Relative to the contemporaneous original HMX R3+R4 arm, exact refinement costs
1446.248x prefill (95%CI1398.396-1493.567) and31.612x decode
(95%CI25.453-36.687). Almost all extra DSP time is in R4, grouped below with
Gate/Up+SwiGLU. The HMX control has noisy Host boundary measurements in this
interleaving with multi-second scalar work; this run does not overturn the
separately sealed L32-0022 ten-inner-replay speed experiment. Its full16 numerical
failure remains unaccepted, irrespective of speed.

Complete modules: units microseconds, parentheses percentage of each arm's Host
wall. R3 is included with QKV+RoPE; R4/SP2/refinement with Gate/Up+SwiGLU.
Frontend modules are N/A outside this scope, not zero-cost measured operations.

### prefill

| Module | OFF | HMX R3+R4 | Exact R4 + HMX R3 |
|---|---:|---:|---:|
| I/O、metadata | 49.13 (2.35%) | 42.70 (1.99%) | 45.48 (0.00%) |
| Input RMSNorm | 76.93 (3.68%) | 76.90 (3.59%) | 77.08 (0.00%) |
| QKV＋RoPE | 236.58 (11.31%) | 181.00 (8.44%) | 181.98 (0.01%) |
| QK–Softmax–AV | 518.74 (24.80%) | 523.71 (24.43%) | 519.64 (0.02%) |
| O projection | 97.79 (4.68%) | 97.83 (4.56%) | 98.23 (0.00%) |
| Post-attention residual＋RMSNorm | 79.08 (3.78%) | 86.71 (4.05%) | 87.07 (0.00%) |
| Gate/Up＋SwiGLU | 328.49 (15.71%) | 532.93 (24.86%) | 3096713.07 (99.89%) |
| Down | 197.77 (9.46%) | 198.88 (9.28%) | 199.21 (0.01%) |
| Final residual | 0.20 (0.01%) | 0.11 (0.01%) | 0.16 (0.00%) |
| KV carrier conversion | 1.63 (0.08%) | 1.48 (0.07%) | 1.46 (0.00%) |
| KV append DMA | 8.50 (0.41%) | 8.52 (0.40%) | 10.11 (0.00%) |
| Block orchestration | 1.11 (0.05%) | 1.16 (0.05%) | 1.11 (0.00%) |
| Layer bookkeeping | 0.86 (0.04%) | 0.86 (0.04%) | 1.03 (0.00%) |
| Stage-boundary bookkeeping | 0.41 (0.02%) | 0.41 (0.02%) | 0.42 (0.00%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.96 (2.34%) | 49.04 (2.29%) | 49.82 (0.00%) |
| Embedding | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Final model RMSNorm | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| LM head＋greedy（不含 final norm） | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Host–DSP 边界 | 445.37 (21.29%) | 341.27 (15.92%) | 2077.74 (0.07%) |
| 完整 Host wall | 2091.57 (100.00%) | 2143.52 (100.00%) | 3100063.61 (100.00%) |

### decode

| Module | OFF | HMX R3+R4 | Exact R4 + HMX R3 |
|---|---:|---:|---:|
| I/O、metadata | 43.35 (2.91%) | 42.95 (2.46%) | 58.82 (0.11%) |
| Input RMSNorm | 53.03 (3.56%) | 52.98 (3.03%) | 53.16 (0.10%) |
| QKV＋RoPE | 100.64 (6.75%) | 70.37 (4.03%) | 70.26 (0.13%) |
| QK–Softmax–AV | 358.34 (24.04%) | 357.98 (20.48%) | 359.60 (0.65%) |
| O projection | 54.80 (3.68%) | 57.89 (3.31%) | 57.05 (0.10%) |
| Post-attention residual＋RMSNorm | 53.63 (3.60%) | 53.83 (3.08%) | 53.59 (0.10%) |
| Gate/Up＋SwiGLU | 293.64 (19.70%) | 343.96 (19.67%) | 53672.70 (97.12%) |
| Down | 158.08 (10.61%) | 157.89 (9.03%) | 157.70 (0.29%) |
| Final residual | 0.14 (0.01%) | 0.13 (0.01%) | 0.09 (0.00%) |
| KV carrier conversion | 3.14 (0.21%) | 3.18 (0.18%) | 3.12 (0.01%) |
| KV append DMA | 5.73 (0.38%) | 5.82 (0.33%) | 6.28 (0.01%) |
| Block orchestration | 1.08 (0.07%) | 1.07 (0.06%) | 2.24 (0.00%) |
| Layer bookkeeping | 0.82 (0.05%) | 0.85 (0.05%) | 0.95 (0.00%) |
| Stage-boundary bookkeeping | 0.42 (0.03%) | 0.39 (0.02%) | 0.55 (0.00%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.79 (3.27%) | 48.87 (2.80%) | 49.96 (0.09%) |
| Embedding | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Final model RMSNorm | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| LM head＋greedy（不含 final norm） | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Host–DSP 边界 | 314.95 (21.13%) | 550.08 (31.46%) | 719.92 (1.30%) |
| 完整 Host wall | 1490.56 (100.00%) | 1748.25 (100.00%) | 55266.01 (100.00%) |

E2E token/s: N/A. The exact candidate fails the authorized singlelayer cost gate,
so frontend and fullmodel E2E were not run. Do not extrapolate these layer times
into model token/s. No PPL or text-quality measurement in L32-0024.

## Sources, attempts and limitations

Arithmetic candidate/full16 source:66a82019dabdcee29a9a099319e946942bcecb8c.
Formal source:9d3ad57167d1ce981e7bf3f1084d87502ce266e2; only profiling tooling
changed, N1 native binaries hash-identical to arithmetic candidate N1.
Post-formal20091a85a7b5751d81a873b23ce66a6703403a68 fixes one diagnostic counter:
exact stage1 work no longer adds256 to `dense_r4_parallel_prepare_tiles`.
The previous counter overcount is preserved in formal records and excluded
from physical conclusions; no arithmetic, main timing ledger, layout or task
schedule changes. A fresh isolated layer0 audit verifies identical arithmetic
after this bookkeeping-only repair. Formal timing is attributed to its actual
profiled HEAD, not a claim of a new measurement after the counter repair.

Results: `/mnt/d/llm_exp/results/llama32-htp/l32-0024`.
No new model files. Reused immutable0019 isolated OFF/BOTH fixtures,0021 chain3,
and0023 chain16; previous0023 ledger33 files reverified. Source tools:
`check_llama32_r4_exact.py`, `llama32_rotated_fp32.py --r4-mode 3`,
`profile_llama32_r4_refinement.py`. Binary/protocol hashes and exact remote
commands are recorded with every attempt. Preserve failed ideal-exact CLI exit1
for the original HMX arm even when its isolated cosine gate passes.

42CLI/144RPC:12 functional/audit CLI (24RPC),30formalCLI (120RPC,60measured).
30exit0 and12 retained original-HMX exact-reference exit1. No build failure or
native crash. One arithmetic candidate (complete exact recomputation), one
subsequent counter-only repair. No default/quality/speed promotion.

## Conclusion and next direction

Keep exact mode3 as a diagnostic arithmetic control. Keep mode1 as the existing
fast experimental implementation with its full16 numerical failure explicit.
This experiment demonstrates an exact runtime computation can match the frozen
reference; it does not deliver an acceptable fast numerical repair.

A useful next experiment should seek a tighter, independently justified error
bound or an HMX-friendly exact/reproducible accumulation representation that
avoids complete scalar recomputation. Both require new implementation evidence;
no speed claim follows from this experiment. Separately, training-only
calibration for perturbation robustness may reduce the chain's sensitivity,
but changes the frozen quantization contract and cannot be fitted to these
validation trajectories. Do not relax the current gate or use captured outputs
as replacement goldens to manufacture acceptance.
