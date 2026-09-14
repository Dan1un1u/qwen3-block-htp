# L32-0026 H512 fast accumulation and rounding

Two bounded fast-HMX candidates did not resolve independent full16 alignment.
Original native implementation is restored. Preserve speed/correctness-first
priority; mode3 exact dense scalar implementation is diagnostic only. No new
model-quality PPL/text work, no default or baseline promotion.

## Frozen contract and hypotheses

Original first factor uses exact FP16 coefficient181/4096 and unity converter,
followed by FP16 H16 sum/4 and the same SP2 LUT. Frozen isolated0019, chain3
0021 and full160023 models/goldens are unchanged. Components<=1ULP+minnormal,
conditional actual integer/FP32 tail exact; independent cosine>=.999; all
physical/provenance checks and10% speed gate retained.

Candidate A,18afc89afc29d280a606a59227f6fd8ca4c4cc68: multiply H512 coefficients
by16 (bits0x39a8), converter1/16. Same exact real dot product, same XOR generation,
layouts and HMX command count. Hypothesis: a higher intermediate exponent range
could reduce loss. Three isolated layers show byte-identical FP32 output files
and identical valid raw/stage1/finalR4 captures to sealed fast control. No benefit.

Candidate B,cb5eed10206ea256a3ac1496892f1947ed1ca7ee: restore original coefficient
and unity conversion; issue the sixteen matched K32 tiles in reverse order,
one tile per instruction pair, retaining the accumulator until a single final
conversion. No intermediate rounding/output or extra GEMM/layout. Same parent
worker ownership. More load-instruction dispatch is possible overhead, unmeasured.
Only Llama mode1 H512 uses the candidate command; other calls retain original.

## Single-layer hardware results

All layer0/7/15 M64+decode1 component, conditional-tail and singlelayer-cosine
gates pass. H16 conditioned on actual first-stage output remains exact. Counts
below are prefill differences to the unchanged ideal R4 on the same raw input.

| Layer | Original/A H512 differences | Reverse H512 differences | Original/A SP2 differences | Reverse SP2 differences |
|---|---:|---:|---:|---:|
|0|31|27|1|2|
|7|27|21|1|2|
|15|6|4|2|0|

Reverse order reduces first-factor differing halfwords in all three samples,
but can increase differences after the SP2 threshold. Layer15 final block is
exact in both phases; layer0/7 prefill NRMSE worsens from7.6786e-7/1.95997e-7
to1.06650e-6/1.47650e-6. Decode block outputs remain exact for all three layers.
This is not stable numerical improvement and must not be selected by layer.
No tuning of accumulation order or LUT to validation code locations was done.

## Consecutive and full16 validation of B

| Scope/phase | Minimum row cosine | NRMSE | Differing FP32 output values | Gate |
|---|---:|---:|---:|---|
|chain3 prefill|0.9999999999245163|1.0669749884e-7|7927|pass|
|chain3 decode|1.0|0|0|pass|
|full16 prefill|0.609073601365145|0.2967764437|116736|fail|
|full16 decode|0.7699031028646253|0.6505855918|2048|fail|

Full16 KV value mismatch counts352930/363758; append-structure checks0. Results
are finite, but the unchanged independent numerical gate fails. Original0023
cosines0.5051155954/0.7882501259 remain historical evidence: B improves prefill
but worsens decode cosine, neither passes. These are frozen-fixture numerical
alignment results, not PPL or usable-text evidence. No fullmodel acceptance.

## Attribution limits and next fast option

For sealed layer0/7/15 captures, a plain exact-sum->FP32->FP16 double-rounding
model accounts for only8 of68 differing H512 halfwords across six phases.
FP32 BLAS reductions also differ from actual HMX and are not a bit-accurate
hardware simulator. Reordering results show sensitivity to K-tile accumulation
order, but do not identify undocumented accumulator mantissa width or algorithm.

Local v79 intrinsic header exposes FP16 conversion/store; no FP32 output route
was established. The bundled v81 HMX PRM describes a37-bit accumulator and
FP16 conversion, but is NOT evidence for the v79 internal arithmetic contract.
Do not enable reserved control bits or accept v81-specific semantics by inference.

A possible next bounded hardware prototype is exact integer-plane H512: represent
existing FP16 values on their exact binary lattice, perform several native HMX
integer dense products with Hadamard +/-1, combine partial sums and round once
to the same frozen FP16 boundary. This adds no activation quantization error;
it is not proven fast or implemented here. It can avoid scalar dense sums while
using original producer/consumer scheduling. First isolate arithmetic and cost.
Offline prefill row statistics: layer0 needs3byte planes for1022/1024rows and4
for2; layer7 all3; layer15 needs2/3/4 for277/695/52rows. Fixed whole-LUT coverage
needs4/4/5planes respectively. Thus a universal two-plane claim is false; any
native design must cover full declared LUT range, signed digits, accumulation
overflow, normalization and RNE. No model/LUT change is authorized by this analysis.

## Restoration and evidence

Forward commitf1162649cdbff56b134dbc8eab9a2517a0c35fec restores all changed native
src/include files exactly to7478516462bf82878838fb3d23a41e220b0f3351. A fresh layer0
audit is byte-identical to the sealed original output and passes all local gates.
Only launcher result-ID support remains. Five builds succeeded.9CLI/18RPC:
1exit0,8original ideal-exact exit1 retained, no crash. Ordinary VTCM8MiB,
peak<=8229344, intermediateDDR/spill0. Explicit audit captures are untimed and
are not performance evidence. No code-generation speed acceptance is claimed
for the rejected candidates. Existing fast0022 speed result remains unchanged.

No formal profiling or E2E/PPL was run for the failed candidate; token/s N/A.
Do not infer candidate throughput from one-shot audited timings. No new models.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0026. verify_evidence_a01.py checks
63 prior evidence files,5 consumed manifests and876 payload entries, build hashes,
restored source, counters and original outputs. Per-run protocols/results and
binaries preserve both candidates. All final evidence is covered by new ledger.
