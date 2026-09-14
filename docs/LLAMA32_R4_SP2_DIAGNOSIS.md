# L32-0015 low-energy output and SP2 threshold diagnosis

2026-09-14. User authorized bounded diagnostic continuation; weights, scales and
native code stayed frozen. No formal timing, consecutive layers, fullmodel or PPL.
Native captures use64dbdffb64eee132be5067ba1596b6f3cbc25930; diagnostic source
closure: 43da04312097a56bc1fd872b22314fd62ac9043a.

## Main findings

The residual output mismatch is reproducible. It is explained by FP16 R4 rounding
crossing SP2 nearest-neighbor boundaries, followed by crossing a Down Q31 half-LSB
threshold. No SP2 decoding, native packed W4 dot or integer residual implementation
mismatch was found. The resulting output cosine is very sensitive because the
existing U8 scales erase most low-energy signal. This is an implementation/numerical
diagnosis on frozen single-layer stimuli; not a model-quality endorsement.

Layer0 repeat has exactly the same two native output arrays and all three active
R4 audit regions as the prior a05 run. New layer7 and layer15 captures are diagnostic
samples, not promoted subsequent validation gates.

All195 rows (three64-token prefill blocks plus three decode tokens):
-399360 output elements,37 differ from the independent ideal FP16-factor chain;
 maximum2LSB (layer15 output ratio converts a1LSB Down difference into2outputLSB).
-187/195 rows are bitexact;8 have output differences.
-57 rows have both dequantized outputs all-zero (layer7,56prefill+1decode).
-Of138 rows with defined cosine,6 fall below0.999, all layer0; every one of these
 six has exactly one1LSB output change. Prefill rows21,45,46,58,62 and decode row0.
-The whole64-row pooled prefill cosine hides those five low-energy row failures.
-All conditional actual-R4 -> SP2 -> signed Down dot -> Q31 -> Q14 references
 exactly match all399360 native outputs.
-1893/1597440 SP2 values differ versus the ideal R4 chain; every changed pair
 straddles the relevant codebook midpoint. No unknown/non-codebook codes.
-The old isolated probes used old SP2 alpha; this fullblock package uses the
 newly calibrated rotated alpha. Their threshold-crossing counts are not comparable.

| Layer / phase | Exact output rows | Zero / undefined rows | Defined cosine <0.999 | Max LSB | Minimum defined row cosine |
|---|---:|---:|---:|---:|---:|
| 0 / prefill | 58/64 | 0 | 5 | 1 | 0.9899494936611665 |
| 0 / decode | 0/1 | 0 | 1 | 1 | 0.9925092578236596 |
| 7 / prefill | 64/64 | 56 | 0 | 0 | 1.0 |
| 7 / decode | 1/1 | 1 | 0 | 0 | N/A |
| 15 / prefill | 63/64 | 0 | 0 | 2 | 0.9998648042103423 |
| 15 / decode | 1/1 | 0 | 0 | 0 | 1.0 |

## Exact layer0 decode channel514 trace

R4 output relative L2 error is0.01962%; matched floating Down error beforeSP2
is0.01667%. The22 changed SP2 inputs increase matched floating Down error to0.2869%;
after adding the same frozen left residual in floating arithmetic it is0.23443%.
Cosine of that continuous residual remains0.99999725.

One decisive SP2 input is channel4394:
idealFP16=0.03973388671875; actualFP16=0.039764404296875.
SP2 boundary=0.0397639274597168; levels2304 ->2560, alpha1.635029911994934e-5.
Its Down weight code for output514 is2, hence signed-dot delta+512.
All22 changed inputs together contribute net signed-dot delta+364:
527920 ->528284, exactly reproduced by independent integer calculation.

| Boundary | Ideal-factor chain | Actual-HMX chain |
|---|---:|---:|
| Q31 Down before rounding, centered LSB |0.499776266515255|0.5001208614557981|
| Down U8 code (zero81)|81|82|
| Residual Q14 before rounding, centered output LSB|-0.50079345703125|0.48248291015625|
| Final output U8 code (zero83)|82|83|

Thus a0.000344595 Down-LSB change crosses0.5 and becomes one full quantized code.
There is no integer overflow or wrong rounding implementation in this trace.
Reference output has only58 nonzero channels, energy67LSB^2; actual57 and66.
One lost -1 component makes cosine exactly sqrt(66/67)=0.9925092578.
The absolute output RMSE is0.0034053 (0.0220971 outputLSB), but normalized error
is12.2169% because the quantized vector itself has little energy.
Do not call this zero actual error or relabel the original failed gate as passed.

## Baseline quantization floor and local boundary controls

Layer0 decode Down scale0.15153186; residual output scale0.15410540. This is about
4.05x the RMS of the pre-store continuous residual on this diagnostic token.
Layer7 decode output scale2.99705887 versus local continuous residual RMS0.00117354,
a factor2553.87. It becomes all zero; all57 zero rows also occur in the unrotated
L32-0012-mode8 reference. That baseline uses different Down weights, so this is
zero-pattern context, not a paired model-quality comparison.

Software-only local controls retain the same already-U8 left residual, W4 weights,
SP2 input and scales; compare to dequantized signed Down plus that left residual.
They do not recover earlier quantization loss, execute a fullmodel, or test PPL.
No range is fitted on these samples.

| Local tail / storage | Layer0 decode relative L2 | Layer7 decode relative L2 | Layer15 decode relative L2 |
|---|---:|---:|---:|
| Current U8 Down + Q14 residual + U8 output | 89.15727% | 100.00000% | 39.83529% |
| U8 Down + floating add + U8 output | 89.15727% | 100.00000% | 39.83529% |
| Floating Down + floating add + same U8 output | 83.74766% | 100.00000% | 38.78708% |
| Floating Down + floating add + FP16 storage | 0.02024% | 0.02039% | 0.02080% |

Replacing Q14 by floating addition yields identical final codes on all six captures.
Removing only the intermediate Down U8 store helps a little, while the final U8
residual still dominates these low-energy cases. This supports investigating a
higher-precision residual boundary and quantizing after norm, rather than merely
making the adder floating or demanding bitexact floating rotation.
This remains a prospective architecture direction, not a migration or speed result.

## Metric correction and decision boundary

Previous runner returned cosine0 for two exact zero vectors via epsilon denominator,
then labeled the cosine gate false even though the CLI was exactly correct.
New runner reports cosine=null and ideal gate=null with explicit
undefined_cosine_exact_output status. It neither grants a new pass nor alters
max2LSB/cosine0.999 thresholds. Historical raw reports stay immutable.
New analysis computes per-token and pooled cosine separately, reports zero-vector
cases, absoluteRMSE/LSB and pre-store continuous error; no epsilon-derived fakecosine.

Recommended next decision:
1. Keep exact actual-arithmetic validation mandatory. Full captures pass it.
2. Treat zero-vector cosine as not defined and separate metric validity from
   output equality. For nonzero low-energy outputs, discuss an explicitly
   quantization-floor-aware gate before allowing speed progression.
3. Do not prioritize extra R4 recomputation to chase one output code. If accuracy
   is the priority, first test high-precision Down/residual until after prenorm;
   current local controls show why retaining final residual U8 defeats most benefit.
No gate modification, profiling progression, default or quality promotion was made.

## Evidence and counts

Complete inputs and per-coordinate traces:
rounding-diagnosis-a02/summary.json, tokens.csv, and layer*-*.json.
Diagnostic a01 is retained; a02 adds old-baseline zero patterns and tail controls.
Layer0 native repeat, layer7 and layer15 used the same sealed native build and
unchanged package hashes. Three additional device CLIs/six successful block RPCs.
L32-0015 cumulative:23 CLI attempts,42 RPC invocations (28 components,10 completed
block steps,4 crashed block attempts). Of23 CLIs,16 exited0,3 completed arithmetic
but exited1 due ideal-reference differences,4 crashed in earlier preserved attempts.
No native build this continuation; source changes are Python diagnostic tools only.
Active L32-0015 lock retained pending numerical-contract/next-direction discussion.
