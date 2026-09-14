# L32-0015 checkpoint: R4-SP2 arithmetic passes, decode ideal cosine fails

2026-09-14. User requested pause/discussion on a material issue. Retain active L32-0015
lock; do not expand layers, begin formal profiling or change the gate without discussion.
Current source: 64dbdffb64eee132be5067ba1596b6f3cbc25930.

## Scope and retained baseline
L32-0012 SP2 mode8, 241 levels, radix256, per-output W4, existing U8 residual/KV/head.
No R1/R2/smoothing or L32-0013/14 weight reuse. No default or quality promotion.
Fresh original BF16 Down inverse full8192 H16 x H512 folding and per-output RTN,
train-only new SP2 activation-MSE alpha. This is an execution-cost fixture, not the
same quantized Down weight tensor as the unrotated GPTQ baseline, nor a PPL comparison.
Prepared layers0/7/15 only. R3 is component-tested only; fullblock integration pending.

## Evidence
14 isolated component CLIs /28 RPCs: six actual cases each for R3/R4 (layers0/7/15,
prefill/decode) and one stress case each. R3 bitexact to independent dense reference.
Each R4 factor passes <=1 FP16 ULP + minnormal; SP2 lookup on actual FP16 outputs
has zero mismatches. Ideal chained SP2 codes can differ when HMX rounding crosses
a nonuniform-codebook threshold. See probe-910c7388/component-summary.json.

Fresh offline transform relative FP64 identity residuals:
layer0 4.327034974288534e-15, layer7 4.3554810353696485e-15,
layer15 4.303301092578079e-15. Packed W4 unpack and signed accumulator bounds pass.
Models in l32-0015/rotated-down-a01 and layers-a01; calibration uses frozen training
hidden trajectory with fakequant Gate/Up, not native fullmodel calibration.

Same-build old baseline layer0-base-a01: prefill and decode exact old reference.
Initial R4 block runs a01 through a04 crash at FP16 HMX deep-load instruction:
0x8000040d, Bad VA FF3FE800. Operand diagnostic a=FF326800,w=FF526800,
out=FF626800,sc=FF72B800, mt=32,kt=16,nt=16. R4 operands aligned to32KiB in
64dbdff; a05 completes both DSP steps. This fixes observed address exception.
Do not generalize exact undocumented hardware boundary restrictions from this alone.
Audit flushing and allocation/dump sizes were also ported to8192 dimensions.
All failed builds/attempts retained. No device reset or baseline replacement.

## Layer0 R4 audit a05

| Metric | Prefill64 | Decode1 past64 |
|---|---:|---:|
| SwiGLU FP16 LUT bits versus frozen Gate/Up input oracle | exact | exact |
| R4 stage1 maximum absolute error | 0.0009765625 | 0.000030517578125 |
| R4 stage1 outside 1ULP+minnormal | 0 | 0 |
| Stage2 conditioned on actual stage1 | exact | exact |
| Actual R4 -> SP2 -> Down -> residual oracle mismatches | 0 | 0 |
| Final output versus independent ideal chain max LSB | 1 | 1 |
| Final output cosine (dezeroed physical output) | 0.9996919891 | 0.9925092578 |
| Retained max2LSB AND cosine>=0.999 gate | PASS | FAIL |

Decode has exactly one different output among2048: channel514 actual83,reference82;
zero-point83, output scale0.1541053951. Reference has58 nonzero channels and squared
norm67 LSB^2; actual57 nonzero channels and squared norm66. This explains the
cosine sensitivity to one LSB; it does not authorize relabeling the gate as passing.
Prefill has31 differing output elements. CLI exit1 remains recorded (its stricter
exact output comparison also fails); conditional arithmetic pass is a separate claim.

VTCM peak8360416/8388608, no recorded timed intermediateDDR or spill; audit uses
explicit diagnostic DDR. R4 HMX calls9 prefill/2decode. Full physical assembly audit
is pending; no formal speed or E2E claim from these diagnostic/repeat1 runs.

## Counts and next decision
20 device CLI attempts:14 component passes,1 baseline pass,4 DSP-crash R4 attempts,
1 completed R4 audit with failed numerical gate (CLI exit1).
36 RPC invocations:28 components,2 baseline,4 failed R4,2 completed R4.
Seven R4 builds (a01 failed unused-variable compilation; a02-a07 successful), plus
the earlier successful isolated-probe build. No formal profiling, fullmodel or PPL.

Recommended discussion: keep the failed result as-is and first investigate the
low-amplitude decode output and SP2 threshold crossing across more diagnostic
samples. Decide whether to repair rounding sensitivity or define an explicitly
amplitude-aware numerical contract. Do not silently loosen cosine or claim speed.
R3 fullblock, layers7/15 fullblock, consecutive3/full16 and fixed-ten speed remain pending.
