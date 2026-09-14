# L32-0023 full16 dense rotation validation

The repaired L32-0022 R3+R4/SP2/FP32-residual implementation fails the unchanged
full16 independent numerical gate. OFF remains bit-exact for both M64 prefill
and the following decode token. Frontend generation, formal profiling, E2E and
PPL were not run. No default, quality or performance baseline is promoted.

## Implementation and provenance

Tested source/build HEAD: `022ded9aeeeeb13189831a259e43c0bebfa97a6e`, N16.
Native source is unchanged from L32-0022: normalized dense H512 weights and
unity conversion, dense R3 OPT2, R4 OPT6, SP2 mode8, FP32 residual. The only new
source changes route model-generation/reference tools to L32-0023 and reuse
verified prior folds. No native arithmetic repair was attempted in this experiment.
The successful three-layer result and single-layer speed of L32-0022 remain
valid at their measured scopes; they do not establish full16 correctness.

Generated missing Down folds for layers3-6 and8-14 from original Llama BF16;
reused verified0/1/2/7/15. Same frozen eight training windows (1024 tokens),
five-range grid, per-output-channel signed[-7,7] absmax RTN and normalized
H16 tensor H512 method. Transform relative L2 is about4.26e-15 to4.39e-15.
No heldout fitting, smoothing changes, grouped weights or Qwen weight reuse.
The full16 reference independently evaluates FP64 dense factors rounded to
FP16, followed by the integer/FP32 runtime contract. No native captures are
reference goldens. Prior L32-0022 ledger and all249 files/six model manifests
were reverified; see `prior-verification.json`.

## Full16 hardware result

Shape: M64 prefill, then decode1 with past64/capacity80. These are transformer
boundary replays, not embedding-to-token frontend runs. Threshold: minimum
row cosine>=0.999, with existing component and conditional-exact rules retained.

| Arm | Phase | Minimum row cosine | NRMSE | Unequal FP32 output values | Result |
|---|---|---:|---:|---:|---|
| OFF | Prefill | 1 (bit-exact) | 0 | 0 | Pass |
| OFF | Decode | 1 | 0 | 0 | Pass |
| R3+R4 | Prefill | 0.5051155954 | 0.3883987963 | 129024 | Fail |
| R3+R4 | Decode | 0.7882501259 | 0.6913102269 | 2048 | Fail |

All outputs finite. Original BOTH CLI exit1 and reference outputs are retained.
Rotated cache mismatch counts against the independent trajectory are581478
and592410 (prefill/decode); structural, preserved-prefix and append checks
report zero mismatches. The generic FP16 cache comparator fields report zero
compared elements and are not evidence of cache-value equality.

Both ordinary runs execute16 blocks per token-boundary RPC, acquire8388608
VTCM bytes, report zero intermediate DDR reads/writes/spill and zero
unattributed ledger ticks. BOTH peak plan8229344 bytes; OFF8098272. Rotation
calls are16 each per phase. These observed physical counters do not override
the failed numerical gate. No new physical implementation was introduced.

## Boundary diagnosis

1. Reused the sealed L32-0022 layer0 audit, whose input/native arithmetic and
   fold match the full16 first layer. Compared actual R4 with independent dense
   factors on identical SwiGLU input:31 first-stage FP16 and48 final FP16 values
   differ, but only one SP2 reconstruction code differs. At token48/channel3719,
   native0.00017976760864257812 versus ideal0.0001800060272216797 differs by
   2.384185791015625e-7; SP2 becomes10 versus12. First-layer output max absolute
   error is3.2186508178710938e-6.
2. Injected only the audited actual layer0 output into otherwise independent
   software layers1-15. Prefill first fails at layer index3 (cosine0.9831499745),
   ending at0.5574805806; decode ends at0.7860993849. Decode layer0-2 outputs
   are exact in this diagnostic, but earlier prefill KV differences affect the
   next layer. This demonstrates amplification without requiring subsequent
   native pipeline errors. It is not a complete conditional hardware proof.
3. Captured all16 actual prefill hidden boundaries using existing untimed
   `QBH_HIDDEN_CAPTURE=1`. Final capture is byte-identical to the ordinary
   full16 native output. For each layer, independently computed the layer on
   the same preceding actual hidden input. All local minimum cosines are at
   least0.9999999710; worst local NRMSE3.7257169e-6. Layer index3, where the
   global trajectory first fails, is locally bit-exact. This separates local
   implementation discrepancy from accumulated trajectory divergence; the
   local comparisons do not replace the fullmodel gate.
4. At layer index3 input RMSNorm, only five A8 codes cross a rounding threshold,
   each by one code (tokens31,44,48). For example104.500022888 versus
   104.499900818 becomes105 versus104. FP32 residual storage preserves small
   differences; it does not prevent post-Norm A8 discontinuities.

| Layer index | Same-input local min cosine | Global independent min cosine |
|---:|---:|---:|
| 0 | 0.999999999801 | 0.999999999801 |
| 1 | 1.000000000000 | 0.999999999949 |
| 2 | 0.999999998751 | 0.999999998751 |
| 3 | 1.000000000000 | 0.976865913688 |
| 4 | 0.999999999908 | 0.950921599727 |
| 5 | 0.999999971005 | 0.929062858668 |
| 6 | 0.999999976671 | 0.776439206693 |
| 7 | 0.999999999975 | 0.789114785862 |
| 8 | 1.000000000000 | 0.778343721148 |
| 9 | 0.999999999988 | 0.768693806235 |
| 10 | 0.999999999998 | 0.697073505423 |
| 11 | 0.999999999693 | 0.675210374834 |
| 12 | 0.999999978800 | 0.654542620279 |
| 13 | 0.999999999983 | 0.603885865347 |
| 14 | 1.000000000000 | 0.554526796952 |
| 15 | 1.000000000000 | 0.505115595377 |

The evidence supports R4 rounding -> SP2 threshold crossing -> small residual
perturbation -> post-Norm A8 crossing -> attention/MLP and KV amplification.
It does not prove every native component exact across16 layers and does not
establish model quality or a general conclusion that rotation is ineffective.

## Diagnostic capture accounting

Three hardware CLI invocations, five RPC boundaries: OFF2, BOTH2, capture1.
Two CLI exit0, one retained numerical exit1; no build failures or native crashes.
Hidden capture writes8388608 diagnostic DDR bytes using16 descriptors. It is
explicitly excluded from formal physical/performance evidence. Legacy filenames
`actual_hidden_stack_u8.bin` and `actual_full_stack_output_u8.bin` contain FP32,
with shapes16x64x2048 and64x2048. Capture `gate_pass` means capture integrity,
not the independent numerical gate. The ordinary replay tool's generic scope
string mentions conditional arithmetic, but no full16 native-boundary audit
was run; only the explicitly described diagnostics support attribution.

## Reproduction and artifacts

Results: `/mnt/d/llm_exp/results/llama32-htp/l32-0023`.
Models: `/mnt/d/llm_exp/models/llama32-htp/l32-0023`.
Use the frozen manifests and source HEAD above. Under an authorized preflight,
from the source worktree, scientific Python is
`/home/daniuniu/work/rotation-quant/.venv/bin/python`:

```sh
python tools/prepare_llama32_rotation_extension.py --result-experiment l32-0023 --layers 3,4,5,6,8,9,10,11,12,13,14 --attempt NEW_UNIQUE_FOLD_ATTEMPT
python tools/prepare_llama32_rotated_chain.py --result-experiment l32-0023 --layers 16 --extension NEW_UNIQUE_FOLD_ATTEMPT --attempt NEW_UNIQUE_CHAIN_ATTEMPT
bash scripts/build_llama32.sh 16
python tools/llama32_fp32_residual.py run --layers 16 --result-experiment l32-0023 --package chain16-a01 --attempt NEW_UNIQUE_OFF_ATTEMPT
python tools/llama32_rotated_fp32.py run --result-experiment l32-0023 --arm both --package /mnt/d/llm_exp/models/llama32-htp/l32-0023/NEW_UNIQUE_CHAIN_ATTEMPT --attempt NEW_UNIQUE_BOTH_ATTEMPT
```

Original attempts are `rotated-down-extension-a01`, `chain16-both-a01` and
`chain16-off-a01`; do not overwrite them. Exact deployed commands/binary seals
are in each `protocol.json`. Diagnostic JSONs: `layer0-sp2-threshold-crossings-a01`,
`layer0-injection-propagation-a01`, `prefill-local-layer-comparison-a01`, and
`layer3-norm-threshold-crossings-a01`. `hidden-diagnostic-a01` holds the actual
boundary capture and exact invocation. Reconstruct local comparisons with
`llama32_rotated_fp32.layer` on the preceding captured FP32 hidden, comparing
separately to each frozen `rotation_0_hidden{i}.npy`. Injection uses the sealed
0022 layer0 outputs, then independent layers and persistent per-layer KV for
prefill/decode; neither procedure changes frozen golden files.

## Next decision

Stop frontend/formal expansion under the registered numerical-failure rule.
Retain the L32-0022 native fix; reverting to older code already reproduced a
three-layer failure. Investigate a general R4-to-SP2 boundary refinement that
recomputes ambiguous dense dot products more accurately, with a justified
error bound and unchanged frozen reference. It must address both R4 stages;
correcting one observed code or fitting a LUT to validation inputs is invalid.
Its speed cost and whether it resolves all16-layer failures are unmeasured.
If that is insufficient or too costly, separately consider training-only
calibration for robustness to rounding perturbations; that changes the frozen
quantization contract and requires a new aligned experiment. Do not accept
native captures as goldens, loosen0.999, or infer E2E from layer timings.

Formal profiling: N/A. E2E token/s: N/A (numerical gate failed before frontend).
PPL and text usability: not measured in this experiment.
