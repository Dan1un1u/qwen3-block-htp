# L32-0013: fresh C-RTN/SP2 adapted to the native HMX contract

Status: COMPLETE. Fresh original-weight construction, new C native gates, full16
text and matched device PPL all ran. Native arithmetic alignment PASSED, but
model quality with the existing integer boundaries remains UNUSABLE.
No quality acceptance, default or baseline promotion is made.

## Scope and provenance

The user requested a new original-weight construction, using the committed
rotation-quant implementation as a read-only method reference. Pinned reference
commit: d9a636ba273fa812b1d6098e9f9df0f8a33eb154. The original model is
`/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin`. Existing rotated/quantized
model artifacts are not reused. The original reference checkout is unchanged.

The historical 17.6423999759 PPL candidate is not available locally. It also
used a different arithmetic contract. The current candidate is explicitly a
fresh native-contract adaptation, not a reproduction of that checkpoint.
Qwen3 and the floating W16A16/W4A16 controls remain frozen.

## Contract decisions

| Boundary | Pinned software method | This hardware candidate |
|---|---|---|
| Backbone112 weights | C learned per-channel SW, RTN signed W4 | Same method, native symmetric codes[-7,7] |
| Ordinary96 input sites | Independent fixed signed INT8 scales | 48 learned scales shared by Q/K/V and Gate/Up, plus separate O |
| Down16 input sites | SP2-A8, 187 signed reconstruction values | Same codebook, producer LUT emits two native U8 planes |
| Rotations | Learned dense R1 and per-layer R2, folded offline | Same topology; no online R3/R4 |
| Residuals/projection outputs | BF16 carriers | Existing native U8 carriers and fixed-point conversions |
| Nonlinear/attention | Floating arithmetic | Existing RMS/RoPE, integer log2 softmax with exact division, native AV |
| KV | High precision | Existing native quantized KV |
| Embedding/head | High precision | Fresh native U8 embedding and per-channel W4 head |
| Runtime fallback | Software simulation | No CPU tensor fallback, W4-to-S8 expansion, or intermediate DDR spill |

Initial training learns R1/R2 and shared SA for100 updates. B only initializes
112 SW tensors. C jointly learns R1/R2+SA+SW for100 updates. Effective batch is
8 sequences of2048 tokens, seed42. Down remains A16 during training; SP2 alpha
is fitted afterward using train-only response-MSE calibration. No validation
sample participates in fitting or checkpoint selection.

Final folding uses FP64 products and the reference BF16 casts after gamma/R1/R2
stages. Reference embedding mean-centering is retained and explicitly recorded;
it is not claimed to be an exact RMSNorm invariance. Training-only R1 products
use the audited FP32 implementation except O, which retains FP64; R2 stays FP64.

## Exact native SP2 representation

For normalized codebook values, let the signed reconstruction integer be v in
[-32768,+32768]. The sparse codebook admits this exact representation:

    v = low + 257 * high - 32770
    low, high are unsigned bytes
    stored_u16 = low | (high << 8)

The native W4 path computes two U8-by-W4 products and merges their raw S32
accumulators with the precomputed output-channel weight sum:

    DownAcc = LowAcc + 257 * HighAcc - 32770 * SumW

The real reconstruction scale is alpha/32768. Both signed endpoints are
representable. No per-group scale or W4 expansion is introduced. The existing
fused SwiGLU lookup directly produces the required physical byte planes.
Mode9 is opt-in; all previous SP2 modes retain their original241-level contract.

## Evidence already established

- All187 codebook values roundtrip, with tie/endpoint and exact dot proofs.
- Independent raw-S32 device probes:4352 exact elements, including extremal
  +/-1879048192 dot outputs; no output saturation masks accumulator errors.
- Transport-only single-layer0/7/15 M64+1 device gates pass exactly, peak VTCM
  8229344 bytes, no intermediate DDR traffic/spill. These use old sealed stimuli
  solely for kernel validation, not for the new candidate's model quality.
- Input-only software SP2 matches the pinned reference on790932 FP32/BF16/FP16
  values. Batched offline attention matches the original scalar integer oracle
  on random, constant and endpoint stimuli; actual candidate stimuli receive
  additional per-config/shape cross-checks.
- Initial100-update history, shared SA identities and17 orthogonal matrices
  verified; maximum orthogonality error6.9098e-7.

All evidence resides under `/mnt/d/llm_exp/results/llama32-htp/l32-0013`.

## Evaluation and interpretation

The first device/software alignment uses a frozen validation-only bridge:
128 evenly spaced M64+16 windows,2048 scored targets, identical token IDs,
positions, resets, weights, quantization parameters and codebooks. The gate
requires exact target codes and per-token NLL absolute difference <=5e-5,
plus the existing physical gates. A8 has no model-quality acceptance threshold.

Compare four matched bridge results: original BF16 teacher, fresh C input-only
software control, exact native integer oracle, and real device. The second
comparison isolates the extra native carrier/head/nonlinear quantization.

Teacher and input-only control additionally evaluate complete WT2 validation,
2048-token disjoint windows including the948-token tail:252852 input tokens,
252728 scored targets. The device bridge is not full WT2-2048 PPL and must not
be compared directly with historical17.6424. Full2048 device evaluation is not
implemented in this experiment's initial bridge.

Before full16 deployment, new C weights must pass layer0/7/15 and consecutive
three-layer gates. All stages retain immutable manifests, source/build seals,
logs and failed attempts. No quality or default baseline is promoted.

## Final model results

| Evaluation scope | Original BF16 | C input-only BF16 control | Native integer reference | Real device |
|---|---:|---:|---:|---:|
| Full WT2 validation2048+tail,252728 targets | 13.64087054 | 17.29825810 | Not run | Not run |
| Matched M64+16 bridge,2048 targets | 23.85653711 | 31.45663199 | 138395.59496349 | 138395.59980512 |

The control preserves BF16 residual/nonlinear/KV/head/embedding, using the fresh
candidate's C weights, shared SA and fitted SP2 alpha. It is not the historical
17.6424 checkpoint. Its full-validation PPL is26.812% above its matched teacher.
The hardware contract additionally quantizes the previously implemented
boundaries. Its quality does not match this high-precision-boundary control.

All3 single layers(0/7/15), consecutive3 layers and full16 generation passed.
Single/three-layer output maximum code difference is0. All2048 target tokens
have exact matching target codes; maximum per-token NLL difference is
1.4944266819583163e-6, below the unchanged5e-5 threshold. Native arithmetic
alignment is therefore established, independently of the very poor quality.

Tested native source: e2b3b91c78b3bd9667d14d16484a129223ce6174.
Peak observed VTCM plan8229344/8388608bytes; no intermediate DDR traffic/spill,
FP16 HMX fallback or W4 expansion. Eleven successful device CLI executions
include the earlier transport/raw probes and six new C validations. There are
2078 recorded model-step boundaries and4 raw-probe RPCs. All20 sequential C
pipeline stages completed successfully; failed earlier precision experiments
and the deliberately stopped slow FP64 training attempt remain preserved.

The device and independent integer oracle produce the same unusable text:

    ,,,,,,,,,tees behindilinicheluthor elim,

Auxiliary speed from the one functional M64+15 run: prefill2075.9773token/s,
decode46.0603token/s. This is not formal profiling, a paired speed comparison,
or a10% speed-gate result. Previous formal performance baselines are unchanged.

## Localized quality problem

Native oracle chat-prefill layer1(secondlayer) output has98.50235% zero entries
and30 entirely zero rows out of64, with U8step0.3745098. The independent BF16
control diagnostic applies each native quantizer in isolation to the normal
floating trajectory, without changing weights or running a cumulative ablation:

| Layer1 boundary | Position0 RMS | Positions1..63 RMS | Ordinary-position zero fraction after U8 | Ordinary-position NRMSE |
|---|---:|---:|---:|---:|
| Down output | 15.16736 | 0.034631 | 99.99845% | 0.999913 |
| Block output/residual | 15.15917 | 0.054952 | 99.89382% | 0.997441 |

The large first-position signal determines the shared minmax scale and removes
almost all ordinary-position information. SP2 protects Down's input, while its
output and the residual still use uniform U8. This loss exists before any HMX
implementation is involved. Global energy-weighted NRMSE conceals it: Down's
aggregate NRMSE is only0.01948, dominated by the accurately represented first
position, despite near-total ordinary-position loss.

The next focused direction is to separate the first-position and ordinary-token
scales at Down output and residual boundaries, then test a controlled PPL
ablation under the native W4 constraint. R1 preserves each token's L2 norm and
cannot by itself eliminate this cross-token amplitude difference. No such
contract change or new optimization has been applied in L32-0013.

C training completed100 updates, mean training loss2.8492821, runtime3651.062s.
All376832 learned output-channel scales are positive, none at the floor;
maximum rotation orthogonality error6.5090e-7. Initial training also completed
100 updates. Original reference source, Qwen3 and floating controls are unchanged.

Artifacts: fresh deployment package
`/mnt/d/llm_exp/models/llama32-htp/l32-0013/frontend-a01`;
complete result/evidence root `/mnt/d/llm_exp/results/llama32-htp/l32-0013`.
See summary.json, device-e2e-a01/alignment.json, the individual gate records,
float-carrier-diagnostic-a01/result.json and evidence-ledger.json. Project memory
records the final report-only source closure and evidence-ledger checksum.
