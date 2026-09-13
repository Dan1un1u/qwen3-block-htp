# L32-0013: fresh C-RTN/SP2 adapted to the native HMX contract

Status: initial100-update R1/R2+SA training and B scale initialization complete;
C100-update joint training running. Fresh candidate model/device PPL is pending.
This document must not be read as an accuracy acceptance or baseline promotion.

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
