# Llama 3.2 1B Instruct W4A8 OFF

L32-0003 connects the unrotated native integer chain, with optimization deferred.
Model quality has no acceptance threshold in this experiment. This does not
relax exact implementation, provenance, KV lifetime or physical checks. The
W4A16 quality failure in L32-0002 remains separate and has not been promoted.

Weights are the freshly original-derived C64 per-output-channel GPTQ W4 tensors
sealed by L32-0002; no group scales. A8 uses fresh static affine min-max ranges
from the same frozen Llama 65536-token calibration set, with zero included and
no clipping search. Transformer ranges use every calibration position. Head
ranges use the last position of all512 calibration documents. Heldout documents
are never calibration or model-selection inputs. The tokenizer/chat template,
embedding, final norm and tied-head source all come from the original Llama.

Llama has no Q/K RMSNorm. The head64 RoPE kernel dequantizes and rotates in SF32,
then requantizes, once before KV publication. The integer GQA4 attention uses
log2 exponents (fraction3, widened differences), exact integer normalization,
U8 probability and recentered S8 K/V carriers. QK/AV use integer HMX; all seven
linear projections per layer and the LM head use native packed W4 HMX in both
prefill and decode. SwiGLU emits its freshly calibrated U8 LUT output directly.
No intermediate FP16 attention or W4-to-S8 weight expansion is used.

The initial supported execution is M64 prefill followed by continuous one-token
decode through length79 in a capacity80 cache. KV is head-major U8; the older
head128 segmented layouts and dense R3/R4 are explicitly rejected for Llama.
Decode score/probability VTCM planes are sized for GQA4 physical rows, rather
than being incorrectly equated to the smaller head64 K/V planes. FFN8192 bundle
and tile-count metadata are model-specific. Exactly8MiB VTCM is acquired, with
peak7668960B; no timed intermediate DDR or spill. One FastRPC per token-boundary
unit, one HMX owner, full16-layer execution. Scope is functional integration,
not formal speed profiling and not general arbitrary-length chat serving.

## Evidence and reproduction

Authoritative results are under /mnt/d/llm_exp/results/llama32-htp/l32-0003;
model payloads are under /mnt/d/llm_exp/models/llama32-htp/l32-0003. Always run
project-memory bootstrap/preflight and obtain an active approved experiment
before executing tools. Never overwrite an existing output directory.

1. calibrate_llama32_u8.py creates the immutable fresh calibration report.
2. export_llama32_u8.py creates independent arithmetic layer0/7/15 fixtures.
3. prepare_llama32_u8_stack1.py --output PATH wraps layer0 in the native replay
   ABI. Do not use the standalone floating-layer launcher for direct W4A8.
4. export_llama32_u8_stack.py --layers 3|16 --output PATH creates exact consecutive
   references; run_llama32_stack.py runs the corresponding build. Build with
   scripts/build_llama32.sh 1|3|16.
5. prepare_llama32_u8_frontend.py creates fresh A8 embedding/head metadata and a
   16-token independent greedy oracle; run_llama32_frontend.py checks every
   generated token AND U8 logit code, then evaluates the frozen independent PPL.
   The historical native A8 host pass field alone does not check expected IDs;
   the Llama runner therefore enforces the additional arithmetic comparison.
6. audit_llama32_u8_nll.py computes an independent first-target NLL for the first
   English and Chinese heldout documents, validating code-to-NLL conversion.

Actual projection accumulators are independently computed from signed W4 codes,
then converted with SDK19.0.07 libnative HMX emulation. RMSNorm uses Llama eps1e-5,
RoPE omits gamma, residual uses the runtime Q14 rounding, and SwiGLU uses the
fresh LUT. The1/3/16-layer prefill and decode references match the hardware
outputs and persistent KV exactly. Full feedback generation matches all16 IDs
and selected logit codes; current output repeats Sleep and is not usable text.
W4A16 shared-runtime regression retains exactly its previous error metrics.

Failed CLI/allocation attempts and the failed decode scratch attempt remain in
the evidence directory. Later successful attempts do not replace them. Refer
to validation_summary.json for final PPL, physical audit and auxiliary timing.
