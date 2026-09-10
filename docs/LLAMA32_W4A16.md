# L32-0002 W4A16 chain

Fresh original Llama weights, per-channel signed[-7,7], one FP32 scale/output.
C64 means65536 freshly tokenized EN/ZH calibration tokens, not group size.
True-sequential GPTQ/act-order/1% damping and per-row selection among three fixed
clipping ranges on calibration output error. Head uses fresh absmax GPTQ from
original tied embedding, while lookup remains FP16 as the established recipe.
NumPy dense-Schur/final-output oracle and packed roundtrip pass.

K8192 adaptation uses DMA2 outside Gate/Up8, exact64-K-tile Down stream regions,
phase-disjoint O/Gate and norm/Down storage. Full per-head M64 scores are kept.
LM-head ping-pong slots and the PPL histogram have separate owners. Peak8330752B,
exact8MiB VTCM, zero timed intermediate DDR/spill. No performance promotion.

Layer0/7/15 prefill/decode and continuous3/16 pass original numerical criteria.
Full16 output NRMSE .00186959/.00254469. Through first EOS all8 tokens match:
The capital of France is Paris. The longer16-token forced-beyond-EOS greedy
replay diverges at step12 and remains failed. It is not relabeled as passed;
software full-context predicts the device token There at that position, while
cached HF generation predicts I. User-facing text ends at EOS. Independent
teacher-forced PPL is a separate run with2048 unique verified target pairs.

Frozen128 EN/ZH Wikipedia documents (M64+16 targets): device PPL31.039101 versus
original BF16 26.697999 (+16.26%). English20.604136/17.787442 (+15.84%);
Chinese46.758855/40.072268 (+16.69%). W4 software31.027335, device+0.038% relative
to the same quantized software. Functional text works, but inherited PPL quality
criteria fail; do not claim quality acceptance or baseline promotion.

Result root /mnt/d/llm_exp/results/llama32-htp/l32-0002, authoritative
validation_summary.json. All failed attempts retained. Model root
/mnt/d/llm_exp/models/llama32-htp/l32-0002. quant-a01, layers-a02, stack3-a01,
stack16-a01 and frontend-a01 are fresh packages. Source entrypoints:
prepare_llama32_quant_data.py, audit_llama32_quant_data.py, quantize_llama32.py,
then existing Llama exporters with --quantized; frontend accepts --dataset.
A8 follows separately without a model-quality threshold per user authorization.

Subsequent L32-0004 speed work is recorded in [LLAMA32_PIPELINE_SPEED.md](LLAMA32_PIPELINE_SPEED.md). Earlier functional timings above remain historical evidence.
