# EXP-0305: vector floating-intermediate motivation profiling

Status: completed; own-reference numerical and physical gates pass. Diagnostic result only, adoption pending; no production baseline, workbook, quality or PPL promotion. No speed gate.

## Configuration and attribution

Qwen3-1.7B, all 28 layers, fixed 64-token prompt plus one fixed subsequent decode. Use the verified EXP0284 uniform-INT16 Down package: native per-output-channel W4 projections, no rotation, FP32 residual and RMSNorm. Q/K Norm/RoPE are unchanged. QK and AV remain integer matrix operations; ordinary linear boundaries remain A8, Down input uniform INT16. This is a floating-nonlinearity diagnostic inside the existing low-bit runtime, not an all-floating Transformer or a fully integer residual implementation.

Replace the lookup-based SwiGLU with a fused HVX FP32 DQ -> SiLU/multiply -> INT16 quantization -> native low/high-byte production. Use a vector range-reduced exponential polynomial and reciprocal refinement, not scalar per-element exp/division loops. Softmax uses existing QHL HVX FP32 exponential and normalization between native U8 score/probability boundaries. Preserve projection weight DMA/HMX paths, native buffers, and vector parallelism. No synthetic standalone QDQ tensor pass or intermediate DDR tensor is introduced.

To obtain additive percentages, use explicit QK -> Softmax -> AV phase boundaries, with four vector contexts, and Gate/Up -> SwiGLU boundary with three SwiGLU workers. Qwen retains eight group-owned native K/V workspaces; Llama uses four scratch slots per group batch. Phase timers include dispatch and join. These are costs of the operator-boundary diagnostic schedule, NOT critical-path shares of the fully overlapped production schedule. Worker time sums must not be stacked with wall-clock stages. The Llama pipelined run is retained separately as a demonstration of this attribution limitation, not a speedup baseline.

## Formal results

Five short rounds, ten formal rounds, repeat10 per round after warmup. The formal sample contains 100 complete prefill invocations and 100 subsequent single-token decode invocations. Confidence intervals resample the ten round means (10,000 bootstrap draws), rather than treating all repetitions as independent rounds. Cold loading, external tokenization and ADB overhead are excluded; measured Host wall includes embedding, all blocks, final norm, head, greedy and FastRPC.

| Model | FP SwiGLU + fused QDQ | FP Softmax + fused QDQ | Combined | Prefill Host wall |
|---|---:|---:|---:|---:|
| Llama3.2-1B (L32-0063) | 17.046 ms (35.42%) | 3.154 ms (6.55%) | 41.97% | 48.130 ms |
| Qwen3-1.7B (EXP-0305) | 22.216 ms (29.25%) | 2.858 ms (3.76%) | 33.02% | 75.946 ms |

These data support a substantial floating nonlinear/boundary cost in this diagnostic, principally SwiGLU. They do not isolate QDQ-only cost, establish that Softmax dominates, or prove integer arithmetic is the only way to remove the cost. Keep FP arithmetic and fused QDQ combined in the motivation figure. Full additive modules are in MODULES.md; plotting data in figure-b-qwen17b.csv and the L32-0063 figure-b-llama1b.csv. Draw one normalized stage bar per model, emphasizing these two measured portions. Do not label the bars as a measured speedup of our production runtime.

Diagnostic E2E prefill / one-step decode: Qwen 842.71 / 22.98 token/s; Llama 1329.72 / 43.14 token/s. One-step decode is a correctness/diagnostic supplement, not a sustained decode benchmark.

## Correctness, physical constraints and provenance

- Independent FP32 SwiGLU reference: all 28 x 65,536 = 1,835,008 Gate/Up code pairs exact; at most one integer code from the float64 mathematical expression. Component source predates scheduling-only repairs; FP kernel unchanged.
- Independent complete reference starts from original embedding and follows all 65 causal rows, verified frozen prefix K/V seeds, unchanged W4 projections and SDK output conversion, FP nonlinear boundaries, FP32 residual, final norm and head. The reference is computed from arithmetic and weights, not copied device values.
- Final audit28-a02: all 56 layer-boundary byte hashes exact (28 prefill and 28 decode); selected head token and code exact at both steps. Earlier one/three-layer checks also passed; the final full audit covers their prefixes under the final implementation.
- 8,388,608 bytes requested/granted; peak 8,365,824 bytes. All 300 short/formal invocations and 8,400 layer ledgers pass additive accounting, zero timed hidden/intermediate DDR and zero spills. One HMX owner retained; no CPU fallback.
- Runtime/formal source: 12f672a47c41448824e32373f645cdf67d7b0ab2, branch codex/exp-0305-fp-islands. Immutable full-model seal: binaries-28-a04/seal.json. Source remains opt-in QBH_FP_ISLANDS; default OFF.
- Historical model payload recovered from device into models/qwen3-block-htp/exp0305/recovered-int16. All 909 deployment payloads match the retained EXP0284 hashes, locally and on device again after timing. The 32 retired historical chain .npy arrays were not recovered or used. New independent reference is retained under reference/. No old hashes accepted or rewritten. Original prefix seed SHA256 7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91.
- All failed attempts retained. Owned repairs: reference syntax/argument and token-extent errors; native K scratch ownership and prefix hook during Qwen port; runner initially launched the old binary directory despite recording a newer seal; reference byte-hash offset corrected to runtime 1469598103934665603, without changing expected arrays. Final decode discrepancy was an accidentally duplicated legacy LUT after FP SwiGLU, diagnosed with layer14 replay: attention/post-norm/Gate/Up exact, five INT16 middle codes differed by one. Removed duplicated call; all full-model values then exact. No tolerance widening or formal failed-sample exclusion.
- Final formal-* recordings all retained and all passed. No package hashing, builds, or competing hardware experiments overlapped formal measurement. Postflight verification ran after timing.

Llama companion source closure 14d7858900244f18ac475c27df2fd8fa134ef956; its REPORT.md documents isolated-phases-* formal data selection and original excluded collection, with no altered raw evidence. Existing AV scale-fold research was not silently promoted into these packages.

## Figure (a) historical evidence remains separate

Historical mllm Qwen S32 QNN dominant-path profiles provide weight-expansion/HMX contributions: W4A16 36.1085% / 21.2847%; W4A8 39.4738% / 19.1624%. These labels describe dominant-path contribution, not accelerator utilization. Their stage bars and internal overlays must not be mixed numerically with this M64 standalone-HTP Host-wall diagnostic. Old model/context artifacts were retired; only authorized retained historical reports are used. The new measurement supplies Figure (b); no Figure (c) or production-vs-reference speedup panel is required.
