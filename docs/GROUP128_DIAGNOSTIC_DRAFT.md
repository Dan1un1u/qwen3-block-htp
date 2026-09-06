# Conditional group128 software diagnostic draft

Prepared under approved PC051 while EXP0230 reserve runs. This is a proposed executable protocol, not an active experiment. Activate only after EXP0230 closes with insufficient fixed independent PPL acceptance; otherwise stop escalation. No next experiment or model has been started.

## Question and matched controls

Test whether per-output-channel range freedom is limiting W4A16. Use one ordinary symmetric static group128 candidate, unrotated, fresh original Qwen3 weights. Signed[-7,7], FP32 group scales, no second-level scale quantization/LPBQ and no upstream folded weights. Groups are contiguous128 original input columns per output row; global act-order is undone before packing and group lookup follows original column indices. Official GPTQ static_groups with actorder is the algorithm reference (IST-DASLab/gptq commit2d65066eeb06a5c9ff5184d8cebdf33662c67faf, retained EXP0221 references/gptq.py).

Use the already frozen EXP0230 C8 calibration (64documents,8192tokens,128context), because it provides an exact existing per-channel matched control and bounds repeated export cost. Include the stronger EXP0230 C64 as a descriptive reference on all evaluation sets, so improvement over C8 alone is not confused with improvement over the best current per-channel candidate. Main group attribution is C8 vs group128, not the unmatched C64 comparison. All four F/C8/C64/group128 get matched software evaluation; no candidate search or development-based algorithm adjustment.

Preserve original CPU Gram, act-order, damping0.01, true sequential calibration, FP16 exported execution and final three output-aware range categories. For grouped ranges, absmax and weight-L2.4 clipping are computed per group; midpoint is their mean. Each output row chooses one of the three complete group-scale vectors by its full-projection output SSE, keeping three full GPTQ trials and the original row-level output objective. Codes are newly quantized at every group scale. Use original-coordinate static groups, not group boundaries in the permuted act-order matrix. Preserve original local dead-column and zero-row semantics. Freeze original embedding/norms and inherited per-channel LM head byte-identically to the controls. No rotations, reconstruction training, new scale/code learning, or runtime change.

## Correctness and artifact boundary

Require an independent dense-elimination oracle for group-aware GPTQ, group-index permutation tests spanning multiple groups, zero/dead-column cases, independent NumPy output-SSE/range-choice checks, all-projection packing roundtrip, staged vs unchanged HF layer forward checks and finite exported FP16 weights. Export clearly identified software-only group files with a group128 format marker and separate scale filename; never provide them as an ordinary per-channel device package. Original shard hashes, all codes/scales, selected ranges, calibration inputs, source archives and commands are retained. Original unquantized tensors are loaded fresh before every export.

## Data and decision

Reuse EXP0230 fixed development128documents for descriptive results only; there is exactly one group candidate. Freeze new primary512documents/8192targets and reserve512documents/8192targets before any group scoring, balanced en/zh x wiki/news, M64+16, with document/text/32gram exclusion against prior experiments including all EXP0230 roles. Independently reconstruct source windows. All variants use identical tokenizer, positions and masks. Canonical packed FP16 GPU software scoring must pass repeat, future-token causal-mask and independent-CE checks. This is a host-only diagnostic and cannot establish DSP accuracy or speed for a grouped implementation.

Use the unchanged overall5% and each language/domain/cell10% PPL ratio limits versus named FP16 software, paired stratified document bootstrap5000seed231, and a predeclared bounded reserve rule. Retain all control and candidate results and distinguish improvement from absolute gate passage. Freeze the final activation protocol under its own experiment before execution. If sufficient, stop escalation and discuss grouped DSP deployment as a separate decision; if insufficient, proceed to the separately registered AWQ-style phase already authorized by PC051.

## Performance scope

No group DSP build/deployment or new physical performance measurement is implied. A dequantized PyTorch FP16 forward is not evidence of a W4 group runtime speed. Group speed/profiling is N/A with the completed EXP0230 per-channel profiling as an explicitly named reference. FP32 metadata is0.25bit/weight (6.25% of raw4bit payload), excluding alignment/headers and the existing per-row metadata. Actual DMA/VTCM/scale-lookup and E2E costs require a later deployment experiment. No baseline promotion.
