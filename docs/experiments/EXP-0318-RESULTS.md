# EXP-0318 — Qwen3-1.7B 2048-token context extension

Completed implementation and hardware correctness validation. Existing weights/scales, uniform INT16 Down, FP32 residual, no rotation, prefix patch and native U8 LM head are preserved. No existing performance/quality baseline is promoted or overwritten.

## Implementation
- Host and DSP support W4A8 prefill up to2048; persistent row-major KV capacity up to2112;64continuous decode steps after2048. A16 limits remain768/832.
- Existing M64 chunking, absolute RoPE and valid-row causal masks are retained. All KV remains persistent between chunks.
- Fixed attention overlay is2359296bytes inside8MiB VTCM. Adaptive clients:4 through1408validtokens;3 for1409–1920;2 from1921. Each group computes QK/global softmax/AV over its full valid context. No segmented independently normalized softmax, new quantizer, weight change or intermediate DDR spill.
- Optional QBH_LONG_TARGET_FILE supplies decode+1 little-endian uint32 targets. Final prefill head scores target0; subsequent decode consumes target[i-1] and scores target[i]. Existing native head histogram/NLL path is reused. Targets, vocabulary count, finite scores and NLL identity are checked.
- Optional QBH_LONG_CACHE_GUARD verifies all previous and future K/V bytes remain unchanged around each append, outside timing. Boundary audit must be enabled for nonzero layer hashes. Audit-mode latency is excluded from performance reporting.

## Correctness evidence
64,768,769,1024,2047,2048prefill boundary executions pass.2048+64 reaches2112validtokens. All calls retain8MiB VTCM and zero intermediate DDR reads/writes/spill.
Independent CPU/ISA-simulator integer reference for2048+3 matches all980nonzero layer hashes and all4head selections/logit codes. Reference uses no device activations; its vectorized attention arithmetic first matched original scalar reference on48cases covering lengths1,31,64,769,2048,2112.
2048+64 parallel versus original sequential attention matches2688layer hashes and65head outputs. KV guards check11505500160unchanged bytes per trajectory.2047tail/padding poison matches zero padding,980layer hashes. See final-audit-checks.json and audited-oracle-comparison.json.
New long scoring reproduces43historical EXP0317 targets exactly (target codes, NLL, logits, normalization and cache lengths); final binary recheck also exact.

## WikiText2 quality
Same frozen8192target tokens as EXP0317: positions64..8255, original native tokenizer, no chat, no test-set tuning. Long schedule uses2048-token windows,64warmup tokens,1984scored targets per full window, final256targets retained. Effective context64..2047; reset only between5windows. Old schedule reset every43targets with64warmup (context64..106). This is a lightweight test subset, not full WikiText2 PPL or a matched floating-point teacher comparison.

| Context schedule | Target tokens | PPL |
|---|---:|---:|
| EXP0317 short windows |8192|49.6269961128|
| EXP0318 2048-token windows |8192|28.4670476589|

Mean NLL3.348747195839262; exact8192unique target coverage, independent aggregate from logsumexp-target agrees; native LM head saturation count0. Improvement reflects context/reset policy, not repaired quantization or new weights. Long prompt text is readable English but repeats library background and does not answer the France question within65outputs. Long-context task quality remains unaccepted.

## Timing
One uninstrumented2048+64complete-model diagnostic, including host staging and FastRPC, excluding cold load/tokenization: prefill1316.821ms / 1555.26token/s;64decode calls1881.411ms / 34.02token/s. Not formal repeat10, no speed-parity gate or paper-table update.

## Attempts and interpretation
Original build launch lacked executable permission, retried via bash. A first769host-limit check used recipe before parsing; fixed before accepted runs. Original scalar reference was stopped and replaced by equivalent validated vectorization. Early production-mode layer hash comparisons used uncomputed zero fields and are NOT correctness evidence; final audited comparisons supersede them. Raw attempts retained in attempts.json.

Source: 0deb09e6ad504d3bafedbb006f5d816dc59fe4ba on codex/exp-0318-qwen17-context2048. PPL campaign binary-a3 at7c28b30; finalbinary-a4 at0deb09e adds cache auditing and64decode allowance only;43score regression exact. Reproduction scripts, fixture manifests, sealed binaries, commands, raw logs, independent references and result checks are in this folder.

Next: measure a matched same-token/same-context floating teacher to quantify remaining quantization loss; separately diagnose repetition and long-context instruction following. Do not attribute all old high PPL to arithmetic error or claim2048quality acceptance.

Evidence ledger SHA256: 26b70e3ba70899d3747e483b6014fc7afabb7fd7fe41be162c8a510d860d1181 (522files,558721866bytes).
