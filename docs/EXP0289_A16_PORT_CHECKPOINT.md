# EXP-0289 implementation checkpoint — formal baseline not yet accepted

Request: Qwen3-0.6B W16A16/W4A16 baselines, SIMD instead of slow elementwise scalar loops, basic DMA/compute overlap, no deep tuning.

## Implemented
- Original0.6B FP16 weights and verified original-derived per-output-channel W4 codes, original tied FP16 embedding; FP16 or W4 head according to recipe. No1.7B weights; no A16 rotations or SP2.
- Hidden1024 versus attention2048 shape fixes, O input/Q stride/audit bounds and expanded-weight slot capacities.
- Fixed native FP16 KV journal append past32 decode tokens; tested64+42/cache128.
- Safe64KiB alignment of0.6B A16 expanded-weight DMA/HMX slots, after an observed HMX operand-load fault. Exact undocumented hardware root cause not claimed.
- W4 head scale residency607744B accommodated; 594 head commands,593 weight prefetches in both recipes.
- Existing HVX norms/SwiGLU/packing and FP16 HMX, inherited basic double-buffer/overlap. New SIMD dynamic decode softmax; no per-element expf/division loop.
- Current numerical investigation uses0.6B SIMD FP32 nonlinear intermediates with FP16 tensor boundaries (native measured commit d35d35d). This is a diagnostic candidate, not an accepted performance improvement. Original fast candidate is archived at b28fde2. Other model branches and prior SP2 evidence are unchanged.

## Validation
All selected original layers0/14/27 and consecutive3 pass unchanged NRMSE<=0.003/cosine>=0.99999 prefill/decode checks.
Full28 runs complete, with finite hidden states, 8MiB requested/acquired, planned peak F16=5260032/W4F16=6342144B, no explicit intermediate DDR/spill and closed additive ledgers. These are necessary physical checks, not a blanket numerical pass.
Primary shape:64 prefill tokens,42 decode steps,cache128. Audit-enabled timings include diagnostic tensor extraction and MUST NOT be reported as normal E2E.
Full independent hidden-state checks FAIL:

| Candidate | W16A16 maximum NRMSE | W4A16 maximum NRMSE |
|---|---:|---:|
| Original fast |0.0065323461|0.0054536203|
| SIMD FP32 intermediate diagnostic |0.0063284242|0.0040269522|

No numerical threshold has been changed. No five-short/ten-formal campaign has run. These are not accepted paper baselines.

## Reference corrections and retained evidence
- Initial W4 reference rounded fully dequantized weights to FP16 before the dot product; actual HMX uses raw signed4 operands and half channel scales at accumulator conversion. Corrected reference-a03 resolves the original chain3 failure without changing W4 bytes.
- reference-a04 additionally modeled original half Q/K inverse and half sigmoid internal boundaries; this did not resolve fullstack differences. Preserve all files, do not present it as a passing reference.
- reference-a05 uses independent float64 dot products; current F16 max NRMSE remains0.0052395630. Changing GPU accumulation precision alone does not explain away the full failure.
- Earlier HMX fault, compiler failures, all failed gates, and all reference attempts retained.
- GQA-without-QKV-overlap diagnostic is NOT an accepted control: F16 prefill hidden/norm is exact but decode differs materially; W4 CLI rejects its qkv_norms/attention combination. Do not infer pipeline equivalence from this attempt.

## Greedy functional diagnostic
Both current vector candidates produce, before first EOS:
"The capital of France is Paris."
The harness deliberately continues to43 steps after EOS for a fixed-length experiment; subsequent text is not normal-response quality evidence. No PPL or quality acceptance.
One unaudited greedy run observed F16 prefill49603.021us/decode46629.045us and W4 prefill46542.552us/decode45833.172us. Different greedy trajectories, repeat1, no formal warm campaign: diagnostic only, not a fair published speed comparison.

## Pending user decision
Asked whether to permit speed measurements of explicitly provisional/unaccepted candidates while retaining numerical failures, or continue numerical diagnosis under the unchanged gate before timing. Do not infer approval from elapsed time. No formal timing or automatic Selected promotion until this is resolved.
Device idle. EXP-0289 stays active, source branch codex/exp-0289-qwen3-06b-a16. Native archived build seals and raw run protocol.json files identify exact code; Python-only commits after native build do not change the archived binaries.
