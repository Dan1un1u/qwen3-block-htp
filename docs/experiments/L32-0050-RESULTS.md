# L32-0050: full-model chunked long prefill
Scope: Llama-3.2-1B-Instruct W4A8-SP2 mode8, FP32 residual, no rotation.
Weights and static quantization are unchanged from the sealed existing SP2 package. No PPL, natural-text usability, or model-quality acceptance is inferred.
## Implementation
Opt-in long frontend splits prompt into M64 chunks, keeps absolute RoPE positions and persistent per-layer KV, and dispatches the complete16-layer model once per chunk. Only the final prompt chunk runs final norm/head/greedy. Decode then continues from the accumulated prompt KV.
741 = 11x64 +37;536 =8x64 +24. Tail uses M64 physical projection tiles with valid-row attention/append/head semantics. A one-token tail reuses M1. All real prompt tokens, every chunk RPC and host ID/RoPE staging are included in prompt timing; padding never contributes to token/s numerator.
The vectorized cached multi-query attention path preserves the existing integer attention arithmetic for longer prefixes. SP2/SwiGLU/Down and projection pipelines remain the existing implementation.
Host metadata must rebind layer0 on each chunk. Q/K RoPE must process every live query row, rather than treating nonempty KV as decode. MLP must initialize its worker context directly, rather than relying on attention pool setup; this fixed the second-chunk DSP failure. Failed attempts preserved.
## Correctness
Reference is an independent implementation of actual HMX conversion, integer attention, SP2 and ordered FP32 boundaries, not a floating teacher.
64/65/128/129 and536+3/741+3 passed full-chain outputs, all-layer KV prefixes, absolute positions, cache bounds and head IDs/codes.65+3 has one negative-zero versus positive-zero representation difference at step2/layer4/channel1838; values are exactly equal, with unchanged subsequent quantization and output. No nonzero numerical tolerance was relaxed.
Future KV poison (K=A5,V=5A) and padded IDs/RoPE leave all valid outputs unchanged for536/741 and untouched future cache intact.
Audit-only row copies and KV dumps are excluded from profiling; failed/audit runtimes are not performance results.
## Measurement
Fixed project-owned input trajectory, not the three named Quant.npu datasets. Counts +D denote D decode RPC steps, with the first selected token produced by prefill. No claim of matched dataset or generated-token convention against external work.
Five auxiliary repeat1 processes and ten formal repeat10 processes per target shape. All formal replays, including the first invocation of each prepared process, are retained; model loading, session preparation, external tokenization and printing are excluded. Per-phase host staging is included.
No cross-shape10%comparison, no optional resampling, no automatic baseline promotion.
## Known scope
Only the1B W4A8-SP2 long frontend is implemented/validated here. Other sizes/recipes remain later bounded phases. Current build is1B/16layers;3B or other recipes require correct rebuild/seal.
This establishes long-context support, not a tuned long-attention throughput result. Long KV attention dominates current wall time; short M64 speed cannot be extrapolated.

## Completed results

536+46 independent chain also passes:880 layer outputs bit-exact, all KV/head exact. Across the seven shape/trajectory checks,1631 layer hashes match bit-for-bit and one differs only in signed zero. Counts include reused trajectory prefixes.

| Prompt + decode steps | Prefill token/s (95% CI) | Decode token/s (95% CI) | Prompt Host wall ms | Decode total Host wall ms |
|---|---:|---:|---:|---:|
| 536+46 | 638.96 [638.56,639.48] | 23.04 [23.02,23.05] | 838.866 | 1996.884 |
| 741+3 | 519.04 [518.75,519.33] | 19.39 [19.37,19.40] | 1427.642 | 154.741 |

7000 timed RPC boundaries plus350 auxiliary boundaries reconcile exactly to complete additive Host/DSP ledgers. VTCM peak8,098,272 bytes within8,388,608 requested/acquired; no timed intermediateDDR/spill. Per-process means are the bootstrap sampling unit; both sample0 and later samples are retained. See MODULES.md and profiling_summary.json.

741 attention accounts for79.13%prefill and69.11%decode Host wall. Its firstM64chunk takes24.17ms without LMhead, compared with59.30ms atpast64 and191.42ms atpast640; final37-rowtail takes150.17ms includinghead. The cached multiquery path is functionally established but remains less pipelined than the original no-past M64 path. These observations motivate further long-KV attention scheduling/layout work; this experiment does not tune on its final measurement.

The current actual-arithmetic long attention reference is preserved; no reversion to another attention approximation or scalar inference fallback. Current bounded frontend limits prompt<=768,decode<=63,cache832; tested targets above are the supported evidence scope, not arbitrary long-context coverage. Long-prefill capability is opt-in via QBH_LONG_PREFILL_TOKENS/QBH_LONG_DECODE_STEPS; models need the matching long prompt/position/KV fixtures.

Measured/closure source fad2d40fa909c2c42f251b04a1c1eed64ad2af1c; archive binaries-a8. Audited binaries-a7 and measured-a8 share identical DSP skeleton/stub hashes; the last change only adds host staging counters/oracle resume. Final-a8 separately revalidatesM64 and536+46. Source branch codex/llama32-no-rotation; other recipe/model sources were not migrated in this phase. Original short baselines are not replaced.
