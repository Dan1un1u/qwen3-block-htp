# L32-0051 implementation and evidence notes

Model: Llama-3.2-1B-Instruct; W4A8-SP2 mode8, FP32 residual, no rotation.
Frozen arithmetic and fixtures from L32-0050. Current long frontend supports prompt<=768 and decode<=63, cache832; evidence targets536+46 and741+3. No other model/recipe migration or model-quality claim.

## What changed
1. Four query rows fit one native128-byte vector. Max/sum reductions stay vector-valued across KV tiles; integer exact-fast normalization and causal masks are unchanged. Requantized scores reuse the dead score carrier.
2. Head64 K packing consumes two rows together with signed horizontal sums; V recentering uses the exact256-entry register lookup. Saturation counting is vectorized. No change to stored row-major KV bytes.
3. Four GQA clients own disjoint slots in the existing attention overlay. Their HVX work overlaps; one mutex serializes HMX/DMA submission to the original single HMX owner. Cached QK RoPE reuses the existing4-context pool. No intermediate DDR spill.
4. Identical V recenter tables are prepared once and read by clients. The shared table is explicitly reserved beyond every live slot; it cannot alias the former up buffer. Layout capacity is checked before launch.

## Interpretation
The M64 no-past prefill dispatch/kernel is unchanged. Only the opt-in cached path changes; M64 is freshly measured with the same complete frontend/head boundary.
The elapsed attention module is additive in the full-model ledger. Parallel attention subcounters record summed worker service/queue durations; they overlap and must not be summed to infer wall time.
Long prefill performs one LM-head evaluation at the end, amortizing the head cost across the prompt. Throughput parity at these bounded lengths does not imply constant cost for arbitrary context lengths.
Full Host wall includes each model RPC and per-phase host input/RoPE staging. Cold loading, session preparation, external tokenization, report formatting and untimed audit transfers are excluded consistently.
Fixed project-owned prompt/token trajectory, not a named Quant.npu dataset. Decode+D counts D decode RPC steps, plus the first selected token from prefill.

## Failed candidates retained
See failed_attempts.json: wrong full-vector predicate cleared K; fixed explicitly. Initial shared LUT overlapped a growing slot atpast512; fixed by independent tail reservation. Neither failing implementation is in the formal campaign.

## Reproduction
Use tools/long_prefill_loop.py with QBH_LONG_OPT=31 and the immutable0050 model manifests. The active archive is binaries-a10, source e31a286fac5b43cfccd93706bcc21a00400920ce.
Use formal_protocol.json / formal_runner.py / analyze.py for the frozen campaign. Do not rerun into existing result directories.
Historical control:0050 binaries-a8, fad2d40fa909c2c42f251b04a1c1eed64ad2af1c.
No automatic baseline promotion.
