# L32-0052 — Llama3B long-prefill migration results

Implementation and independent hardware-arithmetic checks pass. The inherited long/M64 throughput-parity target remains unmet; this is not a baseline promotion.

## Frozen measurement contract

Llama-3.2-3B-Instruct, 28 layers, W4A8-SP2 mode8, FP32 residual (`QBH_LLAMA_FP32_RESIDUAL=1`), no rotation. Original weights/calibration remain frozen. M64 chunks, absolute RoPE, persistent capacity832 KV, valid tail rows, final-prompt-chunk head only. Complete warm Host wall includes embedding, all layers, final norm/head/greedy, FastRPC and host input/RoPE staging; excludes cold loading, tokenization, audit and log output. Fixed project-owned prompt/teacher-forced decode trajectory; these are not named-dataset or free-generation quality results. Decode count means RPC steps after the prefill-selected first token.

Five short rotated rounds are auxiliary. Ten frozen rotated five-arm formal rounds × repeat10; every sample retained. Controls use the same measured binary. M64 is the long frontend M64+3 at capacity832; it is not a claim about a newly measured historical short-paper baseline. opt0 is the existing sequential long-path design adapted to 3B geometry in this same binary; opt31 is the migrated native-layout/parallel-GQA path. These opt0 numbers are newly measured experimental controls, not a previously published 3B long-prefill baseline.

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| m64 | 1167.87 | 23.77 |
| original536 | 432.07 | 15.94 |
| opt536 | 1103.85 | 16.28 |
| original741 | 370.23 | 14.26 |
| opt741 | 1106.97 | 14.75 |

| Prompt | Long/M64 prefill throughput | 95% paired bootstrap CI | Parity | Prefill speedup over opt0 | Decode wall ratio over opt0 |
|---|---:|---|---|---:|---:|
| 536 | 0.945187 | [0.940273, 0.949967] | FAIL | 2.5548x | 0.978955 |
| 741 | 0.947860 | [0.942891, 0.953162] | FAIL | 2.9900x | 0.966762 |

Both matched-context decode 10% slowdown guards pass. Long-context decode is reported at its actual KV length, not compared to M64 as an optimization regression. Parity remains a 95% lower-bound >=1 criterion; neither controls nor gates were weakened.

## Implementation and correctness

Migrated head128/GQA3 native K/V packing, four-query softmax and disjoint parallel GQA workspaces while retaining one HMX/DMA owner. The immutable shared V LUT is outside every live slot. Parallel worker subcounters overlap and are not added as wall-clock module time; MODULES.md uses elapsed attention time.

Final measured binary independently validated at 64,65,128,129,536+46,741+3 with poisoned future cache: 2,520/2,520 FP32 layer hashes exact; all actual integer KV values and selected head IDs/codes exact; no signed-zero exceptions needed. Every one of 14,400 formally timed RPC boundaries passes status, exact8MiB VTCM, no intermediate DDR/spill, head and additive wall-ledger checks.
Maximum planned VTCM: 8360416 / 8,388,608 bytes; attention overlay peak 1904640 bytes.

An old head128 RoPE scalar boundary repair allowed compiler FMA contraction. A long-context quantization threshold exposed a one-code difference, then propagated through later layers. Explicit separate rounding fixes it; the independent reference and tolerances were unchanged. Partial final chunks now use the existing native head. Failed candidates and diagnostic evidence are preserved in IMPLEMENTATION.md.

Measured source: `5cedece9dd1050fda6bce6e083cd903d95c77ce0`, archive `binaries-a9`. Closure source: `0360df4` (full hash in closure-binary-equivalence.json), limits partial-head admission to head128 to preserve 1B dispatch. All executable/loadable sections and addresses are byte-identical to the measured 3B binaries; only DSP non-loadable debug metadata differs. Closure build archive `binaries-closure-a02`.

No PPL/text-quality assessment, no paper edits, no automatic baseline promotion. Historical L32-0051 1B evidence is unchanged. A16 and ordinary A8 long paths are outside this migration. Runtime remains opt-in and bounded to prompt<=768, decode<=63, KVcapacity832.

Full module profiles: MODULES.md. Raw timings and CIs: profiling_summary.json. Protocol: formal-protocol.json. Independent checks: audit*-final-validation.json.

## Remaining prefill cost
At741 tokens, attention grows from103.66 to169.57 microseconds/token (+65.91), while final-head amortization saves56.82 microseconds/token. Norm and chunk-tail work add further cost. Thus the migrated pipeline substantially improves the original long path but does not erase context growth and M64 tail padding. The measured net gap is47.10 microseconds/token versus the matched M64 reference. See prefill_per_token_attribution.json; these are elapsed module ledgers, not overlapping worker-time sums.

Closure staged build also passes a poisoned64+3 full hardware audit:112 extra layer hashes, all KV/head/padding exact; no new speed samples were substituted.
