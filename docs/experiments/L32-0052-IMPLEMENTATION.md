# L32-0052 implementation and diagnostic provenance

Scope: frozen Llama-3.2-3B-Instruct W4A8-SP2, no rotation, FP32 residual, original calibration and weights. This migrates the opt-in L32-0051 long-prefill frontend and native-layout/pipeline method. It does not promote a paper baseline or establish model quality.

- Full 28-layer M64 chunks preserve absolute RoPE and persistent KV. Only the final prompt chunk runs final normalization/head/greedy. Tail rows are logical; padded rows never enter attention/KV. Fixed teacher-forced decode follows, and the first selected token belongs to prefill.
- Head128/GQA3 replaces head64/GQA4 geometry: each attention plane uses max(GQA*64, head_dim), aligned worker slots are disjoint, four GQA clients share one serialized HMX/DMA owner. The immutable V LUT lives after all four slots.
- Head128 K packing uses vector centering, horizontal signed sums and word scatter; V packing uses the register LUT and native stores. Existing 3B head128 carrier-to-row conversion remains vectorized.
- Exactly 8 MiB VTCM; no timed hidden/intermediate DDR or spill. Existing DDR KV is persistent model state, not an intermediate spill. Top-level elapsed attention is additive; summed worker service counters overlap and must not be added to elapsed wall.
- Limits remain prompt<=768, decode<=63, cache832; validation shapes64/65/128/129/536+46/741+3. Project-owned fixed text/trajectory, not a named benchmark dataset.

## Repairs retained with their failed evidence

The initial fixture accidentally generated RoPE via the GPU. CPU generation matching the frozen source model restored exact first64 RoPE bytes; no original weights or hashes were replaced. Initial generation-mode3 usage error was corrected to the frozen W4A8 mode9.

The old head selector admitted only rows1 or64 to direct packed W4. A partial final chunk in 3B reached the expanded fallback and produced an incorrect selected token despite exact hidden state. The long partial chunk now uses the already validated native W4 head; all head IDs/codes match.

741 tokens additionally exposed one old rounding-contract defect, independently of the new pipeline: opt0 and opt31 both disagreed at chunk10/layer27/row36 (absolute token676). Layer0..26 and K/V were exact. Untimed diagnostic DMA isolated Q code1474: hardware64 vs reference63, then three AV codes differed. Independent scalar arithmetic in rope-boundary-arithmetic.json gives separately rounded y=-5.676424980163574,z=63.999996185302734,code63; fused-first multiplication gives y=-5.676424503326416,z=64,code64. Clang contraction in the scalar RoPE boundary repair was the cause. Explicit fp contract(off) retains separate SF32 products, vector arithmetic and division contract. The independent reference was unchanged. Final741 all layer hashes and heads match. Temporary row-specific diagnostics were removed; immutable source revisions and failed logs remain.

The first diagnostic used cached DSP memcpy to shared DDR and could not be trusted at the host; it was superseded by coherent audit DMA. Generic untimed final-hidden capture uses the otherwise unused generation reference tensor. None of these copies are enabled in timing runs.

Sequential and independently batched teacher-forced CPU oracles match for128 and741; batched oracle never consumes device outputs. Full arithmetic gates are separate from PPL/quality.

## Measurement

formal-protocol.json freezes source, archived binary hashes, oracle hashes, five short rounds and ten rotated five-arm processes x repeat10. M64 is unchanged; opt0 long controls and opt31 candidates use the same binary. Complete warm Host wall includes input/RoPE staging, embedding, all blocks, norm/head/greedy and FastRPC. Cold loading/session preparation, external tokenizer and audit/logging are excluded. All formal samples are retained. Parity needs the bootstrap95 lower bound of long/M64 throughput>=1; decode guard is compared at matched context with upper wall ratio<=1.1. Migration support is not automatically a parity claim.

## Closure scope guard
After formal timing, limited partial long-head admission to head128 to preserve 1B source dispatch. Closure0360df4 executable/loadable bytes equal measured5cedece; debug-only ELF metadata differs. First closure80a83cf shifted eight source-line diagnostic constants; preserved disassembly diff, then restored line count. No timed or reference records overwritten.
