# L32-0067: residual boundary fusion ablation

Llama3.2-1B, full16, fixed M64+42. Uniform INT16 Down, FP32 residual/Norm, no rotation, frozen L32-0065 M AV-folded package. F=fused, N=split Norm→Q, R=split O/Down→residual, NR=both.

Five short and ten formal four-arm cycles, each process repeat10; fixed Williams arm order. All arms use one binary and identical 40 KiB reserved VTCM scratch, DMA/pipeline scheduling, native output layout, weights, quantizers and arithmetic. No baseline promotion.

## End-to-end results

| Configuration | Prefill ms (64) | Prefill token/s | Decode ms/token | Decode token/s |
|---|---:|---:|---:|---:|
| F | 27.082127 | 2363.18 | 21.526391 | 46.45 |
| N | 27.186095 | 2354.14 | 21.559454 | 46.38 |
| R | 27.180860 | 2354.60 | 21.547331 | 46.41 |
| NR | 27.339049 | 2340.97 | 21.636117 | 46.22 |

## Paired effects

95% paired bootstrap intervals across ten cycle means (20,000 samples, seed 670067). Values below are wall-time changes, not throughput changes.

| Contrast | Prefill wall change [95% CI] | Decode wall change [95% CI] |
|---|---:|---:|
| N/F | +0.384% [+0.089%, +0.646%] | +0.154% [-0.210%, +0.480%] |
| R/F | +0.365% [+0.058%, +0.687%] | +0.097% [-0.299%, +0.478%] |
| NR/F | +0.949% [+0.407%, +1.460%] | +0.510% [+0.022%, +0.970%] |
| NR/R | +0.582% [+0.215%, +0.936%] | +0.412% [+0.008%, +0.841%] |
| NR/N | +0.563% [+0.074%, +1.034%] | +0.356% [-0.125%, +0.850%] |
| interaction | +0.197% [-0.244%, +0.697%] | +0.258% [-0.346%, +0.943%] |

Interaction is NR×F/(N×R). Overlapping execution means separate effects need not add.

## Implementation and correctness

Norm split materializes one normalized FP32 row per worker in VTCM before the same vector quantizer and direct native store. Four workers have disjoint rows. O/Down split materializes one FP32 64×32 tile, then a separate noinline vector residual-add consumer finishes before that slot is released. Existing double buffers and HMX ownership are retained. Neither intervention inserts a whole-model/layer barrier or DDR roundtrip.

Both splits separately add 32 MiB VTCM read+write traffic during prefill and 21 MiB over42 decode steps. These are logical tensor traffic counts, not measured DRAM traffic or critical-path attribution.

All12 fused/split layer-count audits (1,3,16) match the frozen independent layer hashes and selected head codes. Each also matches all43 exported FP32 hidden payloads, all43 U8 norm payloads, and every prefill KV payload byte-for-byte against the unflagged path. 17200 formal invocations pass status, 8MiB VTCM, zero explicit intermediate DDR/spill and exact additive-ledger checks. These tests validate implementation on the fixed trace, not model quality/PPL.

The first implementation passed1/3-layer arithmetic but assembly showed dynamically indexed partial-vector arrays on the thread stack. No timing campaign was started for that build. Revision a02 expands these four lanes identically in every arm; assembly contains no epilogue vector stack stores. Norm stack vectors are constants/address masks. a01 evidence remains retained.

Full Host wall includes embedding, all16 blocks, finalNorm,LM head/greedy and FastRPC; excludes cold package/session loading and external tokenization. Decode is42 steps, not repeat1.

## Interpretation

Norm-to-quantize and projection-to-residual fusion each have a small confirmed prefill effect on this configuration (split penalties0.384% and0.365%). Neither independent decode contrast excludes zero. Splitting both raises prefill wall by0.949% (95%CI0.407–1.460%) and decode by0.510% (0.022–0.970%); the decode interval is close to zero. Multiplicative interaction intervals include1 in both phases, so this experiment does not establish synergy or antagonism.

This is a marginal boundary-fusion ablation inside an already vectorized, tiled, buffered runtime. The controls retain on-chip intermediates and tile release rather than introducing DDR roundtrips or operator-wide barriers. It supports modest incremental savings, not a claim that these two fusions alone explain a large end-to-end speedup. The wider integer/nonlinear and scheduling/layout effects require their own previously archived experiments. No cross-model or long-context conclusion is inferred.

All four full16 AV-padding poison audits also pass the same688 layer hashes and43 head codes. Their hidden, final-normalized and prefill-KV payloads match the respective unpoisoned arms exactly;5376 poison applications per arm. The timed60 runs use one sealed binary, one fixed repeat10 trajectory and a command contract differing only in the two split flags.

## Workbook archive

Desktop HTP workbook H144:L155 contains12 phase contrasts including the interaction; I95:L98 contains the4 configurations; J3576:J3743 contains168 module rows. Existing A–G sheet parts are byte-identical and all previously populated cells/styles are preserved. Workbook SHA256 b1193e77b0f5dc2d1e5ab9fefb8ad18d57b5f97a4fbbc2bbf27e7dd2a4d78b39. See workbook/receipt.json for saved-file checks.
