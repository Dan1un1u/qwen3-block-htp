# L32-0045 results: 3B no-performance-gate optimization

Four pipeline iterations completed. The retained combined candidate improves both full-model E2E phases without changing arithmetic, weights, scales, SP2, FP32 residual or rotations (none).

## Formal paired E2E

SM8750 / HTP V79; Llama-3.2-3B-Instruct, full28, M64+42, cache128. Ten AB/BA pairs, repeat10; preceded by five fixed short pairs. Every prespecified round retained. No performance stopping gate.
Includes embedding, all blocks, final norm, LM head/greedy and FastRPC. Excludes cold loading/session preparation and external tokenization. Fixed trajectory speed; free-greedy63 correctness is separately audited.

| Arm | Prefill token/s | Decode token/s |
|---|---:|---:|
| CONTROL | 1136.492048 | 22.446407 |
| OPT | 1184.718073 | 23.640810 |

prefill: throughput +4.243411%; paired wall ratio 0.959293248, 95% bootstrap CI [0.958162336, 0.960596826]. CI describes uncertainty, not an acceptance/stopping gate.

decode: throughput +5.321132%; paired wall ratio 0.949477074, 95% bootstrap CI [0.948345701, 0.950279265]. CI describes uncertainty, not an acceptance/stopping gate.

## Retained mechanisms

1. QKV batch4 ->32 reuses the existing two1.5MiB weight slots. Two bias slots occupy16KiB of phase-dead Gate storage, not live RoPE.
2. First Q weight DMA overlaps FP32 input Norm; first K/V weight DMA overlaps the preceding projection’s last HMX command.
3. First Down weight DMA overlaps final Up HMX; downstream activation consumption still waits for SwiGLU completion.
All new cross-operator prefetches honor existing pipeline-disable bit16; measured mask0. No extra tensor allocation, full-weight traversal, intermediate DDR/spill or arithmetic change.
Per generated-token invocation:980 fewer HMX commands and1960 fewer weight DMA descriptors across28layers; same physical tile pairs/weight/cache bytes. VTCM peak8,360,416 of8,388,608 bytes.
Stage DMA ownership changes: interpret adjacent stages and Host wall; an isolated stage reduction is not the whole pipeline gain.

## Exploratory pilots

| Candidate | Paired control | Prefill gain | Decode gain |
|---|---|---:|---:|
| A | control-l28 | +1.1917% | +0.5257% |
| B | control-l28 | +1.0240% | +1.1052% |
| C | control-l28 | +3.3679% | +4.1522% |
| D | c-l28 | +0.2028% | +0.4088% |

All pilots use3pairs/repeat10, separate measurement windows. A=QKV8; B=QKV32; C=B+Norm/QKV DMA lookahead; D=C+Down lookahead. Do not add gains across windows. Small D-over-C effect is exploratory; formal evidence supports the combined final candidate versus frozen0044.

## Correctness and provenance

87 validated device processes, 23678 token boundaries. Each A/B/C/D/final checked on selected0/13/27, chain3/28 and full64-step true greedy against the frozen independent reference. Final bounded1B layer7 regression passed.
Additional bitwise audit: 6,108,160 captured FP32 words across 372 checks match exactly. Chain28 covers all valid M64 prefill rows; full-generation audit captures each final hidden vector, selected IDs/logit codes and prefill KV. This is implementation equivalence, not model quality/PPL validation.
Measured source: ff213089c6f5cfc07c620c9509b22a5e00486e80; frozen control measured source: 4686765b98939a34fec218af950eeba573087c71. Same original-derived0041 payloads/0044 fixtures. All2137 prior-ledger files (4,135,058,140 bytes) rehashed;904 current local/device payload files verified.
Current workspace build restored to3B/full28 after bounded1B regression. Measured runtime remains immutable final-l28/runtime.json; build restoration is not a second speed measurement.
Complete modules: MODULES.md. Raw formal/short/pilot runs, binary seals, independent comparisons, physical counters and validation index are preserved alongside this report.

## Scope and remaining opportunities

No PPL/quality claim, automatic paper-baseline promotion, or claim of optimality. No separate formal63 throughput measurement. Existing1B historical speeds and other recipes remain separate.
Remaining full-prefix K/V preparation and longer-lifetime native-cache layout could be investigated in a subsequent measured iteration; they were not implemented here. Prior0044 rejected RMS/K-transpose probes were not repeated.
