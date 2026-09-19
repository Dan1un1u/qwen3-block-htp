# EXP-0301: decode optimization coverage audit and repair

The screen uses decode throughput relative to the same model/recipe A64+42, with a 10% loss threshold. It flagged 11 Qwen B/C rows. Llama rows were below the threshold after L32-0057 and were not rerun. Historical D was not used as the reference.

Two implementation gaps were repaired: Qwen0.6 A8/SP2 retained long option 927 instead of already validated 65439; generic A16 KV look-ahead DMA was admitted only during prefill. The latter is now optionally available during long decode for both floating recipes and both Qwen sizes. No weights, quantization boundaries, residual representation, physical M64 residual/Norm work, token trajectory or model-quality acceptance changed.

Qwen1.7 A8/SP2 already uses 65439. Its historical C attention wall grows from about 1.97 ms at A to 4.73 ms at C, almost entirely explaining the 21.10 ms to 23.85 ms complete decode growth. No corresponding omitted dispatch/vector/packing option was found. Further incremental native long-KV redesign is outside this bounded repair; those measured rows are retained.

Each repaired recipe/model set has five short rounds and ten formal rotated paired repeat10 rounds. Complete Host wall includes embedding, all layers, final norm, LM head, greedy, FastRPC and long Host input/RoPE staging. Cold loading, external tokenizer and audit writes are excluded. Short A retains its established native timing path. B536+46 and C741+3 are fixed project-owned fixtures, not named-dataset quality measurements. Every formal sample is retained.

## Same-round decode comparison

| Model | Recipe | Shape | Original token/s | Repaired token/s | Speed change | Repaired/A decode TPS | Decode wall-ratio95%CI |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | W4A8 | 536+46 | 63.4190 | 79.1785 | +24.85% | 83.38% | [0.799902, 0.802377] |
| Qwen3-0.6B | W4A8 | 741+3 | 58.2469 | 71.7067 | +23.11% | 75.51% | [0.810605, 0.814077] |
| Qwen3-0.6B | W4A8-SP2 | 536+46 | 63.1686 | 78.7974 | +24.74% | 83.33% | [0.800962, 0.802297] |
| Qwen3-0.6B | W4A8-SP2 | 741+3 | 57.9133 | 71.3971 | +23.28% | 75.51% | [0.808806, 0.813536] |
| Qwen3-0.6B | W16A16 | 536+46 | 24.2339 | 25.0440 | +3.34% | 91.80% | [0.966509, 0.968877] |
| Qwen3-0.6B | W16A16 | 741+3 | 23.1410 | 24.0682 | +4.01% | 88.22% | [0.959413, 0.963371] |
| Qwen3-0.6B | W4A16 | 536+46 | 23.9607 | 24.7708 | +3.38% | 91.04% | [0.966428, 0.968186] |
| Qwen3-0.6B | W4A16 | 741+3 | 23.0549 | 23.9733 | +3.98% | 88.11% | [0.957941, 0.965560] |
| Qwen3-1.7B | W16A16 | 536+46 | 12.0330 | 12.3234 | +2.41% | 96.65% | [0.970568, 0.981970] |
| Qwen3-1.7B | W16A16 | 741+3 | 11.7597 | 11.9225 | +1.38% | 93.51% | [0.977793, 0.994633] |
| Qwen3-1.7B | W4A16 | 536+46 | 14.1217 | 14.4415 | +2.26% | 94.88% | [0.973916, 0.981831] |
| Qwen3-1.7B | W4A16 | 741+3 | 13.7328 | 14.0581 | +2.37% | 92.37% | [0.973986, 0.979504] |

## Remaining cross-context losses above 10%

| Model | Recipe | Shape | Decode loss versus A |
|---|---|---|---:|
| Qwen3-0.6B | W4A8 | 536+46 | 16.62% |
| Qwen3-0.6B | W4A8-SP2 | 536+46 | 16.67% |
| Qwen3-0.6B | W16A16 | 741+3 | 11.78% |
| Qwen3-0.6B | W4A16 | 741+3 | 11.89% |
| Qwen3-0.6B | W4A8 | 741+3 | 24.49% |
| Qwen3-0.6B | W4A8-SP2 | 741+3 | 24.49% |
| Qwen3-1.7B | W4A8 | 741+3 | 11.50% |
| Qwen3-1.7B | W4A8-SP2 | 741+3 | 11.34% |

These remaining cross-context losses are not failures of the same-length regression guard. All repaired same-length prefill/decode wall-ratio confidence upper bounds are <= 1.10. Cross-context parity is separately reported, not relaxed or silently marked pass. Qwen0.6 A8/SP2 long prefill now passes all four 10% throughput gates. Qwen0.6 A16 prefill remains outside the gate and was not optimized by this decode-only change.

## Correctness, hardware attribution and limits

A8/SP2: 5,236 layer outputs match frozen independent actual-arithmetic references, with exact valid KV and unchanged poisoned future storage. A16: all captured bytes and live layer hashes match the pinned original hardware references; independent actual-operand norm/head and the existing in-DSP softmax component checks pass. Historical full-floating/PPL alignment failures remain failures. This is no new model-quality acceptance.

For Qwen0.6 A8, C decode attention rises from 1.918 ms at native A to 4.778 ms, explaining 2.861 ms of the 3.415 ms complete Host-wall increase after repair. For Qwen0.6 W16/W4 A16, attention rises from 3.020/3.209 ms to 8.401/8.967 ms. These counters localize the remaining cost; they do not prove that no further attention optimization is possible.

Source locations at closure commit 22fba3060abd606ec33efb7fabb942b33e14a20e: src/dsp/long_attention_parallel.inc lines 10/33/39/49/56/64 contain the existing separated DMA lock, dead-clear suppression, register K/V packing and four-row decode AV conversion. src/dsp/block_imp.c lines 17751-17758 add generic A16 decode admission to the existing KV prefetch path.

Explicit reproduction settings: QBH_LONG_OPT=65439 for Qwen A8/SP2 long paths; A16 decode-prefetch candidates use 4 on Qwen0.6 and 7 on Qwen1.7, preserving prior controls 0 and 3. Native A retains its established native dispatcher. Experiment runners and archived runtime seals set these flags explicitly; historical experiment scripts and their defaults are not rewritten.

Every formal RPC checks exactly 8 MiB acquired VTCM, bounded scratch, no timed intermediate DDR spill/read/write, and complete additive module accounting. Same-shape HMX command counts, cache DMA descriptor counts, KV bytes and attention overlay sizes are unchanged. A8 gains come from existing register K/V packing, omitted dead-row work and resource overlap. A16 reduces exposed DMA waiting using an existing phase-dead expanded-weight slot; overlapped HMX intervals can lengthen while total attention/Host wall falls. Service-time counters that overlap are not summed as additive elapsed modules.

Shared-source A8/SP2 regressions on the final A16 binary passed native64+42 and long741+3 for both Qwen sizes. These repeat1 runs validate head/physical compatibility only and do not replace the sealed formal A8 timing samples.

Module tables and full per-round arrays are archived alongside commands, package manifests, inherited hash verification, exact captures and runtime seals. No new weight export or selected-baseline promotion. Llama numbers, Qwen1.7 A8/SP2 numbers, and curated D are retained. C still has 3 decode RPCs excluding the token produced by prefill; no 741+63 measurement was introduced.

The initial unchanged 0.6B source build failed an unused pool parameter check, repaired without changing execution. The first reference verifier selected a nested model ledger rather than its authority-bound experiment-root ledger; directory selection was fixed without accepting any new hash. Failed attempts/tool versions are retained.

Measured sources: A8 7682a576a90c812d083ed058e26ab8bdc892ef22; A16 22fba3060abd606ec33efb7fabb942b33e14a20e. Source closure 22fba3060abd606ec33efb7fabb942b33e14a20e.

