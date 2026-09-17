# EXP-0285: Original latest FP32 baseline versus FP16 residual

Qwen3-1.7B latest no-rotation native-W4 SP2 baseline, parent EXP0284 SP2 arm. Original sealed EXP0284 binary mode2 FP32 control versus EXP0285 binary mode3 FP16 storage; projection scale/add and ordered RMSNorm remain FP32. Norm still produces A8. Embedding begins in FP16; both residual additions round to FP16 at storage in candidate. No rotations, quantizer fitting, weight changes or model-quality/PPL evaluation. No baseline promotion. This original-binary comparison is primary; the first same-binary comparison remains supplementary because common storage dispatch slowed its FP32 control.

Primary fixed original M64 prompt +15 decode steps, with identical token trajectory. Complete Host wall includes embedding,28layers, finalNorm, full head,greedy and FastRPC; excludes cold loading,tokenizer andADB. One warmup and one auxiliaryrepeat1 per arm, five short and ten alternatingAB/BA formalrepeat10 pairs, no optional stopping. Headline/module values use arithmetic mean of all formal measurements. Paired bootstrap20000 seed285.

| Residual storage | Prefill token/s | Decode token/s | Prefill Host ms | 15-decode Host ms |
|---|---:|---:|---:|---:|
| FP32 | 1847.4690 | 47.1516 | 34.641990 | 318.122561 |
| FP16 | 1636.5189 | 46.8402 | 39.107402 | 320.238035 |

| Phase | FP16 Host wall change | Throughput change | Paired95% wall-ratio CI |
|---|---:|---:|---|
| prefill | +12.890% | -11.418% | [1.126562, 1.131365] |
| decode | +0.665% | -0.661% | [1.005196, 1.008051] |

Independent implementation checks: actual layer14 prefill/decode and chain3 outputs exact; Q/K, AV, postNorm, Gate/Up and low/high SP2 live carriers exact; repeat10 deterministic. Full28 fixed trajectory: all448 per-layer FP16 byte hashes exact to independent CPU, all16 final hidden and normalized vectors exact, full-vocabulary head selected IDs/codes exact. FP32 full-model control88 exported files exact against sealed EXP0284. These establish implementation correctness, not model quality.
All 4800 short/formal token profiles have complete additive ledgers and declared physical gates. Both arms have equal weight DMA bytes, HMX commands/tile pairs and VTCM arena peak 8365824 bytes; grant8388608. No explicit timed intermediate DDR or spills reported by runtime counters.

Memory qualification: FP16 live residual occupies262144 bytes versus524288 for M64; it is physically packed, not FP32 with fake rounding. A reserved unused262144-byte hole keeps subsequent HMX/DMA addresses identical to the control. Total VTCM plan is therefore NOT reduced in this experiment. Initial compact allocation caused a cDSP HMX memory access fault at raw native-W4 QKV accumulation (BadVA0xFF3FE800); restoring downstream placement fixed it. Exact hardware address constraint not isolated, no general bank-boundary claim. Full compaction is future work.

Compiler qualification: both inherited FP32 and candidate assembly contain vector spills on worker stacks; runtime intermediate-DDR counters do not instrument compiler stack traffic. No full-sized FP32 residual tensor or explicit conversion buffer is introduced. Do not interpret zero reported spills as proof of zero compiler stack traffic.

Retained recovery evidence: initial build script permission invocation corrected by explicit bash; missing global NumPy corrected using existing quantization environment; helper forward declarations fixed; first two candidate device attempts faulted before producing outputs. Header-rejection hypothesis was disproved by logcat DSP_RUNNING/precise HMX exception. Padded-placement candidate passes independent references. First CPU hash export used standard FNV offset instead of project-specific basis; a02 recomputed all hashes from the independent CPU trajectory, old results retained. Original-baseline fairness check additionally found altered generated code slowed the new FP32 branch. A fresh fixed5short/10formal campaign pairs the unchanged original sealed binary with the unchanged candidate; both output references and binary hashes are verified. Initial same-binary campaign is retained in the parent directory, not pooled. No threshold relaxation or discarded timing rounds.

Repeat1 auxiliary timing is retained in SUMMARY.json; only repeat10 formal timing supports the speed conclusion.

## Attribution and conclusion

Compared with the actual original baseline, formal FP16 prefill adds4.465412ms: InputNorm+0.786859ms, post-attentionNorm+0.638278ms, O epilogue/projection+1.796029ms, Down epilogue/projection+1.272963ms. These are exclusive stage-ledger differences, not a claim that every extra instruction has been individually timed. FP16 storage introduces widening, narrowing, register shuffling and partial vector stores around retained FP32 arithmetic. HMX operand formats, matrix work and weight traffic remain unchanged. The tested implementation fails the10percent prefill speed gate; decode passes. Retain original FP32 speed baseline. This does not prove an optimized FP16 residual path cannot be faster, and no further optimization is authorized or executed in this experiment.

Initial same-binary control1795-range prefill was not the original latest baseline: common helper dispatch changed generated FP32 Norm code. The first4800-profile campaign and its apparent9.52percent prefill overhead remain as supplementary evidence; they do not support a10percent acceptance against the original baseline. Final original-binary4800-profile paired campaign is authoritative for speed. Both are fully retained.
