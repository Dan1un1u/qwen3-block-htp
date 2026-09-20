# L32-0060: SP2 HVX shift-add full-model comparison

Same-binary paired native HMX versus four-worker HVX shift/add on Down only. No rotation, SP2mode8, frozen per-channel W4 and FP32 residual. Fixed token trajectories, not named datasets. Full embedding, all layers, final norm, LM head/greedy and FastRPC; cold loading/tokenizer and audit captures excluded; host staging included.

| Model | P+D | HMX prefill TPS | HVX prefill TPS | Prefill wall ratio | HMX decode TPS | HVX decode TPS | Decode wall ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2-1B | 64+42 | 2318.61 | 85.42 | 27.143x | 44.51 | 17.08 | 2.607x |
| Llama3.2-1B | 536+46 | 2496.05 | 79.81 | 31.273x | 50.75 | 17.92 | 2.832x |
| Llama3.2-3B | 64+42 | 1181.46 | 32.96 | 35.851x | 23.64 | 7.29 | 3.243x |
| Llama3.2-3B | 536+46 | 1148.73 | 30.70 | 37.421x | 23.74 | 7.28 | 3.260x |

All five short and ten formal repeat10 rounds retained, AB/BA paired; bootstrap20k seed600060. Numerical/physical implementation checks are separate from speed parity. Detailed confidence intervals and per-round values in SUMMARY.json.

The component0059 result is not used as an E2E denominator. This integration reuses persistent workers and dead-stage VTCM, consumes the unchanged fused SP2 byte planes, includes metadata preparation, native-W4 gather/unpack, DMA, vector shifts/signed32 accumulation and the same compiled FP32 epilogue. It does not establish an algorithmic lower bound on every possible HVX implementation.

3B producer Down prefetch is sized to the HVX N128 task batch and reused without duplicate DDR reads; all metadata handling remains timed. Whole backend, not an intrinsic-only comparison.

No model-quality/PPL claim or baseline promotion. Native HMX default retained.

Source builds: {"1B": "5ec72782c410fbbe85ec5d92e031c8fc8fd14797", "3B": "4806798f979834f6913cd97f09c401e476ad709d"}. Within each model both arms use one identical binary.

Single-layer, chain3 and full-model audits pass. Full-model independent layer/head references and paired activation/KV/residual capture files are bit-exact. Both arms stay within8MiB VTCM, with no intermediate tensor DDR spill.

HVX Down occupies96.3-97.2percent of prefill Host wall and59.3-65.8percent of decode Host wall across these four cases. The3B/1B HVX Down time ratio is2.60-2.62, close to the2.625 ratio of dot-product terms. These are measured implementation costs, not claimed hardware peak throughput. Long536 prefill comprises nine M64 chunks including tail padding.
