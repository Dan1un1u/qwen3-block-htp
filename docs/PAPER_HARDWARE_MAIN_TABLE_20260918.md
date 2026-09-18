# Hardware main-table inventory and EXP0293 promotion

User-promoted Qwen3-0.6B W16A16 baseline: EXP0293 full-M64 FP32 residual. Selection applies to paper speed only; independent full-floating alignment remains failed. No new native changes, builds or device experiments.

## Intended paper structure

Model × four recipes (W16A16/W4A16/W4A8/W4A8-SP2) × three Table12 workloads =48 conditions. Header: Model | Configuration | Dataset | Prefill Len | Decode Len | Prefill TPS | Decode TPS. Same SoC/measurement boundary, batch1, matched actual prompts/cache/stopping and repetition policy required before comparison. Memory/energy omitted from current speed main table.

| Dataset | Prefill Len | Decode Len |
|---|---:|---:|
| HellaSwag | 64 | 42 |
| Persona-Chat | 536 | 46 |
| DroidTask | 741 | 3 |

## Currently measured inventory

All values below are original directly measured E2E, not new HEAD timings and not assertions of named-dataset evaluation. Matching64+42 lengths alone does not establish HellaSwag provenance. No values are extrapolated to536+46 or741+3. All rotations disabled.

| Model | Configuration | Measured I/O | Prefill TPS | Decode TPS | Evidence |
|---|---|---|---:|---:|---|
| Qwen3-0.6B | W16A16 | 64+42 | 1687.09 | 27.38 | EXP-0293; formal; user-selected baseline |
| Qwen3-0.6B | W4A16 | 64+42 | 1737.47 | 27.67 | EXP-0291; diagnostic repeat3; formal pending |
| Qwen3-0.6B | W4A8 | 64+42 | 3074.86 | 93.47 | EXP-0294; formal |
| Qwen3-0.6B | W4A8-SP2 | 64+42 | 3020.43 | 92.40 | EXP-0288; formal |
| Qwen3-1.7B | W16A16 | 64+15 | 797.80 | 12.50 | EXP-0283; formal |
| Qwen3-1.7B | W4A16 | 64+15 | 1040.24 | 15.30 | EXP-0260; formal |
| Qwen3-1.7B | W4A8 | 64+15 | 1857.85 | 47.22 | EXP-0284; formal |
| Qwen3-1.7B | W4A8-SP2 | 64+15 | 1846.79 | 47.12 | EXP-0284; formal |
| Llama-3.2-1B-Instruct | W16A16 | 64+15 | 974.17 | 17.47 | L32-0039; formal |
| Llama-3.2-1B-Instruct | W4A16 | 64+7 | 1261.16 | 23.60 | L32-0006; formal |
| Llama-3.2-1B-Instruct | W4A8 | 64+15 | 2373.11 | 45.70 | L32-0040; formal |
| Llama-3.2-1B-Instruct | W4A8-SP2 | 64+15 | 2335.24 | 45.74 | L32-0040; formal |
| Llama-3.2-3B-Instruct | W16A16 | — | — | — | unsupported; user deferred |
| Llama-3.2-3B-Instruct | W4A16 | 64+42 | 493.22 | 7.93 | L32-0047; formal |
| Llama-3.2-3B-Instruct | W4A8 | 64+42 | 1206.68 | 23.73 | L32-0046; formal |
| Llama-3.2-3B-Instruct | W4A8-SP2 | 64+42 | 1184.72 | 23.64 | L32-0045; formal |

Qwen0.6 W4A16 archive row is one coherent latest/decode-best repeat3 candidate (1737.47/27.67), not a formal paper result. The older full ten-round EXP0289 value1678.35/25.65 remains valid historical formal evidence. Prefill-best EXP0291 is a different candidate1747.22/27.10; do not splice best columns from different binaries.
Other measured A16 rows retain their historical FP16 residuals; only the promoted Qwen0.6 W16A16 row uses FP32. Measured A8/SP2 rows use FP32 residual. This heterogeneity must stay visible and must not be labeled all-recipes FP32.
A8/SP2 comparisons for1.7B/1B use the same fixed-trajectory paired experiments0284/0040; do not choose the larger greedy or earlier noisy values per cell. These are archived numerical implementations, not software-side precision validation.
Completed missing implementations: Qwen0.6 ordinaryA8 (EXP0294), Llama3B ordinaryA8 (L32-0046), Llama3B W4A16 (L32-0047). User-deferred: Llama3B W16A16. Pending: Qwen0.6 latest W4A16 formal repetition; named-dataset alignment across all48 conditions and runtime support/measurement for536+46/741+3. Existing1B and1.7B M64+7/15 results cannot fill64+42 decode cells.
References: authoritative per-experiment status/index/reports; exact current values and target48 placeholders in paired JSON. Quant.npu Table12 was read and page21 visually inspected.

## EXP0294 update
Qwen3-0.6B ordinary W4A8 completed: M64+42,3074.86/93.47 TPS (prefill/decode), ten formal repeat10; project prompt only, named datasets remain pending. 3B W16A16 user-deferred due uint32-offset ABI. All other existing rows unchanged. Machine-readable inventory updated.

## L32-0046 / L32-0047 completion
Both remaining authorized Llama3B recipes implemented and formally measured. W4A8 only removesDownSP2 and retains latest0045 sharedoptimizations/FP32residual; W4A16 uses genericFP16baseline with capacity-correct existingpipeline, no specialization tuning. All times remain project-owned prompts. L32-0047 full mathematical FP16alignment FAILS (max hidden/normNRMSE0.006557, gate0.003); independent native-vs-rowcache exact checks and components/slices pass. Do not equate timing or implementation checks with model-quality acceptance. Existing SP2 rows remain original selected formal records, not replaced by new matched-control fluctuations.
