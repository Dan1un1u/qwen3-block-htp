# EXP-0299: generic A16 ABC completion

Fixed-shape hardware measurements, not named-dataset evaluation. A=64+42, B=536+46, C=741+3; batch1, complete model including embedding, final norm, LM head and greedy selection. Decode count means decode RPCs, excluding the token produced by prefill. Full Host wall includes FastRPC and long-path Host input/RoPE staging. Cold loading, external tokenization and audit output are excluded.

Five short and ten formal paired rounds per model; repeat10, all samples retained. Shape order and W16/W4 order rotate. TPS = total tokens / total Host wall. Ratio intervals are two-sided paired Student-t 95% intervals over round wall ratios, not bootstrap intervals.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Qwen3-0.6B | 64+42 | W16A16 | 1674.05 | 27.18 |
| Qwen3-0.6B | 64+42 | W4A16 | 1740.55 | 27.28 |
| Qwen3-0.6B | 536+46 | W16A16 | 1102.28 | 23.91 |
| Qwen3-0.6B | 536+46 | W4A16 | 1082.80 | 23.93 |
| Qwen3-0.6B | 741+3 | W16A16 | 1002.08 | 22.85 |
| Qwen3-0.6B | 741+3 | W4A16 | 982.41 | 23.22 |
| Qwen3-1.7B | 64+42 | W16A16 | 790.10 | 12.66 |
| Qwen3-1.7B | 64+42 | W4A16 | 1036.90 | 15.34 |
| Qwen3-1.7B | 536+46 | W16A16 | 641.36 | 11.86 |
| Qwen3-1.7B | 536+46 | W4A16 | 753.51 | 14.24 |
| Qwen3-1.7B | 741+3 | W16A16 | 612.47 | 11.57 |
| Qwen3-1.7B | 741+3 | W4A16 | 706.93 | 13.87 |

## Scope of changes

Missing long A16 execution now uses generic M64 chunks, absolute RoPE and persistent row-major FP16 KV with correct valid tails. Basic HVX copies and packed-K inverse gathers replace scalar carrier conversion. Generic HVX vector softmax supports long attention and the existing native decode path for both A16 recipes. Existing DMA/HMX overlap and head buffering remain. No new per-model worker, tile or scheduling specialization was introduced. W16 and W4 share the applicable basic vector implementation.

Qwen0.6 W16 keeps the promoted FP32 residual and original full-M64 decode scheduling; other A16 residual storage remains FP16. Weight payloads/encodings are inherited unchanged from pinned parents. Llama3B retains existing resident segmented-weight mapping, with mapping costs inside timing; no new disk paging.

The preliminary Qwen0.6 five-round short campaign triggered the >10% W4 slowdown investigation: long-prefill W4 was about 32-34% slower. Module accounting identified scalar KV carrier conversion rather than weight-expansion overlap as the main differential. Sharing basic vector conversion/copies repaired this; 8498 before/after audit tensor files were byte-exact. This generic repair was applied to both source trees and both A16 recipes. No optimization loop was started merely to force staircase speedups.

Qwen0.6 final measurements use dee3b14, where its native vector softmax was already enabled. Later source changes share that same kernel with other model sizes and add audits; they do not change the Qwen0.6 timed kernel or row schedule. Other models use their explicitly archived source/binary seals.

## Implementation checks and limitations

Each model/recipe: native 64+42; long 64+42, 65+3, 128+3, 129+3, 536+46, 741+3; and 741+3 with poisoned unused KV. Audits and profiling are separate. Actual-operand scalar softmax references enforce the existing 0.003 normalized-error tolerance; independent FP32 norm and full-vocabulary frozen-weight head references check consumed operands and selected half-precision logits. On the new long path, KV historical prefix/future suffix remain byte-exact at every update, new entries are finite, and poisoned unused memory cannot alter live layer hashes/head outputs. Timing ledgers are additive; requested/acquired VTCM=8MiB, planned peak<=8MiB, no counted tensor-intermediate DDR/spill, one HMX owner.

These checks validate implemented arithmetic and changed boundaries, not equivalence to the original teacher, PPL or text quality. Previously recorded full-floating alignment failures remain failures. Fixed token trajectories prevent divergent model outputs from changing lengths. No gate tolerance was relaxed, no baseline deliberately slowed, and no automatic recipe/quality promotion is made.

## W4/W16 paired wall ratios

User-requested investigation trigger: mean W4/W16 Host wall >1.10 in either stage. Long/M64 throughput is descriptive for basic A16; the existing A8/SP2 long-throughput gate is separate.

| Model | Shape | Stage | W4/W16 wall | 95% interval | >10% trigger |
|---|---|---|---:|---|---|
| Qwen3-0.6B | A | prefill | 0.961795 | [0.960637, 0.962953] | False |
| Qwen3-0.6B | A | decode | 0.996343 | [0.995078, 0.997609] | False |
| Qwen3-0.6B | B | prefill | 1.017994 | [1.016124, 1.019863] | False |
| Qwen3-0.6B | B | decode | 0.999525 | [0.996779, 1.002270] | False |
| Qwen3-0.6B | C | prefill | 1.020033 | [1.017957, 1.022108] | False |
| Qwen3-0.6B | C | decode | 0.984007 | [0.980033, 0.987981] | False |
| Qwen3-1.7B | A | prefill | 0.761989 | [0.760443, 0.763534] | False |
| Qwen3-1.7B | A | decode | 0.825107 | [0.823514, 0.826701] | False |
| Qwen3-1.7B | B | prefill | 0.851160 | [0.850238, 0.852082] | False |
| Qwen3-1.7B | B | decode | 0.833016 | [0.830679, 0.835352] | False |
| Qwen3-1.7B | C | prefill | 0.866384 | [0.865118, 0.867650] | False |
| Qwen3-1.7B | C | decode | 0.834753 | [0.832948, 0.836558] | False |

## Provenance

Source closure: 009b4fb8c3b5edd94a7521d34b4dc5103409403b; branch codex/exp-0299-a16-tables.

- Qwen3-0.6B: source dee3b14d83149da720658e3cb006e9cb352781cf; 0.6B/vector/formal-summary.json; audit 0.6B/audits-vector. Full additive modules: 0.6B/RESULTS.md.
- Qwen3-1.7B: source 009b4fb8c3b5edd94a7521d34b4dc5103409403b; 1.7B/shared/formal-summary.json; audit 1.7B/audits-shared. Full additive modules: 1.7B/RESULTS.md.

All preliminary/failed attempts remain in the evidence tree and are not mixed into final measurements. Each final-campaign.json identifies accepted measurements. Parent authority/fixture pins, model manifests, uploaded binary SHA256 checks, raw commands/logs and per-replay records are retained. EVIDENCE_SHA256.json seals the completed experiment.

Desktop workbook: update only the 24 A16 ABC rows across the companion experiments; preserve all A8/SP2 values and eight curated historical D rows.
