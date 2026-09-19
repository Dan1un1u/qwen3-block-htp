# L32-0055: generic A16 ABC completion

Fixed-shape hardware measurements, not named-dataset evaluation. A=64+42, B=536+46, C=741+3; batch1, complete model including embedding, final norm, LM head and greedy selection. Decode count means decode RPCs, excluding the token produced by prefill. Full Host wall includes FastRPC and long-path Host input/RoPE staging. Cold loading, external tokenization and audit output are excluded.

Five short and ten formal paired rounds per model; repeat10, all samples retained. Shape order and W16/W4 order rotate. TPS = total tokens / total Host wall. Ratio intervals are two-sided paired Student-t 95% intervals over round wall ratios, not bootstrap intervals.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Llama3.2-1B | 64+42 | W16A16 | 978.44 | 18.13 |
| Llama3.2-1B | 64+42 | W4A16 | 1255.25 | 23.93 |
| Llama3.2-1B | 536+46 | W16A16 | 827.54 | 17.26 |
| Llama3.2-1B | 536+46 | W4A16 | 968.17 | 22.29 |
| Llama3.2-1B | 741+3 | W16A16 | 772.98 | 16.78 |
| Llama3.2-1B | 741+3 | W4A16 | 885.27 | 21.29 |
| Llama3.2-3B | 64+42 | W16A16 | 405.72 | 6.85 |
| Llama3.2-3B | 64+42 | W4A16 | 493.95 | 8.12 |
| Llama3.2-3B | 536+46 | W16A16 | 357.64 | 6.68 |
| Llama3.2-3B | 536+46 | W4A16 | 402.12 | 7.74 |
| Llama3.2-3B | 741+3 | W16A16 | 348.57 | 6.58 |
| Llama3.2-3B | 741+3 | W4A16 | 386.69 | 7.60 |

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
| Llama3.2-1B | A | prefill | 0.779487 | [0.777422, 0.781553] | False |
| Llama3.2-1B | A | decode | 0.757906 | [0.756911, 0.758900] | False |
| Llama3.2-1B | B | prefill | 0.854754 | [0.853848, 0.855661] | False |
| Llama3.2-1B | B | decode | 0.774582 | [0.771559, 0.777605] | False |
| Llama3.2-1B | C | prefill | 0.873160 | [0.872310, 0.874011] | False |
| Llama3.2-1B | C | decode | 0.788060 | [0.786229, 0.789892] | False |
| Llama3.2-3B | A | prefill | 0.821385 | [0.819059, 0.823712] | False |
| Llama3.2-3B | A | decode | 0.844088 | [0.843085, 0.845091] | False |
| Llama3.2-3B | B | prefill | 0.889387 | [0.888731, 0.890042] | False |
| Llama3.2-3B | B | decode | 0.863080 | [0.862064, 0.864097] | False |
| Llama3.2-3B | C | prefill | 0.901447 | [0.899998, 0.902896] | False |
| Llama3.2-3B | C | decode | 0.866886 | [0.864076, 0.869695] | False |

## Provenance

Source closure: 5a818c744e3aa63b9f5c6401248ec17983e6a86e; branch codex/llama32-no-rotation.

- Llama3.2-1B: source 5a818c744e3aa63b9f5c6401248ec17983e6a86e; 1B/shared/formal-summary.json; audit 1B/audits-shared. Full additive modules: 1B/RESULTS.md.
- Llama3.2-3B: source 5a818c744e3aa63b9f5c6401248ec17983e6a86e; 3B/shared/formal-summary.json; audit 3B/audits-shared. Full additive modules: 3B/RESULTS.md.

All preliminary/failed attempts remain in the evidence tree and are not mixed into final measurements. Each final-campaign.json identifies accepted measurements. Parent authority/fixture pins, model manifests, uploaded binary SHA256 checks, raw commands/logs and per-replay records are retained. EVIDENCE_SHA256.json seals the completed experiment.

Desktop workbook: update only the 24 A16 ABC rows across the companion experiments; preserve all A8/SP2 values and eight curated historical D rows.
