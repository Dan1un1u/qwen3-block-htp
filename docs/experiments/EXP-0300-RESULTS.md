# EXP-0300 generic A16 long-context throughput repair

Scope: Qwen1.7B, Llama1B and Llama3B, sequentially; Qwen0.6 excluded. Fixed A64+42/B536+46/C741+3 token fixtures. Existing A8/SP2 and curated D remain unchanged. Basic shared HVX scheduling and cache-DMA overlap only; no model-specific specialization.

Ten formal rotated paired rounds with ten retained replays each, following five short rounds. Mean TPS is total tokens/total complete Host wall. M64 control unchanged. Higher-is-better prefill ratio is long/fastest eligible same-campaign M64; bootstrap95% lower>=0.90. Same-length candidate/control decode wall upper<=1.10. No intra-phase stopping or sample deletion.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Qwen3-1.7B | 64+42 | W16A16 | 778.91 | 12.50 |
| Qwen3-1.7B | 64+42 | W4A16 | 1038.81 | 15.32 |
| Qwen3-1.7B | 536+46 | W16A16 | 766.14 | 11.71 |
| Qwen3-1.7B | 536+46 | W4A16 | 954.48 | 14.19 |
| Qwen3-1.7B | 741+3 | W16A16 | 777.17 | 11.42 |
| Qwen3-1.7B | 741+3 | W4A16 | 948.60 | 13.78 |

## Gate results

| Model | Recipe | Shape | Prefill ratio lower95% | Decode wall upper95% | Pass |
|---|---|---|---:|---:|---|
| Qwen3-1.7B | w16 | B | 0.97686 | 1.01059 | True |
| Qwen3-1.7B | w16 | C | 0.98640 | 1.01540 | True |
| Qwen3-1.7B | w4 | B | 0.91703 | 1.00524 | True |
| Qwen3-1.7B | w4 | C | 0.91175 | 1.00411 | True |

## Decode across contexts (descriptive)

The regression gate compares equal long shapes. This table describes the change from A64+42 to B536+46 or C741+3, where context and decode counts both differ.

| Model | Recipe | B/A decode TPS | C/A decode TPS |
|---|---|---:|---:|
| Qwen3-1.7B | w16 | 93.68% | 91.37% |
| Qwen3-1.7B | w4 | 92.62% | 89.96% |

## Method and implementation evidence

C1 completed all five engineering short rounds; Qwen W4 prefill confidence failed (B lower0.89936, C lower0.89028). Formal hardware had not started when the candidate was stopped at the completed phase boundary. All short records and phase-stop proof retained. C2 passed Qwen W4 B but missed C with lower95%0.89697. C3 native probability production was byte-exact but slower: eliminating an explicit pack did not compensate for its more expensive producer. These negative candidates are retained. The final candidate and its gates are identified per model below. Selected C4 A16 long runs use QBH_LONG_OPT=3, control0; option1 retains C2. C4 restores C2 row-major probability layout and alternates asynchronous V and next-group K cache reads in one reusable VTCM slot, behind QK/softmax and AV. The reusable slot follows the raw KV extent; only padding is cleared, while DMA overwrites every valid element. This removes redundant full-plane clearing without changing the existing M64 compute policy. DMA staging counters measure exposed issue/join time rather than engine occupancy. C2 fuses conversion/scale/mask and omits entirely masked causal tiles, retaining reduction order over every contributing tile. The bottleneck was serialized long-context floating softmax. Main plus three existing HVX workers process disjoint rows. Original per-row FP32 reduction/exp sequence and FP16 rounding are retained; scratch occupies 13,312 bytes of phase-dead expanded-weight VTCM. No extra pool, tensor DDR, HMX owner or per-model tile specialization. Original serial control remains selectable. Native M64 and single-row decode retain original dispatch.

Each A16 recipe passed unchanged64/65/128/129/536/741, native and poisoned-KV audits. Original full capture files are byte-exact; independent actual-operand normalization/head checks and physical accounting remain enabled in untimed audits. A8/SP2 frozen parent capture fields are all byte-exact. Old A8 archives omit final_hidden captures added by EXP0299; only these named additions are accepted, after exact hash/size/finite validation. Failed initial schema assertion and reconciliation are retained.

Implementation checks are not teacher alignment, PPL or quality acceptance; existing mathematical-alignment failures remain failures. Weight encodings, residual policies, KV capacity832,8MiB VTCM and full-model costs retained. Llama3B includes existing resident segmented mapping. No baseline promotion.

## Provenance

Source closure: d2af6c7d3d5bccf467b5dd194be08c16ae6ed68c.

- Qwen3-1.7B: measured source d2af6c7d3d5bccf467b5dd194be08c16ae6ed68c, 1.7B/c4/formal-summary.json, 1.7B/audits-c4-pass.json, 1.7B/c4/a8-regressions-pass.json.

Full additive module tables: companion MODULES report. All original/failed/successful samples retained and sealed in EVIDENCE_SHA256.json.
