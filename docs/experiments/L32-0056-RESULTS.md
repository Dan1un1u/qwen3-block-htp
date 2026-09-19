# L32-0056 generic A16 long-context throughput repair

Scope: Qwen1.7B, Llama1B and Llama3B, sequentially; Qwen0.6 excluded. Fixed A64+42/B536+46/C741+3 token fixtures. Existing A8/SP2 and curated D remain unchanged. Basic shared HVX scheduling and cache-DMA overlap only; no model-specific specialization.

Ten formal rotated paired rounds with ten retained replays each, following five short rounds. Mean TPS is total tokens/total complete Host wall. M64 control unchanged. Higher-is-better prefill ratio is long/fastest eligible same-campaign M64; bootstrap95% lower>=0.90. Same-length candidate/control decode wall upper<=1.10. No intra-phase stopping or sample deletion.

| Model | Shape | Recipe | Prefill token/s | Decode token/s |
|---|---|---|---:|---:|
| Llama3.2-1B | 64+42 | W16A16 | 962.35 | 17.88 |
| Llama3.2-1B | 64+42 | W4A16 | 1254.24 | 23.94 |
| Llama3.2-1B | 536+46 | W16A16 | 1094.36 | 17.05 |
| Llama3.2-1B | 536+46 | W4A16 | 1380.14 | 22.31 |
| Llama3.2-1B | 741+3 | W16A16 | 1100.74 | 16.51 |
| Llama3.2-1B | 741+3 | W4A16 | 1378.00 | 21.65 |
| Llama3.2-3B | 64+42 | W16A16 | 400.98 | 6.80 |
| Llama3.2-3B | 64+42 | W4A16 | 493.81 | 8.14 |
| Llama3.2-3B | 536+46 | W16A16 | 413.17 | 6.62 |
| Llama3.2-3B | 536+46 | W4A16 | 476.28 | 7.70 |
| Llama3.2-3B | 741+3 | W16A16 | 421.37 | 6.55 |
| Llama3.2-3B | 741+3 | W4A16 | 489.17 | 7.63 |

## Gate results

| Model | Recipe | Shape | Prefill ratio lower95% | Decode wall upper95% | Pass |
|---|---|---|---:|---:|---|
| Llama3.2-1B | w16 | B | 1.12113 | 1.00514 | True |
| Llama3.2-1B | w16 | C | 1.13227 | 1.01314 | True |
| Llama3.2-1B | w4 | B | 1.09926 | 1.00055 | True |
| Llama3.2-1B | w4 | C | 1.09749 | 0.98214 | True |
| Llama3.2-3B | w16 | B | 1.01776 | 1.00035 | True |
| Llama3.2-3B | w16 | C | 1.03414 | 1.00328 | True |
| Llama3.2-3B | w4 | B | 0.94150 | 1.01405 | True |
| Llama3.2-3B | w4 | C | 0.98901 | 0.99993 | True |

## Decode across contexts (descriptive)

The regression gate compares equal long shapes. This table describes the change from A64+42 to B536+46 or C741+3, where context and decode counts both differ.

| Model | Recipe | B/A decode TPS | C/A decode TPS |
|---|---|---:|---:|
| Llama3.2-1B | w16 | 95.32% | 92.34% |
| Llama3.2-1B | w4 | 93.16% | 90.43% |
| Llama3.2-3B | w16 | 97.32% | 96.28% |
| Llama3.2-3B | w4 | 94.62% | 93.83% |

## Method and implementation evidence

C1 completed all five engineering short rounds; Qwen W4 prefill confidence failed (B lower0.89936, C lower0.89028). Formal hardware had not started when the candidate was stopped at the completed phase boundary. All short records and phase-stop proof retained. C2 passed Qwen W4 B but missed C with lower95%0.89697. C3 native probability production was byte-exact but slower: eliminating an explicit pack did not compensate for its more expensive producer. These negative candidates are retained. The final candidate and its gates are identified per model below. Selected C4 A16 long runs use QBH_LONG_OPT=3, control0; option1 retains C2. C4 restores C2 row-major probability layout and alternates asynchronous V and next-group K cache reads in one reusable VTCM slot, behind QK/softmax and AV. The reusable slot follows the raw KV extent; only padding is cleared, while DMA overwrites every valid element. This removes redundant full-plane clearing without changing the existing M64 compute policy. DMA staging counters measure exposed issue/join time rather than engine occupancy. C2 fuses conversion/scale/mask and omits entirely masked causal tiles, retaining reduction order over every contributing tile. The bottleneck was serialized long-context floating softmax. Main plus three existing HVX workers process disjoint rows. Original per-row FP32 reduction/exp sequence and FP16 rounding are retained; scratch occupies 13,312 bytes of phase-dead expanded-weight VTCM. No extra pool, tensor DDR, HMX owner or per-model tile specialization. Original serial control remains selectable. Native M64 and single-row decode retain original dispatch.

Each A16 recipe passed unchanged64/65/128/129/536/741, native and poisoned-KV audits. Original full capture files are byte-exact; independent actual-operand normalization/head checks and physical accounting remain enabled in untimed audits. A8/SP2 frozen parent capture fields are all byte-exact. Old A8 archives omit final_hidden captures added by EXP0299; only these named additions are accepted, after exact hash/size/finite validation. Failed initial schema assertion and reconciliation are retained.

Implementation checks are not teacher alignment, PPL or quality acceptance; existing mathematical-alignment failures remain failures. Weight encodings, residual policies, KV capacity832,8MiB VTCM and full-model costs retained. Llama3B includes existing resident segmented mapping. No baseline promotion.

## Provenance

Source closure: 68d9401265b8f77aa6de9d1d2135d5608b61e9db.

- Llama3.2-1B: measured source 68d9401265b8f77aa6de9d1d2135d5608b61e9db, 1B/c4/formal-summary.json, 1B/audits-c4-pass.json, 1B/c4/a8-regressions-pass.json.
- Llama3.2-3B: measured source 68d9401265b8f77aa6de9d1d2135d5608b61e9db, 3B/c4/formal-summary.json, 3B/audits-c4-pass.json, 3B/c4/a8-regressions-pass.json.

## Transport interruption disclosure

Llama3B formal round10 W4A16 B was interrupted by loss of the ADB device. All five completed replays, including the slow ones, were retained. The five missing replays were completed after reconnection with the same frozen source, binary, model and runtime settings. The partial sixth replay and original exit255 remain in raw evidence. The final round spans this interruption; its combined mean includes all ten completed replays, not a replacement faster run. A second disconnection before any W4 C candidate completion was recovered after user-confirmed charging/cooling (85%,36.5 C). That arm and final control were then measured without replacing earlier samples. See 3B/FORMAL_TRANSPORT_RECOVERY.json, c4/recovery-transport and c4/recovery-after-charge.

Full additive module tables: companion MODULES report. All original/failed/successful samples retained and sealed in EVIDENCE_SHA256.json.
