# EXP-0290: Qwen3-0.6B W4A16 speed inversion diagnosis

The inversion is real in EXP-0289 formal measurements. None of the configurations here restores the intended W16A16 < W4A16 < W4A8 throughput staircase. No new baseline is selected.

## Unchanged formal reference

| Recipe | Prefill token/s | Decode token/s |
|---|---:|---:|
| W16A16 | 1710.94 | 27.68 |
| W4A16 | 1678.35 | 25.65 |

EXP-0289: ten formal rounds, repeat10, M64+42/cache128/batch1/all28layers; embedding, final norm, head/greedy and FastRPC included, loading/tokenization excluded. Full floating-reference alignment remains failed; user-authorized provisional timing, no PPL claim.

## Diagnosis

Historical exclusive module means (ms), from EXP-0289:

| Module | Prefill F16 | Prefill W4A16 | Decode F16 | Decode W4A16 |
|---|---:|---:|---:|---:|
| QKV including overlapped Q/K Norm/RoPE work | 7.631 | 11.479 | 7.623 | 11.610 |
| Gate/Up + activation | 10.066 | 9.533 | 10.061 | 9.671 |
| Down | 3.847 | 3.588 | 3.848 | 3.629 |
| LM head + greedy, excluding final norm | 6.355 | 4.335 | 6.340 | 4.808 |
| Host wall | 37.406 | 38.133 | 36.133 | 38.993 |

Decode KV carrier conversion is additionally 0.425 ms F16 versus 1.136 ms W4A16.
Complete module table: /mnt/d/llm_exp/results/qwen3-block-htp/exp0289/MODULES.md.
Overlapping expansion/work/DMA counters are not additive exclusive timings.

- W4F16 expands packed W4 with HVX and uses FP16 HMX plus channel scales. Compression reduces DDR traffic, not the underlying FP16 matrix arithmetic. Representative decode weight DDR bytes: F16 1,191,970,816; W4F16 300,666,368; both 4402 HMX commands. The weights are not accidentally loaded as full FP16 from DDR.
- The inherited QKV group2/region32 at K1024 has two regions per group. Main takes one and the pool shares the other, with worker start/join and publication overhead retained. Adaptive mode caps non-Down expansion workers at two. Head4 mildly improves performance; these diagnostics do not establish the group size as the dominant cause.
- Decode direct grouped V is unpacked for all64 physical rows before one row is appended to KV. Native Q also unpacks full heads before repacking. This is concrete avoidable work; its isolated benefit is not measured yet.
- Both A16 recipes retain M64 backbone HMX projections and substantial M64 packing/common work for logical M1 decode. This is shared inefficiency, not by itself an explanation of the inversion.
- Existing decode optimization is active: 224 calls (28 layers x8 KV groups), head prefetch593. No missing OPT flag or scalar fallback was identified as the primary cause.

Source anchors: src/dsp/block_imp.c qbh_w4f16_expand_with_main, qbh_w4f16_projection_worker_count, qbh_w4f16_projection_group_tiles, qbh_run_w4f16_projection, qbh_scan_append_f16_kv_hmx_native, qbh_scan_f16_attention; src/dsp/w4_u8_expand.c; src/dsp/hmx_fp16.c.

## Same-binary bounded diagnostics

Each row is one launch with repeat3. Same archived EXP-0289 binary and package, local and device SHA checks, fixed token trajectory. Small deltas are not formal results and order effects are not ruled out.

| Configuration | Prefill token/s | Decode token/s | Prefill delta | Decode delta |
|---|---:|---:|---:|---:|
| control | 1691.17 | 25.88 | +0.00% | +0.00% |
| head4 | 1697.27 | 26.24 | +0.36% | +1.38% |
| row | 1648.96 | 26.08 | -2.50% | +0.77% |
| head4row | 1655.87 | 26.21 | -2.09% | +1.26% |

Control = group2/native QKV; head4 = head-aligned batch4; row = row-major QKV retaining native input/post norms; head4row = both.
516 profiles passed physical ledger/VTCM/no-spill checks; every token ID and selected-logit half bit matched the frozen own-hardware output. This does not establish full hidden/KV bytewise equivalence, independent arithmetic agreement, or quality. No full boundary audit or formal campaign was run because no configuration restored the target ordering.
The region8 attempt was rejected before computation: selected adaptive pipeline requires region32. Failed protocol/exit/stdout/stderr retained; no condition bypassed. Remaining valid configurations were tested independently.

## Next implementation priorities

1. Remove decode full-row conversion: directly gather the live V row from grouped tiles and extract active Q rows. Check full hidden/norm/KV equivalence.
2. Adapt QKV producer/consumer scheduling at K1024; reduce repeated expansion dispatch/join while retaining head-release overlap. Group4 alone is insufficient.
3. Reduce A16 physical decode work toward logical M1: active-row HVX preparation first, minimum legal HMX row tile and correct strides next. Share applicable improvements with W16A16 for fairness.
4. Validate complete paired E2E after exact boundary checks. Do not infer speed ordering solely from bitwidth.

No native kernel/model changes, no numerical threshold changes, no promoted configuration. EXP-0289 remains the measured provisional baseline.

Documentation correction: the exact dynamic attention symbol is qbh_scan_f16_attention. The sealed external/source report used qbh_scan_attention_f16_dynamic as an incorrect source anchor; measurements, hashes and conclusions are unchanged. Original sealed evidence is preserved.
