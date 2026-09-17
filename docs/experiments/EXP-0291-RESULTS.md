# EXP-0291 W4A16 archive — user stopped further optimization

Qwen3-0.6B, M64+42/cache128/batch1/full28. User requested stopping and archiving on2026-09-18. No inference/build process remained. No additional device tests were started after the stop. The last proposed M64-to-M32 projection change failed at source-pattern lookup before any file write; it was neither implemented nor built.

## Retained implementations

There is no uniquely established formal winner. Two non-dominated observed configurations are preserved separately; do not combine their maxima into one result.

| Archive | Prefill E2E token/s | Decode token/s | 64-token Host ms | 42-step Host ms | Evidence |
|---|---:|---:|---:|---:|---|
| prefill_best_observed | 1747.22 | 27.10 | 36.630 | 1549.807 | one launch, repeat3 |
| decode_best_observed | 1737.47 | 27.67 | 36.835 | 1517.921 | one launch, repeat3 |

- Prefill best observed: native source da3c0e00c6c2716666f8d1436ce1ff7a1751645e, archive binaries/l28-a2. W4 nibble lookup directly produces FP16 Crouton pairs, plus opt4 Q/V active-row extraction.
- Decode best observed/latest frozen source: cd12773a133134c7e7f2cba73bbe470e0988fe40, archive binaries/l28-a3. Adds QK active-row preparation and exact positive inverse SF-to-HF HVX conversion. Mode4 is the retained configuration; mode5 main-only expansion was not selected.
- Both use unchanged original W4 bytes/scales and the EXP0289 package. Full hidden/final-norm/KV payloads match the original implementation byte-for-byte across2451 files each. Complete selected-token IDs/logit half bits and physical/VTCM/ledger checks pass.
- The independent full-model floating-reference alignment failures from EXP0289 remain failed. Exact regression does not establish PPL, text quality, or independent whole-model arithmetic acceptance.
- No5short/10formal campaign was run. Small differences between these two repeat3 observations are not proven reproducible. The10-20percent-over-F16 goal was not reached in these observations.
- The original formal floating baseline remains1710.94 prefill/27.68 decode token/s, EXP0289 ten formal rounds. Its measurement has not been replaced by a slower rebuild. New shared-conversion/row-preparation F16 repeat3 was1691.20/27.51; this is retained as a diagnostic, not a replacement baseline.
- No artificial slowdown, sleeps, dummy work, lower clocks, or removal of useful float optimizations has been introduced. A separately labeled conservative implementation would be a different experimental control, not a substitute for the real fastest float baseline.

## Reproduction and provenance

ARCHIVE_MANIFEST.json records exact commands, package paths, immutable measured source commits, each binary SHA256 and both observed timing denominators. Device helper scripts/device_exp0291.py and measurement scripts/measure_exp0291.py are retained. Original runtime binaries and both candidates remain archived; results under exp0289/exp0290 are unchanged.
Full profiling report contains diagnostic means for all exported numeric counters; R1 timed/R10 formal sections are N/A. Audit-enabled Host times are excluded from E2E claims. HMX/HVX/DMA counters overlap and must not be added.

## Failed or unselected attempts

Standalone row extraction and main-only expansion did not achieve the target. The latter stays an explicit experimental option, not the retained launch mode. Initial direct-lookup build failed only because the obsolete helper became unused under -Werror; the old helper was guarded for non0.6 builds, without disabling compiler diagnostics. Original logs retained.
