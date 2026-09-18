# L32-0048: single-session delayed mapping feasibility

## Conclusion
PASS on the current SM8750/PJZ110 V79 device (serial 3B15C8007Z300000), SDK 6.6.
A single DSP session successfully accesses more than 4 GiB of fully touched resident rpcmem buffers through bounded mapping windows. Maximum tested data allocation is 6.73388671875 GiB. There is no observed driver blocker requiring multiple sessions in this bounded test. This does not make DSP addresses 64-bit and does not automatically repair the existing monolithic uint32 weight-offset ABI.

No model export, inference kernels, quantization recipe, PPL, text quality or E2E baseline was changed or measured. This is a synthetic mapping/DMA probe; complete FP16 model support remains pending.

## Fixed campaign and results
Measured source: 4a5fbbfda4dc14ea4b3937e711b01922bc868c59.
Parent inference source: 38b001a6127d7083d7d20f0b552ba23f20fde4ec.
Sealed binaries: runtime-a02/seal.json. Frozen plan: fixed-final-plan.json.
One small control: 3 x 16 MiB, two windows, four sweeps / twelve mappings.
Fifteen independent large-buffer processes: five per configuration, twenty alternating forward/reverse sweeps per process. All fifteen processes and 2,200 large mappings pass; small control also passes. This is not a five-short/ten-formal E2E campaign.

All data buffers are allocated and memset-touched before registration and remain allocated throughout the one top-level probe RPC. Each movable buffer is 768 MiB. The pinned control, when present, is 751.5 MiB, equivalent in size to one FP16 128256 x 3072 vocabulary layout. The largest arm provides 6 GiB movable plus 751.5 MiB pinned, slightly exceeding the 5.25 GiB backbone plus two vocabulary layouts (6.717773 GiB), but it does not emulate all KV/runtime allocations or the full inference address-space layout.

| Configuration | Allocated data GiB | Moving windows | Mappings | Mean map us | Mean unmap us | Map + unmap per sweep ms |
|---|---:|---:|---:|---:|---:|---:|
| single-final | 5.250000 | 1 | 700 | 34.701 | 65.294 | 0.699967 |
| double-pinned-final | 5.983887 | 2 | 700 | 29.787 | 56.148 | 0.601545 |
| fullsize-final | 6.733887 | 2 | 800 | 30.137 | 56.998 | 0.697083 |

The large configurations reuse only two/six/six distinct DSP virtual base addresses per process respectively, rather than consuming a fresh VA region on every map. All map/unmap returns, DSP sample mismatches, retained-window checks, host readback mismatches, and host/DSP cleanup errors are zero. Each complete data window fits within the 32-bit address range. The small probe control and existing runtime metadata are additional to the reported data-buffer total.

Costs use the SDK HAP_perf.h definition of a 19.2 MHz QTimer. Map/unmap costs include initial mapping and final draining, averaged over twenty sweeps and five processes. Per-process sweep-cost ranges: single 0.698443–0.702271 ms; double+pinned 0.600724–0.602279 ms; full-size 0.696328–0.698479 ms. These are descriptive component measurements, not a demonstrated throughput benefit from two windows.

Full probe Host wall is approximately 150.17/149.41/170.80 ms per twenty-sweep process. This includes deliberate scalar diagnostic comparisons, DMA and control overhead, and must not be presented as inference latency. Allocation/touch and registration are separately retained in SUMMARY.json; session-open/prepare are outside probe Host wall. The isolated mapping costs cannot simply be subtracted from or converted to model TPS. Actual inference must include its actual remapping, synchronization and DMA costs.

## Mechanism and correctness checks
- Host registers weight-equivalent buffers with FASTRPC_MAP_FD_DELAYED, avoiding simultaneous permanent DSP mapping of the full allocation.
- DSP uses HAP_mmap2/HAP_munmap2 within one RPC, after leasing the existing 8 MiB VTCM. There is no HMX compute in this probe.
- Before releasing a window, all accesses/DMA are drained; alternate forward/reverse sweeps exercise mapping reuse, including repeated mappings of the same buffer at sweep boundaries.
- Each buffer has 33 evenly distributed 4 KiB sample pages, including first/middle/last. Each is DMA-read into VTCM; the first 2 KiB is an immutable per-buffer/per-page sentinel. The second 2 KiB receives an epoch-specific acknowledgement via DMA writeback.
- Later sweeps verify the previous epoch; host independently validates both halves of every sample after the RPC. The pinned buffer is also checked by host at one byte per 4 KiB page.
- After every new map, DSP rechecks the retained other window and three pinned locations through DMA. Distinct simultaneously mapped data buffers must not overlap.
- Checks are distributed sampled checks, not exhaustive read/write validation of every allocated byte. All allocated bytes are initially touched.

## Preserved failures and evidence
control-a01 failed at the Android dynamic linker before any DSP run because /vendor/lib64 in LD_LIBRARY_PATH selected incompatible libraries. Correcting the launcher to the established remote runtime directory fixed it. This is not a mapping failure. Its command/stdout/stderr/exit are retained.
Earlier successful control/single/double runs and both build logs/binary seals are preserved separately; final conclusions above use only the final fixed plan and strengthened source version.

Audit: python3 tools/report_llama32_mapping_probe.py /mnt/d/llm_exp/results/llama32-htp/l32-0048.
The auditor verifies exact frozen-plan commands, binary hashes, allocation and registration count, alternating buffer order, check counts, map/unmap results, address bounds, retained checks, completion and cleanup. SUMMARY.json is generated directly from archived JSONL. An external EVIDENCE_LEDGER.json records hashes of archived files.

## Next integration, not executed
Use segmented resident FP16 weights identified by buffer ID plus uint32 offset within that buffer. Resolve a segment into an active DSP base before using the existing generic FP16 HMX/HVX/DMA kernels. Keep activation/KV/control memory in bounded permanent mappings and drain before recycling a weight window. A generic one-window path is sufficient to establish correctness; use a second window only where useful, without deep specialization.
Preserve model-level RPC granularity rather than adding a host RPC per layer. Export fresh raw FP16 weights into bounded shards, then verify selected blocks and the complete 28-block sequence before matched E2E. The FP16 vocabulary/head layouts, KV allocations, complete address-space budget and longer inference lifetime still need integration validation. No need to attempt multi-session now; it remains a fallback only if actual model integration reveals a further limit.

## Upstream reference
The mechanism follows upstream llama.cpp's documented large-model handling and V79 HAP mapping implementation:
- https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#large-model-handling
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c

This experiment establishes behavior of our current device/software stack, not all Qualcomm devices or all historical FastRPC mapping APIs.
