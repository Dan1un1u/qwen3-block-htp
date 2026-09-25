# A-table W16A16 inference DRAM peak reruns (2026-09-25)

EXP-0313 / L32-0074. PJZ110 SM8750, Windows ADB port 5038. Each model launched once with its exact sealed historical A protocol: M64 prefill, 42 decode steps, cache capacity 128, repeat10. Runtime, all manifest package entries and fixed sequence SHA256 verified before execution. No runtime code changed. Existing spreadsheet throughput is not replaced by instrumented timing.

Metric: maximum observed post-model-ready sum of unique DMA-BUF backing allocation bytes (deduplicated by inode) and ordinary process RSS with DMA-BUF mappings subtracted. Covers resident weights, activation/work buffers, KV, head/embedding and attributable host process memory. Excludes cold loading, sampler, VTCM, hidden driver/DSP-private allocations and unmapped filesystem cache. This is a sampled process-attributable DRAM peak, not an exact instantaneous whole-device physical-memory maximum. Per-process smaps/fdinfo reads are not atomic. DMA-BUF allocator backing size is counted even where ordinary RSS reports zero. Cross-runtime memory definitions must be checked before direct comparison.

Units: decimal MB = bytes / 1,000,000; MiB retained below for clarity. Source raw samples, peak smaps/status, ready fdinfo/maps and dmabuf_dump are retained per model. Historical selected tokens and all non-timing eval_step fields match exactly across all 430 steps/model. This confirms reproduction, not a new model-quality assessment.

| Model | Peak bytes | Peak MB | Peak MiB | Inference samples | Median interval ms |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1541705728 | 1541.71 | 1470.29 | 413 | 37.84 |
| Llama-3.2-1B-Instruct | 3013881856 | 3013.88 | 2874.26 | 535 | 44.48 |
| Qwen3-1.7B | 4101902336 | 4101.90 | 3911.88 | 708 | 47.97 |
| Llama-3.2-3B-Instruct | 7252299776 | 7252.30 | 6916.33 | 1097 | 55.98 |

Spreadsheet: Paper_Hardware_Tables_ABC.xlsx / A_HellaSwag, cells F7/F12/F17/F22. Only these four numeric cells are updated. All throughput, other memory entries, B/C and styles are preserved.

Evidence ledger SHA256: `3a341997f45ffabd38e02a071026e964d153d8e4fc4673f1adae6aaee0cd2f9b`. Workbook SHA256: `3725e5b055b06c0264076cb58db782852fe8ef50b25300cdf64f1c7b7cd47aa3`.
