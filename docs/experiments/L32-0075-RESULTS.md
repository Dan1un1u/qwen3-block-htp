# A-table W4A16 inference DRAM peak reruns (2026-09-25)

EXP-0314 / L32-0075. PJZ110 SM8750, Windows ADB port 5038. Each model launched once with its exact sealed historical A protocol: M64 prefill, 42 decode steps, cache capacity 128, repeat10. Runtime, all manifest package entries and fixed sequence SHA256 verified before execution. No runtime code changed. Existing spreadsheet throughput is not replaced by instrumented timing.

Metric: maximum observed post-model-ready sum of unique DMA-BUF backing allocation bytes (deduplicated by inode) and ordinary process RSS with DMA-BUF mappings subtracted. Covers resident weights, activation/work buffers, KV, head/embedding and attributable host process memory. Excludes cold loading, sampler, VTCM, hidden driver/DSP-private allocations and unmapped filesystem cache. This is a sampled process-attributable DRAM peak, not an exact instantaneous whole-device physical-memory maximum. Per-process smaps/fdinfo reads are not atomic. DMA-BUF allocator backing size is counted even where ordinary RSS reports zero. Cross-runtime memory definitions must be checked before direct comparison.

Units: decimal MB = bytes / 1,000,000; MiB retained below for clarity. Source raw samples, peak smaps/status, ready fdinfo/maps and dmabuf_dump are retained per model. Historical selected tokens and all non-timing eval_step fields match exactly across all 430 steps/model. This confirms reproduction, not a new model-quality assessment.

| Model | Peak bytes | Peak MB | Peak MiB | Inference samples | Median interval ms |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 650719232 | 650.72 | 620.57 | 474 | 33.53 |
| Llama-3.2-1B-Instruct | 1164480512 | 1164.48 | 1110.54 | 459 | 40.07 |
| Qwen3-1.7B | 1526743040 | 1526.74 | 1456.02 | 706 | 38.61 |
| Llama-3.2-3B-Instruct | 2437009408 | 2437.01 | 2324.11 | 1153 | 47.64 |

Spreadsheet: Paper_Hardware_Tables_ABC.xlsx / A_HellaSwag, cells F8/F13/F18/F23. Only these four numeric cells are updated. All throughput, other memory entries, B/C and styles are preserved.

Evidence ledger SHA256: `e1440d15a078f06f9ab8252ea31fe214befefb6aafd3bb3950c27942b1499c9b`. Workbook SHA256: `a6b2e6f20ad04b582221b70eceb482f1d6bf981ca4f1b8bca0e4cc78930fbcc6`.
