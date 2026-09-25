# A-table W4A8-INT16-Down inference DRAM peak reruns (2026-09-25)

EXP-0315 / L32-0076. PJZ110 SM8750, Windows ADB port 5038. Each model launched once with its exact sealed historical A protocol: M64 prefill, 42 decode steps, cache capacity 128, repeat10. Runtime, all manifest package entries and fixed sequence SHA256 verified before execution. No runtime code changed. Existing spreadsheet throughput is not replaced by instrumented timing.

Metric: maximum observed post-model-ready sum of unique DMA-BUF backing allocation bytes (deduplicated by inode) and ordinary process RSS with DMA-BUF mappings subtracted. Covers resident weights, activation/work buffers, KV, head/embedding and attributable host process memory. Excludes cold loading, sampler, VTCM, hidden driver/DSP-private allocations and unmapped filesystem cache. This is a sampled process-attributable DRAM peak, not an exact instantaneous whole-device physical-memory maximum. Per-process smaps/fdinfo reads are not atomic. DMA-BUF allocator backing size is counted even where ordinary RSS reports zero. Cross-runtime memory definitions must be checked before direct comparison.

Units: decimal MB = bytes / 1,000,000; MiB retained below for clarity. Source raw samples, peak smaps/status, ready fdinfo/maps and dmabuf_dump are retained per model. Historical selected tokens and all non-timing eval_step fields match exactly across all 430 steps/model. This confirms reproduction, not a new model-quality assessment.

| Model | Peak bytes | Peak MB | Peak MiB | Inference samples | Median interval ms |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1076690944 | 1076.69 | 1026.81 | 137 | 35.38 |
| Llama-3.2-1B-Instruct | 2190331904 | 2190.33 | 2088.86 | 238 | 40.08 |
| Qwen3-1.7B | 2915618816 | 2915.62 | 2780.55 | 224 | 41.99 |
| Llama-3.2-3B-Instruct | 4040740864 | 4040.74 | 3853.55 | 392 | 47.84 |

Spreadsheet: Paper_Hardware_Tables_ABC.xlsx / A_HellaSwag, cells F9/F14/F19/F24. Only these four numeric cells are updated. All throughput, other memory entries, B/C and styles are preserved.

Evidence ledger SHA256: `fb9ff176cad4dd212a34c0b43b5ff019cb61a28c92258fe6a65679bf2e82973d`. Workbook SHA256: `7d1c098d5696d5baa23d5f65975ef55a11b82d1832786301fcdd4e89a75062b9`.
