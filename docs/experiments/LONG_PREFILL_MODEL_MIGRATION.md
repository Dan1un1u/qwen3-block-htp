# Long-prefill migration across remaining models

Implemented and verified L32-0051-style native-layout/parallel-GQA long prefill for Llama3.2-3B and both Qwen3 sizes. W4A8-SP2/no rotation/FP32 residual and frozen model-specific arithmetic retained. Correctness passes; inherited throughput parity is a separate gate.

| Model | Prefill+decode steps | Prefill token/s | Decode token/s | Long/M64 prefill | Parity |
|---|---|---:|---:|---:|---|
| Llama-3.2-3B-Instruct | 536+46 | 1103.85 | 16.28 | 0.945x | FAIL |
| Llama-3.2-3B-Instruct | 741+3 | 1106.97 | 14.75 | 0.948x | FAIL |
| Qwen3-0.6B | 536+46 | 2486.77 | 24.84 | 0.846x | FAIL |
| Qwen3-0.6B | 741+3 | 2410.92 | 21.11 | 0.820x | FAIL |
| Qwen3-1.7B | 536+46 | 1693.52 | 19.95 | 0.925x | FAIL |
| Qwen3-1.7B | 741+3 | 1677.19 | 17.51 | 0.917x | FAIL |

Each model: six final candidate arithmetic audits at64/65/128/129/536+46/741+3, five short rounds, ten formally rotated paired rounds × repeat10. Qwen also verifies sequential control741; Llama closure adds an equivalent-binary64+3 audit. Exact8MiB VTCM, no timed intermediateDDR/spill, all44,000 formal RPC ledger/head/physical checks pass.

M64 reference: Llama long-frontend64+3/capacity832; Qwen faster of retained native segmented-cache64+3 and long-row-major64+3. These references are explicitly different; no global-best historical M64 claim for Llama. All long TPS include host input/RoPE staging. Native short control retains prior boundary, conservatively favoring the control. Fixed project-owned/teacher-forced trajectories; named-dataset speed and PPL/text quality remain unmeasured. Decode step count excludes the prefill-produced first token.

The migration accelerates long prefill but does not establish parity for every model. Longer-context attention, row preparation and chunk tails remain visible. One-row long-cache decode still uses sequential dynamic attention/full-prefix preparation and does not inherit the native short-cache efficiency. Long frontend is bounded to prompt<=768/decode<=63/cache832. A16 and ordinary A8 long paths are outside this migration.

No paper baseline or quality status promoted. Historical Llama1B L32-0051 result is unchanged. See Llama l32-0052/RESULTS.md and MODULES.md; Qwen size-specific RESULTS.md/MODULES.md; protocol, failed candidates and all raw evidence remain alongside them.
