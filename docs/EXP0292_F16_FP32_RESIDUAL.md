# EXP0292 temporary FP32 residual result

Temporary FP32-residual W16A16; original FP16-residual implementation retained. Identical original weights, FP16 HMX O/Down outputs, KV16, head, OPT4 and pipeline settings; candidate Norm/residual handles only valid rows on decode, while the unchanged control retains its original full-M64 row work. This compares complete implementations, not isolated dtype arithmetic cost. Embedding widens to FP32; O/Down widen/add once in FP32; input/post/final RMSNorm read FP32 and produce FP16. This aligns residual storage/add precision with A8, not the A8 raw projection-output contract. No deliberate slowdown, extra delay, removed control optimization, PPL or quality claim. Original historical formal F16 remains1710.94/27.68; samebinary paired control below is separately measured.
Selected0/14/27 and chain3 M64/M1 pass unchanged wholeblock thresholds. Independent actual-operand residual additions exact; norm outputs <=1 FP16 ULP. Full-model software reference remains failed for BOTH arms; see measured maxima below. Do not call this full-model numerical/quality acceptance. Final norm and independent entire LMhead checks pass all43steps, argmax43/43. FP16 full audit matches2451 original files; FP32 repeated audit matches2451 files. Both KV streams finite, prefix preserved, valid length extends once, padding untouched. Timed paths have full8MiB grant, zero intermediateDDR/spill/output audit, oneRPC/pass, exact own token/logit repeats and complete additive ledgers.
Five short/ten balanced formal rounds, repeat10; repeat1 auxiliary. Complete Host wall includes embedding, all28blocks, finalnorm, head/greedy, FastRPC; excludes coldloading, tokenizer and audit I/O. Audit-enabled times are not speed evidence. HMX/HVX/DMA/worker counters overlap and cannot be added.

| 配置 | Prefill token/s | Decode token/s | 64-token Host ms | 42-step Host ms |
|---|---|---|---|---|
| fp16 | 1709.12 | 27.69 | 37.446 | 1516.971 |
| fp32 | 1690.59 | 28.14 | 37.857 | 1492.801 |

## Paired speed comparison

{
  "prefill": {
    "wall_ratio": 1.0109605861561304,
    "wall_overhead_percent": 1.096058615613038,
    "paired_bootstrap95_ratio": [
      1.0091633075439272,
      1.0124544807456175
    ],
    "throughput_change_percent": -1.0841754175407226
  },
  "decode": {
    "wall_ratio": 0.9840667598048936,
    "wall_overhead_percent": -1.5933240195106446,
    "paired_bootstrap95_ratio": [
      0.9830748133846683,
      0.9850743061043769
    ],
    "throughput_change_percent": 1.619121877286589
  }
}
## Full floating alignment, NOT passed

{
  "fp16": 0.00653234609135029,
  "fp32": 0.012067306152894927
}

Native flag: QBH_FP32_RESIDUAL=1 with F16F16, QBH_F16F16_OPT=4, Qwen0.6 build ABI137. Flag0 retains original. Packages: /mnt/d/llm_exp/models/qwen3-block-htp/exp0292/f16f16; exact commands and binary hashes in each protocol.json. Other recipes and original artifacts unchanged.

## Reproduction and implementation scope
Source branch: codex/exp-0292-qwen3-06b-f16-fp32-residual.
Measured native source: e2d63b9b9a52e653adde35773adaa511267dfa1d.
Full runtime seal and immutable binaries: runtime-l28.json and binaries/l28-a1.
No default baseline promotion. Explicit temporary FP32 flag1 only supports Qwen0.6 F16F16, vector/fused pooled norms and the declared current fast schedule. Flag0 preserves original.
The existing optimized QK row handling, KV layout, MLP DMA/HMX schedule and FP16 head prefetch remain available in both paired arms. The new FP32 norm/residual path handles logical rows rather than computing unused M64 decode rows; its decode benefit is not evidence FP32 arithmetic is intrinsically faster.
311 weight/norm/embedding/head payload hashes equal the original package. New references are independently generated, no weight requantization. Original whole-model floating mismatch is not hidden or relabeled; maxima differ under the changed residual contract and are not a PPL metric.
