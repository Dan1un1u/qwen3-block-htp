# EXP0293 matched full-M64 floating residual result

Temporary FP32-residual W16A16; original FP16-residual implementation retained. Identical original weights, FP16 HMX O/Down outputs, KV16, head, OPT4 and pipeline settings; both arms retain full-M64 input/post Norm and final residual work on decode. Existing HVX arithmetic and basic pipeline remain, without FP32-only valid-row specialization. Final head Norm already used identical row selection and stays unchanged. Both implementations differ in required residual dtype arithmetic; the task/row budgets now match. Embedding widens to FP32; O/Down widen/add once in FP32; input/post/final RMSNorm read FP32 and produce FP16. This aligns residual storage/add precision with A8, not the A8 raw projection-output contract. No deliberate slowdown, extra delay, removed control optimization, PPL or quality claim. Original historical formal F16 remains1710.94/27.68; samebinary paired control below is separately measured.
Selected0/14/27 and chain3 M64/M1 pass unchanged wholeblock thresholds. Independent actual-operand residual additions exact; norm outputs <=1 FP16 ULP. Full-model software reference remains failed for BOTH arms; see measured maxima below. Do not call this full-model numerical/quality acceptance. Final norm and independent entire LMhead checks pass all43steps, argmax43/43. Each arm full audit matches2451 frozen EXP0292 files, including all43 final hidden/norm boundaries and4816 combined KV snapshots. These are original-versus-current checks, not two newly collected candidate audits. Independent head proof is inherited through byte-identical actual operands and ID/logit outputs. Both KV streams finite, prefix preserved, valid length extends once, padding untouched. Timed paths have full8MiB grant, zero intermediateDDR/spill/output audit, oneRPC/pass, exact own token/logit repeats and complete additive ledgers.
Five short/ten balanced formal rounds, repeat10; repeat1 auxiliary. Complete Host wall includes embedding, all28blocks, finalnorm, head/greedy, FastRPC; excludes coldloading, tokenizer and audit I/O. Audit-enabled times are not speed evidence. HMX/HVX/DMA/worker counters overlap and cannot be added.

| 配置 | Prefill token/s | Decode token/s | 64-token Host ms | 42-step Host ms |
|---|---|---|---|---|
| fp16 | 1713.05 | 27.73 | 37.360 | 1514.490 |
| fp32 | 1687.09 | 27.38 | 37.935 | 1533.811 |

## Paired speed comparison

{
  "prefill": {
    "wall_ratio": 1.015385583665615,
    "wall_overhead_percent": 1.5385583665614933,
    "paired_bootstrap95_ratio": [
      1.013240910117915,
      1.0175147218473832
    ],
    "throughput_change_percent": -1.515245431205725
  },
  "decode": {
    "wall_ratio": 1.0127572529277593,
    "wall_overhead_percent": 1.275725292775931,
    "paired_bootstrap95_ratio": [
      1.012051792449153,
      1.0134590745146868
    ],
    "throughput_change_percent": -1.2596555483438632
  }
}
## Full floating alignment, NOT passed

{
  "fp16": 0.00653234609135029,
  "fp32": 0.012067306152894927
}

Native flag: QBH_FP32_RESIDUAL=1 with F16F16, QBH_F16F16_OPT=4, Qwen0.6 build ABI137. Flag0 retains original. Packages: /mnt/d/llm_exp/models/qwen3-block-htp/exp0292/f16f16; exact commands and binary hashes in each protocol.json. Other recipes and original artifacts unchanged.
