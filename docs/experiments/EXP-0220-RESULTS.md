# EXP-0220: gamma folding attribution and R2-only

Gamma folding is equivalent in unquantized algebra, while requantizing the folded weights is not equivalent to folding an already quantized operator. This completed quality study finds no aggregate quality loss from pure FP16 folding, but all three W4 folding-location ablations regress. QKV-only has the largest isolated NLL regression. Retaining gamma and using R2-only also regresses; fixed H128 R2 with this quantizer is not an eligible replacement. These findings do not reject other rotations or quantizers.

## Scope and provenance

Only W4F16 package transforms change. The frozen F16F16 and W4U8 device recipes are not executed or modified. Host FP16 original/folded inference is a separate precision diagnostic. Every changed tensor is freshly generated from verified Qwen3-origin checkpoint shards; the upstream mllm folded weights, LPBQ groups and QNN implementation are not inputs. Unchanged W4 tensors are retained byte-for-byte from the frozen package. Quantization remains one FP32 scale per output row, symmetric [-7,7] RTN. There is no calibration, group scale, seed search, learned rotation, online rotation or new DSP specialization.

- QKV-only: all 28 input RMSNorm gammas folded into Q/K/V, corresponding norms set to one.
- Gate/Up-only: all 28 post-attention RMSNorm gammas folded into Gate/Up, corresponding norms set to one.
- Head-only: final RMSNorm gamma folded into LM head, final gamma set to one.
- R2-only: normalized positive Sylvester H128 transforms each V output head and compensates each corresponding O input head; all gammas, Q/K, RoPE, embedding and head remain unchanged. R1 is disabled.

The embedding and LM head are explicitly detached before weight transforms, reflecting the independent deployed FP16 embedding and W4 head. R2 changes V-cache coordinates but preserves shape, ownership and append semantics. Changed packages contain 197, 141, 4 and 113 files respectively, including independent speed token references. Inherited layer-replay references are historical placeholders and cannot certify new layer replay.

## Frozen lightweight quality

All four variants complete qbh-lite-v1 quick, full and repeat on the actual DSP: 512 bilingual NLL targets, 24 strict short tasks and four open prefixes. Every overlapping token, target code, NLL, rank, tie and saturation field matches; repeated scores are identical. Eight holdout windows remain unused. Original W4 and full-gamma-fold rows reuse frozen EXP-0219 quality, not a fresh paired run. This small repeated diagnostic is not a broad capability benchmark.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| FP16 原始（Host） | 3.6344 | 37.88 | 22/24 | 97.46% |
| FP16 全折叠（Host） | 3.6342 | 37.87 | 22/24 | 97.27% |
| W4A16 原始（DSP，冻结） | 4.2317 | 68.83 | 8/24 | 55.08% |
| W4A16 全折叠（DSP，冻结） | 5.6017 | 270.88 | 0/24 | 32.62% |
| W4A16 仅 QKV 折叠 | 5.0649 | 158.37 | 0/24 | 35.35% |
| W4A16 仅 Gate/Up 折叠 | 4.5174 | 91.59 | 0/24 | 53.12% |
| W4A16 仅最终 norm/head 折叠 | 4.4179 | 82.92 | 3/24 | 51.95% |
| W4A16 R2-only，保留 gamma | 5.6464 | 283.28 | 0/24 | 32.42% |


Host R2 software reconstructed from the actual packed codes/scales scores NLL 5.649252 and 0/24, supporting the DSP quality regression without claiming bit-exact software/DSP logits. R2 effectiveness gate (lower NLL AND more correct tasks than original W4) fails; there is no baseline promotion.

## Attribution

For column-vector normalized input u, original output is W D_gamma u. Replacing it by (W D_gamma) u is an exact algebraic identity. But Q(W) D_gamma generally differs from Q(W D_gamma). Gamma scales input columns while this W4 scheme shares one scale across an entire output row, so folding can change the row maximum and integer rounding for every input coordinate.

Pure FP16 original versus folded NLL is 3.6343902312 versus 3.6342090815, both 22/24. The two probe-logit NRMSE values are 0.002885576 and 0.003615400; top-1 agrees at 31 of 32 positions and all values are finite. Maximum layer-hidden NRMSE is 0.003514159. This was predeclared as a report-only FP16 precision control, not exact equality or application of the separate FP32 composition tolerance.

On the same actual incoming hidden states at samples 0 and 20, last 16 positions, average conditional projection NRMSE is:

| Projection | Original W4 | Folded W4 | Folded FP16 |
|---|---:|---:|---:|
| q | 12.2758% | 18.2449% | 0.0293% |
| k | 12.7169% | 19.1702% | 0.0308% |
| v | 19.6631% | 22.6455% | 0.0357% |
| gate | 9.9419% | 12.5941% | 0.0299% |
| up | 16.6271% | 18.5641% | 0.0352% |
| head | 14.8782% | 21.0640% | 0.0401% |

These are untimed host projection diagnostics averaged across 28 layers and two prompts (head: two prompts), not DSP activation captures. The original/gamma-folded operators use the same real norm input and preserve the FP16 cast order. Together with the FP16 quality control and W4 location ablations, the evidence supports requantization as the main source of gamma-fold regression. It does not establish additive contributions or explain every R2 error by weight RMSE.

Unquantized FP32 full-model checks for all four transforms pass the predeclared cosine >=0.99999 / NRMSE <=0.003 gate. R2 probe NRMSE is 9.61e-7 / 1.51e-6 and all 32 top-1 positions match. Every changed packed W4 chunk round-trips to the independent quantizer codes. The R2-only negative result therefore remains after removing gamma folding and checking the unquantized transform; its quantized V/O errors can still alter attention output and accumulate across layers. Establishing their precise distribution is a follow-up, not a result of this run.

If pursuing full R1 next, the supported direction is an input-error-aware per-channel quantizer such as controlled clipping/GPTQ, with separately declared calibration data and a fixed protocol. Do not repeat the same fixed Hadamard transform, introduce LPBQ silently, or tune on qbh-lite-v1/holdout outcomes.

## Runtime and retained evidence

Same frozen EXP-0218 ABI108 speed binary from d981072513d06ed61731c14743c76ac6bc81617f for original/R2, outer experiment EXP-0220. Full 28-layer token-in/token-out execution, embedding, final norm, head, greedy selection and persistent KV remain unchanged. Exactly 8 MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC and one HMX owner, no QNN.

Android tar initially truncated long PAX symlink targets. The failed original deployment/archive/checksum are preserved with attempt01 names and deployment_recovery.json. Short relative links fixed packaging; all 1276 files and frozen runtime binaries pass hashes on every successful deployment. No old hash or numerical gate was changed.

Evidence: /mnt/d/llm_exp/results/qwen3-block-htp/exp0220. Packages: /mnt/d/llm_exp/models/qwen3-block-htp/exp0220. Scripts experiment_exp0220.py, measure_exp0220.py and summarize_exp0220.py refuse to overwrite retained results. Reproduction requires a newly declared experiment and isolated paths. Full source/runtime/manifests and dependency hashes are retained at closure.

## Completed profiling

Original/R2 each pass one warmup, five alternating short rounds and ten alternating formal rounds. All 320 formal invocation ledgers and 8,960 layer ledgers close exactly with zero unattributed ticks. Every selected token agrees with its independently generated software sequence, including the candidate's first EOS. This fixed 16-step benchmark deliberately continues after EOS; R2 speed is a diagnostic of full-model execution, not usable generated-text throughput. Host wall covers the complete model token-in/token-out pass, excluding cold staging and separate WSL tokenizer/detokenizer work (retained in execution JSON).

Measured script/source state: 1bebf5dcc47e5304dfe93b69cd9f4187fae3dfca on codex/exp-0220-gamma-fold-attribution-r2-only. Frozen runtime source d981072513d06ed61731c14743c76ac6bc81617f. Full numeric original/R2 repeat-one and repeat-ten tables and signed changes are in full_profiling_report.md. Other recipe columns below are frozen EXP-0218 nonpaired references. R2 quality failed; this is not a promoted Selected Baseline or an activation-only comparison.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 R2-only EXP-0220 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 376.3 (0.60%) | 247.4 (0.63%) | +52.12% |
| Input RMSNorm | 489.7 (0.61%) | 492.1 (0.78%) | 554.0 (1.40%) | -11.17% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11716.4 (18.61%) | 7052.7 (17.82%) | +66.13% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3962.8 (6.30%) | 3214.5 (8.12%) | +23.28% |
| O projection | 5757.7 (7.14%) | 5019.1 (7.97%) | 1256.4 (3.17%) | +299.48% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.9 (0.75%) | 654.0 (1.65%) | -27.54% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22381.2 (35.56%) | 14442.7 (36.49%) | +54.97% |
| Down | 13447.9 (16.67%) | 8559.6 (13.60%) | 3428.5 (8.66%) | +149.66% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.85% |
| KV carrier conversion | 174.0 (0.22%) | 172.0 (0.27%) | 203.2 (0.51%) | -15.35% |
| KV append DMA | 343.6 (0.43%) | 338.8 (0.54%) | 463.9 (1.17%) | -26.96% |
| Block orchestration | 16.1 (0.02%) | 19.0 (0.03%) | 34.6 (0.09%) | -44.95% |
| Layer bookkeeping | 23.9 (0.03%) | 23.9 (0.04%) | 23.2 (0.06%) | +3.37% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.4 (0.01%) | 22.5 (0.06%) | -62.73% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 95.1 (0.15%) | 105.6 (0.27%) | -9.96% |
| Embedding | 68.1 (0.08%) | 67.6 (0.11%) | 62.4 (0.16%) | +8.26% |
| Final model RMSNorm | 49.7 (0.06%) | 48.2 (0.08%) | 3.7 (0.01%) | +1194.41% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6735.4 (10.70%) | 5284.7 (13.35%) | +27.45% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2334.1 (3.71%) | 2466.6 (6.23%) | -5.37% |
| 完整 Host wall | 80692.2 (100.00%) | 62947.1 (100.00%) | 39575.9 (100.00%) | +59.05% |


Direct E2E medians and paired speed changes:

```json
{
  "times": {
    "original": {
      "prefill_tokens": 64,
      "prefill_host_us": 62953.151,
      "prefill_tokens_per_second": 1016.6290167111731,
      "decode_tokens": 15,
      "decode_total_host_us": 1391449.112,
      "decode_tokens_per_second": 10.780128335731764
    },
    "r2_only": {
      "prefill_tokens": 64,
      "prefill_host_us": 62947.057,
      "prefill_tokens_per_second": 1016.7274381072335,
      "decode_tokens": 15,
      "decode_total_host_us": 1387858.9324999999,
      "decode_tokens_per_second": 10.808014884466655
    }
  },
  "paired_speed_percent": {
    "prefill": -0.011636250434343687,
    "decode": 0.29325381936629036
  }
}
```

Sub-percent timing changes are reported without a performance-improvement claim. No baseline is promoted.
