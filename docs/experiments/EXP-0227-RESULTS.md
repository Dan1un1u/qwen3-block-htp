# EXP0227 fixed-rotation final-format block reconstruction

A0=EXP0224 A; R0=EXP0225 step100. A/R learn only transformer row scales; GPTQ codes, rotations, norms, embedding and head stay fixed. Frozen8192 calibration and independent64-window Wiki/news validation; qbh/holdout never selects parameters. This is an OmniQuant-inspired scale-only block reconstruction adaptation, not full OmniQuant.

Both coordinates reduce qbh conditional PPL but each loses one strict short task; independent validation NLL does not improve. The rotated candidate remains worse in PPL than the reconstructed original-coordinate candidate, with one more correct task. This bounded scale-only intervention does not establish consistent quality recovery or incremental rotation benefit. No baseline is promoted. Training/export GPU/CPU weights are exactly equal, so this result is not explained by a second quantization pass changing trained weights.

## Independent actual-package validation

| Variant | NLL | PPL | English NLL | Chinese NLL |

|---|---:|---:|---:|---:|

| A0 | 3.541112 | 34.5053 | 3.380727 | 3.701497 |

| R0 | 3.561694 | 35.2228 | 3.399672 | 3.723716 |

| A | 3.548497 | 34.7610 | 3.381949 | 3.715046 |

| R | 3.562357 | 35.2462 | 3.387347 | 3.737366 |

## Frozen DSP qbh-lite-v1

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| A0 EXP0224（冻结） | 3.6603 | 38.87 | 19/24 | 66.80% |
| R0 EXP0225 step100（冻结） | 3.9165 | 50.22 | 20/24 | 66.02% |
| A 原坐标 block 重建 | 3.6548 | 38.66 | 18/24 | 69.92% |
| R 固定学习旋转 block 重建 | 3.7616 | 43.02 | 19/24 | 68.55% |


These are512 conditional targets and24 strict short tasks, not a broad capability certification. Software metrics are retained separately. PPL is conditional on the fixed short benchmark. Both content and format failures count under unchanged scoring.

## Reconstruction diagnostics

| Coordinate | selected0/50/100 blocks | final train relative MSE | final validation relative MSE | elapsed seconds |

|---|---|---:|---:|---:|

| A | [1, 8, 19] | 0.028418977 | 0.10756134 | 137.07 |

| R | [0, 8, 20] | 0.013659898 | 0.071010046 | 181.19 |

Each local selection compares checkpoints on the same incoming student stream at that block. Improvements are not additive causal contributions. Full-model quality determines whether reconstruction transfers.

Numerical recovery: unscaled FP16 backward erased Q/K/Gate/Up scale gradients in a real block3 audit. Original attempts are preserved under training_unscaled/smoke_unscaled. Final runs use standard GradScaler65536, unscale before Adam/gradient clipping, with5600 finite successful updates and no overflow retries. The forward, objective, data, budget and thresholds are unchanged. See gradient_precision_audit.json, training_audit.json and recovery_loss_scaling.json.

## Effectiveness

```json

{
  "per_coordinate": {
    "A": false,
    "R": false
  },
  "incremental_rotation": false,
  "baseline_promoted": false
}

```

## Complete profiling

| 模块 | F16A16 冻结 EXP-0218 | W4A16 block scale reconstruction A EXP-0227 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 380.9 (0.60%) | 247.4 (0.63%) | +53.95% |
| Input RMSNorm | 489.7 (0.61%) | 492.4 (0.78%) | 554.0 (1.40%) | -11.11% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11731.7 (18.61%) | 7052.7 (17.82%) | +66.34% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3959.1 (6.28%) | 3214.5 (8.12%) | +23.16% |
| O projection | 5757.7 (7.14%) | 5046.3 (8.01%) | 1256.4 (3.17%) | +301.64% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.4 (0.75%) | 654.0 (1.65%) | -27.62% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22464.7 (35.64%) | 14442.7 (36.49%) | +55.54% |
| Down | 13447.9 (16.67%) | 8653.0 (13.73%) | 3428.5 (8.66%) | +152.39% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.83% |
| KV carrier conversion | 174.0 (0.22%) | 172.5 (0.27%) | 203.2 (0.51%) | -15.13% |
| KV append DMA | 343.6 (0.43%) | 341.9 (0.54%) | 463.9 (1.17%) | -26.29% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.35% |
| Layer bookkeeping | 23.9 (0.03%) | 22.9 (0.04%) | 23.2 (0.06%) | -1.01% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.0 (0.01%) | 22.5 (0.06%) | -60.19% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.2 (0.15%) | 105.6 (0.27%) | -11.76% |
| Embedding | 68.1 (0.08%) | 67.0 (0.11%) | 62.4 (0.16%) | +7.34% |
| Final model RMSNorm | 49.7 (0.06%) | 47.8 (0.08%) | 3.7 (0.01%) | +1182.52% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6630.5 (10.52%) | 5284.7 (13.35%) | +25.47% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2312.0 (3.67%) | 2466.6 (6.23%) | -6.27% |
| 完整 Host wall | 80692.2 (100.00%) | 63034.1 (100.00%) | 39575.9 (100.00%) | +59.27% |


The F16A16 and W4A8 columns are frozen nonpaired EXP0218 references. Current A0/R0/A/R use1warmup5short10four-way rotating formal rounds with identical ABI108 runtime. Full additive and overlapping counters are retained in full_profiling_report.md.

## Direct E2E throughput

```json

{
  "A0": {
    "prefill_tokens": 64,
    "prefill_host_us": 62962.8645,
    "prefill_tokens_per_second": 1016.4721778184028,
    "decode_tokens": 15,
    "decode_total_host_us": 1388082.8635,
    "decode_tokens_per_second": 10.806271292895332
  },
  "R0": {
    "prefill_tokens": 64,
    "prefill_host_us": 63048.4115,
    "prefill_tokens_per_second": 1015.0929813671831,
    "decode_tokens": 15,
    "decode_total_host_us": 1387413.098,
    "decode_tokens_per_second": 10.811487956703722
  },
  "A": {
    "prefill_tokens": 64,
    "prefill_host_us": 63034.0885,
    "prefill_tokens_per_second": 1015.3236371459548,
    "decode_tokens": 15,
    "decode_total_host_us": 1387262.447,
    "decode_tokens_per_second": 10.812662039859859
  },
  "R": {
    "prefill_tokens": 64,
    "prefill_host_us": 63159.4795,
    "prefill_tokens_per_second": 1013.3079073268804,
    "decode_tokens": 15,
    "decode_total_host_us": 1388433.0724999998,
    "decode_tokens_per_second": 10.803545591859994
  }
}

```
