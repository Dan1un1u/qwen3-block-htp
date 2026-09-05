# EXP-0219: offline R1/R2 with per-channel W4

This completed experiment found a quality regression, not an eligible rotated
baseline. On frozen qbh-lite-v1 the actual DSP original W4F16 scores NLL 4.2317
and 8/24 tasks; identity gamma folding scores 5.6017 and 0/24; fixed Hadamard
R1/R2 scores 8.5625 and 0/24. No baseline is promoted. F16F16 and W4U8 remain
frozen. These 512 targets and 24 short tasks are a lightweight diagnostic,
not a general benchmark; the eight holdout windows were not evaluated.

## Provenance and quantization

The mathematical reference is
[mllm 2782f7ad](https://github.com/Dan1un1u/mllm/blob/2782f7ad9d1ee5e61f65d418cb89ae7217b5fb06/scripts/qwen3_block_rotation.py),
whose Layer-5 POC used LPBQ G32. Its checkpoint, already-folded weights,
group quantizer, QNN runtime, QDQ parameters and HWIO carriers are **not** reused.
Only the canonical OI rotation method is referenced. This is the fixed
Sylvester offline subset of the method discussed by
[SpinQuant](https://arxiv.org/abs/2405.16406), not learned SpinQuant or R3/R4.

Both B and C reload `/mnt/d/llm_exp/models/Qwen3-origin` as FP32 and recompute
all folds. The two original checkpoint shard hashes match the frozen EXP-0218
records. All 196 transformer projections plus the LM head are freshly quantized
using this project's existing `pack_w4_chunk`: one FP32 scale per output
channel, `scale=max(abs(row))/7`, round-to-nearest-even and clamp to [-7,7].
There are no group scales, scale1/scale2, or imported LPBQ tensor assets.
The embedding remains FP16. The previous W4 package supplies cache and metadata
scaffolding; all affected inference tensors are replaced atomically, preserving
hard-linked historical originals. Inherited replay references are historical
placeholders and must not be used to validate rotated layer replay.

## Transform and checks

R1 is normalized positive Sylvester H2048; R2 is shared H128 for all value
heads and corresponding GQA output heads. No random signs or seed search.
For row-vector hidden states, `x_rot=x R1`:

- Fold input RMSNorm gamma into Q/K/V and post-attention gamma into Gate/Up;
  then right-multiply these matrices by R1.
- Left-multiply each V head by R2 transpose; compensate with headwise R2 on
  the input axis of O. Also left-multiply O and Down by R1 transpose.
- Rotate embedding rows by R1. Fold final RMSNorm gamma into LM head before
  multiplying its input axis by R1. Set these residual/final norm gammas to one.
- Preserve Q/K head norm gamma and RoPE. V cache coordinates change, while
  physical layout, shape, cache lifecycle and K semantics remain unchanged.
- Detach Transformers' tied embedding/head before the final gamma fold; the
  existing deployment already stores independent FP16 embedding and W4 head.

The adapted Layer-5 transform is exactly equal to all seven upstream FP32
matrix results. An independently materialized dense Hadamard oracle passes
the FP64 1e-10 algebra bound. Unquantized full-model FP32 logits on 16 Chinese
and 16 English positions have NRMSE 1.52e-6 / 2.03e-6, cosine above
0.99999999999 and identical top-1 at all 32 positions. Every exported packed
W4 chunk reconstructs the quantizer's expected integer codes exactly.

Actual DSP A/B/C quick, full and repeat suites all pass physical checks and
score determinism, including agreement of overlapping quick/repeat/full
samples. Independent FP16 software diagnostics reconstructed from the same
packed codes/scales score B NLL 5.6092 and C 8.5744, both 0/24. These support
the negative quality conclusion but do not imply bit-exact software/DSP logits.
The precise cause of the increased composed quantization error is unresolved;
lower local weight RMSE is insufficient evidence of model-quality improvement.
The negative result is specific to these fixed rotations and this quantizer.

## Execution and reproduction

Use project-memory bootstrap and successful EXP-0219 preflight before stateful
work. Evidence lives at `/mnt/d/llm_exp/results/qwen3-block-htp/exp0219`; packages
at `/mnt/d/llm_exp/models/qwen3-block-htp/exp0219`. Export/run commands refuse
to overwrite retained outputs. To reproduce elsewhere, declare a new experiment
and new output paths; do not overwrite this record.

`rotation_exp0219.py B|C` exports and performs independent software diagnostics.
`measure_exp0219.py deploy --variant A|B|C` checks package hashes and deploys to
isolated directories. `quick`, `full`, and `repeat` use the frozen qbh-lite-v1
binary suites and unchanged EXP-0218 scoring. `warmup`, `short`, `formal` run
alternating A/C sessions, with one warmup, five short and ten formal rounds.
`summarize_exp0219.py` produces quality and complete profiling archives.

The exact EXP-0218 speed binaries from d9810725 are reused for both A and C;
ABI remains 108 and embedded runtime experiment label remains 218. The external
experiment and model provenance explicitly identify EXP-0219. There is no DSP
code change, online rotation, group quantization, or changed execution schedule.
The 320 formal invocations and 8,960 layer ledgers close exactly. All normal
generated tokens match each package's independently generated software sequence.
Exactly 8 MiB VTCM, one complete-model FastRPC per step, one HMX owner, no QNN
and zero timed intermediate hidden/logits DDR or spill are preserved.

Direct M64 prefill / 15-feedback-decode medians: original A 1017.924 / 10.821
token/s, rotated C 1015.548 / 10.801 token/s. Median paired speed changes are
-0.234% / -0.186%, reported without a speed-improvement claim. Full numeric
fields and the stable module table are retained in `full_profiling_report.md`.
Frozen F16F16 and W4U8 table columns are nonpaired historical references;
different W4 values prohibit attributing the ratio solely to activation width.


| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| BF16 teacher | 3.6420 | 38.17 | 22/24 | 100.00% |
| F16A16（冻结参照） | 3.6336 | 37.85 | 22/24 | 97.27% |
| W4A16 原始 | 4.2317 | 68.83 | 8/24 | 55.08% |
| W4A16 仅 gamma 折叠 | 5.6017 | 270.88 | 0/24 | 32.62% |
| W4A16 R1/R2 | 8.5625 | 5231.70 | 0/24 | 28.52% |


| 模块 | F16A16 冻结 EXP-0218 | W4A16 R1/R2 EXP-0219 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 377.5 (0.60%) | 247.4 (0.63%) | +52.59% |
| Input RMSNorm | 489.7 (0.61%) | 492.3 (0.78%) | 554.0 (1.40%) | -11.14% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11688.8 (18.55%) | 7052.7 (17.82%) | +65.74% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3935.5 (6.24%) | 3214.5 (8.12%) | +22.43% |
| O projection | 5757.7 (7.14%) | 5000.9 (7.94%) | 1256.4 (3.17%) | +298.03% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.4 (0.75%) | 654.0 (1.65%) | -27.61% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22415.0 (35.57%) | 14442.7 (36.49%) | +55.20% |
| Down | 13447.9 (16.67%) | 8555.2 (13.58%) | 3428.5 (8.66%) | +149.53% |
| Final residual | 140.1 (0.17%) | 140.1 (0.22%) | 183.8 (0.46%) | -23.80% |
| KV carrier conversion | 174.0 (0.22%) | 172.6 (0.27%) | 203.2 (0.51%) | -15.07% |
| KV append DMA | 343.6 (0.43%) | 336.5 (0.53%) | 463.9 (1.17%) | -27.46% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.20% |
| Layer bookkeeping | 23.9 (0.03%) | 23.3 (0.04%) | 23.2 (0.06%) | +0.45% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.4 (0.01%) | 22.5 (0.06%) | -62.50% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.5 (0.15%) | 105.6 (0.27%) | -11.47% |
| Embedding | 68.1 (0.08%) | 63.5 (0.10%) | 62.4 (0.16%) | +1.75% |
| Final model RMSNorm | 49.7 (0.06%) | 48.1 (0.08%) | 3.7 (0.01%) | +1191.61% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6727.1 (10.67%) | 5284.7 (13.35%) | +27.29% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2415.1 (3.83%) | 2466.6 (6.23%) | -2.09% |
| 完整 Host wall | 80692.2 (100.00%) | 63020.2 (100.00%) | 39575.9 (100.00%) | +59.24% |


## Direct E2E throughput

{
  "A": {
    "prefill_host_us": 62873.072499999995,
    "decode_host_us_per_token": 92414.22736666666,
    "prefill_tokens_per_second": 1017.9238496734831,
    "decode_tokens_per_second": 10.820844673973815
  },
  "C": {
    "prefill_host_us": 63020.182,
    "decode_host_us_per_token": 92587.28816666667,
    "prefill_tokens_per_second": 1015.5476859778031,
    "decode_tokens_per_second": 10.800618743686465
  }
}
