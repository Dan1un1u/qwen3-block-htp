# EXP0226 sampling ablation results

Training uses the same Wikipedia corpora,100steps,batch8,400x128 tokens/language, seed225,lr1.5cosine, exact Cayley, original BF16 checkpoint, per-channel W4 and frozen GPTQ8192. Only document coverage/body sampling changes. English30->200documents,max46->2windows; Chinese same400documents, random body windows. Samplingseed226 and all64validationwindows frozen before training. Wiki32 reused; independent XL-Sum BBC news32 added. News is expository prose, not a full general-domain or fiction benchmark. Validation never includes qbh scoring/holdout. Other recipes and runtime frozen.

| 模型 | 全部 NLL | 英文 NLL | 中文 NLL | 英文百科 | 中文百科 | 英文新闻 | 中文新闻 |
|---|---|---|---|---|---|---|---|
| control_A | 3.541112 | 3.380727 | 3.701497 | 3.235651 | 3.173332 | 3.525804 | 4.229663 |
| step000 | 3.667078 | 3.557169 | 3.776987 | 3.441085 | 3.270374 | 3.673254 | 4.283600 |
| old100 | 3.561694 | 3.399672 | 3.723716 | 3.272592 | 3.260804 | 3.526752 | 4.186628 |
| step050 | 3.581662 | 3.445253 | 3.718070 | 3.346376 | 3.220652 | 3.544130 | 4.215489 |
| step100 | 3.573788 | 3.431109 | 3.716467 | 3.299876 | 3.213084 | 3.562341 | 4.219850 |


## Primary attribution: fixed old100 versus new100

| 子集 | 新−旧 NLL | PPL变化 | 改善片段 | 描述性 paired bootstrap95% NLL |
|---|---|---|---|---|
| all | +0.012094 | +1.217% | 28/64 | [-0.019239434151066374, 0.04299372120203896] |
| en | +0.031437 | +3.194% | 12/32 | [-0.01675094835437538, 0.0787911312571108] |
| zh | -0.007249 | -0.722% | 16/32 | [-0.04728467624371862, 0.03271967989469968] |
| wiki_en | +0.027284 | +2.766% | 6/16 | [-0.04973204504559872, 0.10633291945543084] |
| wiki_zh | -0.047720 | -4.660% | 10/16 | [-0.10588084367904156, 0.011836711985345631] |
| news_en | +0.035589 | +3.623% | 6/16 | [-0.018920007527731885, 0.08672110303906926] |
| news_zh | +0.033222 | +3.378% | 6/16 | [-0.010007902166210702, 0.08014261427209712] |


## Same-rotation training surrogate versus actual export

| 旋转 | STE RTN NLL | 实际 GPTQ NLL | GPTQ−STE |
|---|---|---|---|
| step000 | 5.635196 | 3.667078 | -1.968117 |
| old100 | 3.692371 | 3.561694 | -0.130677 |
| step050 | 3.807138 | 3.581662 | -0.225476 |
| step100 | 3.660152 | 3.573788 | -0.086364 |


STE uses unchanged GPU training FP16 path; actual packages use CPU software FP16 scoring with identical64x127targets. This diagnoses objective mismatch, not bitwise CPU/GPU equivalence. The small validation and one training seed do not establish broad statistical generalization.

## Final qbh per-language comparison

```json
{
  "selected_checkpoint": "step100",
  "role": "final evaluation after selection; distinct from primary fixed-step100 validation attribution",
  "cases": [
    {
      "id": 0,
      "language": "zh",
      "old_nll": 4.037484169825,
      "new_nll": 3.9376192079101373,
      "delta": -0.09986496191486305
    },
    {
      "id": 1,
      "language": "zh",
      "old_nll": 3.195009707750625,
      "new_nll": 3.3238972426893754,
      "delta": 0.12888753493875038
    },
    {
      "id": 2,
      "language": "zh",
      "old_nll": 4.29820275300625,
      "new_nll": 5.052316069850001,
      "delta": 0.7541133168437506
    },
    {
      "id": 3,
      "language": "zh",
      "old_nll": 4.30758667001875,
      "new_nll": 4.30796194090625,
      "delta": 0.00037527088750000104
    },
    {
      "id": 4,
      "language": "zh",
      "old_nll": 3.991136430579375,
      "new_nll": 3.46947872624375,
      "delta": -0.5216577043356248
    },
    {
      "id": 5,
      "language": "zh",
      "old_nll": 4.00231278148125,
      "new_nll": 4.615347746025,
      "delta": 0.6130349645437496
    },
    {
      "id": 6,
      "language": "zh",
      "old_nll": 3.9243978249125,
      "new_nll": 3.63733565798125,
      "delta": -0.28706216693124986
    },
    {
      "id": 7,
      "language": "zh",
      "old_nll": 4.1485823408125,
      "new_nll": 3.66466545724375,
      "delta": -0.4839168835687504
    },
    {
      "id": 8,
      "language": "zh",
      "old_nll": 3.387921571875,
      "new_nll": 3.6189384464375,
      "delta": 0.23101687456249964
    },
    {
      "id": 9,
      "language": "zh",
      "old_nll": 4.143774512449999,
      "new_nll": 4.155815719174999,
      "delta": 0.012041206724999753
    },
    {
      "id": 10,
      "language": "zh",
      "old_nll": 3.715526342475,
      "new_nll": 3.6541244981437497,
      "delta": -0.06140184433125029
    },
    {
      "id": 11,
      "language": "zh",
      "old_nll": 3.96428048775,
      "new_nll": 3.811367389125,
      "delta": -0.15291309862500002
    },
    {
      "id": 12,
      "language": "zh",
      "old_nll": 3.5507966279025,
      "new_nll": 3.49742841694375,
      "delta": -0.053368210958749884
    },
    {
      "id": 13,
      "language": "zh",
      "old_nll": 5.1718921636125,
      "new_nll": 4.9911665905625,
      "delta": -0.18072557305000014
    },
    {
      "id": 14,
      "language": "zh",
      "old_nll": 4.256417512175,
      "new_nll": 4.141017556625,
      "delta": -0.11539995555000004
    },
    {
      "id": 15,
      "language": "zh",
      "old_nll": 3.888993144755,
      "new_nll": 4.361101985238751,
      "delta": 0.47210884048375057
    },
    {
      "id": 20,
      "language": "en",
      "old_nll": 2.9001535161500005,
      "new_nll": 2.744024395878125,
      "delta": -0.15612912027187553
    },
    {
      "id": 21,
      "language": "en",
      "old_nll": 2.9241642981375,
      "new_nll": 2.3746360543625,
      "delta": -0.5495282437749998
    },
    {
      "id": 22,
      "language": "en",
      "old_nll": 3.6935253135624997,
      "new_nll": 3.36526572638125,
      "delta": -0.3282595871812495
    },
    {
      "id": 23,
      "language": "en",
      "old_nll": 5.2159704013,
      "new_nll": 4.806102871768751,
      "delta": -0.4098675295312493
    },
    {
      "id": 24,
      "language": "en",
      "old_nll": 3.08374512185625,
      "new_nll": 2.7421389814625,
      "delta": -0.34160614039375004
    },
    {
      "id": 25,
      "language": "en",
      "old_nll": 2.2779304979125,
      "new_nll": 2.2847591636875,
      "delta": 0.006828665775000076
    },
    {
      "id": 26,
      "language": "en",
      "old_nll": 7.029691938526533,
      "new_nll": 3.331353427440625,
      "delta": -3.6983385110859075
    },
    {
      "id": 27,
      "language": "en",
      "old_nll": 4.695209264564999,
      "new_nll": 5.000483872295625,
      "delta": 0.30527460773062565
    },
    {
      "id": 28,
      "language": "en",
      "old_nll": 3.5929155363075003,
      "new_nll": 3.255551219,
      "delta": -0.33736431730750027
    },
    {
      "id": 29,
      "language": "en",
      "old_nll": 4.181092619206249,
      "new_nll": 4.56370461125,
      "delta": 0.38261199204375096
    },
    {
      "id": 30,
      "language": "en",
      "old_nll": 3.5854740140375,
      "new_nll": 3.4334431881312497,
      "delta": -0.1520308259062504
    },
    {
      "id": 31,
      "language": "en",
      "old_nll": 3.42391824524375,
      "new_nll": 2.7230907683393752,
      "delta": -0.700827476904375
    },
    {
      "id": 32,
      "language": "en",
      "old_nll": 4.679065943431063,
      "new_nll": 3.487908363061644,
      "delta": -1.1911575803694188
    },
    {
      "id": 33,
      "language": "en",
      "old_nll": 3.7167168854375,
      "new_nll": 3.9832600375,
      "delta": 0.2665431520624999
    },
    {
      "id": 34,
      "language": "en",
      "old_nll": 2.73860108785625,
      "new_nll": 2.85688221520625,
      "delta": 0.11828112734999996
    },
    {
      "id": 35,
      "language": "en",
      "old_nll": 3.6056008318687502,
      "new_nll": 3.2056859740875,
      "delta": -0.39991485778125035
    }
  ],
  "language": {
    "en": {
      "improved_cases": 11,
      "total": 16,
      "nll_change": -0.44909279034662186
    },
    "zh": {
      "improved_cases": 9,
      "total": 16,
      "nll_change": 0.015954225607469508
    }
  }
}
```

Frozen GPTQ8192 calibration contains32 English windows from only3 documents (15/16/1). This experiment changes rotation-training sampling only. GPTQ calibration coverage remains an untested possible factor, not an established cause and not a new authorized experiment.

## Collection recovery

Initial package reload retained FP32 RoPE inv_freq while direct export had called model.half(). Canonical reload now converts buffers as well as parameters to FP16. Failed initial-control scores and interrupted command logs remain retained. A bilingual oracle and all32 historical Wiki scores reproduce the original exports exactly; CPU scoring uses8threads after exact8vs16 checks, while GPTQ export retains16threads. Training and quantized weights were not altered by this repair.

## Selection before qbh

```json
{
  "selected": "step100",
  "rule": "minimum_equal_language_equal_domain_actual_export_validation_NLL_tie_earliest",
  "scores": {
    "step000": {
      "nll": 3.6670784148148945,
      "language_nll": {
        "en": 3.5571693563829934,
        "zh": 3.7769874732467956
      }
    },
    "step050": {
      "nll": 3.5816618938992115,
      "language_nll": {
        "en": 3.4452534648433772,
        "zh": 3.7180703229550454
      }
    },
    "step100": {
      "nll": 3.5737880625555354,
      "language_nll": {
        "en": 3.4311086848217798,
        "zh": 3.716467440289291
      }
    }
  },
  "evaluation_used": false
}
```

## Final DSP quality and immutable historical controls

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| A EXP0224（冻结） | 3.6603 | 38.87 | 19/24 | 66.80% |
| 旧采样 EXP0225 step100 | 3.9165 | 50.22 | 20/24 | 66.02% |
| 新采样 R1R2 step100 | 3.6999 | 40.44 | 9/24 | 68.75% |


## Three-recipe profile, historical other-recipe columns

| 模块 | F16A16 冻结 EXP-0218 | W4A16 learned R1R2 GPTQ step100 EXP-0226 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 381.5 (0.60%) | 247.4 (0.63%) | +54.19% |
| Input RMSNorm | 489.7 (0.61%) | 492.7 (0.78%) | 554.0 (1.40%) | -11.06% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11702.8 (18.51%) | 7052.7 (17.82%) | +65.93% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3959.8 (6.26%) | 3214.5 (8.12%) | +23.18% |
| O projection | 5757.7 (7.14%) | 5024.8 (7.95%) | 1256.4 (3.17%) | +299.94% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 474.6 (0.75%) | 654.0 (1.65%) | -27.44% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22438.5 (35.49%) | 14442.7 (36.49%) | +55.36% |
| Down | 13447.9 (16.67%) | 8652.1 (13.69%) | 3428.5 (8.66%) | +152.36% |
| Final residual | 140.1 (0.17%) | 140.0 (0.22%) | 183.8 (0.46%) | -23.83% |
| KV carrier conversion | 174.0 (0.22%) | 172.3 (0.27%) | 203.2 (0.51%) | -15.21% |
| KV append DMA | 343.6 (0.43%) | 341.7 (0.54%) | 463.9 (1.17%) | -26.33% |
| Block orchestration | 16.1 (0.02%) | 19.9 (0.03%) | 34.6 (0.09%) | -42.47% |
| Layer bookkeeping | 23.9 (0.03%) | 23.2 (0.04%) | 23.2 (0.06%) | +0.11% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.2 (0.01%) | 22.5 (0.06%) | -59.14% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 94.9 (0.15%) | 105.6 (0.27%) | -10.14% |
| Embedding | 68.1 (0.08%) | 65.0 (0.10%) | 62.4 (0.16%) | +4.05% |
| Final model RMSNorm | 49.7 (0.06%) | 48.4 (0.08%) | 3.7 (0.01%) | +1199.30% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6778.8 (10.72%) | 5284.7 (13.35%) | +28.27% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2329.2 (3.68%) | 2466.6 (6.23%) | -5.57% |
| 完整 Host wall | 80692.2 (100.00%) | 63222.4 (100.00%) | 39575.9 (100.00%) | +59.75% |


## Direct E2E

```json
{
  "A0": {
    "prefill_tokens": 64,
    "prefill_host_us": 63220.6505,
    "prefill_tokens_per_second": 1012.3274514551222,
    "decode_tokens": 15,
    "decode_total_host_us": 1390199.6875,
    "decode_tokens_per_second": 10.789816840611252
  },
  "old100": {
    "prefill_tokens": 64,
    "prefill_host_us": 63098.880000000005,
    "prefill_tokens_per_second": 1014.2810775722168,
    "decode_tokens": 15,
    "decode_total_host_us": 1390063.6705,
    "decode_tokens_per_second": 10.790872618521542
  },
  "step100": {
    "prefill_tokens": 64,
    "prefill_host_us": 63222.422,
    "prefill_tokens_per_second": 1012.2990859160694,
    "decode_tokens": 15,
    "decode_total_host_us": 1390274.374,
    "decode_tokens_per_second": 10.789237204195206
  }
}
```

## Interpretation and next discussion

The sampling ablation does not establish consistent quality recovery. On fixed qbh-lite-v1, DSP PPL improves from 50.22449369 to 40.44461663 (-19.4723%), English PPL 46.24650852 to 29.51483975, but strict short-task correctness falls from20/24 to9/24. There are both wrong answers and format-following failures: for example, case40 returns3 instead of5; case45 returns an equation instead of only7; JSON answers can include code fences. Scoring rules remain unchanged; no relaxed score is substituted.

On independent64-window validation at matched100steps, actual GPTQ NLL rises from3.561694003186048 to3.5737880625555354, while the STE training surrogate improves3.6923705115914345 to3.6601522751152515. The descriptive confidence interval for the actual mean difference crosses zero: evidence supports no reliable improvement, not a universal claim of degradation. Thus broader rotation-training sampling alone is insufficient in this bounded run. The result also warrants checking transfer between training fake quantization and actual GPTQ/output-scale export.

Suggested next discussion: freeze rotations and compare broader GPTQ calibration document coverage at the same8192-token budget; the current4096 English calibration tokens come from only3 documents. This is a hypothesis, not a proven cause or authorization to start another experiment. No baseline is promoted. Speed is effectively unchanged; complete5short/10three-way formal evidence is retained.
