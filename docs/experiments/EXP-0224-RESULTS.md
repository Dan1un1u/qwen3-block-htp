# EXP-0224: post-GPTQ output-aware per-row scales

Only the approved first direction: original-coordinate A and fresh gamma-fold/fixed H2048 R1/H128 R2 C. Controls are frozen EXP0221 A and EXP0223 C. All196transformer projections per candidate regenerated from verified Qwen3-origin. Head, embedding and all norms byte-identical to the corresponding control. No LPBQ/group scales, learned rotations, calibration expansion, runtime or other-recipe changes.

## Observed outcome and limits

Actual DSP A improves from EXP0221 absmax NLL3.873038/8of24 to3.660316/19of24; C improves from EXP0223 clipping3.956556/9of24 to3.892162/17of24. Both predeclared effectiveness conjunctions pass. New original-coordinate A is better than new rotated C on both metrics; cross-coordinate C>A diagnostic is false. No incremental rotation benefit over the improved unrotated method is established. No baseline promotion.

Frozen F16A16 diagnostic is NLL3.633608/22of24. New A is close in this512target NLL diagnostic but still misses5strict tasks versus2for F16. The fixed lightweight suite cannot certify broad quality. All four open outputs are nonempty for both candidates, but A repeats the English quiet-activity instruction rather than answering, and C's Chinese quiet-activity answer suggests reading in the rain. Exact texts and token IDs are retained; readable output is not general usability acceptance. Software A3.661083/19of24 and C3.888549/16of24 remain distinct from actual DSP (C17of24); no bit-exact software/DSP quality claim.

A selects wider-than-clipping candidates on6492of573440rows (1.1321%); on the same current calibration inputs those rows account for91.5298% of the summed clipping-candidate projection SSE. C selects wider candidates on1681rows (0.2931%), accounting for39.1738%. Both largest within-projection improvements are layer2/down (zero-based, the third layer): A clipping SSE4.328490e9 versus selected7.416848e6; C3.065701e8 versus1.456459e8. This directly shows that weight-local clipping can be damaging for a small set of output rows despite looking favorable for most rows. Cross-projection sums are unnormalized and input scales differ; these diagnostics are not comparable whole-model errors, and no isolated-layer ablation proves that layer2/down alone causes the task improvement.

Most rows selecting clipping does not mean most final model bytes are identical to EXP0223: true-sequential GPTQ observes altered predecessor activations and therefore can alter subsequent integer codes. Local selection plus that propagation is the complete intervention. No scores or candidate settings were adjusted after evaluation.

Current-run paired throughput changes: A prefill -0.3838%, decode+0.0189%; C prefill-0.0585%, decode+0.0330%. No confirmed speed improvement is claimed. A marginal prefill median looks faster (62.347ms versus63.026ms), while the median of paired ratios is slightly slower. These are distinct nonlinear summaries; retained rounds show Host-wall variation and the overview Host-DSP boundary shifts. Ranking uses paired evidence, not the apparent marginal-median gain. No repeat confirmation or baseline promotion was performed.

Detailed selection diagnostics (same current candidate inputs):

```json
{
  "A": {
    "rows": 573440,
    "wider_rows": 6492,
    "wider_percent": 1.1321149553571428,
    "wider_rows_share_of_clipping_sse": 0.9152983548253183,
    "clipping_sse": 4828367734.726038,
    "selected_sse": 481548182.33042884,
    "top_projection_improvements": [
      {
        "projection": "layer2/down",
        "rows": 2048,
        "wider_rows": 523,
        "clipping_sse": 4328489656.702142,
        "selected_sse": 7416847.874061054,
        "improvement": 4321072808.82808
      },
      {
        "projection": "layer27/down",
        "rows": 2048,
        "wider_rows": 13,
        "clipping_sse": 20455562.73708816,
        "selected_sse": 13653027.16876295,
        "improvement": 6802535.568325208
      },
      {
        "projection": "layer26/down",
        "rows": 2048,
        "wider_rows": 6,
        "clipping_sse": 10065354.674345072,
        "selected_sse": 6830363.967734422,
        "improvement": 3234990.7066106503
      },
      {
        "projection": "layer27/up",
        "rows": 6144,
        "wider_rows": 141,
        "clipping_sse": 5457868.742694486,
        "selected_sse": 2504577.7659612563,
        "improvement": 2953290.97673323
      },
      {
        "projection": "layer25/v",
        "rows": 1024,
        "wider_rows": 345,
        "clipping_sse": 26207624.80544918,
        "selected_sse": 24098731.455289103,
        "improvement": 2108893.3501600735
      },
      {
        "projection": "layer25/k",
        "rows": 1024,
        "wider_rows": 234,
        "clipping_sse": 17388131.529697567,
        "selected_sse": 15810209.349759284,
        "improvement": 1577922.1799382842
      },
      {
        "projection": "layer26/v",
        "rows": 1024,
        "wider_rows": 203,
        "clipping_sse": 23173661.167926177,
        "selected_sse": 21852892.444140818,
        "improvement": 1320768.723785357
      },
      {
        "projection": "layer24/k",
        "rows": 1024,
        "wider_rows": 219,
        "clipping_sse": 15167385.239043508,
        "selected_sse": 13852529.8429753,
        "improvement": 1314855.3960682102
      }
    ],
    "scope": "same candidate inputs, unnormalized projection SSE sums, not whole-model attribution"
  },
  "C": {
    "rows": 573440,
    "wider_rows": 1681,
    "wider_percent": 0.2931431361607143,
    "wider_rows_share_of_clipping_sse": 0.3917377221452322,
    "clipping_sse": 608676395.1360729,
    "selected_sse": 447747800.22370183,
    "top_projection_improvements": [
      {
        "projection": "layer2/down",
        "rows": 2048,
        "wider_rows": 1398,
        "clipping_sse": 306570092.2729726,
        "selected_sse": 145645923.2900291,
        "improvement": 160924168.98294353
      },
      {
        "projection": "layer26/o",
        "rows": 2048,
        "wider_rows": 14,
        "clipping_sse": 3589271.7742287135,
        "selected_sse": 3586573.0205577808,
        "improvement": 2698.753670932366
      },
      {
        "projection": "layer23/down",
        "rows": 2048,
        "wider_rows": 3,
        "clipping_sse": 2554597.578367616,
        "selected_sse": 2553551.9824226354,
        "improvement": 1045.5959449807824
      },
      {
        "projection": "layer11/o",
        "rows": 2048,
        "wider_rows": 212,
        "clipping_sse": 38339.49481931769,
        "selected_sse": 38020.48995270871,
        "improvement": 319.0048666089756
      },
      {
        "projection": "layer26/down",
        "rows": 2048,
        "wider_rows": 1,
        "clipping_sse": 3934236.857184209,
        "selected_sse": 3933958.834455242,
        "improvement": 278.02272896730165
      },
      {
        "projection": "layer21/v",
        "rows": 1024,
        "wider_rows": 1,
        "clipping_sse": 3091398.816528756,
        "selected_sse": 3091352.4686924024,
        "improvement": 46.347836353622824
      },
      {
        "projection": "layer7/down",
        "rows": 2048,
        "wider_rows": 3,
        "clipping_sse": 15423.90197140899,
        "selected_sse": 15415.30736804275,
        "improvement": 8.594603366240495
      },
      {
        "projection": "layer23/gate",
        "rows": 6144,
        "wider_rows": 1,
        "clipping_sse": 1139744.6882376792,
        "selected_sse": 1139736.8863008146,
        "improvement": 7.801936864571331
      }
    ],
    "scope": "same candidate inputs, unnormalized projection SSE sums, not whole-model attribution"
  }
}
```

## Fixed method

Three candidates per output row: absmax/7, midpoint between absmax and EXP0223 L2.4 clipping scale, and that clipping scale. Each runs complete GPTQ before selection. Score=sum over all8192calibration positions of squared linear output error using FP16(codes*scale).float() minus original FP32 weights; FP32 GEMM in512token chunks, FP64 sum. Undamped actual-input error, not RTN or GPTQ damped proxy. Ties choose earlier/larger range. Per-row choices, all three scores/scales and selected scores retained.

Frozen64x128 bilingual EXP0221 training calibration, same act-order,1% damping,FP64 factor/FP32 error compensation,block128 computational only. True sequential QKV->O->Gate/Up->Down and layer ordering uses the newly quantized predecessors. Included endpoints guarantee nonincreasing selected local score on these identical current inputs; this does not guarantee whole-model quality or imply that old/new projection inputs are identical. No evaluation/holdout selection or tuning.

## Actual DSP quality

Frozen qbh-lite-v1:512conditional targets,24strict tasks,4open prefixes; a lightweight diagnostic, not general quality certification. Software results remain separate.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| A0 EXP0221 absmax（冻结） | 3.8730 | 48.09 | 8/24 | 62.50% |
| C0 EXP0223 clipping（冻结） | 3.9566 | 52.28 | 9/24 | 60.55% |
| output-selected GPTQ A | 3.6603 | 38.87 | 19/24 | 66.80% |
| output-selected GPTQ C | 3.8922 | 49.02 | 17/24 | 66.21% |

Predeclared lower-NLL AND more-task effectiveness: {"A": true, "C": true}. Aggregate pass iff any effectiveness true. Candidate C versus candidate A both-metric diagnostic: False. No promotion.

## Scale selection

| Variant | Rows | Absmax | Midpoint | L2.4 clip | Mean range/absmax |
|---|---|---|---|---|---|
| A | 573440 | 0.06% | 1.07% | 98.87% | 0.7064 |
| C | 573440 | 0.10% | 0.20% | 99.71% | 0.7204 |

Summed projection SSEs are unnormalized local sums from different projection inputs/shapes, retained for diagnostics rather than whole-model comparison. All selected row scores no worse than both endpoints on identical inputs.

```json
{
  "A": {
    "rows": 573440,
    "candidate_names": [
      "absmax",
      "midpoint",
      "weight_L2.4_clip"
    ],
    "choice_histogram": [
      339,
      6153,
      566948
    ],
    "mean_selected_ratio": 0.706375977654378,
    "summed_projection_sse": [
      837812627.9120908,
      1460933357.9753928,
      4828367734.726042
    ],
    "summed_selected_sse": 481548182.33042884
  },
  "C": {
    "rows": 573440,
    "candidate_names": [
      "absmax",
      "midpoint",
      "weight_L2.4_clip"
    ],
    "choice_histogram": [
      558,
      1123,
      571759
    ],
    "mean_selected_ratio": 0.7203844178340659,
    "summed_projection_sse": [
      1095039788.8033412,
      687558006.5351466,
      608676395.136073
    ],
    "summed_selected_sse": 447747800.22370183
  }
}
```

## Checks and provenance

Independent NumPy full-input scoring and choices, explicit-scale dense GPTQ parity, original absmax parity, zero-row ties and packing pass.392projection packing and112staged/HF forward checks pass. FP32 transformed equivalence, independent16speed tokens and all quick/full/repeat consistency pass.

```json
{
  "A": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 0.0,
          "cosine": 0.9999999999997751,
          "max_abs": 0.0,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 0.0,
          "cosine": 0.9999999999997475,
          "max_abs": 0.0,
          "top1_equal": 16,
          "passed": true
        }
      ],
      "original_shards": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
      }
    },
    "forward_checks": 56,
    "max_forward_nrmse": 0.0,
    "frozen_file_count": 883,
    "determinism": {
      "repeat_equal": true,
      "overlap_mismatches": []
    },
    "execution": {
      "source_head": "09aec376d74638703a497625e4e531a77c8db47e",
      "files": {
        "experiment_exp0224.py": "97f7c2092d181549c4d05f521d97163f43f64c938f064eebdb09aa87b4c75533",
        "output_scale_exp0224.py": "fae7c192987ec91124b620acb1d25d62a84326c56e14935e4ae78cb513fc4a2d",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1"
      }
    }
  },
  "C": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 1.5193639915537854e-06,
          "cosine": 0.9999999999986376,
          "max_abs": 5.7220458984375e-05,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 2.0345006134564764e-06,
          "cosine": 0.9999999999977212,
          "max_abs": 5.054473876953125e-05,
          "top1_equal": 16,
          "passed": true
        }
      ],
      "original_shards": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
      }
    },
    "forward_checks": 56,
    "max_forward_nrmse": 0.0,
    "frozen_file_count": 883,
    "determinism": {
      "repeat_equal": true,
      "overlap_mismatches": []
    },
    "execution": {
      "source_head": "dfee1b1c3b3ef720ba2fc1d224e522f048218bab",
      "files": {
        "experiment_exp0224.py": "97f7c2092d181549c4d05f521d97163f43f64c938f064eebdb09aa87b4c75533",
        "output_scale_exp0224.py": "fae7c192987ec91124b620acb1d25d62a84326c56e14935e4ae78cb513fc4a2d",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1"
      }
    }
  }
}
```

Software diagnostic:

```json
{
  "A": {
    "nll": 3.661082625389099,
    "ppl": 38.903437930257674,
    "tasks_correct": 19,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.66796875,
    "language_nll": {
      "zh": 3.9930894672870636,
      "en": 3.3290757834911346
    }
  },
  "C": {
    "nll": 3.888548944145441,
    "ppl": 48.83996556307159,
    "tasks_correct": 16,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.66796875,
    "language_nll": {
      "zh": 4.127618625760078,
      "en": 3.6494792625308037
    }
  }
}
```

## Complete profiling

A0/A/C0/C rotating four-way1warmup,5short,10formal;640formal invocation and17920layer ledgers. M64 prefill plus15feedback decodes, scoring off; direct complete28-layer Host wall includes final norm/head/greedy/persistent KV, excludes cold staging and separately logged WSL tokenizer/detokenizer. Fixed16tokens continue after EOS if encountered and are diagnostic throughput. No partial-model extrapolation.

Frozen EXP0218 source d981072513d06ed61731c14743c76ac6bc81617f ABI108, embedded218 intentionally retained, outer224. Exact8MiB VTCM,zero timed intermediate hidden/logits DDR/spill,one FastRPC/one HMX owner,no QNN. Offline model work is not DSP timing. Scaffolds named layer14 execute all28layers; inherited replay references remain historical and unused.

Overview candidate chosen by lowest new DSP NLL, ties more tasks then alphabetical; report selection does not promote. F16/U8 are frozen EXP0218 nonpaired references with different weights; no activation-only attribution. Full report retains every repeat1/repeat10 numeric control/candidate field, exclusive additive ledgers and overlapping engine/wait counters.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 output-selected GPTQ A EXP-0224 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 392.6 (0.63%) | 247.4 (0.63%) | +58.71% |
| Input RMSNorm | 489.7 (0.61%) | 492.6 (0.79%) | 554.0 (1.40%) | -11.08% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11746.1 (18.84%) | 7052.7 (17.82%) | +66.55% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3964.9 (6.36%) | 3214.5 (8.12%) | +23.34% |
| O projection | 5757.7 (7.14%) | 5050.1 (8.10%) | 1256.4 (3.17%) | +301.95% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.7 (0.76%) | 654.0 (1.65%) | -27.57% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22481.7 (36.06%) | 14442.7 (36.49%) | +55.66% |
| Down | 13447.9 (16.67%) | 8639.3 (13.86%) | 3428.5 (8.66%) | +151.99% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.73% |
| KV carrier conversion | 174.0 (0.22%) | 173.2 (0.28%) | 203.2 (0.51%) | -14.79% |
| KV append DMA | 343.6 (0.43%) | 342.4 (0.55%) | 463.9 (1.17%) | -26.17% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.28% |
| Layer bookkeeping | 23.9 (0.03%) | 23.1 (0.04%) | 23.2 (0.06%) | -0.34% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.0 (0.01%) | 22.5 (0.06%) | -60.07% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 94.2 (0.15%) | 105.6 (0.27%) | -10.80% |
| Embedding | 68.1 (0.08%) | 66.6 (0.11%) | 62.4 (0.16%) | +6.59% |
| Final model RMSNorm | 49.7 (0.06%) | 48.0 (0.08%) | 3.7 (0.01%) | +1189.51% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6725.6 (10.79%) | 5284.7 (13.35%) | +27.27% |
| Host–DSP 边界 | 2374.1 (2.94%) | 1533.8 (2.46%) | 2466.6 (6.23%) | -37.82% |
| 完整 Host wall | 80692.2 (100.00%) | 62347.1 (100.00%) | 39575.9 (100.00%) | +57.54% |

## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63025.937,
      "prefill_tokens_per_second": 1015.4549546800074,
      "decode_tokens": 15,
      "decode_total_host_us": 1388596.094,
      "decode_tokens_per_second": 10.802277253129015
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62347.08349999999,
      "prefill_tokens_per_second": 1026.5115288031077,
      "decode_tokens": 15,
      "decode_total_host_us": 1388479.7644999998,
      "decode_tokens_per_second": 10.80318228865337
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63194.297000000006,
      "prefill_tokens_per_second": 1012.7496156813011,
      "decode_tokens": 15,
      "decode_total_host_us": 1390223.592,
      "decode_tokens_per_second": 10.789631312773752
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63358.516,
      "prefill_tokens_per_second": 1010.1246689553145,
      "decode_tokens": 15,
      "decode_total_host_us": 1389896.4829999998,
      "decode_tokens_per_second": 10.79217062814886
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.38375244825754606,
      "decode": 0.018878336428107545
    },
    "C": {
      "prefill": -0.05846968245281081,
      "decode": 0.03303114342818603
    }
  }
}
```

## Retained artifacts and reproduction

Models D:/llm_exp/models/qwen3-block-htp/exp0224; results D:/llm_exp/results/qwen3-block-htp/exp0224. experiment_exp0224.py A|C; measure_exp0224.py deploy/quick/full/repeat and warmup/short/formal; summarize_exp0224.py; close_exp0224.py report then commit/sync/archive. Fresh paths and registered experiment required.392row-search NPZs,56hidden checkpoints, source archive, runtime binaries, manifests and all logs bound by closure and evidence ledgers. No overwrite or automatic further optimization.
