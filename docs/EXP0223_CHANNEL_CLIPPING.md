# EXP-0223: transformer per-channel clipping before GPTQ

User-approved bounded scale/clipping trial. A uses original coordinates/gamma, B fresh gamma folding, C fresh fixed H2048 R1/H128 R2. All196 transformer projections per variant are generated afresh from verified Qwen3-origin. Each variant retains byte-identical EXP0221 LM head, embedding and all norm files. No LPBQ/group quantization, new rotation, new calibration, other-recipe change or runtime change.

## Observed outcome and limits

Transformer clipping helps the folded/rotated GPTQ variants but hurts original-coordinate A. Actual DSP A: NLL3.873038->3.969016 and tasks8/24->4/24; B:4.601481->3.911864 and0/24->2/24; C:4.320988->3.956556 and2/24->9/24. Predeclared per-variant effectiveness passes B/C and fails A; clipped C beats clipped A on both metrics, so that incremental gate passes. Aggregate local_gate passes because at least one variant improves; this does not certify usable quality or promote any baseline.

C is still worse in NLL than frozen original-coordinate GPTQ A (3.956556 versus3.873038), and its9/24 versus8/24 is only one extra answer on a tiny suite. The C-to-frozen-A NLL gap shrinks from0.447949 to0.083518 (81.36%), an arithmetic description of this suite rather than a statistical significance claim. Compared with clipped B, clipped C has worse NLL (3.956556 versus3.911864) but more correct tasks (9 versus2); no configuration dominates all quality metrics. Frozen F16A16 scored22/24, so recovered rotated quality remains far below that task reference.

Open-text checks reinforce this limitation: clipped A emits immediate EOS on3/4 open prompts; clipped C does so on1/4, and one other C English answer restates the instruction within the16-token cap. B's four open outputs are nonempty but it still gets only2/24 strict tasks. Exact texts and token IDs are retained in quality_summary.json; short outputs are diagnostic, not general usability certification.

All588projection probes show GPTQ improves over clipped RTN on each identical candidate input, yet A's model quality worsens. The scale search minimizes weight-local L2.4 before GPTQ; it does not minimize activation-weighted post-GPTQ or whole-model loss. This objective mismatch and accumulated quantization error are plausible explanations, not independently proven causes. The experiment supports range selection as a material contributor to folded/rotated degradation, not the claim that fixed absmax alone explains it. No learned rotations or complete SpinQuant reproduction were attempted here.

All six current-run speed comparisons differ by less than0.4% in paired median throughput. No speed benefit is claimed. The overview uses B only because the predeclared report rule selects lowest new NLL; its2/24 task score prevents reading that selection as a recommendation. Keep the frozen baseline identities and all candidates for discussion; no automatic further optimization.

## Fixed quantization intervention

For each output row of transformed FP32 weights, evaluate80 ratios p=1-i/100 for i=0..79. Scale=p*absmax(row)/7, nearest-even signed[-7,7]. Select the minimum sum(abs(weight-reconstructed_weight)^2.4); ties choose the largest range. This adapts the weight-range search approach in SpinQuant's utils/quant_utils.py and its w_clip evaluation option to our fixed integer grid. It is not a full SpinQuant reproduction. No activation-weighted clipping search or post-GPTQ grid search is performed. Original unclipped range is a candidate, so selected weight-local objective cannot increase; this does not guarantee model quality.

GPTQ then keeps that chosen scale fixed while doing activation-weighted error compensation. Existing true-sequential QKV->O->Gate/Up->Down ordering,1% damping,FP64 factor/FP32 compensation, act-order/inverse export and computational block128 remain. Calibration is the frozen EXP0221 independent64x128 bilingual training tokens (8192 total); no evaluation or holdout tuning. Each later projection/layer sees the newly quantized predecessor. The old head stays frozen even though transformer outputs change, isolating the transformer intervention.

## Actual DSP quality

Frozen qbh-lite-v1:512 conditional targets,24 strict tasks,4 open prefixes. EXP0221 absmax A/B/C DSP quality is a frozen historical control. All new candidates complete actual DSP quick/full/repeat and independent software quality. Software scores are separate and not assumed bit-exact DSP. This lightweight diagnostic does not certify general model quality.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| absmax GPTQ A（冻结） | 3.8730 | 48.09 | 8/24 | 62.50% |
| absmax GPTQ B（冻结） | 4.6015 | 99.63 | 0/24 | 46.09% |
| absmax GPTQ C（冻结） | 4.3210 | 75.26 | 2/24 | 57.81% |
| clipping GPTQ A | 3.9690 | 52.93 | 4/24 | 62.70% |
| clipping GPTQ B | 3.9119 | 49.99 | 2/24 | 60.16% |
| clipping GPTQ C | 3.9566 | 52.28 | 9/24 | 60.55% |

Predeclared lower-NLL AND more-task effectiveness per variant: {"A": false, "B": true, "C": true}. Incremental clipped C versus clipped A on BOTH metrics: True. No automatic baseline promotion.

## Scale selection

| Variant | Output rows | Rows clipped | Mean selected range / absmax |
|---|---|---|---|
| A | 573440 | 100.00% | 0.7044 |
| B | 573440 | 99.99% | 0.6709 |
| C | 573440 | 100.00% | 0.7197 |

Per-row choices, scales and before/after objectives are retained as hashed NPZ artifacts. Local probe errors use every16th calibration input (512 positions) on identical candidate inputs, comparing clipped RTN and clipped GPTQ against unquantized transformed weights. They are unweighted projection summaries, not whole-model errors, and are not used to select settings. Input Gram hashes inherited from the GPTQ utility identify diagonals, not full Gram matrices.

## Independent checks and identities

NumPy exhaustive scale selection, dense explicit-scale GPTQ elimination, original absmax parity and integer packing oracles pass. All588projection packing checks and168staged/HF forward checks pass. Nonprojection package files remain byte-identical to each EXP0221 variant.

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
      "source_head": "d6ab67420cc0c823c0705cd6820e323ef5c2f4c6",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
      }
    }
  },
  "B": {
    "FP32": {
      "checks": [
        {
          "sample": 0,
          "finite": true,
          "nrmse": 1.3316402355909908e-06,
          "cosine": 0.9999999999988802,
          "max_abs": 8.0108642578125e-05,
          "top1_equal": 16,
          "passed": true
        },
        {
          "sample": 20,
          "finite": true,
          "nrmse": 1.2410618960000922e-06,
          "cosine": 0.9999999999990077,
          "max_abs": 3.814697265625e-05,
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
      "source_head": "b396ac62de2d58b29de555432df016878b8982cb",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
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
      "source_head": "b396ac62de2d58b29de555432df016878b8982cb",
      "files": {
        "experiment_exp0223.py": "9a5bc75564f1793f130e7a1d22ddf503543b1bd5aa3841fe5a304014a01f9178",
        "clipping_exp0223.py": "26457b3cfd60f07e05903e01ca3e3f6efc96fdc5e1b0e45047d9d0a9af662ff1",
        "gptq_exp0221.py": "cb8af8fbae866ace133713934f5e441d3f14aea293e366c60c0658ce5aa13ee0"
      }
    }
  }
}
```

Software diagnostic:

```json
{
  "A": {
    "nll": 3.969574511051178,
    "ppl": 52.96199130284732,
    "tasks_correct": 3,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.630859375,
    "language_nll": {
      "zh": 4.270898096263409,
      "en": 3.6682509258389473
    }
  },
  "B": {
    "nll": 3.912194263190031,
    "ppl": 50.00856362136663,
    "tasks_correct": 2,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.599609375,
    "language_nll": {
      "zh": 4.345175914466381,
      "en": 3.479212611913681
    }
  },
  "C": {
    "nll": 3.957964614033699,
    "ppl": 52.35066363034929,
    "tasks_correct": 9,
    "tasks_total": 24,
    "teacher_top1_agreement": 0.599609375,
    "language_nll": {
      "zh": 4.258320279419422,
      "en": 3.657608948647976
    }
  }
}
```

## Complete profiling scope

All six variants A0/A/B0/B/C0/C complete one warmup,5short and10formal sessions with rotating order. A0/B0/C0 are the corresponding frozen EXP0221 packages, freshly timed here.960formal invocation ledgers and26880layer ledgers close. All16tokens per speed session match independent per-package references. Scope is one M64 prefill plus15feedback decode steps; quality scoring is off. Host wall is complete28-layer token-in/token-out execution with final norm/head/greedy and persistent KV, excluding cold staging and separately logged WSL tokenizer/detokenizer. No per-layer throughput extrapolation. Fixed speed sequences continue after EOS if present; such throughput describes diagnostic execution.

Runtime remains EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108 (embedded218 intentionally retained, outer experiment223).8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC/one HMX owner, no QNN. Offline quantization elapsed time is not DSP inference time. Package scaffolds named layer14 execute all28layers. Historical inherited replay references are not valid new replay references and are not consumed in generation/evaluation.

The overview W4A16 column uses the lowest-NLL clipped candidate under the predeclared tie rule. Other recipe columns are frozen EXP0218 nonpaired references, with different W4 weights, so no activation-only attribution. Full report retains every numeric repeat-one/repeat-ten control/candidate field; additive accounting fields are exclusive, engine/wait counters overlap. Detailed raw evidence is retained.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 clipped GPTQ B EXP-0223 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 380.8 (0.60%) | 247.4 (0.63%) | +53.93% |
| Input RMSNorm | 489.7 (0.61%) | 491.5 (0.78%) | 554.0 (1.40%) | -11.29% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11748.4 (18.62%) | 7052.7 (17.82%) | +66.58% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3960.5 (6.28%) | 3214.5 (8.12%) | +23.21% |
| O projection | 5757.7 (7.14%) | 5057.3 (8.01%) | 1256.4 (3.17%) | +302.52% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 474.2 (0.75%) | 654.0 (1.65%) | -27.48% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22406.6 (35.51%) | 14442.7 (36.49%) | +55.14% |
| Down | 13447.9 (16.67%) | 8643.2 (13.70%) | 3428.5 (8.66%) | +152.10% |
| Final residual | 140.1 (0.17%) | 140.1 (0.22%) | 183.8 (0.46%) | -23.77% |
| KV carrier conversion | 174.0 (0.22%) | 171.9 (0.27%) | 203.2 (0.51%) | -15.42% |
| KV append DMA | 343.6 (0.43%) | 341.3 (0.54%) | 463.9 (1.17%) | -26.43% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.35% |
| Layer bookkeeping | 23.9 (0.03%) | 23.7 (0.04%) | 23.2 (0.06%) | +2.36% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.9 (0.01%) | 22.5 (0.06%) | -60.42% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.4 (0.15%) | 105.6 (0.27%) | -11.52% |
| Embedding | 68.1 (0.08%) | 66.5 (0.11%) | 62.4 (0.16%) | +6.55% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.71% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6697.0 (10.61%) | 5284.7 (13.35%) | +26.72% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2387.8 (3.78%) | 2466.6 (6.23%) | -3.19% |
| 完整 Host wall | 80692.2 (100.00%) | 63099.3 (100.00%) | 39575.9 (100.00%) | +59.44% |

## Direct E2E

```json
{
  "times": {
    "A0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63069.2965,
      "prefill_tokens_per_second": 1014.7568397246987,
      "decode_tokens": 15,
      "decode_total_host_us": 1386578.6195,
      "decode_tokens_per_second": 10.817994586855088
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 63140.2865,
      "prefill_tokens_per_second": 1013.615926497261,
      "decode_tokens": 15,
      "decode_total_host_us": 1387815.6775,
      "decode_tokens_per_second": 10.80835174525545
    },
    "B0": {
      "prefill_tokens": 64,
      "prefill_host_us": 62837.109,
      "prefill_tokens_per_second": 1018.5064370163816,
      "decode_tokens": 15,
      "decode_total_host_us": 1384688.49,
      "decode_tokens_per_second": 10.832761381586987
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 63099.3485,
      "prefill_tokens_per_second": 1014.2735467387591,
      "decode_tokens": 15,
      "decode_total_host_us": 1387656.2759999998,
      "decode_tokens_per_second": 10.809593311708555
    },
    "C0": {
      "prefill_tokens": 64,
      "prefill_host_us": 63101.198000000004,
      "prefill_tokens_per_second": 1014.2438183186315,
      "decode_tokens": 15,
      "decode_total_host_us": 1387046.3785,
      "decode_tokens_per_second": 10.81434639281602
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63242.3695,
      "prefill_tokens_per_second": 1011.9797930721112,
      "decode_tokens": 15,
      "decode_total_host_us": 1388028.5915,
      "decode_tokens_per_second": 10.806693818741845
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": -0.1627881196463843,
      "decode": -0.08914060633274934
    },
    "B": {
      "prefill": -0.33245670972336105,
      "decode": -0.2061125939613384
    },
    "C": {
      "prefill": -0.2367729934462548,
      "decode": -0.1645508532618134
    }
  }
}
```

## Reproduction and retained evidence

Models: D:/llm_exp/models/qwen3-block-htp/exp0223. Results: D:/llm_exp/results/qwen3-block-htp/exp0223. Frozen dataset/calibration/source/package hashes, source archive, runtime binaries, per-row NPZs,84hidden checkpoints, manifests and full logs are bound by closure/evidence ledgers. experiment_exp0223.py A|B|C generates fresh models; measure_exp0223.py deploy/quick/full/repeat and warmup/short/formal collects results; summarize_exp0223.py and close_exp0223.py retain reports. Fresh output paths and a registered experiment are required; do not overwrite or tune retained outputs. No baseline promotion or automatic further direction.


Environment:

```json
{
  "python": "3.10.12",
  "torch": "2.13.0+cpu",
  "transformers": "4.51.0",
  "numpy": "2.2.6",
  "threads": 16
}
```
