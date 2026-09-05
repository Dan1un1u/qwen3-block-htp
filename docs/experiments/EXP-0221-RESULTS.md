# EXP-0221: per-channel GPTQ with original, folded and rotated weights

This completed experiment tests A=original+GPTQ, B=fresh gamma fold+GPTQ and C=fresh gamma fold+fixed Sylvester H2048 R1/H128 R2+GPTQ. All changed weights originate in verified Qwen3-origin shards. No upstream folded weights, LPBQ/group scales or QNN assets are used. The other two recipes remain frozen. Historical RTN A/B/C quality comes from EXP-0219; these controls were not re-evaluated.

## Outcome

Unrotated GPTQ A is the best candidate in this fixed trial: actual DSP NLL improves from4.231668 to3.873038 while strict short tasks remain8/24. B reaches4.601481 and0/24; C reaches4.3210 and2/24. GPTQ substantially reduces the damage of folded/rotated RTN, but fixed R1/R2 provides no incremental benefit over GPTQ A. Both predeclared conjunction gates fail; evidence remains valid and no baseline is promoted. This does not establish that all rotations or GPTQ configurations fail. Four-way inference speeds are essentially unchanged; no speed improvement is claimed.

## Fixed calibration and quantizer

64 sequences of128 tokens,32 English WikiText-2-raw train rows and32 Chinese Wikipedia20231101.zh train rows. First128 tokens from first32 eligible distinct rows per language, skipping truncated/short rows and exact32-token overlaps with evaluation samples. No padding, chat-template addition, parameter search or evaluation-guided selection. Raw HTTP snapshots, dataset metadata revisions, exact IDs and tokenizer hashes are retained. Eight holdout windows were only included in the duplicate-exclusion check and never scored. This8192-token calibration is a bounded first trial, not evidence of calibration sufficiency across domains or long contexts.

GPTQ method reference: IST-DASLab/gptq at2d65066eeb06a5c9ff5184d8cebdf33662c67faf, Apache-2.0, archived with its license. Adaptation uses CPU, FP64 damped-Hessian factorization and FP32 error compensation. Fixed transformed-row absmax/7 scale, nearest-even signed[-7,7], groupsize=-1, act-order enabled/inverted on export, damping1% of mean Hessian diagonal. Computational blocksize128 does not add quantization groups. No clipping search. Row chunks share the same input factor and preserve per-row independence.

True-sequential FP16 calibration processes QKV, O, Gate/Up, Down in order; subsequent projections/layers observe previously quantized outputs. The final head observes the quantized stack. All transforms precede calibration, so input statistics use the correct folded/rotated coordinates. Original FP32 transformed weights determine scales before quantization; deployed activations and reconstructed weights use FP16. Tied HF embedding/head are explicitly detached. All197 transformer/head projections are GPTQ quantized, embedding stays FP16, Q/K head norm and RoPE semantics stay intact.

## Actual DSP quality

Frozen qbh-lite-v1:512 bilingual conditional targets,24 strict short tasks and4open prefixes. Quick/full/repeat suites are deterministic across all overlapping token/code/NLL/rank/tie/saturation fields. This is a lightweight diagnostic, not broad model-quality certification. Software results are independently computed from exported integer codes/scales and are not assumed bit-exact DSP logits.

| 实现 | NLL ↓ | 条件 PPL ↓ | 短题 | Teacher top-1 |
|---|---|---|---|---|
| RTN 原始（冻结） | 4.2317 | 68.83 | 8/24 | 55.08% |
| RTN gamma 折叠（冻结） | 5.6017 | 270.88 | 0/24 | 32.62% |
| RTN R1/R2（冻结） | 8.5625 | 5231.70 | 0/24 | 28.52% |
| GPTQ 原始 A | 3.8730 | 48.09 | 8/24 | 62.50% |
| GPTQ gamma 折叠 B | 4.6015 | 99.63 | 0/24 | 46.09% |
| GPTQ R1/R2 C | 4.3210 | 75.26 | 2/24 | 57.81% |

Predeclared gates: A improves original RTN NLL AND short-task count = False; C improves A on BOTH metrics = False. No automatic baseline promotion.

## Calibration-local diagnostics

| Variant | Projections | Mean RTN NRMSE | Mean GPTQ NRMSE | GPTQ lower count |
|---|---|---|---|---|
| A | 197 | 0.156287 | 0.070592 | 197 |
| B | 197 | 0.187472 | 0.097075 | 197 |
| C | 197 | 0.154998 | 0.064659 | 197 |

These errors use every16th calibration activation,512 positions, and compare linear output against unquantized transformed FP32 weights on identical inputs. Means are unweighted across projections, not whole-model errors; they are neither evaluation scores nor used to choose grid/settings. Input-statistic hashes in weight_stats are hashes of the Gram diagonal, not the full Gram.

All dense inverse-Hessian oracle, packed-code roundtrip, FP32 transform invariance and staged-versus-unchanged-HF calibration-forward checks pass. Full details:

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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000467,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000442,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003662,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000003506,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000305,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000002809,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000002343,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000002158,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001652,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000001776,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001432,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000146,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001215,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000001221,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000915,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000968,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000566,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000684,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000469,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000389,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000369,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000315,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000302,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000222,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000218,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000153,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000158,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000075,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000084,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000006,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000033,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000022,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 3.8741880133748055,
      "ppl": 48.14359046418558,
      "tasks_correct": 9,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.630859375,
      "language_nll": {
        "zh": 4.30016877502203,
        "en": 3.448207251727581
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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000004858,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000921,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003788,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000635,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000003135,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000484,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000002336,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000346,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001568,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000255,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.00000000000014,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000226,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000001215,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000155,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000968,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000124,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000068,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000084,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000475,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000067,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000433,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000058,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000335,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000275,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000003,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000178,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000098,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000001,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000056,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000003,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000016,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000016,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 4.600959472358227,
      "ppl": 99.57981389951235,
      "tasks_correct": 0,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.46875,
      "language_nll": {
        "zh": 5.093510314822197,
        "en": 4.108408629894257
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
    "calibration_forward": [
      {
        "layer": 0,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 0,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 1,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000063,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 2,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000682,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 3,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.00000000000005,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000039,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 4,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000402,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000318,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 5,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000333,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000253,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 6,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000025,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000209,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 7,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000209,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000017,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 8,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000187,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000012,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 9,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000012,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 10,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000073,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000056,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 11,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000049,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000047,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 12,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000044,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000033,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 13,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 14,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000027,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000024,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 15,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000018,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 16,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 17,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000009,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 18,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000007,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 19,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 20,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 21,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 22,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000004,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 23,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 24,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999999,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 25,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 26,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 0,
        "nrmse": 0.0,
        "cosine": 1.0000000000000002,
        "max_abs": 0.0,
        "finite": true
      },
      {
        "layer": 27,
        "sample": 32,
        "nrmse": 0.0,
        "cosine": 0.9999999999999998,
        "max_abs": 0.0,
        "finite": true
      }
    ],
    "software": {
      "nll": 4.323169119656086,
      "ppl": 75.42728802273471,
      "tasks_correct": 2,
      "tasks_total": 24,
      "teacher_top1_agreement": 0.572265625,
      "language_nll": {
        "zh": 4.538219437003136,
        "en": 4.108118802309036
      }
    }
  }
}
```

## Complete profiling and provenance

Original RTN plus A/B/C each pass one warmup,5short,10formal sessions with rotating order. All640formal invocation ledgers and17920layer ledgers close exactly. Each16-token speed sequence matches its independent software reference. Fixed M64+15 feedback steps continue after EOS if present; timing then describes diagnostic execution, not usable text throughput. Host wall covers the complete model token-in/token-out pass, excluding cold staging and separately logged WSL tokenizer/detokenizer work. Low-NLL GPTQ candidate is chosen solely for the overview column under the predeclared rule; this does not promote a baseline. Other columns are frozen EXP0218 nonpaired references and do not support activation-only attribution with changed W4 weights.

Runtime is unchanged EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108, embeddedlabel218 intentional; outer experiment221. Full28-layer model, final norm/head/greedy, persistent KV;8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC and one HMX owner, no QNN. Offline calibration/quantization costs are not DSP inference timings.

Evidence D:/llm_exp/results/qwen3-block-htp/exp0221; models D:/llm_exp/models/qwen3-block-htp/exp0221. Source archives, package manifests, frozen runtime binaries, dependency hashes and final evidence ledger are identified in closure.json/evidence_sha256.json. Inherited layer-replay references remain historical placeholders and cannot validate new layer replay. Every successful deployment checks1276files and frozen binary hashes. Tokenizer metadata and license-filename recovery preserved downloaded data and exact calibration IDs; see calibration_recovery.json/reference_recovery.json. Downloads used the existing Windows Clash proxy without global proxy changes.

Reproduction requires a newly registered experiment and fresh output paths. gptq_exp0221.py freezes calibration and checks the independent oracle; experiment_exp0221.py generates each candidate; measure_exp0221.py deploys/runs suites; summarize_exp0221.py reports. Do not overwrite retained outputs or tune against this evaluation/holdout.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 GPTQ A EXP-0221 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 384.9 (0.61%) | 247.4 (0.63%) | +55.57% |
| Input RMSNorm | 489.7 (0.61%) | 491.6 (0.78%) | 554.0 (1.40%) | -11.26% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11726.5 (18.66%) | 7052.7 (17.82%) | +66.27% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3953.7 (6.29%) | 3214.5 (8.12%) | +22.99% |
| O projection | 5757.7 (7.14%) | 5024.6 (7.99%) | 1256.4 (3.17%) | +299.92% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.5 (0.75%) | 654.0 (1.65%) | -27.60% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22410.4 (35.66%) | 14442.7 (36.49%) | +55.17% |
| Down | 13447.9 (16.67%) | 8570.6 (13.64%) | 3428.5 (8.66%) | +149.98% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.72% |
| KV carrier conversion | 174.0 (0.22%) | 172.6 (0.27%) | 203.2 (0.51%) | -15.08% |
| KV append DMA | 343.6 (0.43%) | 340.2 (0.54%) | 463.9 (1.17%) | -26.65% |
| Block orchestration | 16.1 (0.02%) | 19.3 (0.03%) | 34.6 (0.09%) | -44.13% |
| Layer bookkeeping | 23.9 (0.03%) | 23.4 (0.04%) | 23.2 (0.06%) | +1.12% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.3 (0.01%) | 22.5 (0.06%) | -62.96% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.7 (0.15%) | 105.6 (0.27%) | -11.27% |
| Embedding | 68.1 (0.08%) | 65.9 (0.10%) | 62.4 (0.16%) | +5.50% |
| Final model RMSNorm | 49.7 (0.06%) | 47.9 (0.08%) | 3.7 (0.01%) | +1186.01% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6695.7 (10.65%) | 5284.7 (13.35%) | +26.70% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2281.9 (3.63%) | 2466.6 (6.23%) | -7.49% |
| 完整 Host wall | 80692.2 (100.00%) | 62850.8 (100.00%) | 39575.9 (100.00%) | +58.81% |


## Direct E2E

```json
{
  "times": {
    "original": {
      "prefill_tokens": 64,
      "prefill_host_us": 62982.682,
      "prefill_tokens_per_second": 1016.1523448620368,
      "decode_tokens": 15,
      "decode_total_host_us": 1389468.3575000002,
      "decode_tokens_per_second": 10.79549593125585
    },
    "A": {
      "prefill_tokens": 64,
      "prefill_host_us": 62850.834,
      "prefill_tokens_per_second": 1018.2840214976304,
      "decode_tokens": 15,
      "decode_total_host_us": 1391317.4985,
      "decode_tokens_per_second": 10.78114809608283
    },
    "B": {
      "prefill_tokens": 64,
      "prefill_host_us": 62818.255000000005,
      "prefill_tokens_per_second": 1018.8121271436145,
      "decode_tokens": 15,
      "decode_total_host_us": 1387194.112,
      "decode_tokens_per_second": 10.81319468576291
    },
    "C": {
      "prefill_tokens": 64,
      "prefill_host_us": 63085.10400000001,
      "prefill_tokens_per_second": 1014.5025678328119,
      "decode_tokens": 15,
      "decode_total_host_us": 1391000.937,
      "decode_tokens_per_second": 10.78360165044231
    }
  },
  "paired_speed_percent": {
    "A": {
      "prefill": 0.13881673030360364,
      "decode": -0.14955646429521963
    },
    "B": {
      "prefill": 0.20515272293613052,
      "decode": 0.04212258862772433
    },
    "C": {
      "prefill": -0.18861969605361573,
      "decode": -0.18218790759139125
    }
  }
}
```


## Source identity

```json
{
  "source_branch": "codex/exp-0221-w4f16-gptq-offline-rotation",
  "reporting_input_source_head": "bd4944cf2401f9ca732a79bca982d2088aa0d8b2",
  "implementation_files": {
    "gptq_exp0221.py": "1bd1df1bcee446d62035c11f6ea571105ee3909c46b921613f8e7067e8a76179",
    "experiment_exp0221.py": "1a677db15826ce409b49d7ceb272758b52f4cd2c10886b5a6c8b55b5b0d1ff92",
    "measure_exp0221.py": "f11dad4a1836e05cf26f8a7defd099a1401f37a132907115bc8db693a974158a",
    "summarize_exp0221.py": "8a9a021e0261bb140690843f56bdff909e3d716eed5c678170a516be983bca6f"
  },
  "calibration_sha256": "e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669",
  "calibration_ids_sha256": "a20ed2093b0e7fb075fe47a9532843402e9f69784c8ad0b0467194bee2fa8efd",
  "final_archive_source_head": "Recorded after report commit in closure.json"
}
```
