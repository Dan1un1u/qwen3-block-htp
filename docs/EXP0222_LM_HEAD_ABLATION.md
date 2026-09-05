# EXP-0222: LM head precision ablation

User-approved host-only diagnostic on frozen EXP-0221 GPTQ A/B/C. Each row compares identical transformer weights, embedding, normalization and arithmetic, changing only LM head precision. A retains original gamma and uses original FP16 head; B retains identity final norm and uses freshly folded W*gamma; C retains rotated coordinates and uses freshly folded/rotated (W*gamma)*H2048. Original checkpoint only; no LPBQ, clipping, calibration, rotation learning or other-recipe changes.

## Outcome

FP16 head substitution gives only small NLL gains for unrotated A (3.874188 to3.853870, tasks9 to8) and rotated C (4.323169 to4.293221, tasks2 to2). Fold-only B benefits more (4.600959 to4.413664), but tasks remain0. The C-minus-A NLL gap is0.448981 with W4 heads and0.439351 with FP16 heads: nearly all of the observed gap persists. W4 LM-head quantization alone therefore does not explain the rotated candidate's deficit on this diagnostic set. This intervention does not distinguish transformer quantization, accumulated drift, embedding rounding or rotation selection from each other. A's one changed task is a strict failure from apple text containing an extra character; it does not establish that FP16 heads are generally worse.

All W4 software controls reproduce exactly, including NLL, and all bilingual repeat/non-head hash checks pass. No new optimization direction is started. Discuss these findings with the user first.

## Paired software quality

These are software-versus-software pairs. They must not be substituted for DSP results. Fixed qbh-lite-v1:512 conditional targets,24 strict tasks,4 open prefixes; eight holdout samples unscored. Conditional scores use the same teacher-forced inputs; greedy samples use each head's own generated feedback. NLL lower is better. This lightweight set is diagnostic, not a broad quality certification.

| Transformer | W4 head NLL | FP16 head NLL | NLL delta | W4 tasks | FP16 tasks | W4 teacher top1 | FP16 teacher top1 |
|---|---|---|---|---|---|---|---|
| A | 3.874188 | 3.853870 | -0.020318 | 9/24 | 8/24 | 63.09% | 63.67% |
| B | 4.600959 | 4.413664 | -0.187296 | 0/24 | 0/24 | 46.88% | 49.22% |
| C | 4.323169 | 4.293221 | -0.029948 | 2/24 | 2/24 | 57.23% | 56.25% |

## Gaps to unrotated A

```json
{
  "W4": {
    "B": 0.7267714589834213,
    "C": 0.4489811062812805
  },
  "FP16": {
    "B": 0.5597932897508144,
    "C": 0.4393511451780796
  }
}
```

A gap that persists with FP16 heads cannot be attributed solely to W4 head quantization. Differences in paired head gains identify representation-dependent head sensitivity; these are finite interventions and need not add linearly to independent transformer ablations. No hypothesis about clipping or learned rotation was tested here.

## Task changes

```json
{
  "A": [
    {
      "id": 48,
      "language": "zh",
      "old_correct": true,
      "new_correct": false,
      "old_text": "苹果",
      "new_text": "苹苹果"
    }
  ],
  "B": [],
  "C": []
}
```

## Validation and evidence

All consumed package files are checked against frozen manifest identities from EXP-0221 closure. Original checkpoint shards are rehashed. All non-head parameter hashes match before and after both quality evaluations. W4 full controls reproduce previous software token/top1/text/grades exactly, NLL tolerance1e-5; four fixed bilingual NLL/task cases repeat exactly for both heads. A dense explicit Hadamard head-row oracle checks construction independently of the exporter butterfly. Fresh FP16 head tensors and hashes are retained in models/exp0222/A|B|C. No inherited replay caches are consumed.


### A

```json
{
  "control": {
    "samples": 60,
    "max_nll_difference": 0.0,
    "other_fields_exact": true
  },
  "repeats": {
    "W4": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    },
    "FP16": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    }
  },
  "consumed_files": 508,
  "non_head_unchanged": true,
  "head_construction": {
    "nrmse": 8.116195158796543e-09,
    "cosine": 1.0,
    "max_abs": 2.9802322387695312e-08,
    "finite": true
  },
  "head_formula": "W",
  "head_path": "/mnt/d/llm_exp/models/qwen3-block-htp/exp0222/A/lm_head_weight_f16.bin",
  "head_sha256": "bccd2bc5ec321a16d2c273d27634c9fe272fa5814fecc1fa48790fe7793edead",
  "execution_source_head": "412a5b761ca18e064edacef78d5bb5bcf4bd761c"
}
```

### B

```json
{
  "control": {
    "samples": 60,
    "max_nll_difference": 0.0,
    "other_fields_exact": true
  },
  "repeats": {
    "W4": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    },
    "FP16": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    }
  },
  "consumed_files": 508,
  "non_head_unchanged": true,
  "head_construction": {
    "nrmse": 0.00020894145837940217,
    "cosine": 0.9999999781729227,
    "max_abs": 0.000244140625,
    "finite": true
  },
  "head_formula": "W*final_gamma",
  "head_path": "/mnt/d/llm_exp/models/qwen3-block-htp/exp0222/B/lm_head_weight_f16.bin",
  "head_sha256": "eedbcda271db32c70ad6501d111b141bfb6ec8f35036868eefd79cf032812102",
  "execution_source_head": "412a5b761ca18e064edacef78d5bb5bcf4bd761c"
}
```

### C

```json
{
  "control": {
    "samples": 60,
    "max_nll_difference": 0.0,
    "other_fields_exact": true
  },
  "repeats": {
    "W4": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    },
    "FP16": {
      "samples": 4,
      "max_nll_difference": 0.0,
      "other_fields_exact": true
    }
  },
  "consumed_files": 508,
  "non_head_unchanged": true,
  "head_construction": {
    "nrmse": 0.00020730610784619932,
    "cosine": 0.9999999785126858,
    "max_abs": 0.00012087821960449219,
    "finite": true
  },
  "head_formula": "(W*final_gamma)*H2048",
  "head_path": "/mnt/d/llm_exp/models/qwen3-block-htp/exp0222/C/lm_head_weight_f16.bin",
  "head_sha256": "21f95198ddb77da05c3acdceb9a23a5655454cc8734c8e3336efea6f8070f67e",
  "execution_source_head": "412a5b761ca18e064edacef78d5bb5bcf4bd761c"
}
```

## Profiling boundary

Host-only semantic attribution under PC-042. No DSP runtime integration, device execution, warmup/short/formal measurements or module profiling were performed. All repeat-one/repeat-ten module, engine, physical runtime ledger and complete accelerator Host-wall sections are N/A. E2E token/s is N/A. CPU evaluation elapsed time is retained only for reproducibility and is not accelerator throughput. Frozen EXP-0221 performance is not relabeled as mixed-head performance. No automatic baseline promotion. Discuss these results with the user before choosing another direction.

## Reproduction

scripts/experiment_exp0222.py A|B|C reconstructs frozen packed transformer packages, verifies controls and performs each head swap. scripts/close_exp0222.py report then a committed source archive binds the result. Output paths refuse overwrite; future work needs a newly registered experiment. Source and result copies of this report match. Evidence ledger and closure.json record final source HEAD; provenance.json records actual execution source HEAD per variant.


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
