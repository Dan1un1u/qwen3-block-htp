# EXP-0229 independent W4A16 PPL acceptance

Status: A0 **fail**, A **fail**. 512 distinct documents; 8192 conditional target tokens. Recommended candidate: None. No baseline promoted.

Fixed M64 raw-text contexts plus16 continuation targets. Same actual DSP FP16/W4A16 runtime, weights and masks. English/Chinese x Wikipedia/news balanced. Additional original BF16 GPU reference is named separately. This is short-context conditional PPL, not a published full benchmark or long-context acceptance.

| Stratum | DSP F16 PPL | A0 PPL | A PPL | A0 ratio vs F16 [95% CI] | A ratio vs F16 [95% CI] | Limit |
|---|---:|---:|---:|---|---|---:|
| overall | 25.23355 | 30.82934 | 30.99304 | 1.22176 [1.18827, 1.25804] | 1.22825 [1.19547, 1.26278] | 1.05 |
| en | 20.52100 | 25.87114 | 25.90314 | 1.26072 [1.20767, 1.31785] | 1.26227 [1.21078, 1.31578] | 1.10 |
| zh | 31.02833 | 36.73779 | 37.08310 | 1.18401 [1.14286, 1.22788] | 1.19514 [1.15336, 1.23957] | 1.10 |
| wiki | 23.37693 | 28.93534 | 28.85595 | 1.23777 [1.18537, 1.29524] | 1.23438 [1.18404, 1.28985] | 1.10 |
| news | 27.23763 | 32.84732 | 33.28841 | 1.20595 [1.16611, 1.24787] | 1.22215 [1.18185, 1.26322] | 1.10 |
| en_wiki | 20.61049 | 26.73961 | 26.29410 | 1.29738 [1.21474, 1.39309] | 1.27576 [1.19907, 1.36307] | 1.10 |
| zh_wiki | 26.51469 | 31.31138 | 31.66739 | 1.18091 [1.11668, 1.25237] | 1.19433 [1.12923, 1.26632] | 1.10 |
| en_news | 20.43189 | 25.03087 | 25.51799 | 1.22509 [1.16477, 1.29009] | 1.24893 [1.18758, 1.31304] | 1.10 |
| zh_news | 36.31032 | 43.10462 | 43.42498 | 1.18712 [1.13619, 1.24253] | 1.19594 [1.14421, 1.25186] | 1.10 |

Additional original BF16 GPU PPL: 25.272134. A/A0 PPL ratio: 1.005310, paired95% CI [0.995409466342318, 1.0152930041737163].

## Acceptance interpretation

Gates were fixed before scores: overall<=1.05; each language/domain/cell<=1.10. 5000 paired bootstrap resamples use distinct documents, stratified across four equal cells. Point failures are fail; passing points whose upper interval crosses a gate are inconclusive; pass requires point and upper-bound gates. Reserve is evaluated for all variants only if a primary interval straddles a gate. Reserve triggered: False. Short-task scores do not veto PPL acceptance.
A0 is the simpler GPTQ/output-aware scale package; A additionally reconstructs block scales. Candidate recommendations use only the predeclared rule; no test-driven retraining, clipping search, checkpoint or weight changes occurred. No new W4A8 work or automatic promotion is included.

## Evidence and recovery

Independent corpus/token-offset reconstruction passed1024windows; document, text-hash and32gram separation covers all prior calibration/training/checkpoint validation and qbh full/holdout. English Wikipedia explicitly excludes all WikiText training-corpus titles. Original checkpoint, tokenizer, local/device package and runtime hashes verified. Old qbh numerical controls reproduced; balanced sentinel scores match within-run, before/full/after exactly. Every measured target has finite score diagnostics and valid8MiB/no-intermediate-DDR/cache progression. Teacher repeat and causal-mask tests passed; independent CE agrees. See per-step raw JSONL and validated outputs.

The first dataset attempt exposed a legacy WikiText pseudo-heading parser issue; its snapshot and BF16 diagnostic remain under data_attempt_v1 and never enter acceptance. A new explicit-document Wikipedia source was frozen without consulting model quality scores. A symlink-relative deployment checksum path was changed to an absolute path; model hashes did not change. The control fixture ID was corrected from an unused holdout ID to the existing English full-evaluation ID before any device evaluation. Primary/reserve test tokens were unaffected by that fixture correction. All original artifacts and recovery notes are retained.

## Performance scope

No new speed profiling: evaluation-only inputs changed, runtime and packages are frozen. Quality-suite elapsed time is not token-generation throughput. Prior full profiling references:
- exp0218: /mnt/d/llm_exp/results/qwen3-block-htp/exp0218/full_profiling_report.md (SHA256 521be8903db7f87ed69a162628bbfeb578091f7177ae8cc8876d57b341c0450d)
- exp0224: /mnt/d/llm_exp/results/qwen3-block-htp/exp0224/full_profiling_report.md (SHA256 d142794f1f0a0e12a987e3bd22d347b80ae5f7696817ee69939cb4f3018e4aed)
- exp0227: /mnt/d/llm_exp/results/qwen3-block-htp/exp0227/full_profiling_report.md (SHA256 af24ab36cb7698b6c23509e8c40cf49d515d78485917ade309be4090f7a2735a)

Source HEAD: fd41e9c7aa0b2bbfc4c21e1159f7476f2cd7632b. Dataset SHA256: 7adf7f501d3f614ca07bbfe8a3628de538edf23c4c88e4f5cb60f09ad4c21447.
