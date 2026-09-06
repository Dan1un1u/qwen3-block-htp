# EXP0229 completed — W4A16 independent PPL acceptance failed

Source codex/exp-0229-w4f16-independent-ppl-acceptance HEAD fd41e9c7aa0b2bbfc4c21e1159f7476f2cd7632b remains clean/pushed. Active none, next230. User requested acceptance closure then W4A8 discussion only. No model baseline promoted and no W4A8 experiment started.

Authority result: docs/experiments/EXP-0229-RESULTS.md, index EXP-0229 and current_scope.latest_w4f16_independent_ppl_acceptance. Raw evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0229. Final evidence ledger SHA256 62ff0e47a85e45aed2a5f26903951345b9fdcc3ff19f079427a5e5f582357336. Dataset SHA256 7adf7f501d3f614ca07bbfe8a3628de538edf23c4c88e4f5cb60f09ad4c21447; freeze70195c3b22df83b978f678c37cd018a7d55a370be30c53675a0b97397b9736f3.

Primary512documents8192targets complete F/A0/A; no reserve trigger. A0/A both fail. DSPF16 PPL25.233552, A030.829344(+22.18%), A30.993039(+22.82%), originalBF16GPU25.272134. Every language/domain/cell also fails10%. A versus A0 interval crosses1, so no robust block-reconstruction gain. Controls and repeated scores match exactly;296files hash-verified and27PPLaggregates independently reproduced. No new speed profiling; EXP218/224/227 formal evidence retained.

Original failed data/deployment/fixture attempts remain preserved per PC037. All execution stages and final report have completed; no device process needs resuming. Do not rerun EXP229 or overwrite its ledger. Data v2 and all existing qbh outputs are exposed evaluation data, never training/checkpoint selection.

Discuss W4A8 next: distinguish weight quantization error from activation/runtime error. Propose A0 only as a frozen provisional W4-weight control, not accepted quality baseline. First resolve prior repeat nondeterminism and software/device agreement; use identical logical W4 codes/scales, rebuild weight-dependent sums and valid activation qparams for W4U8. Then localize saturation, zero-point correction and accumulated activation/KV error. No automatic rotation, mixed precision, group quantization, retraining or performance specialization. User chooses next scope before EXP230; W4A8 remains frozen.
