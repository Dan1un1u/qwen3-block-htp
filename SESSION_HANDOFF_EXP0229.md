# EXP0229 active — frozen-data device PPL acceptance

Source codex/exp-0229-w4f16-independent-ppl-acceptance HEAD fd41e9c7aa0b2bbfc4c21e1159f7476f2cd7632b, clean/pushed. Active229, next230. User authorizes finishing W4A16 acceptance then W4A8 discussion only. See PC050/docs/experiments/EXP-0229.md.

Data v2 7adf7f501d3f614ca07bbfe8a3628de538edf23c4c88e4f5cb60f09ad4c21447, freeze 70195c3b22df83b978f678c37cd018a7d55a370be30c53675a0b97397b9736f3. Primary512distinctdocs8192targets, reserve512more8192targets; four balanced English/Chinese Wikipedia/news cells, M64+16. Independent data audit passed1024windows. OriginalBF16GPU cache complete1024windows. Device F/A0/A files/binaries verified, new roots exp0229-v2-*. Only F/A0/A runs; W4U8 frozen.

PC037 recovery details and original failed attempt retained under results exp0229; see experiment record. Regression now uses existing qbhfull IDs0/20, never holdout16. No model/threshold changes.

Current stage before gate (run_exp0229_stage.py before) started. Follow source scripts: after before_gate.json pass, run primary; then primary-summary. If reserve_required true, run reserve then combined-summary. Run after sentinel and require equality before/full/after. report_exp0229.py creates final report/closure/ledger after gates. Do not overwrite complete results; one() verifies and resumes existing shards. Stage logs in commands. No new speed profiling (unchanged models/runtime), report N/A with retained prior evidence. Finish by recording actual results, syncing, releasing lock; no automatic baseline promotion.
