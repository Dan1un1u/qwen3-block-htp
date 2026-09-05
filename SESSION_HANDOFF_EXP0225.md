# EXP0225 active — GPU smoke

Source 19aac658379b1518b7792460ed128dc8d40db28b on codex/exp-0225-w4f16-learned-r1r2. All implementation/math gates pass: algebra_oracle.json, output_scale_oracle.json, full_model_oracle.json. FP32 invariance NRMSE2.25299e-6, FP16 dynamic/folded logits exact, chunked CE error1.19949e-8 versus FP64 dense reference. Prior failed diagnostics retained; no thresholds relaxed.

CUDA smoke command currently running via scripts/run_exp0225_stage.py smoke; inspect results/exp0225/commands/smoke_*.log and smoke_complete.json, not chat status. Isolated CUDA venv /home/daniuniu/.cache/qwen3-block-htp-spinquant-py torch2.8cu128 on RTX5070Ti Laptop12GB. CPU venv unchanged. Data bcb67126349e8d086f29619807e9ab7ebba06253792b7fce4e5468f9d39e8280; train800, validation32distinct documents; GPTQ8192 frozen.

After smoke passes: stage train; export step000/step050/step100 and control; select; quality; deploy; quick/full/repeat; quality-report; warmup/short/formal; speed-report. Retain complete evidence/profile and archive before closure. Training checkpoints every10steps contain optimizer and RNG, with --resume recovery; only0/50/100 used for selection. No qbh checkpoint tuning, no other recipe/runtime change, no promotion.
