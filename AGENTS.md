# Qwen3 EXP-0267 source instructions

This branch backports the user-confirmed Llama SP2 method to Qwen3.
Authority: /home/daniuniu/work/qwen3-block-htp-project-memory. Run its bootstrap
and preflight --source-worktree /home/daniuniu/work/qwen3-block-htp before work.
PC081 explicitly supersedes the earlier Qwen research freeze for this migration.
Preserve historical EXP0265 and both Llama branches/artifacts. New Qwen models
and results use D:/llm_exp/{models,results}/qwen3-block-htp/exp0267. No QNN.
Use scripts/build_qwen3_sp2.sh; compile Qwen dimensions and Q/K normalization,
never Llama weights/tokenizer/calibration. All old tools remain provenance.
Routine owned fixes proceed under PC037 with failed evidence retained; no
threshold relaxation, automatic quality promotion, reset/clean or force push.
