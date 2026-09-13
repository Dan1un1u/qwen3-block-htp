# Qwen3 EXP-0268 source instructions

This branch backports shared prefill scheduling to bit-exact Qwen3 U8.
Authority: /home/daniuniu/work/qwen3-block-htp-project-memory. Run its bootstrap
and preflight --source-worktree /home/daniuniu/work/qwen3-block-htp before work.
PC082 explicitly supersedes the earlier Qwen research freeze for this comparison.
Preserve historical EXP0265 and both Llama branches/artifacts. New Qwen models
and results use D:/llm_exp/{models,results}/qwen3-block-htp/exp0268. No QNN.
Use scripts/build_qwen3_sp2.sh; compile Qwen dimensions and Q/K normalization,
never Llama weights/tokenizer/calibration. All old tools remain provenance.
Routine owned fixes proceed under PC037 with failed evidence retained; no
threshold relaxation, automatic quality promotion, reset/clean or force push.
