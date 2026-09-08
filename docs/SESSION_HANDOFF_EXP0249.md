# EXP0249 live checkpoint

Bootstrap and read authority first. Active EXP0249, branch codex/exp-0249-integer-attention-attribution, source 0b85aebb96e9a3ff06e667ec3365d41d69c7dd15. No source/device changes outside scripts. EXP0248 build caches remain experimental layer0 ABI114; not full model.

Numerical reference and four legacy development reproductions pass. Frozen fresh256 docs/4096targets and12 configurations, no calibration or parameter change. GPU final scoring was RUNNING at checkpoint (attention_exp0249.py final; final_attempt1.log). Check process before resuming; never launch a second GPU model process. Python /home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0249. Resume script can reuse completed scores against freeze hash if previous process has exited. Required numerical/causal/CE thresholds unchanged.

Initial freeze attempt asserted every probability scale1/255, but26layers have slight prior clipping. Corrected actual exporter semantics without changing params; original protocol_at_dataset_freeze.md plus amended authority are independently hashed. Failed attempt retained. Actual integer normalizer uses255 and AV retains frozen probability scale. PPL is conditional software attention, not whole DSP. All12 arms remain frozen despite negative results.

Remaining: finish final scoring; run scripts/trace_exp0249.py (same first4 exposed development inputs, must reproduce prior NLL/top1); run scripts/report_exp0249.py; run evidence-local plot_exp0249.py with analysis Python; add interpretation/NEXT_DIRECTION and local Softmax/hardware range analyses to report; verify_integrity.py; seal_exp0249.py; close_memory_exp0249.py after reviewing final report. These evidence-local helpers are prepared but not yet run. Preserve logs/evidence; no hardware, weights, thresholds or promotion changes.

Completed scores at checkpoint (provisional until final integrity closure):

{
  "final_C64": 31.288773041634308,
  "final_OFF_carrier": 46.87527982439223,
  "final_OFF_exponent": 7461.603531223536,
  "final_OFF_legacy": 42.15276953523591,
  "final_OFF_score": 7705.742457053464,
  "final_OFF_sole": 11039.825161999797,
  "final_R3_carrier": 38.290143466076955,
  "final_R3_legacy": 35.21662438189657
}
