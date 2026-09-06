# EXP0227 completed - discuss bounded reconstruction result

Source codex/exp-0227-w4f16-block-scale-reconstruction, HEAD 462fe94dee1ba568ba23b073cec45e2c021558e8, clean and synchronized. EXP0227 completed, evidence valid, local effectiveness fail, adoption pending. No active experiment or running/paused jobs. Next228; baselines unchanged.

Frozen GPTQ codes and fixed coordinates A=EXP0224 A, R=EXP0225 step100. Learn only573440 FP32 transformer row scales per coordinate,100Adam updates per28blocks. Same frozen8192 training and independent64Wiki/news validation.0/50/100 block checkpoint selection; exact final code*scale FP16 forward/export, no new GPTQ pass. Other recipes, embedding/head/norms/rotations/runtime frozen.

DSP qbh-lite-v1: A NLL3.654829994/PPL38.66094796/tasks18of24 versus A0 38.87362375/19of24. R NLL3.761609446/PPL43.01760480/tasks19of24 versus R0 50.22449369/20of24. Both PPLs improve but one strict task is lost; no joint effectiveness. Independent validation A3.541112453->3.548497452, R3.561694003->3.562356519. Presentation-only ranking selects A, using independent NLL, never qbh; both candidates evaluated. Software A38.73994495/18 and R43.10295860/19 agree qualitatively; not bit-exact DSP logits.

Numerical recovery: unscaled FP16 MSE backward erased real block3 Q/K/Gate/Up gradients; original fullA/partialR preserved training_unscaled/smoke_unscaled. Standard GradScaler65536 fixes backward precision, forward/objective/data/budget/thresholds unchanged. Corrected5600updates finite, no overflow retries.392selected GPU/CPU expanded-weight equality checks, fresh R FP32 invariance NRMSE2.84e-6/2.46e-6. Quick8NLL/full16 prefix audit corrected without rerunning measurements or changing thresholds. All recovery evidence retained.

1warmup5short10four-way formal complete,640invocation/17920layer ledgers pass. A prefill64/63034.0885us=1015.323637tok/s, decode15/1387262.447us=10.812662tok/s. R64/63159.4795us=1013.307907tok/s,15/1388433.0725us=10.803546tok/s. Paired changes near zero, no speed win.

Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0227; models same models root.494evidence files/370intermediates. Ledger SHA256 82ea0e55c64554c51949cf772526ac48e529a30d14bd35384e870a66b9e8c5a2; closure SHA256 00e1cd0c48ff112c33e7bb66db016052456880b3429d9b6fbce6dd2f8f7baf43; report SHA256 e3ddf2ddd892b9f1d054d00c17b758f7c39e144d3ea599bc13153736cb0431b8. Authority reports docs/experiments/EXP-0227-RESULTS.md and EXP-0227-PROFILE.md. Do not overwrite closure or rerun completed jobs.

Discuss next direction before another experiment. Scale-only block MSE provides limited correction under frozen integer codes and does not guarantee final-logit/behavior fidelity. No next method is authorized automatically; no return to corpus changes or rotation retraining by assumption.
