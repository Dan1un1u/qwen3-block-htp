# EXP0225 active — actual checkpoint export

Source 19aac658379b1518b7792460ed128dc8d40db28b on codex/exp-0225-w4f16-learned-r1r2. All implementation/math/full-model/CE gates and GPU smoke pass.100steps completed; training_complete.json and training_checkpoint_sha256.json retain evidence. PeakGPU4237730304bytes; final R1orth1.73120e-5,R2orth9.91016e-6. Source/environment archive models/exp0225/artifacts/19aac658379b1518b7792460ed128dc8d40db28b.

Two actual-GPTQ exports currently running: step000 and step050 through run_exp0225_stage.py; inspect commands/*.log and their process state before resuming. Do not launch duplicates or overwrite partial packages. step100, control validation, checkpoint select, final software quality, deploy/quick/full/repeat, quality-report, warmup/5short/10formal/speed-report remain. Memory available was verified before parallel exports. No qbh scores inspected for selection.

Dataset hash bcb67126349e8d086f29619807e9ab7ebba06253792b7fce4e5468f9d39e8280; train800,validation32distinct documents, GPTQ8192 frozen. Initial low-document-coverage data retained and never trained/scored. Prior numerical diagnostics retained; see experiment record for BF16/RoPE/reference-precision fixes. Other recipes/runtime frozen. No promotion.
