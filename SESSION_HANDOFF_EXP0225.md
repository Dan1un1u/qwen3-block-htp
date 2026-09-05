# EXP0225 completed - discuss result

Source branch codex/exp-0225-w4f16-learned-r1r2, HEAD0900322f71b8cbc6071a3a34e3f23679464bbfeb. No active experiment or running/paused jobs. Next available experiment226; user approval required for a new direction. Existing baselines unchanged.

Implementation and100step exact-Cayley recovery complete. All actual exports0/50/100 and independent validation complete; selected step100 before qbh. Final DSP NLL3.916502829899378 /PPL50.22449369429062 /20of24 versus unrotated A3.6603159678909947 /38.87362374795991 /19of24. Validation initialization3.355729741946493 ->step1003.2666980737725555, but A3.2048538800660946 remains better. Correctness/physical/determinism gates pass; local effectiveness fails. This is a valid negative result for this bounded configuration, not a universal rejection of learned rotation.

One warmup,5short and10two-way formal rounds complete. Step100 prefill64tokens/63152.578us =1013.4186tok/s; decode15tokens/1395269.191us =10.7506tok/s.320 invocation and8960 layer ledgers pass. Frozen ABI108 runtime and other recipes unchanged.

Results D:/llm_exp/results/qwen3-block-htp/exp0225; packages/checkpoints/source/env D:/llm_exp/models/qwen3-block-htp/exp0225.334 evidence files and694 intermediates retained. Evidence ledger SHA2561d73989c7a6f4d5d8b91db553db32ee22fa023011f125c4ebe29718234ed95d8; closure SHA256ad8e3d497d1c4da20680bb6c5fb2b825001bc90f243bf61a016ecc302861eaf7. Primary report docs/experiments/EXP-0225-RESULTS.md; full profiling docs/experiments/EXP-0225-PROFILE.md. User profile output: stable three-recipe21-row table then direct E2E tokens/s.

Raw approximate-Cayley training and failed_step050_approx_cayley retained, not eligible. Valid weights live under training_exact except step000 identical raw initialization; identity and stage-specific source provenance verified. Do not rerun completed exploration or overwrite artifacts.
