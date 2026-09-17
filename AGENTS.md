# EXP0285 closure
Bootstrap/read authoritative Qwen project-memory before further work; preflight before any new stateful experiment.
EXP0285 tested packed FP16 residual (QBH_FP32_RESIDUAL=3) on latest no-rotation nativeW4 SP2 Qwen3-1.7B, preserving FP32 add/Norm. Singlelayer/chain3 and independent full28 hardware correctness pass, no PPL/model quality evaluation.
Original sealed EXP0284 FP32 baseline remains faster and is NOT replaced. Final original-binary paired M64+15 result1847.468951/47.151639 versus FP16 1636.518853/46.840157 token/s; Host overhead12.890172percent prefill and0.664987percent decode. Prefill fails10percent gate. New same-binary FP32 control also incurs common-dispatch codegen overhead; its supplementary9.52percent result is not acceptance against the actual original baseline.
Candidate residual storage is genuinely262144bytes, with262144bytes reserved padding keeping downstream HMX addresses fixed after compact-layout HMX faults. TotalVTCM plan8365824 unchanged. No automatic baseline/quality promotion. Llama and other recipes unchanged.
See docs/EXP-0285-RESULTS.md, docs/EXP-0285-PROFILE.md and /mnt/d/llm_exp/results/qwen3-block-htp/exp0285/FINAL_REPORT.md.
