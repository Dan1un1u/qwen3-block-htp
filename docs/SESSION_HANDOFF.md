# L32-0014 running: smooth dense R4 all-A8 precision

Source /home/daniuniu/work/llama32-htp, branch codex/llama32-no-rotation. Approved source implementation is tools/llama32_smooth_accuracy.py, tools/llama32_smooth_native.py and tools/run_llama32_smooth_pipeline.py. Latest source392b7ca.

Preparation completed: original6hashes verified; fresh seed42 signed Hadamard R1/R2 fold; no embedding mean centering; full8192 dense R4 and up-absorbed smooth exponent0.5 clip[1/32,32]. Transform audits max8.6675e-7; custom float model exactly matches HF after fixing its RoPE call. Folded BF16 logits differ from original by relativeL2 .019998; this is recorded, not exact BF16 equivalence. Preserve failed prepare-a01/a02 (custom RoPE rounding divergence), successful prepare-a03. A premature controller attempt pipeline-a01.log failed before launching any child because prepare manifest was still sealing; preserve.

Active durable controller session76735, log results/l32-0014/pipeline-a02.log; individual stages in pipeline-a01/. collect completed8 CE backwards, fit running. Do not rerun/overwrite stages. Models /mnt/d/llm_exp/models/llama32-htp/l32-0014; results /mnt/d/llm_exp/results/llama32-htp/l32-0014. Original rotation-quant is read-only; exact local method snapshot build/l32-0014/reference with manifest in results.

Controller continues fit -> carrier calibration -> train-only RTN/GPTQ selection diagnostics -> input-only bridge/full -> residual-only/down-only/all-carrier BF16 bridge -> native package -> exact SDK integer oracle bridge/generation. All rotations explicit dense matmul. Native oracle is HOST/GPU simulation, not device; default SP2, Qwen and floating branches untouched. Extra transformed-float full/bridge reference and independent arithmetic/contract review remain to do. At most10 affine updates only if needed within approved precision question; no claim all gains come from down. No formal performance task.

Important: BF16 all-carrier diagnostic still uses float attention and BF16 head and is explicitly an intermediate ablation; authoritative full-A8 path is SDK integer oracle with native W4 head/logits, U8 KV, integer softmax/RMS/RoPE/residual plus dense temporary R3/R4. Do not claim device validation or global infeasibility from this one candidate.

## 2026-09-14 later checkpoint

All calibration/GPTQ stages complete. Source e3ec72d248caa79133cc6b66f8e624ffa4f7c7f1. Main controller76735 ended with failure only in initial native cache-equivalence check; all prior stage results valid. Original unrestricted FP32 dense GEMM dispatch produced batch-shape rounding differences. Preserve pipeline-a01/native-bridge.log and native-sample-000.json from that failed attempt; do not use its PPL.

Repair fixes R3/R4 to explicit64-row dense FP32 GEMM tiles with zero padding for small rows, no butterfly. Native-a02 fixed arithmetic passed all128256 vocabulary codes at all16 targets: full80 vs actualM64+15 are exactly equal. Native evaluation currently session18408, log native-a02.log, per-sample results native-a02/, cache-equivalence.json inside it. No real device invocation.

Completed same-weight PPL:
- train-only selection RTN17.87967305,GPTQ17.33661188 (not heldout report).
- input-only A8 bridge26.74915412, fullWT2validation15.15255126.
- extra residual-U8 only bridge18518.31142225.
- extra Down-outputU8 only bridge52.07014431.
- BF16 all-carrier diagnostic bridge28297.09165154; retains floating attention/BF16 head, NOT full native contract.

Pending after native-a02 finishes: run tools/llama32_smooth_diagnostics.py for transformed-float bridge/full, same-weight Down-A16 full, except-residual BF16 ablation, boundary diagnostics and input-control text. Then review exact integer export/provenance, report source/authority and evidence closure. No affine/gain updates yet, no speed/profile or quality promotion.
