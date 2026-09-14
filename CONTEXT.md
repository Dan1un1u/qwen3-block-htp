# L32-0014 completed: Down input A8 repaired; native U8 residual still fails

No active experiment or running job. Next experiment15, no new run approved.
Source /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation.
Closure HEAD b3799364d8be23676b7d12dfe5a394e2d2eec00f; final SDK oracle source e3ec72d248caa79133cc6b66f8e624ffa4f7c7f1.
Frozen rotation branch252aee5ecfbefd4a2df6cd5a5e4b9193d935f86b; Qwen/floating controls and production SP2 defaults untouched.

Read docs/LLAMA32_SMOOTH_R4_ALL_A8_ACCURACY.md and docs/experiments/L32-0014.md.

## Result and interpretation

Fresh original BF16 -> fixed seeded R1/R2 offline -> up-absorbed sqrt(a/w) smooth clip[1/32,32] -> full8192 R4 -> guided static A8 grids -> per-output W4 clipping and token/channel-shrinkage GPTQ. Dense64 R3 after RoPE on both Q/K. All rotations explicit dense matrix multiplication, no butterfly.112 independent frozen A8 input scales,prefix0; differs from old optimized48-shared-scale native input production. No embedding centering or affine/gain optimization updates.8 CE calibration backwards on8x512 train windows, prefix32+64 sampled later positions with frequency balancing.

FullWT2validation2048+948tail/252728targets:
originalBF16 reused13.64087054; transformedBF16 13.64440643; sameW4 DownA16 15.14164350; all112inputA8 15.15255126. Down A8 extra PPL0.072038%. Input-only text: The capital of France is Paris. This is not an overall quantization-quality acceptance.

Matched128 M64+16 windows/2048targets:
input-only26.74915412; extraDownoutputU8 52.07014431; extraresidualU8 18518.31142225; keepDownoutput/residualfloat with other BF16carrier quantizers27.87155044; allBF16carrier diagnostic28297.09165154; SDKinteger51465.45928709. Intermediate BF16 diagnostics retain floatattention/BF16head. SDKinteger includes nativeU8embedding/headW4/logitsU8,RMS/RoPE/residual/log2softmax,KV8 and HMX conversion. Native text unusable.

This is SOFTWARE/SDK simulation of all16layers, hardware runs0. No new DSP R3/R4 implementation, no real-device PPL or speed/profile. Full2048 SDKinteger PPL not run. Do not call it hardware validation. No claim this one failed static U8 residual recipe proves allA8 impossible.

## Validation and failure recovery

Transform vector maxrelativeL2 8.6675e-7; custom floatforward exactly matches HF after RoPE call repair. BF16fold logits relativeL2 .019998 vsoriginal recorded separately; not exact BF16 equivalence. Export audit all112matrices/973078528codes plus scales exact. SDKdot audit113matrices inclhead.
Initial variable-shaped dense FP32 GEMM failed cache equivalence. Final fixed64-row denseGEMM padding passes full80 vs actualM64+15 for all16 positions/all128256 vocabulary codes with zero error. Failed original native sample/log preserved and excluded. Also retain two failed customRoPE attempts and premature controller start before prepare sealing. Main controller stopped at original cachefail; replacement native-a02 completed successfully.

Models /mnt/d/llm_exp/models/llama32-htp/l32-0014. Results /mnt/d/llm_exp/results/llama32-htp/l32-0014.
LedgerSHA256 f19061c6a0e314f6520437b9f52d36dff22b93322789ec890387fcf3a8af62bd;187 result files and256 model files sealed; source files and unchanged read-only reference snapshot verified.
Sessions26511,76735,18408,36209,87076,26141 finished; do not rerun or overwrite.

## Next discussion

Prioritize native residual representation/calibration while holding repaired smooth/R4 weights fixed. Second-layer isolated diagnostic on first heldoutM64: R4 lowers smoothedmiddle peak22.875 to.318359; Downoutput首positionRMS11.0478 vsordinary.07016; residual11.0513 vsordinary.05351. U8 makes96.227% ordinary Downoutput and87.643% residual entrieszero. Equivalent input transforms cannot remove the true large signal afterDown or its propagation alongresidual. No new prefix/dynamic/mixedprecision contract approved or implemented.
