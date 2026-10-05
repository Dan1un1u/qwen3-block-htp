# EXP-0321: Qwen3-1.7B deployment-matched fakequant
## Scope
Frozen EXP0308 INT16Down package and EXP0318 prefix/RoPE. Same EXP0320 WikiText2 raw test subset:8192targets atpositions64..8255,64warmup perwindow,1984targets perfullwindow,4full+256tail. No chat/BOS,training,calibration,weight changes,runtime changes or hardware rerun.
## Results
| Path | PPL | Mean NLL |
|---|---:|---:|
| Original BF16 teacher | 17.8745976615 | 2.8833805799 |
| Deployment-matched software fakequant | 25.7249487596 | 3.2474612899 |
| Historical actual hardware W4A8-Down16 | 28.4670476589 | 3.3487471958 |

Hardware/fakequant PPL ratio1.10659298 (+10.6593%,absolute2.74210).
Fakequant/teacher PPL ratio1.43919037 (+43.9190%).
NLL teacher->fakequant0.36408071, fakequant->hardware0.10128591, total0.46536662.
The first gap is78.24% of the observed NLL difference arithmetically, not an independent causal percentage attribution; different errors can interact.

## Exact software contract
- Actual exported signed W4 integer codes and per-output-channel FP32 scales, no re-quantizing original weights.
- FP16 exported embeddings/norm gamma/RoPE, FP32 residual.
- Frozen U8 scales/zero points and all live projection/Norm/QK boundaries; no artificial U8 rounding of FP32 O/Down residual updates.
- Integer attention retains K/V carrier clipping, frozen score shift/multiplier, wide_nr64 log2/exponent/probability and AV rounding/saturation; unchanged seed patches K/V at first position of each reset window.
- Frozen Gate/Up U8 and exact LUT output reconstructed as signed INT16; no SP2 substituted for INT16.
- Final RMSNorm quantized to actual U8 grid; actual W4 full-vocabulary head with quantized U8 logits before logsumexp.
- Generic floating-point RMSNorm and QK normalization/RoPE; ideal linear QDQ via FP32 reconstruction and floor(code+0.5)/clamp. This replaces HMX FP16 scale/bias representation and hardware converter behavior, and DSP-specific reduction/rsqrt details.
- FP32 GEMM, TF32 disabled. Integer code GEMMs remain exactly representable within proven bounds. Down uses two bounded digit products then int64 reconstruction to avoid introducing FP32 accumulator loss. This is software arithmetic, not a hardware timing result.

This is deployment-matched fakequant, not a conventional model with standard floating-point softmax and ideal floating SwiGLU. Retaining frozen nonlinear algorithms is essential to avoid mislabeling their removal as hardware error. The measured residual gap cannot by itself be called an implementation bug or attributed exclusively to HMX; projection conversion and normalization arithmetic change together.

## Validation and provenance
595 consumed package/RoPE files checked against retained manifests; manifest itself matches pinned SHA256. Prefix and frozen token file hashes exact.
All packed W4 weights including LM head roundtrip independently to byte-identical original layout.
24 attention tests on representative layer0/13/27,groups0/7,KVlength1/31/67/2047 match independent CPU reference for rawscore,probability,AV. Additional9 tests atKV64/769/2047 verify last3queries against CPU, exercising long reductions.
28layers actual-input Q projection checks against independent FP64 dequantized math: one1-LSB threshold difference total among172032codes; otherwise exact. This is expected FP32 versus FP64 boundary rounding and retained.
All28 sampled Down accumulators/reconstructed outputs match independent FP64 integer dot then FP32 scaling exactly.
All8192 targetIDs,positions and effective context lengths exactly match both hardware and BF16 teacher. All NLLs finite. Independent sum in verification script reproduces reported PPL.
Source HEAD486c3cc5244d6d047988cf00f521485a09f6e968 unchanged. No experimental speed, spreadsheet update, quality acceptance or baseline promotion.

Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0321
Ledger SHA256 bc4dac6b33aef1922eb2b0d7a2e6d6d51bced1446dc51fa4692562897a6dc9f5
