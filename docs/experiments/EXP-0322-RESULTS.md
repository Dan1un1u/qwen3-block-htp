# EXP-0322: Qwen3-1.7B module floating fallback PPL
## Protocol and interpretation
Host CUDA-only diagnostics on EXP0321 deployment-matched fakequant. All22distinct configurations use identical8192WikiText2rawtest targets64..8255, max2048context,64warmup,4x1984+256targets. Same W4codes/scales, U8/INT16 package, FP32residual and frozenprefix except explicitly restored sites. No calibration/training/hardware/runtime edits.
Control PPL25.7249487596,NLL3.2474612899 reproduces EXP0321 exactly pertoken in all3stages. Original BF16teacher17.8745976615 and hardware28.4670476589 are historical matched references; this is not a hardware fallback speed/quality measurement.
Original floating weights are original BF16 checkpoint values evaluated in FP32, not newly optimized weights. Every module fallback applies acrossall28layers unless head-only. All unchanged quantized boundaries remain active.
Singleton gains are conditional and nonadditive. Stage2/3 floating-boundary diagnostics necessarily use the FP32attentioncore as common parent. A19.91PPL K fallback is not proof that simply changing K storage while retaining the current integer core achieves19.91.

## Main module fallbacks
| Arm | PPL | NLL reduction vs control | Meaning |
|---|---:|---:|---|
| control | 25.72494876 | 0.00000000 | EXP0321 unchanged |
| attention_full_fp32 | 19.04957768 | 0.30041636 | Original QKV/O weights and float inputNorm output, QKV/QKRoPE/KV/core/concat; keep prefix seed and MLP |
| mlp_full_fp32 | 23.70817088 | 0.08164154 | Original Gate/Up/Down, float postNorm output, projections and SwiGLU/Down input; attention unchanged |
| qkv_w16 | 24.45606613 | 0.05058300 | Only Q/K/V weights restored original BF16 values, all activation grids unchanged |
| down_w16 | 24.68711819 | 0.04117971 | Only Down weight original; frozen INT16 LUT input retained |
| head_full_fp32 | 24.85513322 | 0.03439699 | Original tied head with float finalNorm output and logits; backbone control |
| post_norm_a16 | 25.06674336 | 0.02591929 | Only postattention RMSNorm output removes A8; W4 Gate/Up and output grids unchanged |
| gateup_w16 | 25.20714026 | 0.02033399 | Only Gate/Up weights original; input/output grids and LUT unchanged |
| o_w16 | 25.42149685 | 0.01186614 | Only O weight original; A8 concat retained |
| input_norm_a16 | 25.58981412 | 0.00526690 | Only input RMSNorm output removes A8; W4 QKV and output grids unchanged |
| swiglu_down_input_fp32 | 25.86732789 | -0.00551941 | Exact FP32 SiLU/product on dequantized U8 Gate/Up, no INT16/LUT on Down input; W4Down retained |
| head_w16 | 25.18772129 | 0.02110466 | Original tied head weight only; keep U8 finalNorm/logits |
| head_grids_fp32 | 25.23939052 | 0.01905540 | W4 head retained; remove finalNorm/logit grids |

## Attention refinement
| Arm | PPL | Parent / retained boundaries |
|---|---:|---|
| attention_core_fp32 | 22.67381472 | FP32 QK-softmax-AV from dequantized fixed Q/K/V, quantize concat once; prefix unchanged |
| attention_boundaries_fp32 | 19.91327756 | FP32 attention plus no QKV projection-output/QK-RoPE/KV grids; W4 QKV and concat grid retained; nested versus attention_core |
| qk_projection_grid_off | 22.97150976 | FP32core plus remove Q/K projection output grids only; postRoPE Q/K grids remain |
| qk_rope_grid_off | 19.92944869 | FP32core plus remove Q/K postRoPE and K-cache grids only; projected Q/K grids remain |
| q_rope_grid_off | 22.84011306 | FP32core plus remove only Q postRoPE A8 grid; K/V unchanged |
| k_rope_grid_off | 19.91250679 | FP32core plus remove only K postRoPE/K-cache A8 grid; Q/V unchanged |
| v_grid_off | 22.80741723 | FP32core plus remove V projection/KV grid only |
| concat_grid_off | 22.69732293 | FP32core plus remove AV-concat grid only, W4O consumes floatcontext |
| softmax_only_fp32 | 25.67821228 | Baseline integer rawscore encoding retained; replace log2exponent/NR64 probability with exp(rawscore * multiplier *ln2/8) softmax quantizedtoU8; integerV/AV unchanged |

## Findings
1. Complete Attention fallback25.72495->19.04958 is much larger than completeMLP23.70817 or completehead24.85513.
2. FP32core on the SAME dequantized QKV gives22.67381. Removing just postRoPE K/cache quantization on that parent gives19.91251; removing justQ gives22.84011. Removing bothQK gives19.92945. BothQK/Vprojection and postRoPE/KV boundary restoration together gives19.91328. These near19.91variants are not ranked as robustly different.
3. K-boundary benefit appears inall5windows: FPcore[13.982,24.217,28.376,25.627,39.252] ->Kfloat[12.126,20.962,26.073,22.446,30.581]. This is strongest evidence for prioritizing K postRoPE staticA8 grid rather than upstream QKVoutput grid.
4. Softmax-only exact exponential/normalization on unchanged rawscoreencoding and integerV/AV gives25.67821, only0.04674PPL improvement; window signs mixed. This does not reproduce wholeFPcore benefit. It tests log2 exponent/NR64 versus true exp on encoded scoreunits(sm*ln2/8), not a combined repair of QKscore range/scales and AVcarrier/requantization.
5. FP32SwiGLU/product and Downinput with frozenW4Down gives25.86733, slightly worse thancontrol. CurrentINT16Down is not the primary bottleneck on this subset. DownW16 alone24.68712 indicates weight error remains, but less benefit than attention activation/core.
6. Next precision investigation should fix K grid/range/resolution first and independently split score encoding versus Vcarrier/AV scaling in the current integercore. No new implementation direction is executed or baseline promoted here.
7. Fakequant25.72->hardware28.47 numerical gap fromEXP0321 remains a separate unsolved attribution; these fallbacks diagnose deployed algorithm/boundary loss, not prove a hardware bug.

## Validation
Original checkpoint shard hashes match EXP0320.595consumed package/RoPE files plus prefix/tokens verified at eachstage. ExportedW4 inverse physical packing roundtrips exactly. Original/quantized projection shapes and coordinate similarity checked for196matrices; original gamma castFP16 equals package inalllayers; sampledembedding rows exact and originalhead equals tiedembedding.
FP32causal GQA attention checked against independent NumPyFP64 small reference(maxabs<2e-6). Baseline component arithmetic validation reused from sealedEXP0321.
All32evaluations (22distinct configurations plus10 repeated control/head/core arms) have8192finiteNLLs and exact frozen target,window,position,context masks; threecontrolruns reproduce every EXP0321NLL exactly; all repeated arms agree exactly.
Details/per-window scores:all-results.json/csv, stageanalysis.json, pertokenrows*.json and scoring-verification.json.
No intermediate activation/weight files exported. No workbook modification, device run, performance measurement, hardwarerecipe change or quality acceptance.

Evidence: /mnt/d/llm_exp/results/qwen3-block-htp/exp0322
Ledger SHA256: 8a3c582edfa50904ce86ab0910cff8d170a6ce9cc2afa67b81c836508fe88980
