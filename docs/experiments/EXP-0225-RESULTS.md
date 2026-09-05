# EXP-0225 - Learned offline R1/R2 result

Completed implementation and bounded experiment. Learning improves the actual-export independent validation NLL relative to Hadamard initialization (3.355730 -> 3.266698), but remains worse than unrotated EXP0224 A (3.204854), in both languages. Validation selects step100. Final DSP qbh-lite-v1: 20/24 versus A19/24, conditional PPL50.2245 versus38.8736 (+29.1994%). The predeclared joint NLL/tasks effectiveness gate fails. No baseline promotion; other recipes frozen. This is a valid negative result for this bounded adaptation, not a claim that all learned rotation methods fail.

## Method and separation

Frozen original Qwen3 BF16 checkpoint; no upstream folded weights. Global R1 and28 per-layer shared value-head R2; Hadamard sign initialization; Cayley/Stiefel SGD; W4 signed[-7,7] per-output-row STE, A16. No LPBQ/group scales or online R3/R4. All gamma and rotation products use FP32 before final FP16 rounding.100steps, batch8 (4en/4zh),128tokens, seed225, lr1.5 cosine. Original model weights are frozen; only rotations learn. The100-step valid training took360.314s; peak GPU4,218,659,328bytes. Official SpinQuant reference pinned at8f47aa3f00e8662caf1a484153920a07e5281c3a with license/provenance retained.

800 fixed bilingual training windows,32 validation windows from16 distinct documents per language; document split and32gram exclusion against frozen GPTQ calibration and qbh full/holdout. Data SHA256bcb67126349e8d086f29619807e9ab7ebba06253792b7fce4e5468f9d39e8280. Actual export uses frozen8192-token GPTQ calibration, EXP0224 transformer output-scale selection and absmax GPTQ LM head. Selection checkpoints0/50/100 were fixed before gradients. qbh-lite-v1 (512 conditional targets,24 short tasks,4 open prompts) is final-only; holdout unscored. Validation is distinct from final conditional PPL and is not directly comparable to it. Sequence128 and the STE-to-GPTQ surrogate are declared resource-bounded adaptations; this is not a full reproduction of the paper's training recipe.

## Independent actual-export checkpoint selection

| Actual packed model | Validation NLL | Validation PPL | English NLL | Chinese NLL |
|---|---|---|---|---|
| control_A | 3.204854 | 24.6519 | 3.236187 | 3.173521 |
| step000 | 3.355730 | 28.6665 | 3.441085 | 3.270374 |
| step050 | 3.297332 | 27.0404 | 3.301970 | 3.292694 |
| step100 | 3.266698 | 26.2246 | 3.272592 | 3.260804 |

Selection frozen to step100 before any EXP0225 qbh quality score. All three actual model packages and checkpoints retained. Initial step000 export used the first run's step000 checkpoint; R1/R2 tensors are byte-identical to exact-run initialization, verified in initial_rotation_identity.json. Step050/100 use training_exact.

## Final quality

| DSP implementation | NLL | Conditional PPL | Tasks |
|---|---|---|---|
| F16A16 EXP0218 (frozen) | 3.633608 | 37.8491 | 22/24 |
| control_A | 3.660316 | 38.8736 | 19/24 |
| selected_step100 | 3.916503 | 50.2245 | 20/24 |

| DSP implementation | Chinese PPL | English PPL |
|---|---|---|
| control_A | 54.2530 | 27.8539 |
| selected_step100 | 54.5447 | 46.2465 |

Software selected-step100 NLL3.91720552/PPL50.25979835/20of24 agrees closely with DSP NLL3.91650283/PPL50.22449369/20of24. This is not bitwise software/DSP numerical equivalence. DSP quick/full/repeat overlaps have zero mismatches and repeated cases are identical. English conditional PPL is46.2465 versus A27.8539; Chinese54.5447 versus A54.2530. The final test regression is therefore predominantly English. Short tasks improve by one while likelihood worsens; neither metric alone establishes general quality recovery.

## Implementation and numerical recovery evidence

Independent non-symmetric rotation algebra, STE codes/gradient, output-scale/GPTQ packing and FP64 exact-Cayley update checks pass. All3 exports pass2 full-model FP32 invariance probes,56 independent staged/HF calibration forward checks and197 projection packing roundtrips each. Step100 maximum FP32 invariance NRMSE3.88102264e-6; dynamic/folded FP16 oracle logits exactly equal. Valid100-step training has all finite losses/gradients; final R1/R2 Frobenius Gram errors1.56370487e-5 /3.96988710e-6.

The original official5-iteration approximate Cayley run is preserved as failed evidence: its step050 correlated orthogonality drift failed full-model equivalence despite small entrywise Gram error. The repair uses FP64 exact linear solve for the same tangent, step bound, schedule, data and seed, and strengthens the Gram check with Frobenius norm<1e-4. Full-model thresholds were not relaxed. Earlier BF16 premature rounding, RoPE buffer dtype and low-precision CE oracle defects were corrected with preserved diagnostics; the initial low-document-coverage validation set was replaced before gradients/scoring, with old data retained. See the project-memory experiment record for details. CPU exporters were serialized after measured16-thread contention; scheduling changes affect preparation wall time only.

## Device and profiling

Frozen ABI108 runtime, binary source d981072513d06ed61731c14743c76ac6bc81617f, all1276 deployed package files and binaries SHA-verified. Independent16-token generation matches actual device execution. One warmup,5short and10two-way alternating formal rounds completed.320 full token invocations and8,960 layer ledgers exactly close; exact8MiB VTCM, zero timed intermediate DDR/spill, one FastRPC per invocation, no QNN. Software, quality and speed use the same candidate weights. Runtime source unchanged relative to EXP0224 parentcf8a239a636bb33dd9bf49c7aa03760df9af6c81.

Full numeric profiling: full_profiling_report.md. Module units microseconds, parentheses are each recipe's complete Host wall share. Other recipe columns are frozen historical EXP0218, nonpaired. The current paired A/control comparison is in speed_summary.json: median paired throughput change prefill-0.0582%, decode-0.1720%; no meaningful speed gain is established.

| 模块 | F16A16 冻结 EXP-0218 | W4A16 learned R1R2 GPTQ step100 EXP-0225 | W4A8 冻结 EXP-0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 386.3 (0.61%) | 247.4 (0.63%) | +56.14% |
| Input RMSNorm | 489.7 (0.61%) | 490.9 (0.78%) | 554.0 (1.40%) | -11.38% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11700.5 (18.53%) | 7052.7 (17.82%) | +65.90% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3931.0 (6.22%) | 3214.5 (8.12%) | +22.29% |
| O projection | 5757.7 (7.14%) | 5005.5 (7.93%) | 1256.4 (3.17%) | +298.40% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.3 (0.75%) | 654.0 (1.65%) | -27.62% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22393.7 (35.46%) | 14442.7 (36.49%) | +55.05% |
| Down | 13447.9 (16.67%) | 8628.2 (13.66%) | 3428.5 (8.66%) | +151.66% |
| Final residual | 140.1 (0.17%) | 139.9 (0.22%) | 183.8 (0.46%) | -23.86% |
| KV carrier conversion | 174.0 (0.22%) | 172.7 (0.27%) | 203.2 (0.51%) | -15.03% |
| KV append DMA | 343.6 (0.43%) | 341.7 (0.54%) | 463.9 (1.17%) | -26.33% |
| Block orchestration | 16.1 (0.02%) | 19.2 (0.03%) | 34.6 (0.09%) | -44.58% |
| Layer bookkeeping | 23.9 (0.03%) | 22.9 (0.04%) | 23.2 (0.06%) | -1.01% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 8.0 (0.01%) | 22.5 (0.06%) | -64.24% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 93.5 (0.15%) | 105.6 (0.27%) | -11.44% |
| Embedding | 68.1 (0.08%) | 62.2 (0.10%) | 62.4 (0.16%) | -0.46% |
| Final model RMSNorm | 49.7 (0.06%) | 48.2 (0.08%) | 3.7 (0.01%) | +1193.01% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6761.9 (10.71%) | 5284.7 (13.35%) | +27.95% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2431.1 (3.85%) | 2466.6 (6.23%) | -1.44% |
| 完整 Host wall | 80692.2 (100.00%) | 63152.6 (100.00%) | 39575.9 (100.00%) | +59.57% |

## Direct complete-model E2E

| Recipe | Prefill tokens | Prefill Host us | Prefill tok/s | Decode tokens | Decode Host us | Decode tok/s |
|---|---|---|---|---|---|---|
| F16A16 EXP0218 | 64 | 80692.1875 | 793.1375 | 15 | 1644005.3655 | 9.1241 |
| W4A16 learned step100 | 64 | 63152.5780 | 1013.4186 | 15 | 1395269.1910 | 10.7506 |
| W4A8 EXP0218 | 64 | 39575.9370 | 1617.1443 | 15 | 341462.0840 | 43.9287 |

No block/layer extrapolation. Source branch codex/exp-0225-w4f16-learned-r1r2, final HEAD0900322f71b8cbc6071a3a34e3f23679464bbfeb; step000 export pinned19aac658, step050 pinned89370973, step100 and final evaluation pinned0900322f71b8cbc6071a3a34e3f23679464bbfeb. Stage-specific commands/source heads are retained. Packages/checkpoints/source/environment archives are under D:/llm_exp/models/qwen3-block-htp/exp0225; full results under D:/llm_exp/results/qwen3-block-htp/exp0225.
