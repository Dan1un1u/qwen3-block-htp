# EXP-0320: Qwen3-1.7B matched original BF16 teacher
Original checkpoint shard hashes match the retained EXP0221/0223 authority.
Native tokenizer exactly reproduces all 8256 frozen token IDs. Same 8192 scored target positions64..8255, same per-window64warmup, same reset schedule and tail. WikiText2 raw test subset, not full-corpus standard PPL.
GPU Transformers original BF16, SDPA causal attention, unquantized tied embedding/LM head, FP32 logsumexp and target subtraction, FP64 aggregate. No chat/BOS/artificial prefix-KV.
Hardware historical results remain frozen and include static prefix-KV patch, W4 backbone, quantized activations/KV/head, Down16 and FP32 residual. Thus this measures complete deployed-recipe loss relative to original; it does not isolate W4 or rounding. Prefix patch is not an extra scored corpus token.
| Window protocol | Teacher PPL | Hardware PPL | PPL ratio | Delta NLL |
|---|---:|---:|---:|---:|
| 2048 max,1984 targets/window | 17.87459766 | 28.46704766 | 1.592598 | 0.46536662 |
| 64 warmup+43 targets/window | 27.93590920 | 49.62699611 | 1.776459 | 0.57462203 |

All8192 targets and effective corpus context lengths match hardware for both protocols; long5windows includes256targettail,short191windows includes22targettail.
All scores finite. Independent analyzer reconstructs PPL from retained per-token NLL and verifies target/context correspondence.
Causal cached/uncached smoke:16targets,max absoluteNLLdifference0.1196003,mean0.0237993,consistent with BF16 shape-dependent rounding; not bit-exact. Full-window teacher is the declared scoring authority.
A1 stopped before model inference because default environment torch was CPU-only. A2 reused existing spinquant CUDA environment successfully. No dependencies installed.
Corpus tokenizer emits max-length warning when tokenizing entire corpus; actual model windows never exceed2047input tokens and only the frozen first8256IDs are evaluated.
No source/runtime/package/workbook changes, hardware rerun, E2E speed measurement or baseline promotion. Memory optimization deferred.
Interpretation: longcontext PPL is still59.26% above original; absolute PPL28.47 alone would conceal this. Quality loss remains substantial; future work should separate algorithmic quantization loss from hardware arithmetic/boundary loss on this exact protocol.

Evidence: /mnt/d/llm_exp/results/qwen3-block-htp/exp0320
Ledger SHA256: 12c97c9adb317a4c956bbd0115a8643159d27dc10fb65829c2ce66b2c4e1d657
