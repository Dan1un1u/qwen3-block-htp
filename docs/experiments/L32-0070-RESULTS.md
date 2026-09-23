# L32-0070 full-model HVX butterfly R4 paired E2E

Llama-3.2-1B-Instruct, ordinary W4A8 (no SP2 or INT16 Down), FP32 residual, R3 off. Full H8192 after prequantization SwiGLU, before one frozen symmetric U8 Down boundary. Source 51ded69018c3b5e20ddcfc590d7a32f050019ef0.

**Result:** Full-model prefill wall increases18.829% (throughput decreases15.846%); decode wall increases1.653% (throughput decreases1.626%). Prefill exceeds the descriptive10% wall reference; decode remains inside it. Butterfly core alone is2.616ms prefill and0.160ms/decode step, so the result does not support a core slower than complete inference. End-to-end overhead is measured directly and includes the required boundary work.

## E2E

Same sealed binary; fixed 64+42 trajectory; five short plus ten formal alternating AB/BA rounds, ten measured replays per CLI. Times include embedding, all16 blocks, final norm, LM head, greedy and FastRPC; exclude cold loading, external tokenizer and numerical audit copies. Fixed tokens remove EOS/trajectory confounding; this is a speed/correctness experiment, not a PPL or usable-text result.

| Phase | Ordinary A8 wall ms | + R4 wall ms | A8 token/s | + R4 token/s | Wall increase | Ratio 95% CI |
|---|---:|---:|---:|---:|---:|---|
| prefill | 26.682105 | 31.706155 | 2398.61 | 2018.54 | 18.829% | [1.1845644806997457, 1.1915942000879711] |
| decode | 905.507073 | 920.475943 | 46.38 | 45.63 | 1.653% | [1.0128789545297858, 1.0194872423627956] |

Decode wall above sums42 generated-token steps. Module and subcounter decode below is per step.

## Rotation boundary wall, all16 layers

| Phase | Native FP16 -> FP32 preparation (ms) | FP32 butterfly core (ms) | U8 quantization + native stores (ms) | Sum (ms) |
|---|---:|---:|---:|---:|
| prefill | 0.377226 | 2.616159 | 1.217913 | 4.211298 |
| decode | 0.032044 | 0.159800 | 0.073834 | 0.265678 |

These are elapsed wall intervals, including dispatch/join where used, not sums of worker time. Prefill uses four persistent HVX contexts; decode one. SwiGLU FP16 production remains overlapped with Gate/Up and its changed cost is in Gate/Up, outside these three R4 windows. Their sum is not the measured E2E difference. Mode4 reuses dense_r4_layout_ticks for the core; dense_r4_matmul_ticks and HMX rotation calls remain zero.

## Correctness and interpretation

Original checkpoint SHA pinned by llama_reference.EXPECTED. All16 Down matrices refolded from original BF16 through FP64 normalized Sylvester H8192, then independent per-output RTN W4[-7,7]; all other weights unchanged. FP64 folding invariance, packed-code inverse and signed24 bound checks pass. Symmetric middle scales derive from each original frozen absolute range, with no fit to evaluated tokens. This does not claim ideal-equivalence after W4/A8 quantization or model quality.

Independent actual-arithmetic oracle: staged FP32 butterfly, FP16 LUT and declared U8 rounding. Single1/chain3/full16 all43 boundaries give860 exact layer hashes per arm; full16 padding-poison adds688 per arm. R4 final hidden payloads also compared bytewise; full16 prefill K/V and final-norm U8 payloads match the independent reference, including padding-poison repeats. All cache length/append bookkeeping checks pass. Every timed head/token agrees with its own oracle. 8600 formal token invocations pass physical/additive ledger checks; HMX command counts match between arms. 8MiB grant, peak [8098272] bytes; intermediateDDR reads/writes/spills zero. Rotation is HVX only.

Implementation: streamed FP16 SwiGLU producer, vector format conversion, shared L32-0069 full8192 FP32 butterfly in phase-dead VTCM, vector U8 native stores. No scalar tensor loop or dense HMX Hadamard. Existing no-rotation baseline remains default; no promotion. Results characterize this implementation and M64+42 on1B, not all models or arbitrary context lengths.

## prefill module profile

Microseconds (share of complete Host wall).

| Module | A8 | A8 + R4 |
|---|---:|---:|
| I/O、metadata | 128.657 (0.48%) | 128.691 (0.41%) |
| Input RMSNorm | 1306.639 (4.90%) | 1340.211 (4.23%) |
| QKV＋RoPE | 3867.073 (14.49%) | 3850.484 (12.14%) |
| QK–Softmax–AV | 6187.990 (23.19%) | 6196.895 (19.54%) |
| O projection | 1208.264 (4.53%) | 1206.334 (3.80%) |
| Post-attention residual＋RMSNorm | 1338.283 (5.02%) | 1306.196 (4.12%) |
| Gate/Up + SwiGLU + optional R4 | 5455.989 (20.45%) | 10530.936 (33.21%) |
| Down | 2627.024 (9.85%) | 2622.559 (8.27%) |
| Final residual | 2.611 (0.01%) | 2.438 (0.01%) |
| KV carrier conversion | 29.673 (0.11%) | 29.739 (0.09%) |
| KV append DMA | 116.911 (0.44%) | 116.989 (0.37%) |
| Block orchestration | 25.599 (0.10%) | 25.439 (0.08%) |
| Layer bookkeeping | 12.695 (0.05%) | 12.878 (0.04%) |
| Stage-boundary bookkeeping | 6.187 (0.02%) | 6.264 (0.02%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 92.475 (0.35%) | 93.851 (0.30%) |
| Embedding | 65.256 (0.24%) | 65.221 (0.21%) |
| Final model RMSNorm | 55.507 (0.21%) | 55.454 (0.17%) |
| LM head＋greedy（不含 final norm） | 3305.094 (12.39%) | 3260.820 (10.28%) |
| Host-DSP boundary | 850.179 (3.19%) | 854.754 (2.70%) |
| Complete Host wall | 26682.105 (100.00%) | 31706.155 (100.00%) |

## decode module profile

Microseconds (share of complete Host wall).

| Module | A8 | A8 + R4 |
|---|---:|---:|
| I/O、metadata | 126.565 (0.59%) | 126.560 (0.58%) |
| Input RMSNorm | 873.973 (4.05%) | 873.914 (3.99%) |
| QKV＋RoPE | 1664.744 (7.72%) | 1665.839 (7.60%) |
| QK–Softmax–AV | 5465.559 (25.35%) | 5464.782 (24.94%) |
| O projection | 881.057 (4.09%) | 879.446 (4.01%) |
| Post-attention residual＋RMSNorm | 878.929 (4.08%) | 879.968 (4.02%) |
| Gate/Up + SwiGLU + optional R4 | 4766.952 (22.11%) | 5161.172 (23.55%) |
| Down | 2593.638 (12.03%) | 2597.325 (11.85%) |
| Final residual | 1.125 (0.01%) | 1.051 (0.00%) |
| KV carrier conversion | 49.983 (0.23%) | 49.992 (0.23%) |
| KV append DMA | 93.466 (0.43%) | 93.193 (0.43%) |
| Block orchestration | 18.279 (0.08%) | 17.791 (0.08%) |
| Layer bookkeeping | 9.333 (0.04%) | 9.346 (0.04%) |
| Stage-boundary bookkeeping | 1.263 (0.01%) | 1.262 (0.01%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.162 (0.24%) | 51.189 (0.23%) |
| Embedding | 1.578 (0.01%) | 1.580 (0.01%) |
| Final model RMSNorm | 55.961 (0.26%) | 56.024 (0.26%) |
| LM head＋greedy（不含 final norm） | 3306.982 (15.34%) | 3262.225 (14.89%) |
| Host-DSP boundary | 719.142 (3.34%) | 723.434 (3.30%) |
| Complete Host wall | 21559.692 (100.00%) | 21916.094 (100.00%) |

Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0070. SUMMARY.json contains additive module profiles and rotation subcounters; FOLD_VALIDATION.json, FINAL_HIDDEN_CHECKS.json, BOUNDARY_CHECKS.json and POSTFLIGHT.json record gates. EVIDENCE_LEDGER.json seals all local result files. Package manifests pin model bytes, which are stored separately under models/llama32-htp/l32-0070. No workbook update requested.
