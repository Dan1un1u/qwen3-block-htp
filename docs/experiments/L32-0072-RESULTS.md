# Full-model HVX R4 paired comparison

64+42, same binary per model, fixed trace, five short and ten formal paired repeat10; R3 off, ordinaryA8; no model-quality claim

|Model|A8 prefill tps|R4 prefill tps|Prefill wall increase|A8 decode tps|R4 decode tps|Decode wall increase|
|---|---:|---:|---:|---:|---:|---:|
|Qwen3-0.6B|3055.64|2590.48|17.96%|93.75|90.66|3.41%|
|Qwen3-1.7B|1858.75|1584.25|17.33%|47.39|46.27|2.42%|
|Llama3.2-1B|2398.61|2018.54|18.83%|46.38|45.63|1.65%|
|Llama3.2-3B|1250.14|1077.68|16.00%|24.06|23.50|2.39%|

## R4 component wall
All layers summed; decode normalized per step. Core is normalized Hadamard, includes no HMX matrix calls. Preparation/quantization are separate subsets of Activation.

|Model|Phase|Prepare us|Core us|Quantize us|Total us|Mean single-layer total us|Core/A8 E2E|
|---|---|---:|---:|---:|---:|---:|---:|
|Qwen3-0.6B|prefill|285.67|1363.31|841.83|2490.80|88.96|6.51%|
|Qwen3-0.6B|decode|25.60|80.29|48.69|154.59|5.52|0.75%|
|Qwen3-1.7B|prefill|510.35|2892.18|1614.55|5017.08|179.18|8.40%|
|Qwen3-1.7B|decode|43.92|173.74|96.96|314.62|11.24|0.82%|
|Llama3.2-1B|prefill|377.23|2616.16|1217.91|4211.30|263.21|9.80%|
|Llama3.2-1B|decode|32.04|159.80|73.83|265.68|16.60|0.74%|
|Llama3.2-3B|prefill|659.31|4577.61|2131.19|7368.11|263.15|8.94%|
|Llama3.2-3B|decode|56.25|279.60|129.25|465.10|16.61|0.67%|

## Interpretation and limits
- R3 experiment L32-0071 is excluded because its Q/K boundary implementation differs. No R3 conclusions from this table.
- Llama1B retains the original full L32-0070 five short/ten formal paired experiment, verified against the authority-pinned ledger. Other models are newly measured.
- Qwen widths are not powers of two. Full H12×H256/H512 is orthogonal; HVX butterfly plus vector12-way sign-add stage, not padding or small-block substitution.
- Current implementation measures the integrated boundary including FP16 SwiGLU preparation, full FP32 transform and frozen ordinary U8 quantization. It is not an intrinsic lower bound or model-quality evaluation.
- All formal comparisons use matching flags and nativeW4 arithmetic outside R4. New Down weights are folded from original checkpoints and freshly per-channel quantized.
- Qwen1.7 and Llama3B reuse phase-dead Gate space for Down input to stay within8MiB. Qwen1.7 retains its full per-layer KV atlas. Llama3B QKV wide batches and prefetch are preserved. Down weight prefetch cannot occupy butterfly scratch simultaneously, and that integration cost is included.
- Speed measures include embedding/all layers/final norm/head/greedy/FastRPC, excluding cold load/external tokenizer. Source formal rounds and exact boundary evidence remain in per-project result roots.

## Complete profiling archive
- [Llama3.2-3B full profiling](L32-0072-Llama3_2-3B-FULL_PROFILING.md)

Workbook: M_R4蝶形消融 appended as final sheet; original12 sheets and styles preserved. Receipt SHA256 9c9cb3f01dbc9a1618d53ed0a73fbb54158908a081e84aaed8c5fdbb11513399.
