# EXP-0289 — Qwen3-0.6B A16 provisional speed baselines

Completed user-authorized speed measurement, with full floating-reference gate failures explicitly retained. No PPL/quality acceptance or automatic Selected promotion.
Branch codex/exp-0289-qwen3-06b-a16. Measured source b28fde2500e2422dca77e65d85ca48df0eded57c; restored native-equivalent build428faef; all four rebuilt binaries SHA256-identical.
Primary M64+42/cache128/batch1, full28, identical fixed token trajectory. Five short and ten rotated formal rounds, repeat10;8600formal token passes. Ten-round means:
| Recipe | Prefill tok/s | Decode tok/s |64-token Host ms|42-step Host ms|
|---|---:|---:|---:|---:|
|W16A16|1710.935856|27.675677|37.406429|1517.578036|
|W4A16|1678.352049|25.645407|38.132643|1637.720167|
|W4A8-SP2 EXP0288, historical nonpaired|3020.427339|92.402097|21.189055|454.535139|

Includes embedding/all layers/final norm/head/greedy/FastRPC; excludes cold loading/tokenizer/ADB/audit. W4A16 does not outperform W16A16 on this untuned small-model path: larger QKV time offsets smaller head cost; overlapping expansion counters diagnose but do not sum to wall.
Both recipes use existing HMX FP16, HVX vector kernels, basic DMA/compute overlap and593 head prefetches. Decode softmax is vectorized; no slow per-element scalar exp/div fallback. No claim of exhausted optimization or native W4 HMX use in W4A16: W4 weights are expanded for floating HMX.
Hardware checks: selected0/14/27 andchain3 pass,8MiB grant,peaks5260032/6342144B,zero explicit timed intermediateDDR/spill,zero unattributed ledger; all timed outputs exact to own audited token/logit sequence. KV prefix/append/capacity/finite checks pass4816 snapshots. Independent final norm/head checks pass86 audited boundaries, all head argmax match.
Full floating alignment remains FAILED: original fast max hidden NRMSE0.0065323461/0.0054536203 vs0.003. User explicitly permits provisional speed evidence; no failure relabeled, no quality/implementation-equivalence overclaim.
Full report: /mnt/d/llm_exp/results/qwen3-block-htp/exp0289/FULL_PROFILING_REPORT.md
Modules: /mnt/d/llm_exp/results/qwen3-block-htp/exp0289/MODULES.md
Models: /mnt/d/llm_exp/models/qwen3-block-htp/exp0289/f16f16 and reference-a03/w4f16

## Restored executable closure
After timing, the unsuccessful SIMD FP32-intermediate trial was removed from the current source. All native source/include/CMake files match measured b28fde2 exactly. Rebuild at428faef produces SHA256-identical qwen3_block_cli, llama_sp2_cli, libqwen3_probe.so and libqwen3_probe_skel.so. Both restored executables pass43-step token/logit regression and physical checks. Both original fast recipes greedily output "The capital of France is Paris." before EOS. This functional example is not PPL acceptance.
The FP32-intermediate trial is preserved in history and archived binaries, not selected. No deeper schedule tuning was done. Decode still pays fixed-size HMX tiles/common processing overhead; vectorized does not mean all single-row specialization has been exhausted.


Evidence ledger: 639035334c0afe83b0b606af279ff108e37e3eba775064e1df0692087bbc081b;13019 files. Source closure 7e36dcabbec07270934e443e2d4ae3664ba4f210. Numerical gate stays failed; authorized speed scope completed.
