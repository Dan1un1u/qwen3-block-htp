# L32-0033 softmax interpretation

The retained Llama log2 path is slower than the new FP32 HVX vector path under the frozen contract. Fixed-token full-model FP/LOG2 Host wall is 0.94964867 (95% CI 0.94791243–0.95164266) prefill and 0.98853313 (0.98712262–0.98995377) decode. This does not support a universal log2 speed advantage. Llama retains per-layer original division settings (selected layer7 EXACT), unlike Qwen wide NR64; no cross-model kernel-equivalence claim.

32 independent FP component cases and 32 retained integer cases pass. FP probability max absolute error 2.19360e-7 and row mass error 2.23630e-7 versus Float64; all masked outputs zero, all finite, final U8 equals own FP32 half-up rounding, no Float64 threshold differences. Full selected layer7, chain3, chain16 references are independent; all16-layer greedy and fixed-token CPU predictions match hardware token/logit codes through all rounds. This is implementation evidence, not PPL/model-quality acceptance.

Component formal mean (4 heads): M64/past0 LOG2 175.7707 us, FP 105.4477 us; M1/past64 LOG2 3.5429 us, FP 2.1653 us. Core includes native pack and matched rowmass telemetry; input restore is outside both timers. Separate pack timing N/A. These component numbers are not full-model throughput.

On the same selected-layer QKV, changed probability approximation gives AV max difference 7 U8 codes prefill and 6 decode (RMSE 0.7695/0.8212). Per-component/per-layer correctness checks compare each arm to its own reference, never require cross-arm byte equality. Changed token trajectories are why fixed-input and greedy full-model measurements are reported separately.

Pre-device reference a01 used the wrong AV zero-point placement when multiplier=1, caught and corrected in a02 before any device test or timing. Both preparations retained. FP runtime uses vector probability-code rowmass and caller-owned 512 B VTCM scratch; vendor exp header SHA and no vector stack-spill disassembly recorded. Prototype Qwen scalar rowmass does not enter Llama denominator. No numerical gate was loosened, no timing rerun was selected by speed.

All 9600 timed profiles: same HMX work and weight DDR bytes, no tensor DDR intermediates/spill, 8 MiB VTCM. Untimed incomplete FARF timeline from earlier phases remains N/A. No baseline promotion. Remaining approved campaign work is A8 supported additional context length and Qwen A9 closure.
