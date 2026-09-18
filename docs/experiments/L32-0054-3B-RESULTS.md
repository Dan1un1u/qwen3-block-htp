# L32-0054 3B: paired ordinary A8 and SP2

Five rotated short rounds, ten rotated formal rounds, ten trajectories per formal process. TPS is total tokens divided by mean complete Host wall. Cold loading, tokenization and audit writes excluded. Long runs include Host input/RoPE staging; native M64 is the unchanged short RPC timing contract. All samples retained.

| Shape | A8 prefill TPS | SP2 prefill TPS | SP2 prefill wall overhead | A8 decode TPS | SP2 decode TPS | SP2 decode wall overhead |
|---|---:|---:|---:|---:|---:|---:|
| 64+42 | 1206.28 | 1183.81 | +1.90% | 23.69 | 23.68 | +0.05% |
| 536+46 | 1174.79 | 1149.87 | +2.17% | 23.79 | 23.76 | +0.13% |
| 741+3 | 1192.46 | 1169.48 | +1.96% | 23.16 | 23.18 | -0.08% |

Correctness: 5236 exact layer outputs, 2976210944 exact KV elements; all checked heads and poisoned padding valid. 22600 timed RPC boundaries passed numerical status, VTCM8MiB, overlay, no-spill and additive module checks.

Long/M64 throughput gate uses the paired 95% bootstrap lower bound >=0.90. This gate and model-quality acceptance are separate from the correctness of the A8/SP2 implementation. No PPL or named-dataset evaluation is claimed.

| Recipe | Shape | Long/M64 TPS | 95% CI | 10% gate |
|---|---|---:|---|---|
| a8 | 536 | 0.9739 | [0.9724, 0.9751] | pass |
| a8 | 741 | 0.9885 | [0.9866, 0.9899] | pass |
| sp2 | 536 | 0.9713 | [0.9703, 0.9724] | pass |
| sp2 | 741 | 0.9879 | [0.9869, 0.9889] | pass |

Measured source: ab9d7c2d0bf1061d10a945757589fb6df0ef64fc. Only the long-entry representation whitelist was extended to ordinary A8. Frozen W4 codes/scales, non-Down qparams, FP32 residual and no-rotation schedule were retained. Ordinary Down middle qparams/LUT reuse the independently verified prior A8 control, and native64 expected-token files were generated independently for the common 42-step input trajectory.
