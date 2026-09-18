# L32-0054 1B: paired ordinary A8 and SP2

Five rotated short rounds, ten rotated formal rounds, ten trajectories per formal process. TPS is total tokens divided by mean complete Host wall. Cold loading, tokenization and audit writes excluded. Long runs include Host input/RoPE staging; native M64 is the unchanged short RPC timing contract. All samples retained.

| Shape | A8 prefill TPS | SP2 prefill TPS | SP2 prefill wall overhead | A8 decode TPS | SP2 decode TPS | SP2 decode wall overhead |
|---|---:|---:|---:|---:|---:|---:|
| 64+42 | 2378.88 | 2343.83 | +1.50% | 44.89 | 44.87 | +0.05% |
| 536+46 | 2556.54 | 2503.94 | +2.10% | 37.72 | 37.69 | +0.07% |
| 741+3 | 2544.45 | 2506.66 | +1.51% | 34.17 | 34.22 | -0.16% |

Correctness: 2992 exact layer outputs, 850345984 exact KV elements; all checked heads and poisoned padding valid. 22600 timed RPC boundaries passed numerical status, VTCM8MiB, overlay, no-spill and additive module checks.

Long/M64 throughput gate uses the paired 95% bootstrap lower bound >=0.90. This gate and model-quality acceptance are separate from the correctness of the A8/SP2 implementation. No PPL or named-dataset evaluation is claimed.

| Recipe | Shape | Long/M64 TPS | 95% CI | 10% gate |
|---|---|---:|---|---|
| a8 | 536 | 1.0747 | [1.0713, 1.0778] | pass |
| a8 | 741 | 1.0696 | [1.0665, 1.0726] | pass |
| sp2 | 536 | 1.0683 | [1.0669, 1.0698] | pass |
| sp2 | 741 | 1.0695 | [1.0673, 1.0718] | pass |

Measured source: ab9d7c2d0bf1061d10a945757589fb6df0ef64fc. Only the long-entry representation whitelist was extended to ordinary A8. Frozen W4 codes/scales, non-Down qparams, FP32 residual and no-rotation schedule were retained. Ordinary Down middle qparams/LUT reuse the independently verified prior A8 control, and native64 expected-token files were generated independently for the common 42-step input trajectory.
