# EXP-0046 complete profiling closure

## Identity and comparison

| Field | Value |
|---|---|
| Experiment | EXP-0046 |
| Branch | `codex/exp-0046-w4u8-mlp-native-boundaries` |
| Stage-A commit | `62c7648ee5d69acdafeb3d8577058747bc83a28e` |
| Final commit | `40da9f06f6a5d8335d0b9a0c45f7efa61525c50f` |
| Execution unit | Real Qwen3 layer-14 complete block, M=64 |
| Project Variant | W4U8, integer U8xS8 HMX projections and integer Attention |
| Direct control | `w4u8_mlp_input`: native RMSNorm-to-Gate/Up input, row-major Down unpack |
| Candidate | `w4u8_mlp_io`: native input plus Down-HMX-to-final-residual handoff |
| Formal comparison | Seven interleaved paired rounds at repeat one and repeat ten |
| Formal evidence | `D:\llm_exp\results\qwen3-block-htp\exp0046\20260829T134655Z_40da9f06f6a5_stage_b` |
| Retained artifacts | `D:\llm_exp\models\qwen3-block-htp\exp0046\artifacts\40da9f06f6a5\stage_b` |
| Backend | Standalone FastRPC DSP, no QNN dependency or alternate CPU fallback path |

All qtimer, byte, and count values below are medians normalized per block.
Host wall is milliseconds per block. The additive Block Timing Ledger is the
only summable timing table. Engine work and wait tables overlap and are
diagnostic rather than additive.

## Primary latency

| Metric | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Host wall / block (ms) | 6.4395 | 6.1339 | -4.75% | 6.0279 | 5.7386 | -4.80% |
| DSP block total | 114216.0 | 108893.0 | -4.66% | 114204.3 | 109002.1 | -4.56% |
| Invocation | 114716.0 | 109398.0 | -4.64% | 114255.7 | 109052.2 | -4.55% |
| Runtime setup | 500.0 | 500.0 | 0.00% | 50.5 | 50.2 | -0.59% |
| Runtime teardown | 427.0 | 426.0 | -0.23% | 42.6 | 42.5 | -0.23% |
| Named ledger total | 108106.0 | 109367.0 | +1.17% | 107651.6 | 109026.2 | +1.28% |
| Unattributed ledger | 6610.0 | 30.0 | -99.55% | 6601.9 | 26.0 | -99.61% |

The control's row-major output unpack was intentionally outside a named model
stage and accounts for almost all of its 6.6k unattributed ticks. The candidate
removes that work but makes the fused gather-residual traversal visible inside
the named final-residual interval. Consequently named time rises while total
and Host wall fall; this is attribution closure, not a regression.

## Complete additive Block Timing Ledger

| Metric (qtimer ticks/block) | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Input | 48.0 | 51.0 | +6.25% | 56.8 | 59.0 | +3.87% |
| Metadata | 117.0 | 120.0 | +2.56% | 12.0 | 11.7 | -2.50% |
| Input RMSNorm | 1738.0 | 1740.0 | +0.12% | 1727.0 | 1727.7 | +0.04% |
| QKV projection | 30706.0 | 30704.0 | -0.01% | 31381.4 | 31354.3 | -0.09% |
| Q/K Norm + RoPE | 3.0 | 2.0 | -33.33% | 0.4 | 0.5 | +25.00% |
| Attention | 8028.0 | 8215.0 | +2.33% | 8006.5 | 8033.4 | +0.34% |
| O projection | 9890.0 | 9898.0 | +0.08% | 9899.2 | 9896.7 | -0.03% |
| Post-Attention residual | 2037.0 | 2037.0 | 0.00% | 2032.0 | 2032.2 | +0.01% |
| Post-Attention norm | 0.0 | 0.0 | 0.00% | 0.2 | 0.3 | +50.00% |
| Gate/Up | 44737.0 | 44706.0 | -0.07% | 44766.8 | 44999.7 | +0.52% |
| Activation | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| Down | 9391.0 | 9360.0 | -0.33% | 9418.8 | 9383.6 | -0.37% |
| Final residual | 291.0 | 1383.0 | +375.26% | 290.6 | 1382.6 | +375.77% |
| Output | 125.0 | 123.0 | -1.60% | 12.5 | 12.7 | +1.60% |

Post-Attention norm is fused into the preceding residual/RMSNorm pass.
Activation is fused into the streaming Gate/Up handoff. Q/K Norm and RoPE run
inside the integer-Attention pipeline and are shown in the overlapping
Attention diagnostics below.

## Projection diagnostics (overlapping)

| Metric (per block) | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DMA ticks | 11559.0 | 11618.0 | +0.51% | 11584.4 | 11431.5 | -1.32% |
| HMX compute ticks, all projections | 13559.0 | 13813.0 | +1.87% | 13621.5 | 13787.4 | +1.22% |
| Projection pack ticks | 20523.0 | 20503.0 | -0.10% | 21109.8 | 21100.9 | -0.04% |
| Projection unpack ticks | 13158.0 | 6577.0 | -50.02% | 13153.7 | 6577.8 | -49.99% |
| Projection HMX wait ticks | 207.0 | 212.0 | +2.42% | 215.3 | 218.4 | +1.44% |
| HMX ready wait ticks | 5971.0 | 6047.0 | +1.27% | 6065.5 | 6070.5 | +0.08% |
| QKVO W4 expand ticks | 6354.0 | 6352.0 | -0.03% | 6380.2 | 6385.7 | +0.09% |
| QKVO prefetch wait ticks | 3100.0 | 3104.0 | +0.13% | 3113.5 | 3115.0 | +0.05% |
| QKVO HMX lifetime ticks | 9249.0 | 9269.0 | +0.22% | 9318.3 | 9321.9 | +0.04% |
| QKV batches | 32.0 | 32.0 | 0.00% | 32.0 | 32.0 | 0.00% |
| QKVO prefetches | 44.0 | 44.0 | 0.00% | 44.0 | 44.0 | 0.00% |
| QKVO overlap schedules | 44.0 | 44.0 | 0.00% | 44.0 | 44.0 | 0.00% |

## Attention diagnostics (overlapping)

| Metric (per block) | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Q/K norm + RoPE work ticks | 45801.0 | 45800.0 | 0.00% | 46610.4 | 46591.3 | -0.04% |
| Q/K norm pool wait ticks | 3413.0 | 3400.0 | -0.38% | 3399.1 | 3394.1 | -0.15% |
| V pack ticks | 8127.0 | 8167.0 | +0.49% | 8168.6 | 8176.7 | +0.10% |
| QK HMX ticks | 1275.0 | 1299.0 | +1.88% | 1271.8 | 1285.6 | +1.09% |
| QK requant ticks | 387.0 | 379.0 | -2.07% | 385.0 | 384.5 | -0.13% |
| log2 Softmax ticks | 13996.0 | 13952.0 | -0.31% | 13979.5 | 13988.7 | +0.07% |
| AV HMX ticks | 2515.0 | 2580.0 | +2.58% | 2535.4 | 2536.3 | +0.04% |
| AV requant ticks | 1044.0 | 1061.0 | +1.63% | 1057.8 | 1061.1 | +0.31% |
| Attention pipeline wait ticks | 3450.0 | 3114.0 | -9.74% | 3171.7 | 3180.1 | +0.26% |
| GQA groups | 8.0 | 8.0 | 0.00% | 8.0 | 8.0 | 0.00% |
| QK executions | 32.0 | 32.0 | 0.00% | 32.0 | 32.0 | 0.00% |
| AV executions | 64.0 | 64.0 | 0.00% | 64.0 | 64.0 | 0.00% |

The generic FP16/GQA Attention counters are measured zero because this variant
uses the U8-specific counters shown above; they are not unavailable data.

## MLP diagnostics (overlapping)

| Metric (per block) | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Gate/Up pipeline ticks | 44097.0 | 44066.0 | -0.07% | 44120.6 | 44337.6 | +0.49% |
| Down pipeline ticks | 8489.0 | 8482.0 | -0.08% | 8522.8 | 8495.9 | -0.32% |
| SwiGLU activation work ticks | 7980.0 | 8010.0 | +0.38% | 8011.5 | 8009.9 | -0.02% |
| Weight stage ticks | 8314.0 | 8163.0 | -1.82% | 8298.3 | 8147.0 | -1.82% |
| W4 expand ticks | 35153.0 | 34975.0 | -0.51% | 35165.5 | 34815.7 | -0.99% |
| MLP HMX compute ticks | 26225.0 | 26099.0 | -0.48% | 26201.7 | 25999.8 | -0.77% |
| MLP HMX ready wait ticks | 6829.0 | 6926.0 | +1.42% | 6940.8 | 6940.8 | 0.00% |
| Producer slot wait ticks | 322.0 | 312.0 | -3.11% | 316.2 | 317.7 | +0.47% |
| Expanded slot wait ticks | 32002.0 | 31813.0 | -0.59% | 32012.6 | 32050.2 | +0.12% |
| Input pack ticks | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| Output unpack ticks | 6581.0 | 0.0 | -100.00% | 6575.8 | 0.0 | -100.00% |
| Published Gate/Up pairs | 192.0 | 192.0 | 0.00% | 192.0 | 192.0 | 0.00% |
| Consumed Gate/Up pairs | 192.0 | 192.0 | 0.00% | 192.0 | 192.0 | 0.00% |

Gate/Up and Down each report observed HVX-HMX overlap and parallel-HVX overlap
in both variants. The candidate changes no MLP HMX commands, weight traffic,
expansion schedule, or publication/consumption count.

## Physical contract

| Metric (per block unless static) | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DDR read (bytes) | 25444352.0 | 25444352.0 | 0.00% | 25444352.0 | 25444352.0 | 0.00% |
| Boundary DDR read (bytes) | 304096.0 | 304096.0 | 0.00% | 148374.4 | 148374.4 | 0.00% |
| Boundary DDR write (bytes) | 131072.0 | 131072.0 | 0.00% | 13107.2 | 13107.2 | 0.00% |
| Intermediate DDR read (bytes) | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| Intermediate DDR write (bytes) | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| Intermediate DMA descriptors | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| Weight DMA descriptors | 512.0 | 512.0 | 0.00% | 512.0 | 512.0 | 0.00% |
| Spill/fill | 0.0 | 0.0 | 0.00% | 0.0 | 0.0 | 0.00% |
| HMX commands | 1040.0 | 1040.0 | 0.00% | 1040.0 | 1040.0 | 0.00% |
| U8xS8 tile pairs | 49408.0 | 49408.0 | 0.00% | 49408.0 | 49408.0 | 0.00% |
| VTCM requested (bytes) | 8388608 | 8388608 | 0.00% | 8388608 | 8388608 | 0.00% |
| VTCM acquired (bytes) | 8388608 | 8388608 | 0.00% | 8388608 | 8388608 | 0.00% |
| VTCM peak plan (bytes) | 5306080 | 5306080 | 0.00% | 5306080 | 5306080 | 0.00% |
| FastRPC block invocations / block | 1.0 | 1.0 | 0.00% | 1.0 | 1.0 | 0.00% |
| Attention HVX workers created/locked | 3/3 | 3/3 | unchanged | 3/3 | 3/3 | unchanged |
| MLP pool HVX workers created/locked | 2/2 | 2/2 | unchanged | 2/2 | 2/2 | unchanged |
| Gate/Up configured HVX workers | 3 | 3 | unchanged | 3 | 3 | unchanged |
| Down configured HVX workers | 6 | 6 | unchanged | 6 | 6 | unchanged |
| FastRPC execution units | 1 | 1 | unchanged | 1 | 1 | unchanged |
| HMX ownership domains | 1 | 1 | unchanged | 1 | 1 | unchanged |
| QNN dependency | false | false | unchanged | false | false | unchanged |
| CPU fallback | N/A | N/A | no alternate path | N/A | N/A | no alternate path |
| Intermediate residency | VTCM | VTCM | unchanged | VTCM | VTCM | unchanged |

Boundary traffic includes the legal block input/output plus Prepared Runtime
Session work and is normalized by repeat count; fixed setup is therefore more
amortized at repeat ten. It is identical between paired variants.

## Correctness and numerical gates

| Gate | r1 control | r1 candidate | r10 control | r10 candidate | Result |
|---|---:|---:|---:|---:|---|
| Final mismatches | 0 | 0 | 0 | 0 | pass |
| Final maximum LSB | 0 | 0 | 0 | 0 | pass |
| Output hash | `69f22eeb035e5ec5` | `69f22eeb035e5ec5` | `69f22eeb035e5ec5` | `69f22eeb035e5ec5` | pass |
| Non-finite count | 0 | 0 | 0 | 0 | pass |
| Probability mask violations | 0 | 0 | 0 | 0 | pass |
| Fused K operand mismatches | 0 | 0 | 0 | 0 | pass |
| Independent QK mismatches | 0 | 0 | 0 | 0 | pass |
| Independent log2-Softmax mismatches | 0 | 0 | 0 | 0 | pass |
| Independent AV mismatches | 0 | 0 | 0 | 0 | pass |

The independent Attention audit is unchanged from the parent and is retained
inside the formal evidence directory. EXP-0046 proves hardware implementation
equivalence to the accepted W4U8 experiment reference; it does not establish
full-model quantization accuracy.

## Stage gates and contextual comparisons

| Comparison | repeat one | repeat ten | Pairing |
|---|---:|---:|---|
| Stage A MLP-input boundary | -12.26% | -12.27% | seven-round paired |
| Stage A Host wall | -7.75% | -5.27% | seven-round paired |
| Stage B MLP-output boundary | -33.94% | -33.89% | seven-round paired |
| Stage B Host wall | -4.75% | -4.80% | seven-round paired |
| Final EXP-0046 vs EXP-0045 parent Host wall | -11.81% | -9.83% | non-paired, cross-run context |
| Final EXP-0046 vs selected W4F16 baseline Host wall | +130.44% | +157.37% | non-paired, context only |

The final candidate is therefore a valid local speed improvement over the
EXP-0045 W4U8 parent, but it remains far slower than the user-selected W4F16
baseline. Promotion is a user decision and is not implied by the local pass.

## Closure conclusion

Stage A and Stage B both pass. EXP-0046 removes both synchronous row-major MLP
layout boundaries without changing bytes, HMX work, DMA work, qparams, or model
semantics. The final candidate reduces complete Host wall to 6.1339 ms at
repeat one and 5.7386 ms at repeat ten. The remaining additive critical path is
dominated by Gate/Up (41.3% of repeat-ten DSP total), QKV projection (28.8%),
O projection (9.1%), Down (8.6%), and Attention (7.4%).
