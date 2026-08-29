# EXP-0047 complete profiling closure

## Identity and comparison

| Field | Value |
|---|---|
| Experiment | EXP-0047 |
| Branch | `codex/exp-0047-w4f16-w4u8-profile` |
| Source commit | `f8897405621b769f5e32a2690cea7399b215d59e` |
| Execution unit | Real Qwen3 layer-14 complete block, M=64 |
| Direct control | Current-source reproduction of `W4F16-EXP0038-NORMS` |
| Candidate | `W4U8-EXP0046-NATIVE-MLP-IO` |
| Formal comparison | Seven interleaved paired rounds at repeat one and repeat ten |
| Formal evidence | `D:\llm_exp\results\qwen3-block-htp\exp0047\20260829T141901Z_f8897405621b_formal` |
| Retained artifacts | `D:\llm_exp\models\qwen3-block-htp\exp0047\artifacts\f8897405621b` |
| Backend | Standalone FastRPC DSP, one HMX owner, no QNN or alternate CPU path |

This is a paired **current-best versus current-best** characterization on
one source commit and one device boot. It is not an identical-schedule
single-variable comparison: W4F16 and W4U8 intentionally retain their
own accepted arithmetic, Attention, layout, and tuned pipeline plans.
Both paths consume the same package and the same byte-identical W4 values
and scales; the combined W4 bundle digest is `fd016578c132a7917faf79fc939faf88eb2afb4c922313e40794521e32215b22`.

All qtimer, byte, and count values are medians normalized per block.
Host wall is milliseconds per block. Only the additive Block Timing
Ledger may be summed. Engine work and wait counters overlap and are
diagnostic rather than additive.

## Primary latency

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Host wall / block (ms) | 2.6405 | 6.1353 | +132.35% | 2.2195 | 5.7515 | +159.14% |
| DSP block total | 42113.0 | 109016.0 | +158.87% | 41238.0 | 109033.4 | +164.40% |
| Invocation | 42622.0 | 109522.0 | +156.96% | 41288.7 | 109083.8 | +164.20% |
| Runtime setup | 505.0 | 501.0 | -0.79% | 50.3 | 50.4 | +0.20% |
| Runtime teardown | 431.0 | 424.0 | -1.62% | 43.4 | 42.6 | -1.84% |
| Named ledger total | 42614.0 | 109492.0 | +156.94% | 41281.4 | 109057.0 | +164.18% |
| Unattributed ledger | 8.0 | 30.0 | +275.00% | 7.0 | 26.4 | +277.14% |

## Complete additive Block Timing Ledger

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Input | 91.0 | 50.0 | -45.05% | 97.4 | 59.9 | -38.50% |
| Metadata | 227.0 | 119.0 | -47.58% | 22.6 | 11.8 | -47.79% |
| Input RMSNorm | 826.0 | 1738.0 | +110.41% | 820.2 | 1727.5 | +110.62% |
| QKV projection | 8517.0 | 30799.0 | +261.62% | 8368.8 | 31368.1 | +274.82% |
| Q/K Norm + RoPE | 0.0 | 2.0 | N/A | 0.7 | 0.4 | -42.86% |
| Attention | 2708.0 | 7952.0 | +193.65% | 2683.3 | 8028.2 | +199.19% |
| O projection | 3309.0 | 9908.0 | +199.43% | 3315.0 | 9899.4 | +198.62% |
| Post-Attention residual | 791.0 | 2037.0 | +157.52% | 791.2 | 2032.0 | +156.83% |
| Post-Attention norm | 1.0 | 0.0 | -100.00% | 0.6 | 0.1 | -83.33% |
| Gate/Up | 18568.0 | 44910.0 | +141.87% | 18569.2 | 44962.7 | +142.14% |
| Activation | 1.0 | 0.0 | -100.00% | 0.5 | 0.0 | -100.00% |
| Down | 6336.0 | 9441.0 | +49.01% | 6317.7 | 9397.1 | +48.74% |
| Final residual | 95.0 | 1384.0 | +1356.84% | 95.2 | 1382.6 | +1352.31% |
| Output | 235.0 | 130.0 | -44.68% | 23.4 | 12.6 | -46.15% |

At repeat 1, W4U8 adds 66903.0 additive DSP ticks/block.
The five largest positive ledger deltas are Gate/Up +26342.0 (39.4% of the net gap), QKV projection +22282.0 (33.3% of the net gap), O projection +6599.0 (9.9% of the net gap), Attention +5244.0 (7.8% of the net gap), Down +3105.0 (4.6% of the net gap).

At repeat 10, W4U8 adds 67795.4 additive DSP ticks/block.
The five largest positive ledger deltas are Gate/Up +26393.5 (38.9% of the net gap), QKV projection +22999.3 (33.9% of the net gap), O projection +6584.4 (9.7% of the net gap), Attention +5344.9 (7.9% of the net gap), Down +3079.4 (4.5% of the net gap).

## Projection diagnostics (overlapping, not additive)

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DMA ticks | 41417.0 | 11648.0 | -71.88% | 41353.4 | 11500.3 | -72.19% |
| HMX compute ticks, all projections | 4314.0 | 13776.0 | +219.33% | 4353.4 | 13668.1 | +213.96% |
| Projection pack ticks | 116.0 | 20555.0 | +17619.83% | 113.1 | 21115.6 | +18569.85% |
| Projection unpack ticks | 801.0 | 6579.0 | +721.35% | 819.7 | 6577.7 | +702.45% |
| Projection HMX wait ticks | 27013.0 | 211.0 | -99.22% | 27128.0 | 215.3 | -99.21% |
| HMX ready wait ticks | 0.0 | 6072.0 | N/A | 0.0 | 6107.7 | N/A |
| W4F16 expansion wall ticks | 24777.0 | N/A | N/A | 24855.9 | N/A | N/A |
| W4F16 expansion work ticks | 50956.0 | N/A | N/A | 51032.2 | N/A | N/A |
| W4F16 expansion pool wait | 2552.0 | N/A | N/A | 2693.4 | N/A | N/A |
| W4F16 prefetch wait | 1114.0 | N/A | N/A | 1098.1 | N/A | N/A |
| W4F16 cross-prefetch lifetime | 10126.0 | N/A | N/A | 9949.3 | N/A | N/A |
| W4U8 QKVO expansion ticks | N/A | 6352.0 | N/A | N/A | 6380.3 | N/A |
| W4U8 QKVO prefetch wait | N/A | 3112.0 | N/A | N/A | 3105.4 | N/A |
| W4U8 QKVO HMX lifetime | N/A | 9272.0 | N/A | N/A | 9311.2 | N/A |

Variant-specific W4F16 and W4U8 rows are marked N/A on the other path: the measurement does not exist there because the physical scheduler and telemetry family differ. Zero is reserved for a counter that was present and measured zero.

## Attention diagnostics (overlapping, not additive)

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| FP16 Q/K Norm-RoPE worker work | 6480.0 | N/A | N/A | 6337.9 | N/A | N/A |
| FP16 Q/K Norm-RoPE pool wait | 463.0 | N/A | N/A | 307.6 | N/A | N/A |
| FP16 GQA worker work | 10010.0 | N/A | N/A | 9955.0 | N/A | N/A |
| FP16 GQA HMX wait | 1332.0 | N/A | N/A | 1304.3 | N/A | N/A |
| FP16 GQA queue wait | 2147.0 | N/A | N/A | 2018.4 | N/A | N/A |
| U8 Q/K Norm-RoPE work | N/A | 45935.0 | N/A | N/A | 46631.9 | N/A |
| U8 V pack | N/A | 8213.0 | N/A | N/A | 8180.8 | N/A |
| U8 QK HMX | N/A | 1267.0 | N/A | N/A | 1281.2 | N/A |
| U8 QK requant | N/A | 379.0 | N/A | N/A | 387.8 | N/A |
| U8 log2 Softmax | N/A | 13986.0 | N/A | N/A | 13981.5 | N/A |
| U8 AV HMX | N/A | 2491.0 | N/A | N/A | 2537.9 | N/A |
| U8 AV requant | N/A | 1061.0 | N/A | N/A | 1058.5 | N/A |
| U8 Attention pipeline wait | N/A | 3026.0 | N/A | N/A | 3099.0 | N/A |
| FP16 GQA groups | 8.0 | N/A | N/A | 8.0 | N/A | N/A |
| U8 GQA groups | N/A | 8.0 | N/A | N/A | 8.0 | N/A |
| U8 QK executions | N/A | 32.0 | N/A | N/A | 32.0 | N/A |
| U8 AV executions | N/A | 64.0 | N/A | N/A | 64.0 | N/A |

The FP16 GQA pipeline exposes aggregate worker/HMX/queue intervals rather than separate QK, Softmax, and AV timers. Those component cells are N/A, not zero. W4U8 exposes the integer components separately.

## MLP diagnostics (overlapping, not additive)

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| W4F16 Gate/Up weight DMA | 25002.0 | N/A | N/A | 25055.0 | N/A | N/A |
| W4F16 Gate/Up expansion work | 8432.0 | N/A | N/A | 8416.8 | N/A | N/A |
| W4F16 Gate/Up HMX wait | 12551.0 | N/A | N/A | 12561.8 | N/A | N/A |
| W4F16 Gate/Up prefetch wait | 474.0 | N/A | N/A | 450.7 | N/A | N/A |
| W4F16 Gate/Up stream work | 5467.0 | N/A | N/A | 5466.3 | N/A | N/A |
| W4F16 Gate/Up stream ready wait | 56.0 | N/A | N/A | 36.9 | N/A | N/A |
| W4U8 Gate/Up pipeline | N/A | 44265.0 | N/A | N/A | 44311.4 | N/A |
| W4U8 Down pipeline | N/A | 8555.0 | N/A | N/A | 8491.0 | N/A |
| W4U8 activation work | N/A | 8019.0 | N/A | N/A | 7989.4 | N/A |
| W4U8 weight stage | N/A | 8285.0 | N/A | N/A | 8179.0 | N/A |
| W4U8 weight expansion | N/A | 35132.0 | N/A | N/A | 35087.3 | N/A |
| W4U8 MLP HMX compute | N/A | 26056.0 | N/A | N/A | 26085.0 | N/A |
| W4U8 MLP HMX ready wait | N/A | 6930.0 | N/A | N/A | 6983.6 | N/A |
| W4U8 producer slot wait | N/A | 323.0 | N/A | N/A | 313.4 | N/A |
| W4U8 expanded slot wait | N/A | 31999.0 | N/A | N/A | 32116.0 | N/A |
| W4U8 input pack | N/A | 0.0 | N/A | N/A | 0.0 | N/A |
| W4U8 output unpack | N/A | 0.0 | N/A | N/A | 0.0 | N/A |
| W4U8 published pairs | N/A | 192.0 | N/A | N/A | 192.0 | N/A |
| W4U8 consumed pairs | N/A | 192.0 | N/A | N/A | 192.0 | N/A |

## Physical contract

| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DDR read (bytes) | 25296896.0 | 25444352.0 | +0.58% | 25178931.2 | 25444352.0 | +1.05% |
| Boundary DDR read (bytes) | 303616.0 | 304096.0 | +0.16% | 266291.2 | 148374.4 | -44.28% |
| Boundary DDR write (bytes) | 262144.0 | 131072.0 | -50.00% | 26214.4 | 13107.2 | -50.00% |
| Intermediate DDR read (bytes) | 0.0 | 0.0 | +0.00% | 0.0 | 0.0 | +0.00% |
| Intermediate DDR write (bytes) | 0.0 | 0.0 | +0.00% | 0.0 | 0.0 | +0.00% |
| Intermediate DMA descriptors | 0.0 | 0.0 | +0.00% | 0.0 | 0.0 | +0.00% |
| Weight DMA descriptors | 119.0 | 512.0 | +330.25% | 112.7 | 512.0 | +354.30% |
| Spill/fill | 0.0 | 0.0 | +0.00% | 0.0 | 0.0 | +0.00% |
| HMX commands | 208.0 | 1040.0 | +400.00% | 208.0 | 1040.0 | +400.00% |
| FP16 HMX tile pairs | 98816.0 | N/A | N/A | 98816.0 | N/A | N/A |
| U8xS8 HMX tile pairs | N/A | 49408.0 | N/A | N/A | 49408.0 | N/A |
| VTCM requested (bytes) | 8388608 | 8388608 | +0.00% | 8388608 | 8388608 | +0.00% |
| VTCM acquired (bytes) | 8388608 | 8388608 | +0.00% | 8388608 | 8388608 | +0.00% |
| VTCM peak plan (bytes) | 8171008 | 5306080 | -35.06% | 8171008 | 5306080 | -35.06% |
| FastRPC block invocations | 1.0 | 1.0 | +0.00% | 1.0 | 1.0 | +0.00% |

FP16 and U8xS8 HMX tile-pair counts are carrier-specific and are not
one-for-one measures of arithmetic work. The command count, wall intervals,
waits, and complete ledger must be interpreted together.

## Cross-variant structural attribution

| Indicator | repeat one | repeat ten |
|---|---:|---:|
| W4U8 / W4F16 weight DDR bytes | 1.01x | 1.01x |
| W4U8 / W4F16 HMX commands | 5.00x | 5.00x |
| W4U8 / W4F16 projection-pack ticks | 177.20x | 186.70x |
| W4U8 / W4F16 all-projection HMX ticks | 3.19x | 3.14x |
| W4F16 FP16 tile pairs / HMX command | 475.1 | 475.1 |
| W4U8 U8xS8 tile pairs / HMX command | 47.5 | 47.5 |

The shared W4 bundle means activation precision does not reduce the dominant
weight bytes: W4U8 actually reads about the same amount. W4F16 reports much
more DMA work ticks, yet its projection ledger is substantially shorter; its
DMA, expansion, and HMX work are therefore overlapped effectively rather than
serialized on the complete-block path.

The strongest structural difference is command granularity. W4U8 issues five
times as many HMX commands while carrying only half as many carrier-specific
tile pairs in total, or about one tenth as many tile pairs per command. It also
spends roughly 21k ticks/block in projection packing and about 32k overlapping
ticks waiting for expanded MLP slots. These measurements explain why lower
integer arithmetic width does not become lower latency in the current custom
pipeline: scheduling, carrier construction, and fine-grained submission dominate
before the reduced 8x8 multiply work can become visible.

## Correctness and provenance

| Gate | W4F16 | W4U8 | Result |
|---|---:|---:|---|
| Final mismatches | 0 | 0 | pass |
| Final maximum LSB | 0 | 0 | pass |
| Output hash | `f18b9abbe1487231` | `69f22eeb035e5ec5` | pass against each path's own reference |
| Non-finite count | 0 | 0 | pass |
| Mask violations | 0 | 0 | pass |
| Independent integer QK mismatches | N/A | 0 | pass |
| Independent log2-Softmax mismatches | N/A | 0 | pass |
| Independent integer AV mismatches | N/A | 0 | pass |
| Shared W4 values/scales | `fd016578c132a7917faf79fc939faf88eb2afb4c922313e40794521e32215b22` | same package | pass |
| Intermediate DDR read/write | 0/0 | 0/0 | pass |
| Spill/fill | 0 | 0 | pass |
| VTCM requested/acquired | 8 MiB / 8 MiB | 8 MiB / 8 MiB | pass |
| FastRPC execution units / HMX owners | 1 / 1 | 1 / 1 | pass |
| QNN dependency / CPU fallback | false / none | false / none | pass |

W4U8 correctness here means exact execution of the declared local integer
algorithm and stable agreement with its independent QK/Softmax/AV reference.
It does not claim full-model quantization accuracy or equality to W4F16.

## Closure conclusion

W4F16 measures 2.6405 ms/block at repeat one and 2.2195 ms/block at repeat ten.
W4U8 measures 6.1353 and 5.7515 ms/block, respectively,
or +132.35% and +159.14% versus W4F16.
The experiment has no speed gate and makes no baseline-promotion decision.
Its completion gate passes because both paths satisfy their numerical and
physical contracts and every required PC-027 table is closed.


