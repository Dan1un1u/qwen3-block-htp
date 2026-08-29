# EXP-0048 complete profiling closure

## Identity and comparison

| Field | Value |
|---|---|
| Experiment | EXP-0048 Stage B |
| Branch | `codex/exp-0048-w4u8-native-qkv-o-boundaries` |
| Source commit | `61cd86fdabc3c3cf535bf13bfd7f5693fc502b8e` |
| Execution unit | Real Qwen3 layer-14 complete block, M=64 |
| Control | EXP-0048 Stage A: shared native QKV input carrier |
| Candidate | Stage A plus native O-output-to-residual handoff |
| Formal comparison | Seven interleaved paired rounds at repeat one and repeat ten |
| Stable boot | `77ccafe0-8817-4976-bdae-15af5608d4ff` |
| Formal evidence | `D:\llm_exp\results\qwen3-block-htp\exp0048\20260829T145226Z_61cd86fdabc3_stage_b` |
| Retained artifacts | `D:\llm_exp\models\qwen3-block-htp\exp0048\artifacts\61cd86fdabc3\stage_b` |
| Backend | Standalone FastRPC DSP, one HMX owner, no QNN or alternate CPU path |

The candidate changes only the O-projection output boundary. Integer HMX
writes O directly in native tile order; the fused post-Attention residual
gathers those tiles and writes the unchanged native U8 Gate/Up activation
carrier. Arithmetic, W4 values and scales, qparams, Attention, Stage-A QKV
boundary, MLP, and schedules remain fixed. All tick, byte, and count values
below are medians normalized per block. Only the additive Block Timing Ledger
may be summed; engine work and wait counters overlap.

## Primary latency and paired statistics

| Metric | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Host wall (ns/block) | 5,442,969.0 | 5,150,469.0 | -5.374% | 5,030,682.3 | 4,740,999.9 | -5.758% |
| Invocation ticks | 95,964.0 | 90,698.0 | -5.487% | 95,145.1 | 89,601.9 | -5.826% |
| DSP block total | 95,461.0 | 90,197.0 | -5.514% | 95,095.3 | 89,552.1 | -5.829% |
| O output boundary | 11,913.0 | 6,412.0 | -46.176% | 11,912.0 | 6,406.4 | -46.219% |
| Projection unpack | 6,576.0 | 0.0 | -100.000% | 6,576.7 | 0.0 | -100.000% |
| Post-Attention residual | 2,038.0 | 3,127.0 | +53.435% | 2,031.5 | 3,124.6 | +53.808% |
| Runtime setup | 503.0 | 499.0 | -0.795% | 49.8 | 49.8 | 0.000% |
| Runtime teardown | 424.0 | 425.0 | +0.236% | 42.5 | 42.5 | 0.000% |
| Named ledger total | 95,939.0 | 90,674.0 | -5.488% | 95,120.2 | 89,575.9 | -5.829% |
| Unattributed ledger | 27.0 | 26.0 | -3.704% | 25.1 | 25.2 | +0.398% |

| Paired-round change | r1 minimum | r1 median | r1 maximum | r10 minimum | r10 median | r10 maximum |
|---|---:|---:|---:|---:|---:|---:|
| Host wall | -11.801% | -5.311% | +3.513% | -12.738% | -5.849% | -5.547% |
| Invocation | -5.983% | -5.795% | +3.581% | -13.275% | -5.826% | -5.618% |
| DSP block total | -6.008% | -5.821% | +3.603% | -13.284% | -5.829% | -5.621% |
| O output boundary | -46.431% | -46.177% | -45.914% | -47.288% | -46.278% | -46.140% |
| Projection unpack | -100.000% | -100.000% | -100.000% | -100.000% | -100.000% | -100.000% |

## Complete additive Block Timing Ledger

| Metric | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Input | 48.0 | 50.0 | +4.167% | 61.9 | 61.3 | -0.969% |
| Metadata | 120.0 | 121.0 | +0.833% | 12.1 | 12.0 | -0.826% |
| Input RMSNorm | 1,752.0 | 1,751.0 | -0.057% | 1,745.2 | 1,744.8 | -0.023% |
| QKV projection | 17,691.0 | 17,655.0 | -0.203% | 17,729.7 | 17,746.9 | +0.097% |
| Q/K Norm + RoPE | 3.0 | 3.0 | 0.000% | 0.4 | 0.5 | +25.000% |
| Attention | 7,917.0 | 7,904.0 | -0.164% | 8,072.9 | 7,984.2 | -1.099% |
| O projection | 9,876.0 | 3,287.0 | -66.717% | 9,881.2 | 3,281.1 | -66.795% |
| Post-Attention residual | 2,038.0 | 3,127.0 | +53.435% | 2,031.5 | 3,124.6 | +53.808% |
| Post-Attention norm | 0.0 | 0.0 | N/A | 0.1 | 0.0 | -100.000% |
| Gate/Up | 44,705.0 | 44,852.0 | +0.329% | 44,704.4 | 44,652.5 | -0.116% |
| Activation | 0.0 | 0.0 | N/A | 0.0 | 0.0 | N/A |
| Down | 9,395.0 | 9,439.0 | +0.468% | 9,415.0 | 9,413.1 | -0.020% |
| Final residual | 1,383.0 | 1,383.0 | 0.000% | 1,382.8 | 1,382.5 | -0.022% |
| Output | 130.0 | 130.0 | 0.000% | 12.9 | 13.0 | +0.775% |

The O row alone is not the gate because its former row-major unpack has moved
into the residual consumer. The complete O-output boundary is therefore
`O projection + post-Attention residual`: 11,913 to 6,412 ticks at repeat one
and 11,912.0 to 6,406.4 ticks at repeat ten. This interval is mutually
exclusive and includes all moved work.

## Overlapping projection, Attention, and MLP diagnostics

| Metric | r1 control | r1 candidate | Delta | r10 control | r10 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Projection pack | 8.0 | 8.0 | 0.000% | 8.3 | 8.2 | -1.205% |
| Projection unpack | 6,576.0 | 0.0 | -100.000% | 6,576.7 | 0.0 | -100.000% |
| Weight DMA work | 11,498.0 | 11,596.0 | +0.852% | 11,385.2 | 11,331.6 | -0.471% |
| All-projection HMX compute | 13,729.0 | 13,534.0 | -1.420% | 13,536.1 | 13,367.7 | -1.244% |
| QKVO W4 expansion | 6,354.0 | 6,372.0 | +0.283% | 6,388.7 | 6,386.4 | -0.036% |
| QKVO prefetch wait | 3,104.0 | 3,095.0 | -0.290% | 3,075.4 | 3,061.4 | -0.455% |
| QKVO HMX lifetime | 9,269.0 | 9,277.0 | +0.086% | 9,311.4 | 9,275.8 | -0.382% |
| Projection HMX wait | 211.0 | 212.0 | +0.474% | 221.3 | 219.0 | -1.039% |
| HMX ready wait | 6,147.0 | 6,154.0 | +0.114% | 6,121.3 | 6,115.3 | -0.098% |
| Q/K prep pool wait | 10,935.0 | 10,806.0 | -1.180% | 10,845.1 | 10,878.6 | +0.309% |
| U8 Q/K Norm-RoPE work | 44,732.0 | 44,674.0 | -0.130% | 44,728.6 | 44,716.7 | -0.027% |
| U8 V pack | 8,235.0 | 8,215.0 | -0.243% | 8,189.5 | 8,191.9 | +0.029% |
| U8 QK HMX | 1,216.0 | 1,271.0 | +4.523% | 1,272.3 | 1,258.5 | -1.085% |
| U8 QK requant | 387.0 | 389.0 | +0.517% | 386.9 | 381.8 | -1.318% |
| U8 log2 Softmax | 13,925.0 | 13,956.0 | +0.223% | 13,954.3 | 13,972.6 | +0.131% |
| U8 AV HMX | 2,501.0 | 2,484.0 | -0.680% | 2,540.3 | 2,517.5 | -0.898% |
| U8 AV requant | 1,051.0 | 1,076.0 | +2.379% | 1,060.2 | 1,061.6 | +0.132% |
| Attention pipeline wait | 2,900.0 | 2,909.0 | +0.310% | 3,192.1 | 3,119.9 | -2.262% |
| Gate/Up pipeline | 44,061.0 | 44,220.0 | +0.361% | 44,060.8 | 44,005.0 | -0.127% |
| Down pipeline | 8,506.0 | 8,554.0 | +0.564% | 8,520.2 | 8,529.7 | +0.111% |
| MLP activation work | 8,006.0 | 8,039.0 | +0.412% | 7,998.5 | 7,997.2 | -0.016% |
| MLP weight stage | 8,143.0 | 8,236.0 | +1.142% | 8,112.9 | 8,089.1 | -0.293% |
| MLP W4 expansion | 34,896.0 | 35,125.0 | +0.656% | 34,877.1 | 34,774.6 | -0.294% |
| MLP HMX compute | 25,949.0 | 25,994.0 | +0.173% | 25,943.1 | 25,964.5 | +0.082% |
| MLP HMX ready wait | 7,007.0 | 7,036.0 | +0.414% | 6,996.2 | 6,972.5 | -0.339% |
| MLP producer-slot wait | 328.0 | 329.0 | +0.305% | 323.2 | 325.1 | +0.588% |
| MLP expanded-slot wait | 31,723.0 | 32,039.0 | +0.996% | 31,759.5 | 31,823.3 | +0.201% |

These counters overlap and cannot be added to the ledger. Their near equality
shows that the gain is not a hidden reduction in W4 expansion, weight traffic,
Attention arithmetic, or MLP work. The changed critical-path work is the
eliminated O row-major unpack and its replacement by a cheaper consumer-side
native-tile gather.

## Traffic, commands, resources, and residency

| Metric | r1 control | r1 candidate | r10 control | r10 candidate | Result |
|---|---:|---:|---:|---:|---|
| HMX commands/block | 1,040 | 1,040 | 1,040 | 1,040 | unchanged |
| U8xS8 tile pairs/block | 49,408 | 49,408 | 49,408 | 49,408 | unchanged |
| Weight DMA descriptors/block | 512 | 512 | 512 | 512 | unchanged |
| Weight DDR read bytes/block | 25,444,352 | 25,444,352 | 25,444,352 | 25,444,352 | unchanged |
| Boundary DDR read bytes/block | 304,096 | 304,096 | 148,374.4 | 148,374.4 | unchanged |
| Boundary DDR write bytes/block | 131,072 | 131,072 | 13,107.2 | 13,107.2 | unchanged |
| Intermediate DDR read bytes | 0 | 0 | 0 | 0 | pass |
| Intermediate DDR write bytes | 0 | 0 | 0 | 0 | pass |
| Intermediate DMA descriptors | 0 | 0 | 0 | 0 | pass |
| Spill/fill | 0 | 0 | 0 | 0 | pass |
| Direct O Attention tiles | 64 | 64 | 64 | 64 | unchanged |
| QKV unpacks skipped | 128 | 128 | 128 | 128 | unchanged |
| QKV batches | 32 | 32 | 32 | 32 | unchanged |
| QKVO prefetches | 44 | 44 | 44 | 44 | unchanged |
| QKVO overlap schedules | 44 | 44 | 44 | 44 | unchanged |
| MLP input packs skipped | 1 | 1 | 1 | 1 | unchanged |
| MLP output unpacks skipped | 1 | 1 | 1 | 1 | unchanged |
| MLP pairs published/consumed | 192/192 | 192/192 | 192/192 | 192/192 | balanced |
| VTCM requested/acquired bytes | 8,388,608/8,388,608 | same | same | same | pass |
| VTCM peak-plan bytes | 5,306,080 | 5,306,080 | 5,306,080 | 5,306,080 | unchanged |
| Attention HVX workers created/locked | 3/3 | 3/3 | 3/3 | 3/3 | unchanged |
| MLP HVX workers created/locked | 2/2 | 2/2 | 2/2 | 2/2 | unchanged |
| QKV batch N tiles | 4 | 4 | 4 | 4 | unchanged |
| Gate/Up HVX worker telemetry | 3 | 3 | 3 | 3 | unchanged |
| Down HVX worker telemetry | 6 | 6 | 6 | 6 | unchanged |
| Gate/Up HVX-HMX / parallel overlap | 1/1 | 1/1 | 1/1 | 1/1 | unchanged |
| Down HVX-HMX / parallel overlap | 1/1 | 1/1 | 1/1 | 1/1 | unchanged |

## Correctness, provenance, and gate closure

| Gate | Result |
|---|---|
| Final block output | byte-exact, 0 LSB, hash `69f22eeb035e5ec5` |
| Independent integer QK / log2 Softmax / AV reference | pass, zero mismatches |
| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |
| Intermediate DDR read/write | 0 / 0 bytes |
| Spill/fill | 0 |
| FastRPC execution units / HMX owners | 1 / 1 |
| QNN dependency / CPU fallback | false / none |
| Repeat-one Host-wall and O-boundary speed gate | pass |
| Repeat-ten Host-wall and O-boundary speed gate | pass |
| Stage-B overall local gate | pass |
| Adoption | eligible, pending explicit user decision |

| Evidence item | SHA-256 |
|---|---|
| Gate summary | `1be6838a0be657a8bff6f6bd56e1e69660a243e7f4a6f02946585bd2f11cc571` |
| Formal full report | `bb2ab30f4e360218ae0460ec71ea3ec789b3ed36e2937aae06c8a20367149d0c` |
| Implementation reference | `4e1b458d2219b7b47ceed6a52a115b658083322003aa9c84f5e4038343d6824e` |
| Static gate | `31be39e669306b6ca1476da6eb3c99d4a1004d8ddbdb8a0f0248192f73055230` |
| Package manifest | `043f61337b2ad2758bb95791bc4ca768150f457f1f28bc7d20a7ad57eba42359` |

## Closure conclusion

Stage B strictly improves both required repeat scopes and the complete target
boundary while preserving every correctness and physical constraint. Relative
to Stage A it gains 5.37% at repeat one and 5.76% at repeat ten. Relative to
the pre-EXP-0048 W4U8 candidate, the two-stage QKV/O boundary work gains
16.03% and 17.38%. EXP-0048 is complete and locally adoption-eligible, but it
does not change the user-selected W4F16 baseline without explicit promotion.
