# EXP-0045 complete profiling comparison

The direct control is the Stage-A-selected `qkv_batch4`; the candidate is
`qkvo_batch4`. Values are seven-round medians per block. Tick counters for
repeat ten are divided by ten. Percentage is candidate versus control. The
formal evidence is
`D:\llm_exp\results\qwen3-block-htp\exp0045\20260829T130320Z_25ae2137841b_stage_b`
from source `25ae2137841b66d6b9e69796892c62558814cd3f`.

## Primary latency

| Metric | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Host wall, ns/block | 7,249,740 | 6,752,969 | -6.852% | 6,562,739.5 | 6,370,666.6 | -2.927% |
| DSP block total, ticks | 124,626 | 120,714 | -3.139% | 124,299.2 | 121,159.7 | -2.526% |
| Invocation, ticks | 125,172 | 121,210 | -3.165% | 124,349.6 | 121,210.0 | -2.525% |
| Runtime setup, ticks | 504 | 500 | -0.794% | 50.2 | 50.1 | -0.199% |
| Runtime teardown, ticks | 426 | 426 | 0.000% | 42.8 | 42.9 | +0.234% |
| Named ledger, ticks | 112,005 | 108,044 | -3.536% | 111,183.2 | 108,043.6 | -2.824% |
| Unattributed ledger, ticks | 13,166 | 13,166 | 0.000% | 13,166.1 | 13,166.4 | +0.002% |

## Complete additive Block Timing Ledger

These intervals are mutually exclusive and may be added.

| Interval, ticks/block | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Input stage | 49 | 49 | 0.000% | 58.3 | 57.8 | -0.858% |
| Metadata stage | 121 | 119 | -1.653% | 12.0 | 11.9 | -0.833% |
| Input RMSNorm | 1,660 | 1,660 | 0.000% | 1,648.5 | 1,648.3 | -0.012% |
| QKV projection | 30,686 | 30,698 | +0.039% | 31,340.4 | 31,397.5 | +0.182% |
| QK Norm/RoPE standalone | 0 | 0 | measured zero | 0.2 | 0.2 | 0.000% |
| Attention | 7,943 | 7,924 | -0.239% | 8,042.5 | 8,026.4 | -0.200% |
| O projection | 13,393 | 9,882 | -26.215% | 13,347.8 | 9,899.8 | -25.832% |
| Post-Attention residual | 1,937 | 1,940 | +0.155% | 1,934.9 | 1,934.5 | -0.021% |
| Post-Attention norm standalone | 0 | 0 | measured zero | 0.1 | 0.1 | 0.000% |
| Gate/Up | 45,152 | 44,909 | -0.538% | 45,045.0 | 45,110.6 | +0.146% |
| Activation standalone | 0 | 0 | measured zero | 0 | 0 | measured zero |
| Down | 9,449 | 9,439 | -0.106% | 9,394.6 | 9,460.3 | +0.699% |
| Final residual | 291 | 291 | 0.000% | 290.9 | 290.9 | 0.000% |
| Output stage | 124 | 128 | +3.226% | 12.7 | 12.8 | +0.787% |

## Projection diagnostics

The following counters may overlap and must not be summed into the ledger.

| Counter, ticks/block | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DMA | 12,708 | 11,592 | -8.782% | 12,480.0 | 11,359.8 | -8.976% |
| HMX compute | 14,099 | 13,856 | -1.724% | 13,950.6 | 13,930.3 | -0.146% |
| Projection pack | 27,095 | 27,089 | -0.022% | 27,692.9 | 27,701.3 | +0.030% |
| Projection HMX wait | 2,335 | 212 | -90.921% | 2,403.8 | 217.8 | -90.939% |
| Projection unpack | 13,173 | 13,150 | -0.175% | 13,173.6 | 13,150.0 | -0.179% |
| HMX ready wait | 6,076 | 6,098 | +0.362% | 6,103.4 | 6,123.3 | +0.326% |
| Q/K prep pool wait | 3,397 | 3,423 | +0.765% | 3,374.7 | 3,391.8 | +0.507% |
| QKVO W4 expansion | 4,252 | 6,356 | +49.483% | 4,268.3 | 6,388.0 | +49.661% |
| QKVO prefetch wait | 2,049 | 3,081 | +50.366% | 2,054.8 | 3,105.9 | +51.153% |
| QKVO HMX lifetime | 6,105 | 9,231 | +51.204% | 6,179.9 | 9,315.0 | +50.731% |

The last three counters increase because the candidate instruments QKV plus O,
whereas the control instruments QKV only. The additive O interval and Host wall
are the performance authorities. The decisive diagnostic change is the roughly
91% reduction in synchronous projection-HMX wait.

## Attention diagnostics

These are overlapping engine-work counters.

| Counter, ticks/block | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Q/K Norm-RoPE work | 45,879 | 45,823 | -0.122% | 46,462.2 | 46,591.0 | +0.277% |
| K pack | 0 | 0 | measured zero | 0 | 0 | measured zero |
| V pack | 8,122 | 8,221 | +1.219% | 8,177.8 | 8,187.2 | +0.115% |
| QK HMX | 1,267 | 1,231 | -2.841% | 1,268.4 | 1,268.6 | +0.016% |
| QK requant | 397 | 389 | -2.015% | 384.4 | 388.1 | +0.963% |
| Integer Softmax | 13,996 | 14,015 | +0.136% | 14,003.9 | 13,991.8 | -0.086% |
| AV HMX | 2,487 | 2,459 | -1.126% | 2,510.9 | 2,505.8 | -0.203% |
| AV requant | 1,045 | 1,053 | +0.766% | 1,057.9 | 1,060.9 | +0.284% |
| Attention pipeline wait | 2,930 | 2,866 | -2.184% | 3,166.4 | 3,167.2 | +0.025% |

## MLP diagnostics

These are overlapping engine-work counters.

| Counter, ticks/block | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Gate/Up pipeline | 44,498 | 44,262 | -0.530% | 44,391.5 | 44,472.0 | +0.181% |
| Down pipeline | 8,561 | 8,556 | -0.058% | 8,492.4 | 8,570.7 | +0.922% |
| Activation work | 7,999 | 8,031 | +0.400% | 8,034.2 | 8,040.4 | +0.077% |
| Weight staging | 8,168 | 8,192 | +0.294% | 8,136.8 | 8,121.3 | -0.190% |
| W4 expansion | 34,949 | 35,044 | +0.272% | 34,851.4 | 34,801.6 | -0.143% |
| HMX compute | 26,163 | 26,316 | +0.585% | 26,132.5 | 26,136.5 | +0.015% |
| HMX ready wait | 6,932 | 6,949 | +0.245% | 6,969.6 | 6,985.2 | +0.224% |
| Producer-slot wait | 335 | 328 | -2.090% | 326.7 | 326.8 | +0.031% |
| Expanded-slot wait | 32,285 | 32,161 | -0.384% | 32,096.8 | 32,250.1 | +0.478% |

## Pipeline and task counts

These integer counts are identical at repeat one and after normalization per
block at repeat ten unless the table shows a change.

| Counter per block | Control | Candidate |
|---|---:|---:|
| QKV batch width, output tiles | 4 | 4 |
| QKV batch publications | 32 | 32 |
| QKVO prefetch publications | 29 | 44 |
| QKVO overlap schedules | 29 | 44 |
| Integer-Attention GQA groups | 8 | 8 |
| QK HMX executions | 32 | 32 |
| AV HMX executions | 64 | 64 |
| Direct Attention-to-O tiles | 64 | 64 |
| QKV unpack operations skipped | 128 | 128 |
| MLP Gate/Up pair publications | 192 | 192 |
| MLP Gate/Up pair consumptions | 192 | 192 |
| Attention auxiliary HVX workers created/locked | 3 / 3 | 3 / 3 |
| MLP auxiliary HVX workers created/locked | 2 / 2 | 2 / 2 |
| Gate/Up logical HVX contexts | 3 | 3 |
| Down logical HVX contexts | 6 | 6 |
| Gate/Up and Down HVX-HMX overlap flags | 1 / 1 | 1 / 1 |

## Physical contract and traffic

| Counter per block | r1 control | r1 candidate | r1 delta | r10 control | r10 candidate | r10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Weight DDR read, bytes | 25,444,352 | 25,444,352 | 0.000% | 25,444,352 | 25,444,352 | 0.000% |
| Weight DMA descriptors | 608 | 512 | -15.789% | 608 | 512 | -15.789% |
| Boundary DDR read, bytes | 304,096 | 304,096 | 0.000% | 148,374.4 | 148,374.4 | 0.000% |
| Boundary DDR write, bytes | 131,072 | 131,072 | 0.000% | 13,107.2 | 13,107.2 | 0.000% |
| Intermediate DDR read, bytes | 0 | 0 | measured zero | 0 | 0 | measured zero |
| Intermediate DDR write, bytes | 0 | 0 | measured zero | 0 | 0 | measured zero |
| Intermediate DMA descriptors | 0 | 0 | measured zero | 0 | 0 | measured zero |
| Spill/fill count | 0 | 0 | measured zero | 0 | 0 | measured zero |
| HMX commands | 1,088 | 1,040 | -4.412% | 1,088 | 1,040 | -4.412% |
| U8S8 HMX tile pairs | 49,408 | 49,408 | 0.000% | 49,408 | 49,408 | 0.000% |
| VTCM requested, bytes | 8,388,608 | 8,388,608 | 0.000% | 8,388,608 | 8,388,608 | 0.000% |
| VTCM acquired, bytes | 8,388,608 | 8,388,608 | 0.000% | 8,388,608 | 8,388,608 | 0.000% |
| VTCM peak live plan, bytes | 5,306,080 | 5,306,080 | 0.000% | 5,306,080 | 5,306,080 | 0.000% |

Both modes use one Host-to-DSP FastRPC invocation per Execution Unit, one HMX
owner, no QNN, no CPU fallback, and no graph split. Prepared-session boundary
traffic is normalized per block, so one-time boundary bytes are amortized at
repeat ten.

## Correctness and final gate

| Gate | Control | Candidate |
|---|---:|---:|
| Final mismatches | 0 | 0 |
| Final maximum LSB | 0 | 0 |
| Output hash | `69f22eeb035e5ec5` | `69f22eeb035e5ec5` |
| Independent QK mismatches | 0 | 0 |
| Independent log2-Softmax mismatches | 0 | 0 |
| Independent AV mismatches | 0 | 0 |
| Probability-mask violations | 0 | 0 |
| Fixed 8 MiB VTCM | pass | pass |
| Zero intermediate DDR/spill | pass | pass |
| Stage-B local speed gate | control | pass |

For context, the final candidate versus the identical-build fully serial QKVO
control reduces QKV+O by 15.66% and Host wall by 5.69% at repeat one, and by
15.74% and 5.94% at repeat ten. This is contextual because the Stage-B direct
gate compares against the Stage-A-selected `qkv_batch4` control.
