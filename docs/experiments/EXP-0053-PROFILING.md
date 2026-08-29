# EXP-0053 — Complete profiling report

The candidate changes only the two native-layout W4U8 residual consumers.
Sixty-four rows become sixteen disjoint four-row tasks claimed by the main HVX
context and the three persistent worker contexts. Q14 arithmetic, native
O/Down carriers, Post-Attention RMSNorm, every HMX command, traffic and all
other schedules are fixed.

## Repeat 1

| Metric (per block) | Serial control | Pool4 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| Host wall (ns) | 3,285,365.0 | 3,130,521.0 | -4.713% | -4.955% |
| Invocation ticks | 55,211.0 | 52,213.0 | -5.430% | -5.457% |
| Total ticks | 54,706.0 | 51,715.0 | -5.467% | -5.512% |
| Combined residual ticks | 4,513.0 | 1,293.0 | -71.349% | -71.413% |
| Post-Attention residual/RMSNorm ticks | 3,130.0 | 876.0 | -72.013% | -71.986% |
| Final residual ticks | 1,383.0 | 417.0 | -69.848% | -69.848% |
| Post boundary join wait | 0.0 | 43.0 | n/a | n/a |
| Final boundary join wait | 0.0 | 14.0 | n/a | n/a |

Speed, fixed math/traffic/resources and row-task coverage gates all pass.

## Repeat 10

| Metric (per block) | Serial control | Pool4 candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| Host wall (ns) | 2,907,151.1 | 2,740,229.2 | -5.742% | -5.507% |
| Invocation ticks | 54,369.6 | 51,144.8 | -5.931% | -6.034% |
| Total ticks | 54,319.4 | 51,094.7 | -5.937% | -6.041% |
| Combined residual ticks | 4,507.9 | 1,296.9 | -71.231% | -71.224% |
| Post-Attention residual/RMSNorm ticks | 3,125.0 | 878.3 | -71.894% | -71.889% |
| Final residual ticks | 1,382.9 | 419.0 | -69.701% | -69.699% |
| Post boundary join wait | 0.0 | 31.7 | n/a | n/a |
| Final boundary join wait | 0.0 | 15.3 | n/a | n/a |

Speed, fixed math/traffic/resources and row-task coverage gates all pass.

## Correctness and physical closure

The final block output is byte-exact with zero mismatches and maximum error
0 LSB. QK, probability and AV hashes are unchanged. Both variants request and
acquire exactly 8 MiB VTCM and peak at 5,306,080 bytes. Intermediate DDR
read/write, DMA spill traffic and spill/fill counts remain zero. Both variants
use one FastRPC execution unit, one HMX owner, no QNN dependency, 256 HMX
commands, 49,408 integer tile pairs, 512 weight DMA descriptors and
25,444,352 weight bytes per block.

The additive ledger and overlapping worker-work counters are not summed
together. Host wall is the primary speed metric. EXP-0053 passes its local
gate and is eligible as a W4U8 candidate; baseline promotion remains a user
decision.
