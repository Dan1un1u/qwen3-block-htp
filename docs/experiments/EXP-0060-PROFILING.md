# EXP-0060 — Complete profiling report

The control spills four 32-word U8 RMSNorm square accumulators and sums 128
lanes with scalar code. The candidate adds the four vectors lane-wise and
performs one exact HVX word-reduction tree. Square arithmetic, `sqrtf`,
qparams, gamma, and output encoding are unchanged.

## Repeat 1

| Metric (per block) | EXP-0058 control | HVX-tree candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,963,386.000 | 2,825,781.000 | -4.644% | -5.006% |
| `invocation_ticks` | 48,727.000 | 46,325.000 | -4.930% | -4.767% |
| `input_norm_ticks` | 1,805.000 | 1,561.000 | -13.518% | -13.801% |
| `u8_attention_qk_norm_rope_ticks` | 34,087.000 | 27,641.000 | -18.910% | -18.992% |
| `post_attention_residual_ticks` | 884.000 | 822.000 | -7.014% | -7.281% |
| `qkv_projection_ticks` | 13,070.000 | 10,954.000 | -16.190% | -16.403% |
| `attention_ticks` | 7,435.000 | 7,346.000 | -1.197% | -0.014% |
| `gate_up_ticks` | 13,742.000 | 13,843.000 | 0.735% | 0.988% |
| `down_ticks` | 6,861.000 | 6,827.000 | -0.496% | -0.233% |

Four-target speed gate: **PASS**. Unchanged math, traffic, and resources:
**PASS**.

## Repeat 10

| Metric (per block) | EXP-0058 control | HVX-tree candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,555,671.900 | 2,422,927.100 | -5.194% | -5.131% |
| `invocation_ticks` | 47,641.000 | 45,054.000 | -5.430% | -5.560% |
| `input_norm_ticks` | 1,794.200 | 1,543.500 | -13.973% | -13.949% |
| `u8_attention_qk_norm_rope_ticks` | 33,690.500 | 27,675.400 | -17.854% | -17.826% |
| `post_attention_residual_ticks` | 886.200 | 815.500 | -7.978% | -7.957% |
| `qkv_projection_ticks` | 12,980.900 | 10,890.300 | -16.105% | -16.033% |
| `attention_ticks` | 7,369.600 | 7,341.300 | -0.384% | -0.384% |
| `gate_up_ticks` | 13,822.200 | 13,803.100 | -0.138% | -0.198% |
| `down_ticks` | 6,831.600 | 6,808.700 | -0.335% | -0.449% |

Four-target speed gate: **PASS**. Unchanged math, traffic, and resources:
**PASS**.

## Physical and correctness gates

Final output and QK/probability/AV audit hashes are byte-exact. Exact 8 MiB
VTCM, zero intermediate DDR and spill/fill, one FastRPC execution unit, and
one HMX owner are preserved.

## Decision

EXP-0060 local gate: **PASS**. Baseline promotion remains a user decision.
