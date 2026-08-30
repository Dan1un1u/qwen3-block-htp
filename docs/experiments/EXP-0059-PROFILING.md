# EXP-0059 — Complete profiling report

The control is the complete EXP-0058 V-pack path. The candidate changes only
Softmax: corresponding rows from the two Q heads of one GQA group share one
128-byte HVX vector and one banked `vlut32` operation, while normalization
remains independent per head.

## Repeat 1

| Metric (per block) | EXP-0058 control | Paired candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,952,864.000 | 2,936,875.000 | -0.541% | 1.288% |
| `invocation_ticks` | 48,522.000 | 48,169.000 | -0.728% | -0.452% |
| `u8_attention_softmax_ticks` | 14,090.000 | 12,691.000 | -9.929% | -9.856% |
| `attention_ticks` | 7,238.000 | 6,869.000 | -5.098% | -6.110% |
| `u8_attention_v_pack_ticks` | 2,776.000 | 2,789.000 | 0.468% | 0.472% |
| `qkv_projection_ticks` | 13,107.000 | 13,116.000 | 0.069% | -0.015% |
| `gate_up_ticks` | 13,838.000 | 13,789.000 | -0.354% | -0.582% |
| `down_ticks` | 6,808.000 | 6,770.000 | -0.558% | 0.470% |

Three-target speed gate: **FAIL**. Unchanged math, traffic, and resources:
**PASS**.

## Repeat 10

| Metric (per block) | EXP-0058 control | Paired candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,530,265.600 | 2,516,182.200 | -0.557% | -1.201% |
| `invocation_ticks` | 47,457.600 | 46,885.100 | -1.206% | -0.794% |
| `u8_attention_softmax_ticks` | 14,029.600 | 12,674.700 | -9.657% | -9.746% |
| `attention_ticks` | 7,248.800 | 6,880.600 | -5.079% | -5.085% |
| `u8_attention_v_pack_ticks` | 2,785.500 | 2,788.600 | 0.111% | 0.068% |
| `qkv_projection_ticks` | 12,968.700 | 12,970.000 | 0.010% | 0.094% |
| `gate_up_ticks` | 13,917.200 | 13,750.300 | -1.199% | 0.153% |
| `down_ticks` | 6,808.800 | 6,772.500 | -0.533% | -0.451% |

Three-target speed gate: **PASS**. Unchanged math, traffic, and resources:
**PASS**.

## Physical and correctness gates

Final output and QK/probability/AV audit hashes are byte-exact. The exact
8 MiB VTCM request, zero intermediate DDR and spill/fill, one FastRPC
execution unit, and one HMX owner are preserved.

## Decision

EXP-0059 local gate: **FAIL**. The paired-row Softmax hypothesis is rejected
under the current Attention contract.
