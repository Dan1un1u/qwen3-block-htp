# EXP-0057 — Complete profiling report

The candidate replaces 128 scalar V-recenter table accesses per carrier vector with two 64-lane halfword HVX vgathers. The exact 256-entry table, output S8 carrier, AV HMX math and schedule remain unchanged.

## Repeat 1

| Metric (per block) | EXP-0055 control | Vgather candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,996,927.000 | 2,960,886.000 | -1.203% | -0.993% |
| `invocation_ticks` | 49,442.000 | 48,791.000 | -1.317% | -1.646% |
| `u8_attention_v_pack_ticks` | 8,215.000 | 4,957.000 | -39.659% | -39.210% |
| `attention_ticks` | 8,057.000 | 7,546.000 | -6.342% | -6.342% |
| `qkv_projection_ticks` | 13,173.000 | 13,187.000 | 0.106% | 0.189% |
| `gate_up_ticks` | 13,835.000 | 13,738.000 | -0.701% | -0.820% |
| `down_ticks` | 6,752.000 | 6,760.000 | 0.118% | 0.855% |

Three-target speed gate: **PASS**; unchanged math, traffic and resources: **PASS**.

## Repeat 10

| Metric (per block) | EXP-0055 control | Vgather candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,593,645.900 | 2,549,109.400 | -1.717% | -1.684% |
| `invocation_ticks` | 48,393.200 | 47,760.100 | -1.308% | -1.148% |
| `u8_attention_v_pack_ticks` | 8,168.100 | 4,931.900 | -39.620% | -39.643% |
| `attention_ticks` | 8,085.500 | 7,635.900 | -5.561% | -5.244% |
| `qkv_projection_ticks` | 13,020.100 | 13,009.800 | -0.079% | -0.042% |
| `gate_up_ticks` | 13,896.600 | 13,758.200 | -0.996% | -0.996% |
| `down_ticks` | 6,811.000 | 6,815.300 | 0.063% | -0.139% |

Three-target speed gate: **PASS**; unchanged math, traffic and resources: **PASS**.

## Physical and correctness gates

Final output, V saturation count, and QK, probability, and AV audit hashes are byte-exact. Requested and acquired VTCM is exactly 8 MiB; intermediate DDR and spill/fill remain zero; one FastRPC execution unit and one HMX owner are preserved.

## Decision

EXP-0057 local gate: **PASS**. Baseline promotion remains a user decision.
