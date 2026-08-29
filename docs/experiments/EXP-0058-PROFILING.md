# EXP-0058 — Complete profiling report

The candidate applies five byte-`vdeal` stages to map four contiguous 32-byte rows into 32 adjacent four-row HMX words, then performs one vector store. The exact `vgather` table, arithmetic, qparams, traffic, and scheduling are unchanged.

## Repeat 1

| Metric (per block) | EXP-0057 control | `vdeal` candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,972,656.000 | 2,957,760.000 | -0.501% | -0.576% |
| `invocation_ticks` | 49,014.000 | 48,717.000 | -0.606% | -0.758% |
| `u8_attention_v_pack_ticks` | 4,930.000 | 2,798.000 | -43.245% | -43.215% |
| `attention_ticks` | 7,553.000 | 7,106.000 | -5.918% | -3.800% |
| `qkv_projection_ticks` | 13,108.000 | 13,102.000 | -0.046% | 0.046% |
| `gate_up_ticks` | 13,746.000 | 13,969.000 | 1.622% | 0.480% |
| `down_ticks` | 6,763.000 | 6,873.000 | 1.626% | 0.950% |

Three-target speed gate: **PASS**. Unchanged math, traffic, and resources: **PASS**.

## Repeat 10

| Metric (per block) | EXP-0057 control | `vdeal` candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,563,671.900 | 2,532,328.100 | -1.223% | -0.718% |
| `invocation_ticks` | 47,782.400 | 47,321.800 | -0.964% | -0.764% |
| `u8_attention_v_pack_ticks` | 4,935.300 | 2,784.300 | -43.584% | -43.512% |
| `attention_ticks` | 7,675.700 | 7,258.200 | -5.439% | -5.189% |
| `qkv_projection_ticks` | 12,991.900 | 12,982.900 | -0.069% | 0.024% |
| `gate_up_ticks` | 13,762.200 | 13,672.300 | -0.653% | 0.208% |
| `down_ticks` | 6,832.800 | 6,835.100 | 0.034% | 0.414% |

Three-target speed gate: **PASS**. Unchanged math, traffic, and resources: **PASS**.

## Physical and correctness gates

Final output, V saturation count, and QK/probability/AV audit hashes are byte-exact. Requested and acquired VTCM is exactly 8 MiB; intermediate DDR and spill/fill remain zero; one FastRPC execution unit and one HMX owner are preserved.

## Decision

EXP-0058 local gate: **PASS**. Baseline promotion remains a user decision.
