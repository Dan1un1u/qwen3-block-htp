# EXP-0056 — Complete profiling report

The candidate packs corresponding 64-element Softmax rows from the two Q heads of one GQA group into one 128-byte HVX vector. Max, sum, reciprocal and output remain independent per head. Two 16-entry probability tables share one 32-entry `vlut32` call; QK and AV HMX work is unchanged.

## Repeat 1

| Metric (per block) | EXP-0055 control | Paired-row candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,000,052.000 | 2,985,312.000 | -0.491% | 0.106% |
| `invocation_ticks` | 49,356.000 | 49,283.000 | -0.148% | -0.233% |
| `u8_attention_softmax_ticks` | 13,727.000 | 12,767.000 | -6.994% | -7.966% |
| `attention_ticks` | 8,031.000 | 7,943.000 | -1.096% | -1.696% |
| `qkv_projection_ticks` | 13,125.000 | 13,162.000 | 0.282% | 0.030% |
| `gate_up_ticks` | 13,751.000 | 13,843.000 | 0.669% | 0.196% |
| `down_ticks` | 6,864.000 | 6,816.000 | -0.699% | -0.699% |

Three-target speed gate: **FAIL**; unchanged math, traffic and resources: **PASS**.

## Repeat 10

| Metric (per block) | EXP-0055 control | Paired-row candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,583,218.800 | 2,563,927.100 | -0.747% | -1.167% |
| `invocation_ticks` | 48,124.800 | 48,007.000 | -0.245% | -0.168% |
| `u8_attention_softmax_ticks` | 13,831.000 | 12,749.300 | -7.821% | -8.091% |
| `attention_ticks` | 8,031.300 | 7,773.700 | -3.207% | -3.207% |
| `qkv_projection_ticks` | 13,019.700 | 13,008.400 | -0.087% | -0.115% |
| `gate_up_ticks` | 13,836.900 | 13,804.700 | -0.233% | 0.461% |
| `down_ticks` | 6,817.700 | 6,845.900 | 0.414% | 0.474% |

Three-target speed gate: **PASS**; unchanged math, traffic and resources: **PASS**.

## Physical and correctness gates

Final output and QK, probability, and AV audit hashes are byte-exact. Requested and acquired VTCM is exactly 8 MiB; intermediate DDR and spill/fill remain zero; one FastRPC execution unit and one HMX owner are preserved.

## Decision

EXP-0056 local gate: **FAIL**. EXP-0055 remains the selected W4U8 baseline.
