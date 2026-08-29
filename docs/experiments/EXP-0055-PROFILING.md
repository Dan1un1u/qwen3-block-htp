# EXP-0055 — Complete profiling report

The control is the accepted EXP-0053 schedule with 24 ordered single-head Q/K Norm+RoPE tasks. The candidate executes the same 16 Q heads and 8 K heads as 12 adjacent-head pairs. It converts the invariant FP16 gamma once per pair and each row's RoPE table once per pair; RMS reduction, reciprocal square root, quantization, K carrier construction and all downstream math remain per head.

## Repeat 1

| Metric (per block) | 24-head control | 12-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 3,177,760.000 | 2,983,333.000 | -6.118% | -6.128% |
| `invocation_ticks` | 52,531.000 | 49,437.000 | -5.890% | -5.990% |
| `total_ticks` | 52,038.000 | 48,938.000 | -5.957% | -6.030% |
| `u8_attention_qk_norm_rope_ticks` | 45,377.000 | 34,172.000 | -24.693% | -24.720% |
| `attention_qk_norm_pool_wait_ticks` | 9,246.000 | 6,299.000 | -31.873% | -31.936% |
| `qkv_projection_ticks` | 16,131.000 | 13,093.000 | -18.833% | -18.872% |
| `attention_ticks` | 8,042.000 | 7,955.000 | -1.082% | -0.749% |

Four-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**.

## Repeat 10

| Metric (per block) | 24-head control | 12-pair candidate | Delta | Paired delta |
|---|---:|---:|---:|---:|
| `host_wall_ns_per_block` | 2,747,849.000 | 2,552,526.100 | -7.108% | -6.898% |
| `invocation_ticks` | 51,385.200 | 48,211.700 | -6.176% | -6.451% |
| `total_ticks` | 51,335.300 | 48,162.500 | -6.181% | -6.457% |
| `u8_attention_qk_norm_rope_ticks` | 45,393.900 | 33,724.100 | -25.708% | -25.679% |
| `attention_qk_norm_pool_wait_ticks` | 9,289.500 | 6,169.900 | -33.582% | -33.582% |
| `qkv_projection_ticks` | 16,166.500 | 12,969.700 | -19.774% | -19.829% |
| `attention_ticks` | 8,071.600 | 8,080.700 | 0.113% | -0.445% |

Four-target speed gate: **PASS**; unchanged math/traffic/resources: **PASS**.

## Physical and correctness gates

Final output is byte-exact (0 mismatches, 0 LSB), and QK, probability and AV audit hashes are identical. Requested/acquired VTCM is exactly 8 MiB; intermediate DDR and spill/fill remain zero; the run remains one FastRPC execution unit with one HMX owner and no QNN dependency.

## Decision

EXP-0055 local gate: **PASS**. Baseline promotion remains a user decision.
