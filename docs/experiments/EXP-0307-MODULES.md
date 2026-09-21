# Additive module latency

Units µs. Parentheses show share of complete Host wall. Prefill is per64-token invocation; decode is per single-token invocation, averaged over all42 positions and100 formal trajectories.

## prefill

| Module | F | I |
|---|---:|---:|
| I/O、metadata | 247.986 (0.474%) | 245.755 (0.705%) |
| Input RMSNorm | 2123.734 (4.057%) | 2056.263 (5.903%) |
| QKV＋RoPE | 7040.764 (13.449%) | 7045.304 (20.225%) |
| QK–Softmax–AV | 3857.914 (7.369%) | 3501.186 (10.051%) |
| O projection | 2082.197 (3.977%) | 2085.917 (5.988%) |
| Post-attention residual＋RMSNorm | 2045.572 (3.907%) | 2152.021 (6.178%) |
| Gate/Up＋SwiGLU | 24309.954 (46.435%) | 7145.834 (20.514%) |
| Down | 3841.435 (7.338%) | 3836.805 (11.014%) |
| Final residual | 2.348 (0.004%) | 2.280 (0.007%) |
| KV carrier conversion | 137.521 (0.263%) | 138.478 (0.398%) |
| KV append DMA | 291.266 (0.556%) | 291.456 (0.837%) |
| Block orchestration | 38.768 (0.074%) | 38.347 (0.110%) |
| Layer bookkeeping | 25.141 (0.048%) | 25.762 (0.074%) |
| Stage-boundary bookkeeping | 20.471 (0.039%) | 20.607 (0.059%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 116.898 (0.223%) | 116.807 (0.335%) |
| Embedding | 59.392 (0.113%) | 59.114 (0.170%) |
| Final model RMSNorm | 14.091 (0.027%) | 17.038 (0.049%) |
| LM head＋greedy（不含 final norm） | 5203.014 (9.938%) | 5213.652 (14.967%) |
| Host–DSP boundary | 894.006 (1.708%) | 841.638 (2.416%) |
| Full Host wall | 52352.471 (100.000%) | 34834.264 (100.000%) |

E2E token/s: F=1222.48, I=1837.27

## decode

| Module | F | I |
|---|---:|---:|
| I/O、metadata | 244.922 (1.147%) | 242.036 (1.143%) |
| Input RMSNorm | 365.050 (1.709%) | 365.821 (1.727%) |
| QKV＋RoPE | 2531.797 (11.853%) | 2528.784 (11.939%) |
| QK–Softmax–AV | 1921.360 (8.995%) | 1965.854 (9.281%) |
| O projection | 1368.705 (6.408%) | 1368.913 (6.463%) |
| Post-attention residual＋RMSNorm | 371.236 (1.738%) | 370.589 (1.750%) |
| Gate/Up＋SwiGLU | 6599.720 (30.899%) | 6420.138 (30.311%) |
| Down | 3548.904 (16.615%) | 3545.437 (16.739%) |
| Final residual | 1.809 (0.008%) | 1.811 (0.009%) |
| KV carrier conversion | 164.081 (0.768%) | 163.876 (0.774%) |
| KV append DMA | 112.520 (0.527%) | 112.269 (0.530%) |
| Block orchestration | 29.709 (0.139%) | 29.825 (0.141%) |
| Layer bookkeeping | 16.365 (0.077%) | 16.346 (0.077%) |
| Stage-boundary bookkeeping | 1.772 (0.008%) | 1.771 (0.008%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 73.572 (0.344%) | 73.691 (0.348%) |
| Embedding | 1.725 (0.008%) | 1.705 (0.008%) |
| Final model RMSNorm | 14.875 (0.070%) | 14.398 (0.068%) |
| LM head＋greedy（不含 final norm） | 3274.899 (15.333%) | 3266.026 (15.419%) |
| Host–DSP boundary | 716.147 (3.353%) | 691.908 (3.267%) |
| Full Host wall | 21359.168 (100.000%) | 21181.198 (100.000%) |

E2E token/s: F=46.82, I=47.21
