# L32-0057 full-model module ledgers

Microseconds per complete prefill or per decode token; parentheses are complete Host wall shares. Parallel worker service counters are not added as independent wall components.

## a8 64+42 prefill

| Module | Control31 | Candidate31 |
|---|---:|---:|
| I/O、metadata | 135.9 (0.50%) | 135.9 (0.50%) |
| Input RMSNorm | 1282.3 (4.76%) | 1282.3 (4.76%) |
| QKV＋RoPE | 3885.2 (14.41%) | 3885.2 (14.41%) |
| QK–Softmax–AV | 6212.5 (23.04%) | 6212.5 (23.04%) |
| O projection | 1571.2 (5.83%) | 1571.2 (5.83%) |
| Post-attention residual＋RMSNorm | 1306.9 (4.85%) | 1306.9 (4.85%) |
| Gate/Up＋SwiGLU | 5600.0 (20.77%) | 5600.0 (20.77%) |
| Down | 2717.2 (10.08%) | 2717.2 (10.08%) |
| Final residual | 1.9 (0.01%) | 1.9 (0.01%) |
| KV carrier conversion | 26.2 (0.10%) | 26.2 (0.10%) |
| KV append DMA | 103.3 (0.38%) | 103.3 (0.38%) |
| Block orchestration | 24.3 (0.09%) | 24.3 (0.09%) |
| Layer bookkeeping | 13.4 (0.05%) | 13.4 (0.05%) |
| Stage-boundary bookkeeping | 6.0 (0.02%) | 6.0 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 89.7 (0.33%) | 89.7 (0.33%) |
| Embedding | 60.5 (0.22%) | 60.5 (0.22%) |
| Final model RMSNorm | 56.0 (0.21%) | 56.0 (0.21%) |
| LM head＋greedy（不含 final norm） | 3105.3 (11.52%) | 3105.3 (11.52%) |
| Host input/RoPE staging | 0.0 (0.00%) | 0.0 (0.00%) |
| Host-DSP boundary | 766.0 (2.84%) | 766.0 (2.84%) |
| Complete Host wall | 26963.7 (100.00%) | 26963.7 (100.00%) |

## a8 64+42 decode

| Module | Control31 | Candidate31 |
|---|---:|---:|
| I/O、metadata | 134.2 (0.60%) | 134.2 (0.60%) |
| Input RMSNorm | 870.7 (3.90%) | 870.7 (3.90%) |
| QKV＋RoPE | 1682.6 (7.53%) | 1682.6 (7.53%) |
| QK–Softmax–AV | 6342.8 (28.38%) | 6342.8 (28.38%) |
| O projection | 861.8 (3.86%) | 861.8 (3.86%) |
| Post-attention residual＋RMSNorm | 876.0 (3.92%) | 876.0 (3.92%) |
| Gate/Up＋SwiGLU | 4926.8 (22.05%) | 4926.8 (22.05%) |
| Down | 2675.1 (11.97%) | 2675.1 (11.97%) |
| Final residual | 1.2 (0.01%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.22%) | 48.1 (0.22%) |
| KV append DMA | 87.9 (0.39%) | 87.9 (0.39%) |
| Block orchestration | 16.8 (0.08%) | 16.8 (0.08%) |
| Layer bookkeeping | 9.6 (0.04%) | 9.6 (0.04%) |
| Stage-boundary bookkeeping | 1.4 (0.01%) | 1.4 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.9 (0.23%) | 50.9 (0.23%) |
| Embedding | 2.0 (0.01%) | 2.0 (0.01%) |
| Final model RMSNorm | 56.0 (0.25%) | 56.0 (0.25%) |
| LM head＋greedy（不含 final norm） | 3097.6 (13.86%) | 3097.6 (13.86%) |
| Host input/RoPE staging | 0.0 (0.00%) | 0.0 (0.00%) |
| Host-DSP boundary | 605.8 (2.71%) | 605.8 (2.71%) |
| Complete Host wall | 22347.2 (100.00%) | 22347.2 (100.00%) |

## a8 536+46 prefill

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 1163.9 (0.55%) | 1177.6 (0.56%) |
| Input RMSNorm | 12712.9 (6.04%) | 12712.6 (6.02%) |
| QKV＋RoPE | 34805.5 (16.54%) | 34818.7 (16.50%) |
| QK–Softmax–AV | 47016.8 (22.34%) | 46996.1 (22.27%) |
| O projection | 14125.5 (6.71%) | 14127.1 (6.69%) |
| Post-attention residual＋RMSNorm | 12668.4 (6.02%) | 12668.8 (6.00%) |
| Gate/Up＋SwiGLU | 49894.6 (23.70%) | 49966.4 (23.68%) |
| Down | 24220.7 (11.51%) | 24263.0 (11.50%) |
| Final residual | 10.9 (0.01%) | 10.8 (0.01%) |
| KV carrier conversion | 217.1 (0.10%) | 216.8 (0.10%) |
| KV append DMA | 1135.8 (0.54%) | 1133.1 (0.54%) |
| Block orchestration | 166.0 (0.08%) | 165.8 (0.08%) |
| Layer bookkeeping | 93.1 (0.04%) | 92.9 (0.04%) |
| Stage-boundary bookkeeping | 16.4 (0.01%) | 16.1 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 501.1 (0.24%) | 496.6 (0.24%) |
| Embedding | 506.5 (0.24%) | 504.8 (0.24%) |
| Final model RMSNorm | 56.4 (0.03%) | 56.4 (0.03%) |
| LM head＋greedy（不含 final norm） | 4380.0 (2.08%) | 4381.1 (2.08%) |
| Host input/RoPE staging | 60.5 (0.03%) | 77.6 (0.04%) |
| Host-DSP boundary | 6743.0 (3.20%) | 7149.4 (3.39%) |
| Complete Host wall | 210495.1 (100.00%) | 211031.7 (100.00%) |

## a8 536+46 decode

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 127.5 (0.48%) | 129.4 (0.66%) |
| Input RMSNorm | 869.7 (3.28%) | 869.9 (4.42%) |
| QKV＋RoPE | 1677.9 (6.32%) | 1680.5 (8.55%) |
| QK–Softmax–AV | 10574.0 (39.83%) | 3610.7 (18.36%) |
| O projection | 864.6 (3.26%) | 864.4 (4.40%) |
| Post-attention residual＋RMSNorm | 877.2 (3.30%) | 877.3 (4.46%) |
| Gate/Up＋SwiGLU | 4898.1 (18.45%) | 4909.8 (24.97%) |
| Down | 2660.1 (10.02%) | 2666.1 (13.56%) |
| Final residual | 1.2 (0.00%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.18%) | 48.2 (0.25%) |
| KV append DMA | 123.7 (0.47%) | 122.7 (0.62%) |
| Block orchestration | 16.8 (0.06%) | 17.0 (0.09%) |
| Layer bookkeeping | 9.6 (0.04%) | 9.6 (0.05%) |
| Stage-boundary bookkeeping | 1.3 (0.01%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.9 (0.19%) | 50.9 (0.26%) |
| Embedding | 5.8 (0.02%) | 5.8 (0.03%) |
| Final model RMSNorm | 55.5 (0.21%) | 55.6 (0.28%) |
| LM head＋greedy（不含 final norm） | 3100.2 (11.68%) | 3101.0 (15.77%) |
| Host input/RoPE staging | 7.8 (0.03%) | 8.3 (0.04%) |
| Host-DSP boundary | 576.9 (2.17%) | 635.1 (3.23%) |
| Complete Host wall | 26546.9 (100.00%) | 19664.7 (100.00%) |

## a8 741+3 prefill

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 1551.9 (0.53%) | 1512.7 (0.52%) |
| Input RMSNorm | 18293.2 (6.26%) | 18290.1 (6.29%) |
| QKV＋RoPE | 46601.1 (15.95%) | 46482.3 (15.97%) |
| QK–Softmax–AV | 72725.8 (24.90%) | 72639.0 (24.96%) |
| O projection | 18839.7 (6.45%) | 18807.3 (6.46%) |
| Post-attention residual＋RMSNorm | 18215.9 (6.24%) | 18214.4 (6.26%) |
| Gate/Up＋SwiGLU | 66493.4 (22.77%) | 66082.1 (22.71%) |
| Down | 32210.5 (11.03%) | 31998.3 (11.00%) |
| Final residual | 14.1 (0.00%) | 14.2 (0.00%) |
| KV carrier conversion | 339.8 (0.12%) | 339.7 (0.12%) |
| KV append DMA | 1491.2 (0.51%) | 1456.3 (0.50%) |
| Block orchestration | 219.2 (0.08%) | 219.0 (0.08%) |
| Layer bookkeeping | 123.1 (0.04%) | 122.8 (0.04%) |
| Stage-boundary bookkeeping | 20.1 (0.01%) | 19.8 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 652.8 (0.22%) | 651.7 (0.22%) |
| Embedding | 685.6 (0.23%) | 676.5 (0.23%) |
| Final model RMSNorm | 56.1 (0.02%) | 55.9 (0.02%) |
| LM head＋greedy（不含 final norm） | 4373.2 (1.50%) | 4361.8 (1.50%) |
| Host input/RoPE staging | 92.6 (0.03%) | 96.5 (0.03%) |
| Host-DSP boundary | 9083.2 (3.11%) | 8931.3 (3.07%) |
| Complete Host wall | 292082.5 (100.00%) | 290971.7 (100.00%) |

## a8 741+3 decode

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 131.2 (0.45%) | 127.3 (0.63%) |
| Input RMSNorm | 879.5 (3.00%) | 879.9 (4.36%) |
| QKV＋RoPE | 1676.2 (5.73%) | 1665.3 (8.25%) |
| QK–Softmax–AV | 13261.2 (45.30%) | 4283.6 (21.22%) |
| O projection | 865.0 (2.95%) | 861.7 (4.27%) |
| Post-attention residual＋RMSNorm | 878.9 (3.00%) | 879.1 (4.36%) |
| Gate/Up＋SwiGLU | 4885.0 (16.69%) | 4852.1 (24.04%) |
| Down | 2652.3 (9.06%) | 2636.1 (13.06%) |
| Final residual | 1.2 (0.00%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.16%) | 48.2 (0.24%) |
| KV append DMA | 118.6 (0.40%) | 112.0 (0.55%) |
| Block orchestration | 16.8 (0.06%) | 17.0 (0.08%) |
| Layer bookkeeping | 9.7 (0.03%) | 9.7 (0.05%) |
| Stage-boundary bookkeeping | 1.3 (0.00%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.0 (0.17%) | 51.1 (0.25%) |
| Embedding | 5.8 (0.02%) | 5.7 (0.03%) |
| Final model RMSNorm | 54.7 (0.19%) | 54.6 (0.27%) |
| LM head＋greedy（不含 final norm） | 3093.1 (10.57%) | 3089.2 (15.31%) |
| Host input/RoPE staging | 8.0 (0.03%) | 7.5 (0.04%) |
| Host-DSP boundary | 636.1 (2.17%) | 601.9 (2.98%) |
| Complete Host wall | 29273.8 (100.00%) | 20184.4 (100.00%) |

## sp2 64+42 prefill

| Module | Control31 | Candidate31 |
|---|---:|---:|
| I/O、metadata | 132.9 (0.49%) | 132.9 (0.49%) |
| Input RMSNorm | 1268.2 (4.65%) | 1268.2 (4.65%) |
| QKV＋RoPE | 3872.8 (14.19%) | 3872.8 (14.19%) |
| QK–Softmax–AV | 6204.9 (22.73%) | 6204.9 (22.73%) |
| O projection | 1572.9 (5.76%) | 1572.9 (5.76%) |
| Post-attention residual＋RMSNorm | 1275.7 (4.67%) | 1275.7 (4.67%) |
| Gate/Up＋SwiGLU | 5577.5 (20.43%) | 5577.5 (20.43%) |
| Down | 3123.7 (11.44%) | 3123.7 (11.44%) |
| Final residual | 2.0 (0.01%) | 2.0 (0.01%) |
| KV carrier conversion | 26.1 (0.10%) | 26.1 (0.10%) |
| KV append DMA | 105.4 (0.39%) | 105.4 (0.39%) |
| Block orchestration | 24.5 (0.09%) | 24.5 (0.09%) |
| Layer bookkeeping | 13.3 (0.05%) | 13.3 (0.05%) |
| Stage-boundary bookkeeping | 6.1 (0.02%) | 6.1 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 89.7 (0.33%) | 89.7 (0.33%) |
| Embedding | 59.7 (0.22%) | 59.7 (0.22%) |
| Final model RMSNorm | 60.9 (0.22%) | 60.9 (0.22%) |
| LM head＋greedy（不含 final norm） | 3108.0 (11.38%) | 3108.0 (11.38%) |
| Host input/RoPE staging | 0.0 (0.00%) | 0.0 (0.00%) |
| Host-DSP boundary | 777.2 (2.85%) | 777.2 (2.85%) |
| Complete Host wall | 27301.5 (100.00%) | 27301.5 (100.00%) |

## sp2 64+42 decode

| Module | Control31 | Candidate31 |
|---|---:|---:|
| I/O、metadata | 132.0 (0.59%) | 132.0 (0.59%) |
| Input RMSNorm | 869.3 (3.89%) | 869.3 (3.89%) |
| QKV＋RoPE | 1679.6 (7.52%) | 1679.6 (7.52%) |
| QK–Softmax–AV | 6344.1 (28.40%) | 6344.1 (28.40%) |
| O projection | 863.2 (3.86%) | 863.2 (3.86%) |
| Post-attention residual＋RMSNorm | 877.0 (3.93%) | 877.0 (3.93%) |
| Gate/Up＋SwiGLU | 4912.9 (21.99%) | 4912.9 (21.99%) |
| Down | 2668.6 (11.95%) | 2668.6 (11.95%) |
| Final residual | 1.2 (0.01%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.22%) | 48.1 (0.22%) |
| KV append DMA | 88.4 (0.40%) | 88.4 (0.40%) |
| Block orchestration | 16.8 (0.08%) | 16.8 (0.08%) |
| Layer bookkeeping | 9.6 (0.04%) | 9.6 (0.04%) |
| Stage-boundary bookkeeping | 1.4 (0.01%) | 1.4 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.9 (0.23%) | 50.9 (0.23%) |
| Embedding | 2.0 (0.01%) | 2.0 (0.01%) |
| Final model RMSNorm | 56.1 (0.25%) | 56.1 (0.25%) |
| LM head＋greedy（不含 final norm） | 3100.9 (13.88%) | 3100.9 (13.88%) |
| Host input/RoPE staging | 0.0 (0.00%) | 0.0 (0.00%) |
| Host-DSP boundary | 616.3 (2.76%) | 616.3 (2.76%) |
| Complete Host wall | 22338.3 (100.00%) | 22338.3 (100.00%) |

## sp2 536+46 prefill

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 1194.4 (0.56%) | 1185.1 (0.55%) |
| Input RMSNorm | 12709.3 (5.91%) | 12709.8 (5.91%) |
| QKV＋RoPE | 34841.2 (16.21%) | 34850.9 (16.21%) |
| QK–Softmax–AV | 47005.7 (21.87%) | 47030.7 (21.88%) |
| O projection | 14128.5 (6.57%) | 14124.0 (6.57%) |
| Post-attention residual＋RMSNorm | 12613.3 (5.87%) | 12613.4 (5.87%) |
| Gate/Up＋SwiGLU | 50156.2 (23.33%) | 50163.8 (23.34%) |
| Down | 28109.4 (13.08%) | 28090.9 (13.07%) |
| Final residual | 10.7 (0.00%) | 10.9 (0.01%) |
| KV carrier conversion | 217.3 (0.10%) | 217.4 (0.10%) |
| KV append DMA | 1158.8 (0.54%) | 1156.3 (0.54%) |
| Block orchestration | 166.6 (0.08%) | 166.5 (0.08%) |
| Layer bookkeeping | 93.2 (0.04%) | 93.2 (0.04%) |
| Stage-boundary bookkeeping | 16.3 (0.01%) | 16.5 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 500.9 (0.23%) | 501.7 (0.23%) |
| Embedding | 511.4 (0.24%) | 511.8 (0.24%) |
| Final model RMSNorm | 56.4 (0.03%) | 56.6 (0.03%) |
| LM head＋greedy（不含 final norm） | 4379.1 (2.04%) | 4382.3 (2.04%) |
| Host input/RoPE staging | 76.2 (0.04%) | 62.3 (0.03%) |
| Host-DSP boundary | 6999.4 (3.26%) | 6997.0 (3.26%) |
| Complete Host wall | 214944.3 (100.00%) | 214941.0 (100.00%) |

## sp2 536+46 decode

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 131.2 (0.49%) | 130.9 (0.67%) |
| Input RMSNorm | 871.7 (3.27%) | 871.9 (4.43%) |
| QKV＋RoPE | 1682.4 (6.31%) | 1685.5 (8.57%) |
| QK–Softmax–AV | 10571.5 (39.68%) | 3611.6 (18.36%) |
| O projection | 864.3 (3.24%) | 865.5 (4.40%) |
| Post-attention residual＋RMSNorm | 876.8 (3.29%) | 876.8 (4.46%) |
| Gate/Up＋SwiGLU | 4920.7 (18.47%) | 4928.3 (25.05%) |
| Down | 2674.6 (10.04%) | 2677.4 (13.61%) |
| Final residual | 1.2 (0.00%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.18%) | 48.2 (0.24%) |
| KV append DMA | 128.9 (0.48%) | 130.1 (0.66%) |
| Block orchestration | 16.8 (0.06%) | 16.9 (0.09%) |
| Layer bookkeeping | 9.6 (0.04%) | 9.6 (0.05%) |
| Stage-boundary bookkeeping | 1.3 (0.01%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.9 (0.19%) | 51.0 (0.26%) |
| Embedding | 5.8 (0.02%) | 5.8 (0.03%) |
| Final model RMSNorm | 55.7 (0.21%) | 55.7 (0.28%) |
| LM head＋greedy（不含 final norm） | 3100.6 (11.64%) | 3105.4 (15.79%) |
| Host input/RoPE staging | 9.0 (0.03%) | 7.6 (0.04%) |
| Host-DSP boundary | 622.4 (2.34%) | 590.9 (3.00%) |
| Complete Host wall | 26643.5 (100.00%) | 19671.6 (100.00%) |

## sp2 741+3 prefill

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 1604.8 (0.54%) | 1596.2 (0.54%) |
| Input RMSNorm | 18298.6 (6.15%) | 18300.2 (6.15%) |
| QKV＋RoPE | 46642.8 (15.67%) | 46622.4 (15.66%) |
| QK–Softmax–AV | 72730.4 (24.44%) | 72707.2 (24.42%) |
| O projection | 18823.6 (6.32%) | 18813.9 (6.32%) |
| Post-attention residual＋RMSNorm | 18162.1 (6.10%) | 18164.6 (6.10%) |
| Gate/Up＋SwiGLU | 67040.0 (22.52%) | 67051.7 (22.52%) |
| Down | 37447.7 (12.58%) | 37436.1 (12.57%) |
| Final residual | 14.0 (0.00%) | 14.0 (0.00%) |
| KV carrier conversion | 340.6 (0.11%) | 340.8 (0.11%) |
| KV append DMA | 1624.2 (0.55%) | 1620.4 (0.54%) |
| Block orchestration | 218.8 (0.07%) | 219.2 (0.07%) |
| Layer bookkeeping | 122.6 (0.04%) | 122.7 (0.04%) |
| Stage-boundary bookkeeping | 19.3 (0.01%) | 19.5 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 650.2 (0.22%) | 651.9 (0.22%) |
| Embedding | 708.5 (0.24%) | 709.0 (0.24%) |
| Final model RMSNorm | 56.6 (0.02%) | 56.4 (0.02%) |
| LM head＋greedy（不含 final norm） | 4374.9 (1.47%) | 4373.3 (1.47%) |
| Host input/RoPE staging | 79.6 (0.03%) | 77.1 (0.03%) |
| Host-DSP boundary | 8680.7 (2.92%) | 8892.9 (2.99%) |
| Complete Host wall | 297640.1 (100.00%) | 297789.4 (100.00%) |

## sp2 741+3 decode

| Module | Control31 | Candidate159 |
|---|---:|---:|
| I/O、metadata | 137.2 (0.47%) | 136.0 (0.67%) |
| Input RMSNorm | 873.7 (2.98%) | 874.0 (4.29%) |
| QKV＋RoPE | 1685.7 (5.75%) | 1683.8 (8.26%) |
| QK–Softmax–AV | 13249.5 (45.21%) | 4284.1 (21.01%) |
| O projection | 863.5 (2.95%) | 861.6 (4.23%) |
| Post-attention residual＋RMSNorm | 871.8 (2.97%) | 872.1 (4.28%) |
| Gate/Up＋SwiGLU | 4934.4 (16.84%) | 4938.6 (24.22%) |
| Down | 2681.8 (9.15%) | 2683.0 (13.16%) |
| Final residual | 1.1 (0.00%) | 1.2 (0.01%) |
| KV carrier conversion | 48.1 (0.16%) | 48.2 (0.24%) |
| KV append DMA | 142.4 (0.49%) | 140.7 (0.69%) |
| Block orchestration | 16.8 (0.06%) | 17.0 (0.08%) |
| Layer bookkeeping | 9.7 (0.03%) | 9.7 (0.05%) |
| Stage-boundary bookkeeping | 1.3 (0.00%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.2 (0.17%) | 51.1 (0.25%) |
| Embedding | 5.8 (0.02%) | 5.7 (0.03%) |
| Final model RMSNorm | 55.9 (0.19%) | 55.8 (0.27%) |
| LM head＋greedy（不含 final norm） | 3102.0 (10.59%) | 3103.3 (15.22%) |
| Host input/RoPE staging | 6.3 (0.02%) | 6.4 (0.03%) |
| Host-DSP boundary | 566.3 (1.93%) | 613.8 (3.01%) |
| Complete Host wall | 29304.5 (100.00%) | 20387.4 (100.00%) |
