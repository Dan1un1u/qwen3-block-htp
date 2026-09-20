# L32-0060 additive module tables

Units microseconds; percentage of complete Host wall. Decode per token.

## 1B-64+42 prefill

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 121.211 (0.44%) | 125.809 (0.02%) |
| Input RMSNorm | 1267.978 (4.59%) | 1268.214 (0.17%) |
| QKV＋RoPE | 3882.120 (14.06%) | 4075.744 (0.54%) |
| QK–Softmax–AV | 6297.505 (22.81%) | 6439.013 (0.86%) |
| O projection | 1592.908 (5.77%) | 1638.560 (0.22%) |
| Post-attention residual＋RMSNorm | 1276.383 (4.62%) | 1277.110 (0.17%) |
| Gate/Up＋SwiGLU | 5529.748 (20.03%) | 6318.266 (0.84%) |
| Down | 3141.383 (11.38%) | 721154.116 (96.26%) |
| Final residual | 2.243 (0.01%) | 2.219 (0.00%) |
| KV carrier conversion | 29.470 (0.11%) | 29.130 (0.00%) |
| KV append DMA | 114.156 (0.41%) | 126.448 (0.02%) |
| Block orchestration | 24.106 (0.09%) | 23.945 (0.00%) |
| Layer bookkeeping | 13.479 (0.05%) | 14.318 (0.00%) |
| Stage-boundary bookkeeping | 6.543 (0.02%) | 6.970 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 93.288 (0.34%) | 99.955 (0.01%) |
| Embedding | 62.762 (0.23%) | 61.693 (0.01%) |
| Final model RMSNorm | 60.265 (0.22%) | 60.474 (0.01%) |
| LM head＋greedy（不含 final norm） | 3319.828 (12.03%) | 4384.166 (0.59%) |
| Host input/RoPE staging | 0.000 (0.00%) | 0.000 (0.00%) |
| Host-DSP boundary | 767.360 (2.78%) | 2101.662 (0.28%) |
| Complete Host wall | 27602.735 (100.00%) | 749207.814 (100.00%) |

## 1B-64+42 decode

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 118.595 (0.53%) | 147.594 (0.25%) |
| Input RMSNorm | 869.085 (3.87%) | 865.497 (1.48%) |
| QKV＋RoPE | 1682.303 (7.49%) | 2178.439 (3.72%) |
| QK–Softmax–AV | 6338.146 (28.21%) | 6427.011 (10.97%) |
| O projection | 910.847 (4.05%) | 1180.328 (2.02%) |
| Post-attention residual＋RMSNorm | 876.873 (3.90%) | 877.054 (1.50%) |
| Gate/Up＋SwiGLU | 4846.679 (21.57%) | 6886.919 (11.76%) |
| Down | 2642.967 (11.76%) | 34704.568 (59.26%) |
| Final residual | 0.978 (0.00%) | 0.985 (0.00%) |
| KV carrier conversion | 50.120 (0.22%) | 50.205 (0.09%) |
| KV append DMA | 90.107 (0.40%) | 106.596 (0.18%) |
| Block orchestration | 16.503 (0.07%) | 16.877 (0.03%) |
| Layer bookkeeping | 9.460 (0.04%) | 9.634 (0.02%) |
| Stage-boundary bookkeeping | 1.240 (0.01%) | 1.262 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.223 (0.23%) | 51.638 (0.09%) |
| Embedding | 1.596 (0.01%) | 1.833 (0.00%) |
| Final model RMSNorm | 55.886 (0.25%) | 55.736 (0.10%) |
| LM head＋greedy（不含 final norm） | 3313.321 (14.75%) | 4384.611 (7.49%) |
| Host input/RoPE staging | 0.000 (0.00%) | 0.000 (0.00%) |
| Host-DSP boundary | 591.225 (2.63%) | 615.775 (1.05%) |
| Complete Host wall | 22467.153 (100.00%) | 58562.560 (100.00%) |

## 1B-536+46 prefill

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 1055.757 (0.49%) | 1230.323 (0.02%) |
| Input RMSNorm | 12705.997 (5.92%) | 12713.185 (0.19%) |
| QKV＋RoPE | 34887.841 (16.25%) | 39013.107 (0.58%) |
| QK–Softmax–AV | 47189.107 (21.98%) | 47927.378 (0.71%) |
| O projection | 14303.657 (6.66%) | 15316.559 (0.23%) |
| Post-attention residual＋RMSNorm | 12621.560 (5.88%) | 12635.512 (0.19%) |
| Gate/Up＋SwiGLU | 49786.455 (23.18%) | 67635.914 (1.01%) |
| Down | 28283.380 (13.17%) | 6489760.619 (96.64%) |
| Final residual | 10.683 (0.00%) | 10.766 (0.00%) |
| KV carrier conversion | 245.145 (0.11%) | 243.078 (0.00%) |
| KV append DMA | 1017.783 (0.47%) | 1194.469 (0.02%) |
| Block orchestration | 158.929 (0.07%) | 158.867 (0.00%) |
| Layer bookkeeping | 91.959 (0.04%) | 94.688 (0.00%) |
| Stage-boundary bookkeeping | 16.700 (0.01%) | 17.128 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 488.621 (0.23%) | 505.006 (0.01%) |
| Embedding | 483.080 (0.22%) | 534.993 (0.01%) |
| Final model RMSNorm | 55.268 (0.03%) | 55.768 (0.00%) |
| LM head＋greedy（不含 final norm） | 4433.464 (2.06%) | 5571.009 (0.08%) |
| Host input/RoPE staging | 67.140 (0.03%) | 93.992 (0.00%) |
| Host-DSP boundary | 6836.539 (3.18%) | 20882.860 (0.31%) |
| Complete Host wall | 214739.066 (100.00%) | 6715595.221 (100.00%) |

## 1B-536+46 decode

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 114.990 (0.58%) | 139.898 (0.25%) |
| Input RMSNorm | 871.377 (4.42%) | 867.864 (1.56%) |
| QKV＋RoPE | 1681.168 (8.53%) | 2176.537 (3.90%) |
| QK–Softmax–AV | 3624.869 (18.40%) | 3727.817 (6.68%) |
| O projection | 912.184 (4.63%) | 1177.067 (2.11%) |
| Post-attention residual＋RMSNorm | 876.674 (4.45%) | 876.820 (1.57%) |
| Gate/Up＋SwiGLU | 4835.658 (24.54%) | 6874.847 (12.32%) |
| Down | 2647.726 (13.44%) | 34629.050 (62.06%) |
| Final residual | 0.973 (0.00%) | 0.988 (0.00%) |
| KV carrier conversion | 50.200 (0.25%) | 50.234 (0.09%) |
| KV append DMA | 89.175 (0.45%) | 105.702 (0.19%) |
| Block orchestration | 16.337 (0.08%) | 16.554 (0.03%) |
| Layer bookkeeping | 9.401 (0.05%) | 9.617 (0.02%) |
| Stage-boundary bookkeeping | 1.242 (0.01%) | 1.248 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 49.543 (0.25%) | 49.650 (0.09%) |
| Embedding | 5.654 (0.03%) | 6.078 (0.01%) |
| Final model RMSNorm | 55.277 (0.28%) | 55.076 (0.10%) |
| LM head＋greedy（不含 final norm） | 3282.740 (16.66%) | 4355.613 (7.81%) |
| Host input/RoPE staging | 7.862 (0.04%) | 10.136 (0.02%) |
| Host-DSP boundary | 569.473 (2.89%) | 667.447 (1.20%) |
| Complete Host wall | 19702.525 (100.00%) | 55798.243 (100.00%) |

## 3B-64+42 prefill

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 240.806 (0.44%) | 265.327 (0.01%) |
| Input RMSNorm | 3550.833 (6.55%) | 3556.354 (0.18%) |
| QKV＋RoPE | 9154.079 (16.90%) | 10235.123 (0.53%) |
| QK–Softmax–AV | 6666.584 (12.31%) | 6697.866 (0.34%) |
| O projection | 4157.736 (7.68%) | 4449.046 (0.23%) |
| Post-attention residual＋RMSNorm | 3566.358 (6.58%) | 3568.765 (0.18%) |
| Gate/Up＋SwiGLU | 14035.837 (25.91%) | 18195.707 (0.94%) |
| Down | 7448.353 (13.75%) | 1886637.215 (97.15%) |
| Final residual | 2.698 (0.00%) | 2.852 (0.00%) |
| KV carrier conversion | 81.186 (0.15%) | 80.804 (0.00%) |
| KV append DMA | 236.225 (0.44%) | 273.699 (0.01%) |
| Block orchestration | 37.993 (0.07%) | 38.061 (0.00%) |
| Layer bookkeeping | 23.718 (0.04%) | 25.956 (0.00%) |
| Stage-boundary bookkeeping | 7.017 (0.01%) | 7.355 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 94.009 (0.17%) | 99.092 (0.01%) |
| Embedding | 77.261 (0.14%) | 76.336 (0.00%) |
| Final model RMSNorm | 85.566 (0.16%) | 86.038 (0.00%) |
| LM head＋greedy（不含 final norm） | 3932.981 (7.26%) | 5500.867 (0.28%) |
| Host input/RoPE staging | 0.000 (0.00%) | 0.000 (0.00%) |
| Host-DSP boundary | 771.145 (1.42%) | 2242.067 (0.12%) |
| Complete Host wall | 54170.387 (100.00%) | 1942038.529 (100.00%) |

## 3B-64+42 decode

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 240.748 (0.57%) | 294.996 (0.22%) |
| Input RMSNorm | 2308.623 (5.46%) | 2303.648 (1.68%) |
| QKV＋RoPE | 3921.101 (9.27%) | 5324.551 (3.88%) |
| QK–Softmax–AV | 6588.684 (15.57%) | 6795.688 (4.95%) |
| O projection | 2919.254 (6.90%) | 3973.564 (2.90%) |
| Post-attention residual＋RMSNorm | 2301.802 (5.44%) | 2301.971 (1.68%) |
| Gate/Up＋SwiGLU | 12765.658 (30.17%) | 17883.818 (13.03%) |
| Down | 6348.742 (15.01%) | 90201.976 (65.74%) |
| Final residual | 1.988 (0.00%) | 2.163 (0.00%) |
| KV carrier conversion | 12.497 (0.03%) | 12.542 (0.01%) |
| KV append DMA | 164.026 (0.39%) | 186.508 (0.14%) |
| Block orchestration | 30.062 (0.07%) | 30.279 (0.02%) |
| Layer bookkeeping | 16.495 (0.04%) | 16.800 (0.01%) |
| Stage-boundary bookkeeping | 1.852 (0.00%) | 1.916 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.434 (0.12%) | 52.053 (0.04%) |
| Embedding | 2.068 (0.00%) | 2.374 (0.00%) |
| Final model RMSNorm | 82.879 (0.20%) | 82.817 (0.06%) |
| LM head＋greedy（不含 final norm） | 3931.357 (9.29%) | 5481.415 (4.00%) |
| Host input/RoPE staging | 0.000 (0.00%) | 0.000 (0.00%) |
| Host-DSP boundary | 618.978 (1.46%) | 2253.367 (1.64%) |
| Complete Host wall | 42308.248 (100.00%) | 137202.445 (100.00%) |

## 3B-536+46 prefill

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 2386.156 (0.51%) | 2677.969 (0.02%) |
| Input RMSNorm | 32746.879 (7.02%) | 32899.266 (0.19%) |
| QKV＋RoPE | 83094.724 (17.81%) | 95481.226 (0.55%) |
| QK–Softmax–AV | 68030.999 (14.58%) | 69234.232 (0.40%) |
| O projection | 37391.041 (8.01%) | 40864.341 (0.23%) |
| Post-attention residual＋RMSNorm | 32519.070 (6.97%) | 32552.297 (0.19%) |
| Gate/Up＋SwiGLU | 126728.370 (27.16%) | 174886.846 (1.00%) |
| Down | 66967.354 (14.35%) | 16979954.551 (97.25%) |
| Final residual | 18.047 (0.00%) | 20.003 (0.00%) |
| KV carrier conversion | 673.911 (0.14%) | 671.796 (0.00%) |
| KV append DMA | 3047.754 (0.65%) | 3613.979 (0.02%) |
| Block orchestration | 289.999 (0.06%) | 292.603 (0.00%) |
| Layer bookkeeping | 163.170 (0.03%) | 175.788 (0.00%) |
| Stage-boundary bookkeeping | 22.676 (0.00%) | 23.476 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 509.288 (0.11%) | 537.086 (0.00%) |
| Embedding | 651.228 (0.14%) | 725.936 (0.00%) |
| Final model RMSNorm | 86.738 (0.02%) | 87.203 (0.00%) |
| LM head＋greedy（不含 final norm） | 3938.053 (0.84%) | 5492.639 (0.03%) |
| Host input/RoPE staging | 103.342 (0.02%) | 158.159 (0.00%) |
| Host-DSP boundary | 7235.187 (1.55%) | 20448.446 (0.12%) |
| Complete Host wall | 466603.986 (100.00%) | 17460797.842 (100.00%) |

## 3B-536+46 decode

| Module | HMX | HVX |
|---|---:|---:|
| I/O、metadata | 245.604 (0.58%) | 292.954 (0.21%) |
| Input RMSNorm | 2310.320 (5.49%) | 2305.823 (1.68%) |
| QKV＋RoPE | 3932.625 (9.34%) | 5345.667 (3.89%) |
| QK–Softmax–AV | 6300.243 (14.96%) | 6510.154 (4.74%) |
| O projection | 2921.505 (6.94%) | 3982.108 (2.90%) |
| Post-attention residual＋RMSNorm | 2304.485 (5.47%) | 2304.650 (1.68%) |
| Gate/Up＋SwiGLU | 12785.117 (30.36%) | 17945.596 (13.07%) |
| Down | 6358.623 (15.10%) | 90411.772 (65.84%) |
| Final residual | 1.977 (0.00%) | 2.016 (0.00%) |
| KV carrier conversion | 12.484 (0.03%) | 12.546 (0.01%) |
| KV append DMA | 191.998 (0.46%) | 220.928 (0.16%) |
| Block orchestration | 30.141 (0.07%) | 30.314 (0.02%) |
| Layer bookkeeping | 16.637 (0.04%) | 17.006 (0.01%) |
| Stage-boundary bookkeeping | 1.853 (0.00%) | 1.898 (0.00%) |
| DSP unattributed | 0.000 (0.00%) | 0.000 (0.00%) |
| Runtime setup/teardown | 51.322 (0.12%) | 52.861 (0.04%) |
| Embedding | 9.108 (0.02%) | 9.231 (0.01%) |
| Final model RMSNorm | 83.574 (0.20%) | 83.733 (0.06%) |
| LM head＋greedy（不含 final norm） | 3935.432 (9.34%) | 5491.378 (4.00%) |
| Host input/RoPE staging | 8.481 (0.02%) | 13.173 (0.01%) |
| Host-DSP boundary | 616.576 (1.46%) | 2286.126 (1.66%) |
| Complete Host wall | 42118.106 (100.00%) | 137319.935 (100.00%) |
