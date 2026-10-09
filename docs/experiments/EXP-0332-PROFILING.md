# EXP0332 diagnostic full profiling

One fixed64+42 trajectory per arm, repeat1 auxiliary only. No formal speed gate or baseline promotion. Values are microseconds and Host-wall percentages; decode is per-token mean. All phase ledgers and28 layer ledgers verified. Setup/model loading and external tokenization excluded from timed RPC wall.

## D0 prefill

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 247.031 | 0.622 |
| Input RMSNorm | 2084.688 | 5.247 |
| QKV + Q/K Norm-RoPE | 7034.323 | 17.706 |
| QK-Softmax-AV | 3508.125 | 8.830 |
| O projection | 2109.271 | 5.309 |
| Post-attention residual + RMSNorm | 2107.240 | 5.304 |
| Gate/Up + SwiGLU | 10444.896 | 26.290 |
| Down projection | 3835.677 | 9.655 |
| Final residual | 2.760 | 0.007 |
| KV-cache carrier conversion | 134.844 | 0.339 |
| KV-cache append DMA | 296.719 | 0.747 |
| Block internal orchestration | 38.021 | 0.096 |
| Layer bookkeeping | 24.740 | 0.062 |
| Stage-boundary bookkeeping | 20.729 | 0.052 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 112.396 | 0.283 |
| Token embedding | 61.094 | 0.154 |
| Final model RMSNorm | 14.583 | 0.037 |
| LM head + greedy selection (excluding final norm) | 5200.781 | 13.091 |
| True Host-DSP boundary | 2451.197 | 6.170 |
| Complete Host wall | 39729.114 | 100.000 |

End-to-end token/s: 1610.909

## D0 decode

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 229.670 | 1.045 |
| Input RMSNorm | 365.501 | 1.663 |
| QKV + Q/K Norm-RoPE | 2494.050 | 11.347 |
| QK-Softmax-AV | 1962.779 | 8.930 |
| O projection | 1373.047 | 6.247 |
| Post-attention residual + RMSNorm | 372.912 | 1.697 |
| Gate/Up + SwiGLU | 7119.008 | 32.389 |
| Down projection | 3536.882 | 16.091 |
| Final residual | 1.987 | 0.009 |
| KV-cache carrier conversion | 164.815 | 0.750 |
| KV-cache append DMA | 113.513 | 0.516 |
| Block internal orchestration | 30.913 | 0.141 |
| Layer bookkeeping | 17.333 | 0.079 |
| Stage-boundary bookkeeping | 1.828 | 0.008 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.865 | 0.336 |
| Token embedding | 1.358 | 0.006 |
| Final model RMSNorm | 14.694 | 0.067 |
| LM head + greedy selection (excluding final norm) | 3205.050 | 14.582 |
| True Host-DSP boundary | 900.818 | 4.098 |
| Complete Host wall | 21980.022 | 100.000 |

End-to-end token/s: 45.496

## D2 prefill

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 250.000 | 0.629 |
| Input RMSNorm | 2041.198 | 5.136 |
| QKV + Q/K Norm-RoPE | 7089.167 | 17.839 |
| QK-Softmax-AV | 3495.938 | 8.797 |
| O projection | 2075.521 | 5.223 |
| Post-attention residual + RMSNorm | 2158.698 | 5.432 |
| Gate/Up + SwiGLU | 10401.458 | 26.174 |
| Down projection | 3865.625 | 9.727 |
| Final residual | 2.917 | 0.007 |
| KV-cache carrier conversion | 142.344 | 0.358 |
| KV-cache append DMA | 296.302 | 0.746 |
| Block internal orchestration | 38.021 | 0.096 |
| Layer bookkeeping | 25.000 | 0.063 |
| Stage-boundary bookkeeping | 21.198 | 0.053 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 109.115 | 0.275 |
| Token embedding | 59.375 | 0.149 |
| Final model RMSNorm | 14.688 | 0.037 |
| LM head + greedy selection (excluding final norm) | 5168.802 | 13.006 |
| True Host-DSP boundary | 2484.843 | 6.253 |
| Complete Host wall | 39740.208 | 100.000 |

End-to-end token/s: 1610.460

## D2 decode

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 229.000 | 1.045 |
| Input RMSNorm | 366.285 | 1.671 |
| QKV + Q/K Norm-RoPE | 2496.839 | 11.392 |
| QK-Softmax-AV | 1963.636 | 8.959 |
| O projection | 1365.743 | 6.231 |
| Post-attention residual + RMSNorm | 371.682 | 1.696 |
| Gate/Up + SwiGLU | 7145.568 | 32.601 |
| Down projection | 3526.135 | 16.088 |
| Final residual | 1.990 | 0.009 |
| KV-cache carrier conversion | 164.366 | 0.750 |
| KV-cache append DMA | 114.757 | 0.524 |
| Block internal orchestration | 31.065 | 0.142 |
| Layer bookkeeping | 17.293 | 0.079 |
| Stage-boundary bookkeeping | 1.818 | 0.008 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.744 | 0.336 |
| Token embedding | 1.395 | 0.006 |
| Final model RMSNorm | 14.288 | 0.065 |
| LM head + greedy selection (excluding final norm) | 3199.396 | 14.597 |
| True Host-DSP boundary | 833.099 | 3.801 |
| Complete Host wall | 21918.099 | 100.000 |

End-to-end token/s: 45.624

## RD2 prefill

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 246.875 | 0.622 |
| Input RMSNorm | 2115.312 | 5.332 |
| QKV + Q/K Norm-RoPE | 7027.760 | 17.714 |
| QK-Softmax-AV | 3439.427 | 8.669 |
| O projection | 2118.125 | 5.339 |
| Post-attention residual + RMSNorm | 2069.219 | 5.216 |
| Gate/Up + SwiGLU | 10389.427 | 26.188 |
| Down projection | 3861.250 | 9.733 |
| Final residual | 2.708 | 0.007 |
| KV-cache carrier conversion | 136.198 | 0.343 |
| KV-cache append DMA | 297.500 | 0.750 |
| Block internal orchestration | 39.323 | 0.099 |
| Layer bookkeeping | 25.573 | 0.064 |
| Stage-boundary bookkeeping | 21.146 | 0.053 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 107.917 | 0.272 |
| Token embedding | 61.719 | 0.156 |
| Final model RMSNorm | 16.719 | 0.042 |
| LM head + greedy selection (excluding final norm) | 5212.448 | 13.138 |
| True Host-DSP boundary | 2484.479 | 6.262 |
| Complete Host wall | 39673.125 | 100.000 |

End-to-end token/s: 1613.183

## RD2 decode

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 230.417 | 1.042 |
| Input RMSNorm | 363.694 | 1.645 |
| QKV + Q/K Norm-RoPE | 2670.818 | 12.082 |
| QK-Softmax-AV | 1967.233 | 8.899 |
| O projection | 1375.955 | 6.225 |
| Post-attention residual + RMSNorm | 370.402 | 1.676 |
| Gate/Up + SwiGLU | 7152.572 | 32.357 |
| Down projection | 3520.233 | 15.925 |
| Final residual | 1.980 | 0.009 |
| KV-cache carrier conversion | 156.172 | 0.706 |
| KV-cache append DMA | 117.839 | 0.533 |
| Block internal orchestration | 31.135 | 0.141 |
| Layer bookkeeping | 17.164 | 0.078 |
| Stage-boundary bookkeeping | 1.863 | 0.008 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 74.028 | 0.335 |
| Token embedding | 1.381 | 0.006 |
| Final model RMSNorm | 14.182 | 0.064 |
| LM head + greedy selection (excluding final norm) | 3200.782 | 14.480 |
| True Host-DSP boundary | 837.313 | 3.788 |
| Complete Host wall | 22105.163 | 100.000 |

End-to-end token/s: 45.238

## B2 prefill

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 251.042 | 0.617 |
| Input RMSNorm | 2139.583 | 5.256 |
| QKV + Q/K Norm-RoPE | 7048.333 | 17.313 |
| QK-Softmax-AV | 3503.906 | 8.607 |
| O projection | 2092.083 | 5.139 |
| Post-attention residual + RMSNorm | 2040.000 | 5.011 |
| Gate/Up + SwiGLU | 11444.948 | 28.113 |
| Down projection | 3804.844 | 9.346 |
| Final residual | 2.969 | 0.007 |
| KV-cache carrier conversion | 136.198 | 0.335 |
| KV-cache append DMA | 296.823 | 0.729 |
| Block internal orchestration | 44.219 | 0.109 |
| Layer bookkeeping | 28.125 | 0.069 |
| Stage-boundary bookkeeping | 22.812 | 0.056 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 113.646 | 0.279 |
| Token embedding | 65.625 | 0.161 |
| Final model RMSNorm | 14.740 | 0.036 |
| LM head + greedy selection (excluding final norm) | 5213.958 | 12.807 |
| True Host-DSP boundary | 2447.239 | 6.011 |
| Complete Host wall | 40711.093 | 100.000 |

End-to-end token/s: 1572.053

## B2 decode

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 228.698 | 1.049 |
| Input RMSNorm | 363.342 | 1.667 |
| QKV + Q/K Norm-RoPE | 2496.890 | 11.453 |
| QK-Softmax-AV | 1966.380 | 9.020 |
| O projection | 1364.546 | 6.259 |
| Post-attention residual + RMSNorm | 369.557 | 1.695 |
| Gate/Up + SwiGLU | 6879.567 | 31.556 |
| Down projection | 3531.617 | 16.199 |
| Final residual | 1.982 | 0.009 |
| KV-cache carrier conversion | 164.818 | 0.756 |
| KV-cache append DMA | 114.050 | 0.523 |
| Block internal orchestration | 31.058 | 0.142 |
| Layer bookkeeping | 17.029 | 0.078 |
| Stage-boundary bookkeeping | 1.833 | 0.008 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.684 | 0.338 |
| Token embedding | 1.364 | 0.006 |
| Final model RMSNorm | 14.721 | 0.068 |
| LM head + greedy selection (excluding final norm) | 3206.128 | 14.706 |
| True Host-DSP boundary | 973.890 | 4.467 |
| Complete Host wall | 21801.155 | 100.000 |

End-to-end token/s: 45.869

## RB2 prefill

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 245.469 | 0.605 |
| Input RMSNorm | 2078.229 | 5.119 |
| QKV + Q/K Norm-RoPE | 7036.250 | 17.333 |
| QK-Softmax-AV | 3429.844 | 8.449 |
| O projection | 2106.719 | 5.190 |
| Post-attention residual + RMSNorm | 2033.646 | 5.010 |
| Gate/Up + SwiGLU | 11419.062 | 28.130 |
| Down projection | 3789.062 | 9.334 |
| Final residual | 2.865 | 0.007 |
| KV-cache carrier conversion | 133.802 | 0.330 |
| KV-cache append DMA | 291.042 | 0.717 |
| Block internal orchestration | 37.865 | 0.093 |
| Layer bookkeeping | 25.312 | 0.062 |
| Stage-boundary bookkeeping | 20.469 | 0.050 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 105.781 | 0.261 |
| Token embedding | 60.729 | 0.150 |
| Final model RMSNorm | 17.656 | 0.043 |
| LM head + greedy selection (excluding final norm) | 5213.125 | 12.842 |
| True Host-DSP boundary | 2547.500 | 6.275 |
| Complete Host wall | 40594.427 | 100.000 |

End-to-end token/s: 1576.571

## RB2 decode

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 226.797 | 1.029 |
| Input RMSNorm | 365.246 | 1.657 |
| QKV + Q/K Norm-RoPE | 2670.570 | 12.119 |
| QK-Softmax-AV | 1967.530 | 8.929 |
| O projection | 1364.284 | 6.191 |
| Post-attention residual + RMSNorm | 367.416 | 1.667 |
| Gate/Up + SwiGLU | 6896.714 | 31.297 |
| Down projection | 3535.382 | 16.044 |
| Final residual | 1.918 | 0.009 |
| KV-cache carrier conversion | 155.569 | 0.706 |
| KV-cache append DMA | 118.558 | 0.538 |
| Block internal orchestration | 31.270 | 0.142 |
| Layer bookkeeping | 16.904 | 0.077 |
| Stage-boundary bookkeeping | 2.070 | 0.009 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.852 | 0.335 |
| Token embedding | 1.401 | 0.006 |
| Final model RMSNorm | 14.501 | 0.066 |
| LM head + greedy selection (excluding final norm) | 3199.542 | 14.519 |
| True Host-DSP boundary | 1026.652 | 4.659 |
| Complete Host wall | 22036.176 | 100.000 |

End-to-end token/s: 45.380
