# EXP-0333 complete quality-instrumentation profiling

Full28 W4A8 uniform INT16 Down/FP32 residual2. All eight frozen arms run the same8192 teacher-forced targets with full2048 windows and a retained tail. Repeat1 only, under scoring instrumentation. No paired repeat10 campaign, speed gate, baseline promotion or new three-recipe measurement was authorized; those sections are N/A. Historical formal speed remains separately scoped in the migration frontier. Setup/loading/external tokenization are excluded from each Host wall. The LM-head stage includes the scoring path. These are diagnostics,not a benchmark ranking.

Every invocation and every layer is additively checked. Engine-work/wait counters may overlap and must not be summed as latency. Numeric audit fields in non-audited production records are placeholders,not independent correctness evidence; use long-implementation-gates.json for correctness, cache guards and captured references. Audited boundary writes are excluded from this diagnostic campaign.

## C0 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 254.219 | 0.565 |
| Input RMSNorm | 2101.813 | 4.673 |
| QKV + Q/K Norm-RoPE | 7059.458 | 15.695 |
| QK-Softmax-AV | 3458.313 | 7.689 |
| O projection | 2080.844 | 4.626 |
| Post-attention residual + RMSNorm | 2112.250 | 4.696 |
| Gate/Up + SwiGLU | 7021.417 | 15.610 |
| Down projection | 3851.542 | 8.563 |
| Final residual | 3.312 | 0.007 |
| KV-cache carrier conversion | 139.667 | 0.311 |
| KV-cache append DMA | 448.021 | 0.996 |
| Block internal orchestration | 38.771 | 0.086 |
| Layer bookkeeping | 22.490 | 0.050 |
| Stage-boundary bookkeeping | 8.354 | 0.019 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 108.333 | 0.241 |
| Token embedding | 72.594 | 0.161 |
| Final model RMSNorm | 16.146 | 0.036 |
| LM head + greedy selection (excluding final norm) | 13720.240 | 30.503 |
| True Host-DSP boundary | 2461.760 | 5.473 |
| Complete Host wall | 44979.542 | 100.000 |

## C0 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 247.448 | 0.780 |
| Input RMSNorm | 366.471 | 1.155 |
| QKV + Q/K Norm-RoPE | 2506.274 | 7.897 |
| QK-Softmax-AV | 4593.233 | 14.472 |
| O projection | 1365.708 | 4.303 |
| Post-attention residual + RMSNorm | 370.752 | 1.168 |
| Gate/Up + SwiGLU | 6287.015 | 19.809 |
| Down projection | 3502.656 | 11.036 |
| Final residual | 1.874 | 0.006 |
| KV-cache carrier conversion | 22.085 | 0.070 |
| KV-cache append DMA | 277.454 | 0.874 |
| Block internal orchestration | 31.137 | 0.098 |
| Layer bookkeeping | 16.837 | 0.053 |
| Stage-boundary bookkeeping | 1.987 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.848 | 0.233 |
| Token embedding | 6.310 | 0.020 |
| Final model RMSNorm | 14.448 | 0.046 |
| LM head + greedy selection (excluding final norm) | 11397.181 | 35.910 |
| True Host-DSP boundary | 655.578 | 2.066 |
| Complete Host wall | 31738.296 | 100.000 |

## C2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 252.615 | 0.562 |
| Input RMSNorm | 2089.260 | 4.651 |
| QKV + Q/K Norm-RoPE | 7051.271 | 15.696 |
| QK-Softmax-AV | 3454.000 | 7.688 |
| O projection | 2076.760 | 4.623 |
| Post-attention residual + RMSNorm | 2077.573 | 4.625 |
| Gate/Up + SwiGLU | 7067.510 | 15.732 |
| Down projection | 3855.177 | 8.581 |
| Final residual | 3.083 | 0.007 |
| KV-cache carrier conversion | 139.771 | 0.311 |
| KV-cache append DMA | 449.740 | 1.001 |
| Block internal orchestration | 37.635 | 0.084 |
| Layer bookkeeping | 22.792 | 0.051 |
| Stage-boundary bookkeeping | 8.521 | 0.019 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 109.312 | 0.243 |
| Token embedding | 73.031 | 0.163 |
| Final model RMSNorm | 15.531 | 0.035 |
| LM head + greedy selection (excluding final norm) | 13696.865 | 30.489 |
| True Host-DSP boundary | 2444.156 | 5.441 |
| Complete Host wall | 44924.604 | 100.000 |

## C2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 247.976 | 0.779 |
| Input RMSNorm | 366.365 | 1.151 |
| QKV + Q/K Norm-RoPE | 2521.490 | 7.920 |
| QK-Softmax-AV | 4595.551 | 14.434 |
| O projection | 1370.447 | 4.304 |
| Post-attention residual + RMSNorm | 370.469 | 1.164 |
| Gate/Up + SwiGLU | 6331.971 | 19.888 |
| Down projection | 3526.445 | 11.076 |
| Final residual | 1.876 | 0.006 |
| KV-cache carrier conversion | 22.090 | 0.069 |
| KV-cache append DMA | 281.186 | 0.883 |
| Block internal orchestration | 31.191 | 0.098 |
| Layer bookkeeping | 16.832 | 0.053 |
| Stage-boundary bookkeeping | 1.948 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.778 | 0.232 |
| Token embedding | 6.314 | 0.020 |
| Final model RMSNorm | 14.494 | 0.046 |
| LM head + greedy selection (excluding final norm) | 11403.716 | 35.818 |
| True Host-DSP boundary | 654.198 | 2.055 |
| Complete Host wall | 31838.335 | 100.000 |

## R2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 255.396 | 0.570 |
| Input RMSNorm | 2077.052 | 4.638 |
| QKV + Q/K Norm-RoPE | 7051.052 | 15.744 |
| QK-Softmax-AV | 3414.104 | 7.623 |
| O projection | 2097.354 | 4.683 |
| Post-attention residual + RMSNorm | 2080.667 | 4.646 |
| Gate/Up + SwiGLU | 7013.229 | 15.660 |
| Down projection | 3875.490 | 8.654 |
| Final residual | 3.240 | 0.007 |
| KV-cache carrier conversion | 134.208 | 0.300 |
| KV-cache append DMA | 453.427 | 1.012 |
| Block internal orchestration | 38.573 | 0.086 |
| Layer bookkeeping | 24.521 | 0.055 |
| Stage-boundary bookkeeping | 7.802 | 0.017 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 111.458 | 0.249 |
| Token embedding | 73.688 | 0.165 |
| Final model RMSNorm | 15.708 | 0.035 |
| LM head + greedy selection (excluding final norm) | 13707.521 | 30.608 |
| True Host-DSP boundary | 2350.323 | 5.248 |
| Complete Host wall | 44784.813 | 100.000 |

## R2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 247.377 | 0.772 |
| Input RMSNorm | 366.467 | 1.144 |
| QKV + Q/K Norm-RoPE | 2691.132 | 8.400 |
| QK-Softmax-AV | 4596.669 | 14.347 |
| O projection | 1373.756 | 4.288 |
| Post-attention residual + RMSNorm | 370.363 | 1.156 |
| Gate/Up + SwiGLU | 6343.983 | 19.801 |
| Down projection | 3532.571 | 11.026 |
| Final residual | 1.870 | 0.006 |
| KV-cache carrier conversion | 14.697 | 0.046 |
| KV-cache append DMA | 287.092 | 0.896 |
| Block internal orchestration | 31.162 | 0.097 |
| Layer bookkeeping | 16.649 | 0.052 |
| Stage-boundary bookkeeping | 2.001 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.768 | 0.230 |
| Token embedding | 6.343 | 0.020 |
| Final model RMSNorm | 14.489 | 0.045 |
| LM head + greedy selection (excluding final norm) | 11415.789 | 35.631 |
| True Host-DSP boundary | 652.637 | 2.037 |
| Complete Host wall | 32038.814 | 100.000 |

## D0 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 255.323 | 0.528 |
| Input RMSNorm | 2085.302 | 4.312 |
| QKV + Q/K Norm-RoPE | 7055.542 | 14.591 |
| QK-Softmax-AV | 3470.354 | 7.177 |
| O projection | 2090.583 | 4.323 |
| Post-attention residual + RMSNorm | 2078.521 | 4.298 |
| Gate/Up + SwiGLU | 10415.052 | 21.538 |
| Down projection | 3846.385 | 7.954 |
| Final residual | 2.938 | 0.006 |
| KV-cache carrier conversion | 139.760 | 0.289 |
| KV-cache append DMA | 452.677 | 0.936 |
| Block internal orchestration | 37.708 | 0.078 |
| Layer bookkeeping | 25.510 | 0.053 |
| Stage-boundary bookkeeping | 8.500 | 0.018 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 114.531 | 0.237 |
| Token embedding | 72.292 | 0.149 |
| Final model RMSNorm | 15.979 | 0.033 |
| LM head + greedy selection (excluding final norm) | 13735.917 | 28.406 |
| True Host-DSP boundary | 2453.146 | 5.073 |
| Complete Host wall | 48356.021 | 100.000 |

## D0 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 247.986 | 0.748 |
| Input RMSNorm | 366.429 | 1.105 |
| QKV + Q/K Norm-RoPE | 2525.629 | 7.620 |
| QK-Softmax-AV | 4991.187 | 15.058 |
| O projection | 1373.556 | 4.144 |
| Post-attention residual + RMSNorm | 370.402 | 1.117 |
| Gate/Up + SwiGLU | 7200.537 | 21.724 |
| Down projection | 3539.022 | 10.677 |
| Final residual | 1.951 | 0.006 |
| KV-cache carrier conversion | 22.115 | 0.067 |
| KV-cache append DMA | 280.466 | 0.846 |
| Block internal orchestration | 31.664 | 0.096 |
| Layer bookkeeping | 16.794 | 0.051 |
| Stage-boundary bookkeeping | 1.990 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.745 | 0.222 |
| Token embedding | 6.322 | 0.019 |
| Final model RMSNorm | 14.521 | 0.044 |
| LM head + greedy selection (excluding final norm) | 11426.135 | 34.472 |
| True Host-DSP boundary | 655.657 | 1.978 |
| Complete Host wall | 33146.108 | 100.000 |

## D2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 253.729 | 0.527 |
| Input RMSNorm | 2108.260 | 4.376 |
| QKV + Q/K Norm-RoPE | 7076.542 | 14.688 |
| QK-Softmax-AV | 3445.531 | 7.152 |
| O projection | 2086.521 | 4.331 |
| Post-attention residual + RMSNorm | 2059.615 | 4.275 |
| Gate/Up + SwiGLU | 10389.625 | 21.565 |
| Down projection | 3834.167 | 7.958 |
| Final residual | 2.729 | 0.006 |
| KV-cache carrier conversion | 139.990 | 0.291 |
| KV-cache append DMA | 448.865 | 0.932 |
| Block internal orchestration | 37.552 | 0.078 |
| Layer bookkeeping | 23.677 | 0.049 |
| Stage-boundary bookkeeping | 8.354 | 0.017 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 111.427 | 0.231 |
| Token embedding | 73.427 | 0.152 |
| Final model RMSNorm | 16.240 | 0.034 |
| LM head + greedy selection (excluding final norm) | 13712.333 | 28.462 |
| True Host-DSP boundary | 2349.094 | 4.876 |
| Complete Host wall | 48177.677 | 100.000 |

## D2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 248.759 | 0.749 |
| Input RMSNorm | 366.439 | 1.104 |
| QKV + Q/K Norm-RoPE | 2528.807 | 7.617 |
| QK-Softmax-AV | 4993.796 | 15.041 |
| O projection | 1369.253 | 4.124 |
| Post-attention residual + RMSNorm | 370.715 | 1.117 |
| Gate/Up + SwiGLU | 7240.785 | 21.809 |
| Down projection | 3541.020 | 10.665 |
| Final residual | 1.952 | 0.006 |
| KV-cache carrier conversion | 22.101 | 0.067 |
| KV-cache append DMA | 283.499 | 0.854 |
| Block internal orchestration | 31.469 | 0.095 |
| Layer bookkeeping | 16.802 | 0.051 |
| Stage-boundary bookkeeping | 1.921 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.835 | 0.222 |
| Token embedding | 6.307 | 0.019 |
| Final model RMSNorm | 14.478 | 0.044 |
| LM head + greedy selection (excluding final norm) | 11436.738 | 34.447 |
| True Host-DSP boundary | 652.468 | 1.965 |
| Complete Host wall | 33201.142 | 100.000 |

## RD2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 257.719 | 0.539 |
| Input RMSNorm | 2088.052 | 4.363 |
| QKV + Q/K Norm-RoPE | 7038.333 | 14.707 |
| QK-Softmax-AV | 3426.312 | 7.160 |
| O projection | 2090.427 | 4.368 |
| Post-attention residual + RMSNorm | 2075.427 | 4.337 |
| Gate/Up + SwiGLU | 10381.188 | 21.692 |
| Down projection | 3853.781 | 8.053 |
| Final residual | 3.188 | 0.007 |
| KV-cache carrier conversion | 134.177 | 0.280 |
| KV-cache append DMA | 456.802 | 0.955 |
| Block internal orchestration | 38.062 | 0.080 |
| Layer bookkeeping | 24.896 | 0.052 |
| Stage-boundary bookkeeping | 8.198 | 0.017 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 112.583 | 0.235 |
| Token embedding | 76.115 | 0.159 |
| Final model RMSNorm | 15.938 | 0.033 |
| LM head + greedy selection (excluding final norm) | 13731.625 | 28.693 |
| True Host-DSP boundary | 2043.635 | 4.270 |
| Complete Host wall | 47856.458 | 100.000 |

## RD2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 248.420 | 0.744 |
| Input RMSNorm | 366.266 | 1.097 |
| QKV + Q/K Norm-RoPE | 2704.512 | 8.101 |
| QK-Softmax-AV | 4990.291 | 14.947 |
| O projection | 1371.065 | 4.107 |
| Post-attention residual + RMSNorm | 370.496 | 1.110 |
| Gate/Up + SwiGLU | 7252.687 | 21.724 |
| Down projection | 3542.409 | 10.611 |
| Final residual | 1.958 | 0.006 |
| KV-cache carrier conversion | 14.713 | 0.044 |
| KV-cache append DMA | 284.948 | 0.854 |
| Block internal orchestration | 31.696 | 0.095 |
| Layer bookkeeping | 16.619 | 0.050 |
| Stage-boundary bookkeeping | 2.231 | 0.007 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.799 | 0.221 |
| Token embedding | 6.337 | 0.019 |
| Final model RMSNorm | 14.475 | 0.043 |
| LM head + greedy selection (excluding final norm) | 11438.917 | 34.263 |
| True Host-DSP boundary | 653.644 | 1.958 |
| Complete Host wall | 33385.485 | 100.000 |

## B2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 255.552 | 0.518 |
| Input RMSNorm | 2105.844 | 4.269 |
| QKV + Q/K Norm-RoPE | 7047.854 | 14.287 |
| QK-Softmax-AV | 3467.323 | 7.029 |
| O projection | 2109.948 | 4.277 |
| Post-attention residual + RMSNorm | 2088.146 | 4.233 |
| Gate/Up + SwiGLU | 11391.260 | 23.091 |
| Down projection | 3836.698 | 7.777 |
| Final residual | 3.292 | 0.007 |
| KV-cache carrier conversion | 140.406 | 0.285 |
| KV-cache append DMA | 450.688 | 0.914 |
| Block internal orchestration | 39.135 | 0.079 |
| Layer bookkeeping | 26.667 | 0.054 |
| Stage-boundary bookkeeping | 8.531 | 0.017 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 111.344 | 0.226 |
| Token embedding | 74.969 | 0.152 |
| Final model RMSNorm | 14.719 | 0.030 |
| LM head + greedy selection (excluding final norm) | 13755.688 | 27.884 |
| True Host-DSP boundary | 2403.187 | 4.872 |
| Complete Host wall | 49331.250 | 100.000 |

## B2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 248.559 | 0.757 |
| Input RMSNorm | 366.339 | 1.116 |
| QKV + Q/K Norm-RoPE | 2535.140 | 7.724 |
| QK-Softmax-AV | 4866.731 | 14.828 |
| O projection | 1371.954 | 4.180 |
| Post-attention residual + RMSNorm | 370.459 | 1.129 |
| Gate/Up + SwiGLU | 6987.297 | 21.289 |
| Down projection | 3552.086 | 10.823 |
| Final residual | 1.957 | 0.006 |
| KV-cache carrier conversion | 22.103 | 0.067 |
| KV-cache append DMA | 276.144 | 0.841 |
| Block internal orchestration | 31.753 | 0.097 |
| Layer bookkeeping | 16.434 | 0.050 |
| Stage-boundary bookkeeping | 1.939 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.909 | 0.225 |
| Token embedding | 6.320 | 0.019 |
| Final model RMSNorm | 14.531 | 0.044 |
| LM head + greedy selection (excluding final norm) | 11426.692 | 34.816 |
| True Host-DSP boundary | 650.113 | 1.981 |
| Complete Host wall | 32820.460 | 100.000 |

## RB2 prefill

5 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 254.833 | 0.518 |
| Input RMSNorm | 2072.083 | 4.208 |
| QKV + Q/K Norm-RoPE | 7037.073 | 14.292 |
| QK-Softmax-AV | 3417.927 | 6.942 |
| O projection | 2108.031 | 4.281 |
| Post-attention residual + RMSNorm | 2098.396 | 4.262 |
| Gate/Up + SwiGLU | 11460.208 | 23.275 |
| Down projection | 3826.469 | 7.771 |
| Final residual | 3.312 | 0.007 |
| KV-cache carrier conversion | 134.635 | 0.273 |
| KV-cache append DMA | 448.510 | 0.911 |
| Block internal orchestration | 39.229 | 0.080 |
| Layer bookkeeping | 25.969 | 0.053 |
| Stage-boundary bookkeeping | 9.292 | 0.019 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 113.188 | 0.230 |
| Token embedding | 74.625 | 0.152 |
| Final model RMSNorm | 14.896 | 0.030 |
| LM head + greedy selection (excluding final norm) | 13727.885 | 27.881 |
| True Host-DSP boundary | 2371.344 | 4.816 |
| Complete Host wall | 49237.906 | 100.000 |

## RB2 decode

8187 records;mean per invocation;us and complete Host-wall percent.

| Module | us | Host wall % |
|---|---:|---:|
| I/O and metadata | 245.971 | 0.746 |
| Input RMSNorm | 366.166 | 1.110 |
| QKV + Q/K Norm-RoPE | 2705.101 | 8.201 |
| QK-Softmax-AV | 4865.528 | 14.750 |
| O projection | 1377.548 | 4.176 |
| Post-attention residual + RMSNorm | 370.666 | 1.124 |
| Gate/Up + SwiGLU | 6970.545 | 21.132 |
| Down projection | 3554.611 | 10.776 |
| Final residual | 1.953 | 0.006 |
| KV-cache carrier conversion | 14.696 | 0.045 |
| KV-cache append DMA | 283.364 | 0.859 |
| Block internal orchestration | 31.642 | 0.096 |
| Layer bookkeeping | 16.267 | 0.049 |
| Stage-boundary bookkeeping | 1.983 | 0.006 |
| DSP unattributed residual | 0.000 | 0.000 |
| DSP runtime setup/teardown | 73.712 | 0.223 |
| Token embedding | 6.319 | 0.019 |
| Final model RMSNorm | 14.476 | 0.044 |
| LM head + greedy selection (excluding final norm) | 11435.537 | 34.668 |
| True Host-DSP boundary | 650.001 | 1.971 |
| Complete Host wall | 32986.084 | 100.000 |
