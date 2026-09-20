# Complete additive component ledgers

## prefill

| Stage | HMX us | HVX4 us |
|---|---:|---:|
| Input / residual load | 1072.250 | 1062.958 |
| Input preparation | 3008.849 | 156.757 |
| Uncovered weight DMA / submission | 9.823 | 9.977 |
| Matrix + weight unpack or raw merge + worker wait | 205.374 | 44673.495 |
| FP32 scale / residual update | 27.148 | 27.322 |
| Integer audit + FP32 output publication | 3473.703 | 3473.309 |
| Orchestration | 0.189 | 0.490 |
| Complete DSP component | 7797.336 | 49404.308 |
| Complete warm RPC wall (includes worker setup) | 8832.954 | 50523.679 |

## decode

| Stage | HMX us | HVX4 us |
|---|---:|---:|
| Input / residual load | 26.106 | 25.619 |
| Input preparation | 229.022 | 2.792 |
| Uncovered weight DMA / submission | 54.933 | 10.049 |
| Matrix + weight unpack or raw merge + worker wait | 91.057 | 2122.269 |
| FP32 scale / residual update | 0.414 | 0.584 |
| Integer audit + FP32 output publication | 54.350 | 54.385 |
| Orchestration | 0.114 | 0.134 |
| Complete DSP component | 455.997 | 2215.833 |
| Complete warm RPC wall (includes worker setup) | 930.711 | 2926.418 |
