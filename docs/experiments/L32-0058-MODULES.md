# Complete module ledgers

Microseconds; decode per token. DSP counters are converted at19.2MHz.

## prefill

| Module | SP2 us (% Host) | INT16 us (% Host) |
|---|---:|---:|
| I/O、metadata | 3084.540 (0.490%) | 3072.992 (0.488%) |
| Input RMSNorm | 43706.017 (6.938%) | 43698.024 (6.940%) |
| QKV＋RoPE | 110365.728 (17.519%) | 110392.309 (17.532%) |
| QK–Softmax–AV | 106576.947 (16.917%) | 106571.504 (16.925%) |
| O projection | 49535.287 (7.863%) | 49547.931 (7.869%) |
| Post-attention residual＋RMSNorm | 43263.993 (6.867%) | 43218.080 (6.864%) |
| Gate/Up＋SwiGLU | 164895.440 (26.174%) | 164552.055 (26.133%) |
| Down | 88881.950 (14.109%) | 88883.873 (14.116%) |
| Final residual | 22.039 (0.003%) | 22.146 (0.004%) |
| KV carrier conversion | 967.272 (0.154%) | 967.491 (0.154%) |
| KV append DMA | 4329.353 (0.687%) | 4343.573 (0.690%) |
| Block orchestration | 386.261 (0.061%) | 385.899 (0.061%) |
| Layer bookkeeping | 219.688 (0.035%) | 219.488 (0.035%) |
| Stage-boundary bookkeeping | 30.141 (0.005%) | 29.941 (0.005%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 661.658 (0.105%) | 662.388 (0.105%) |
| Embedding | 893.560 (0.142%) | 889.239 (0.141%) |
| Final model RMSNorm | 82.631 (0.013%) | 86.494 (0.014%) |
| LM head＋greedy（不含 final norm） | 3844.081 (0.610%) | 3862.476 (0.613%) |
| Host input/RoPE staging | 116.595 (0.019%) | 114.701 (0.018%) |
| Host-DSP boundary | 8123.793 (1.290%) | 8151.712 (1.295%) |
| Complete Host wall | 629986.976 (100.000%) | 629672.315 (100.000%) |
## decode

| Module | SP2 us (% Host) | INT16 us (% Host) |
|---|---:|---:|
| I/O、metadata | 253.800 (0.595%) | 251.503 (0.590%) |
| Input RMSNorm | 2313.945 (5.424%) | 2309.524 (5.414%) |
| QKV＋RoPE | 3833.441 (8.985%) | 3837.985 (8.997%) |
| QK–Softmax–AV | 7605.723 (17.828%) | 7601.269 (17.819%) |
| O projection | 2884.397 (6.761%) | 2886.111 (6.766%) |
| Post-attention residual＋RMSNorm | 2299.627 (5.390%) | 2301.776 (5.396%) |
| Gate/Up＋SwiGLU | 12443.184 (29.166%) | 12432.845 (29.146%) |
| Down | 6212.597 (14.562%) | 6207.712 (14.553%) |
| Final residual | 1.654 (0.004%) | 1.657 (0.004%) |
| KV carrier conversion | 15.725 (0.037%) | 15.635 (0.037%) |
| KV append DMA | 209.706 (0.492%) | 204.194 (0.479%) |
| Block orchestration | 30.643 (0.072%) | 30.640 (0.072%) |
| Layer bookkeeping | 17.219 (0.040%) | 17.227 (0.040%) |
| Stage-boundary bookkeeping | 1.920 (0.005%) | 1.914 (0.004%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 52.043 (0.122%) | 52.088 (0.122%) |
| Embedding | 9.058 (0.021%) | 9.128 (0.021%) |
| Final model RMSNorm | 86.073 (0.202%) | 83.924 (0.197%) |
| LM head＋greedy（不含 final norm） | 3841.390 (9.004%) | 3852.946 (9.032%) |
| Host input/RoPE staging | 6.220 (0.015%) | 6.616 (0.016%) |
| Host-DSP boundary | 544.234 (1.276%) | 552.448 (1.295%) |
| Complete Host wall | 42662.600 (100.000%) | 42657.141 (100.000%) |
