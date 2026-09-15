## prefill: F16F16 full-model modules

Microseconds (share of complete Host wall); arithmetic means across all ten formal rounds and all repetitions. Throughput ranking below uses the declared median of round means.

| Module | Current original control | Refreshed FP16 | Wall change |
|---|---:|---:|---:|
| I/O and metadata | 127.328 (0.16%) | 131.545 (0.16%) | +3.31% |
| Input RMSNorm | 490.077 (0.61%) | 490.423 (0.61%) | +0.07% |
| QKV + Q/K Norm-RoPE | 11617.824 (14.44%) | 11664.153 (14.55%) | +0.40% |
| QK-Softmax-AV | 3997.758 (4.97%) | 4004.121 (4.99%) | +0.16% |
| O projection | 5843.502 (7.26%) | 5867.558 (7.32%) | +0.41% |
| Post-attention residual + RMSNorm | 475.204 (0.59%) | 475.661 (0.59%) | +0.10% |
| Gate/Up + SwiGLU | 30090.501 (37.39%) | 30229.903 (37.70%) | +0.46% |
| Down projection | 13787.381 (17.13%) | 13863.481 (17.29%) | +0.55% |
| Final residual | 140.927 (0.18%) | 141.173 (0.18%) | +0.17% |
| KV-cache carrier conversion | 151.238 (0.19%) | 151.551 (0.19%) | +0.21% |
| KV-cache append DMA | 336.501 (0.42%) | 337.740 (0.42%) | +0.37% |
| Block internal orchestration | 16.901 (0.02%) | 17.416 (0.02%) | +3.04% |
| Layer bookkeeping | 25.194 (0.03%) | 25.418 (0.03%) | +0.89% |
| Stage-boundary bookkeeping | 7.060 (0.01%) | 7.266 (0.01%) | +2.91% |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) | N/A |
| DSP runtime setup/teardown | 89.333 (0.11%) | 95.105 (0.12%) | +6.46% |
| Token embedding | 64.354 (0.08%) | 67.245 (0.08%) | +4.49% |
| Final model RMSNorm | 49.551 (0.06%) | 4.660 (0.01%) | -90.59% |
| LM head + greedy selection (excluding final norm) | 12374.834 (15.38%) | 11765.491 (14.67%) | -4.92% |
| True Host-DSP boundary | 782.769 (0.97%) | 839.822 (1.05%) | +7.29% |
| Complete Host wall | 80468.236 (100.00%) | 80179.734 (100.00%) | -0.36% |

## decode: F16F16 full-model modules

Microseconds (share of complete Host wall); arithmetic means across all ten formal rounds and all repetitions. Throughput ranking below uses the declared median of round means.

| Module | Current original control | Refreshed FP16 | Wall change |
|---|---:|---:|---:|
| I/O and metadata | 127.734 (0.12%) | 131.941 (0.17%) | +3.29% |
| Input RMSNorm | 485.256 (0.44%) | 485.119 (0.61%) | -0.03% |
| QKV + Q/K Norm-RoPE | 11493.246 (10.41%) | 11534.495 (14.43%) | +0.36% |
| QK-Softmax-AV | 33328.371 (30.19%) | 4470.987 (5.59%) | -86.59% |
| O projection | 5793.850 (5.25%) | 5804.648 (7.26%) | +0.19% |
| Post-attention residual + RMSNorm | 470.716 (0.43%) | 470.404 (0.59%) | -0.07% |
| Gate/Up + SwiGLU | 29922.019 (27.10%) | 30046.524 (37.60%) | +0.42% |
| Down projection | 13682.250 (12.39%) | 13747.340 (17.20%) | +0.48% |
| Final residual | 139.489 (0.13%) | 139.484 (0.17%) | -0.00% |
| KV-cache carrier conversion | 421.277 (0.38%) | 421.270 (0.53%) | -0.00% |
| KV-cache append DMA | 207.059 (0.19%) | 208.064 (0.26%) | +0.49% |
| Block internal orchestration | 12.605 (0.01%) | 12.444 (0.02%) | -1.27% |
| Layer bookkeeping | 17.245 (0.02%) | 17.355 (0.02%) | +0.64% |
| Stage-boundary bookkeeping | 1.730 (0.00%) | 1.731 (0.00%) | +0.06% |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) | N/A |
| DSP runtime setup/teardown | 50.970 (0.05%) | 51.211 (0.06%) | +0.47% |
| Token embedding | 2.016 (0.00%) | 2.315 (0.00%) | +14.85% |
| Final model RMSNorm | 2.505 (0.00%) | 2.701 (0.00%) | +7.84% |
| LM head + greedy selection (excluding final norm) | 12382.419 (11.22%) | 11653.366 (14.58%) | -5.89% |
| True Host-DSP boundary | 1855.282 (1.68%) | 715.677 (0.90%) | -61.42% |
| Complete Host wall | 110396.037 (100.00%) | 79917.077 (100.00%) | -27.61% |

