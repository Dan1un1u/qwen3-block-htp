## prefill: F16F16 full-model modules

Microseconds (share of complete Host wall); arithmetic means across all ten formal rounds and all repetitions. Throughput ranking below uses the declared median of round means.

| Module | Current original control | Refreshed FP16 | Wall change |
|---|---:|---:|---:|
| I/O and metadata | 67.866 (0.10%) | 68.309 (0.10%) | +0.65% |
| Input RMSNorm | 282.248 (0.43%) | 282.255 (0.43%) | +0.00% |
| QKV + Q/K Norm-RoPE | 5222.676 (7.88%) | 5228.160 (7.95%) | +0.11% |
| QK-Softmax-AV | 11612.619 (17.51%) | 11611.497 (17.66%) | -0.01% |
| O projection | 3354.536 (5.06%) | 3358.187 (5.11%) | +0.11% |
| Post-attention residual + RMSNorm | 272.013 (0.41%) | 271.733 (0.41%) | -0.10% |
| Gate/Up + SwiGLU | 22971.401 (34.64%) | 23005.130 (34.99%) | +0.15% |
| Down projection | 10439.468 (15.74%) | 10467.057 (15.92%) | +0.26% |
| Final residual | 80.844 (0.12%) | 80.749 (0.12%) | -0.12% |
| KV-cache carrier conversion | 87.910 (0.13%) | 88.166 (0.13%) | +0.29% |
| KV-cache append DMA | 172.881 (0.26%) | 172.125 (0.26%) | -0.44% |
| Block internal orchestration | 14.056 (0.02%) | 14.190 (0.02%) | +0.95% |
| Layer bookkeeping | 14.873 (0.02%) | 14.733 (0.02%) | -0.94% |
| Stage-boundary bookkeeping | 8.151 (0.01%) | 7.978 (0.01%) | -2.13% |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) | N/A |
| DSP runtime setup/teardown | 93.527 (0.14%) | 94.808 (0.14%) | +1.37% |
| Token embedding | 50.990 (0.08%) | 51.340 (0.08%) | +0.69% |
| Final model RMSNorm | 48.361 (0.07%) | 3.509 (0.01%) | -92.74% |
| LM head + greedy selection (excluding final norm) | 10661.306 (16.08%) | 10078.498 (15.33%) | -5.47% |
| True Host-DSP boundary | 861.629 (1.30%) | 855.429 (1.30%) | -0.72% |
| Complete Host wall | 66317.354 (100.00%) | 65753.853 (100.00%) | -0.85% |

## decode: F16F16 full-model modules

Microseconds (share of complete Host wall); arithmetic means across all ten formal rounds and all repetitions. Throughput ranking below uses the declared median of round means.

| Module | Current original control | Refreshed FP16 | Wall change |
|---|---:|---:|---:|
| I/O and metadata | 66.789 (0.09%) | 66.865 (0.12%) | +0.11% |
| Input RMSNorm | 277.074 (0.36%) | 277.082 (0.48%) | +0.00% |
| QKV + Q/K Norm-RoPE | 5152.237 (6.68%) | 5164.371 (9.02%) | +0.24% |
| QK-Softmax-AV | 23135.597 (29.98%) | 3757.518 (6.56%) | -83.76% |
| O projection | 3319.719 (4.30%) | 3324.579 (5.81%) | +0.15% |
| Post-attention residual + RMSNorm | 267.960 (0.35%) | 267.795 (0.47%) | -0.06% |
| Gate/Up + SwiGLU | 22813.469 (29.56%) | 22859.404 (39.94%) | +0.20% |
| Down projection | 10354.033 (13.42%) | 10385.331 (18.14%) | +0.30% |
| Final residual | 79.444 (0.10%) | 79.449 (0.14%) | +0.01% |
| KV-cache carrier conversion | 135.214 (0.18%) | 135.260 (0.24%) | +0.03% |
| KV-cache append DMA | 98.107 (0.13%) | 98.204 (0.17%) | +0.10% |
| Block internal orchestration | 8.598 (0.01%) | 8.551 (0.01%) | -0.55% |
| Layer bookkeeping | 9.933 (0.01%) | 9.901 (0.02%) | -0.32% |
| Stage-boundary bookkeeping | 1.180 (0.00%) | 1.195 (0.00%) | +1.24% |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) | N/A |
| DSP runtime setup/teardown | 51.053 (0.07%) | 51.018 (0.09%) | -0.07% |
| Token embedding | 1.816 (0.00%) | 1.863 (0.00%) | +2.57% |
| Final model RMSNorm | 2.515 (0.00%) | 2.461 (0.00%) | -2.15% |
| LM head + greedy selection (excluding final norm) | 10626.582 (13.77%) | 9972.399 (17.42%) | -6.16% |
| True Host-DSP boundary | 772.317 (1.00%) | 772.719 (1.35%) | +0.05% |
| Complete Host wall | 77173.638 (100.00%) | 57235.964 (100.00%) | -25.83% |

