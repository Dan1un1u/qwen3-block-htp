# EXP0330 complete profiling

| Configuration | Prefill64 Host ms | Prefill token/s | Decode42 Host ms | Decode token/s | Prefill / decode wall overhead |
|---|---|---|---|---|---|
| No R4 | 34.599112 | 1849.76 | 875.462527 | 47.97 | Reference |
| HVX butterfly OPT9 | 39.290246 | 1628.90 | 903.562820 | 46.48 | +13.559% / +3.210% |
| Prior dense OPT7 | 41.538493 | 1540.74 | 915.734725 | 45.86 | +20.057% / +4.600% |
| Dense tiled bulk OPT9 | 38.121949 | 1678.82 | 915.509422 | 45.88 | +10.182% / +4.574% |
| Dense tiled bulk / 4 HVX OPT10 | 37.916179 | 1687.93 | 916.351984 | 45.83 | +9.587% / +4.671% |

| Phase | Comparison | Wall delta | Paired95CI ratio |
|---|---|---|---|
| prefill | D9/D7 | -8.225% | [0.915931, 0.919703] |
| prefill | D10/D7 | -8.720% | [0.912124, 0.913486] |
| prefill | D10/D9 | -0.540% | [0.992789, 0.996582] |
| prefill | D9/B | -2.974% | [0.968782, 0.971823] |
| prefill | D10/B | -3.497% | [0.963992, 0.966058] |
| decode | D9/D7 | -0.025% | [0.998557, 1.001181] |
| decode | D10/D7 | +0.067% | [0.998627, 1.003134] |
| decode | D10/D9 | +0.092% | [0.999160, 1.003394] |
| decode | D9/B | +1.322% | [1.012204, 1.014208] |
| decode | D10/B | +1.415% | [1.012455, 1.016075] |

## Prefill

| Module (us; share of complete Host wall) | No R4 | HVX butterfly OPT9 | Prior dense OPT7 | Dense tiled bulk OPT9 | Dense tiled bulk / 4 HVX OPT10 |
|---|---|---|---|---|---|
| I/O and metadata | 232.219 (0.671%) | 231.466 (0.589%) | 230.789 (0.556%) | 230.820 (0.605%) | 232.452 (0.613%) |
| Input RMSNorm | 2055.743 (5.942%) | 2135.140 (5.434%) | 2085.867 (5.022%) | 2086.355 (5.473%) | 2084.623 (5.498%) |
| QKV + Q/K Norm-RoPE | 7033.997 (20.330%) | 7039.411 (17.916%) | 7035.231 (16.937%) | 7035.692 (18.456%) | 7034.515 (18.553%) |
| QK-Softmax-AV | 3497.501 (10.109%) | 3496.377 (8.899%) | 3496.136 (8.417%) | 3498.204 (9.176%) | 3498.407 (9.227%) |
| O projection | 2095.788 (6.057%) | 2093.609 (5.329%) | 2090.097 (5.032%) | 2096.702 (5.500%) | 2093.366 (5.521%) |
| Post-attention residual + RMSNorm | 2153.807 (6.225%) | 2102.147 (5.350%) | 2107.177 (5.073%) | 2107.702 (5.529%) | 2107.754 (5.559%) |
| Gate/Up + SwiGLU + R4 | 7070.435 (20.435%) | 11711.254 (29.807%) | 14029.857 (33.776%) | 10615.081 (27.845%) | 10384.043 (27.387%) |
| Down projection | 3825.351 (11.056%) | 3843.626 (9.783%) | 3823.646 (9.205%) | 3822.704 (10.028%) | 3836.023 (10.117%) |
| Final residual | 2.196 (0.006%) | 2.231 (0.006%) | 2.102 (0.005%) | 2.238 (0.006%) | 2.249 (0.006%) |
| KV-cache carrier conversion | 137.826 (0.398%) | 138.336 (0.352%) | 138.023 (0.332%) | 137.563 (0.361%) | 137.509 (0.363%) |
| KV-cache append DMA | 290.281 (0.839%) | 290.988 (0.741%) | 290.360 (0.699%) | 290.405 (0.762%) | 290.940 (0.767%) |
| Block internal orchestration | 38.528 (0.111%) | 39.135 (0.100%) | 38.549 (0.093%) | 38.561 (0.101%) | 39.129 (0.103%) |
| Layer bookkeeping | 24.992 (0.072%) | 25.508 (0.065%) | 26.403 (0.064%) | 26.377 (0.069%) | 26.173 (0.069%) |
| Stage-boundary bookkeeping | 20.008 (0.058%) | 20.074 (0.051%) | 20.151 (0.049%) | 20.016 (0.053%) | 20.095 (0.053%) |
| DSP unattributed residual | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) |
| DSP runtime setup/teardown | 115.081 (0.333%) | 116.924 (0.298%) | 116.386 (0.280%) | 116.677 (0.306%) | 116.947 (0.308%) |
| Token embedding | 60.560 (0.175%) | 60.462 (0.154%) | 60.565 (0.146%) | 60.535 (0.159%) | 60.654 (0.160%) |
| Final model RMSNorm | 16.947 (0.049%) | 17.010 (0.043%) | 13.953 (0.034%) | 14.009 (0.037%) | 14.129 (0.037%) |
| LM head + greedy selection (excluding final norm) | 5178.380 (14.967%) | 5178.574 (13.180%) | 5173.372 (12.454%) | 5177.517 (13.581%) | 5178.066 (13.657%) |
| True Host-DSP boundary | 749.471 (2.166%) | 747.972 (1.904%) | 759.831 (1.829%) | 744.791 (1.954%) | 759.104 (2.002%) |
| Complete Host wall | 34599.112 (100.000%) | 39290.246 (100.000%) | 41538.493 (100.000%) | 38121.949 (100.000%) | 37916.179 (100.000%) |

## Decode per token

| Module (us; share of complete Host wall) | No R4 | HVX butterfly OPT9 | Prior dense OPT7 | Dense tiled bulk OPT9 | Dense tiled bulk / 4 HVX OPT10 |
|---|---|---|---|---|---|
| I/O and metadata | 227.848 (1.093%) | 229.703 (1.068%) | 229.750 (1.054%) | 229.859 (1.055%) | 231.749 (1.062%) |
| Input RMSNorm | 366.968 (1.761%) | 366.853 (1.705%) | 365.827 (1.678%) | 365.825 (1.678%) | 365.835 (1.677%) |
| QKV + Q/K Norm-RoPE | 2498.784 (11.988%) | 2510.429 (11.669%) | 2511.005 (11.517%) | 2511.607 (11.522%) | 2512.493 (11.516%) |
| QK-Softmax-AV | 1949.297 (9.352%) | 1965.746 (9.137%) | 1968.475 (9.028%) | 1970.067 (9.038%) | 1968.588 (9.023%) |
| O projection | 1364.509 (6.546%) | 1371.077 (6.373%) | 1370.074 (6.284%) | 1372.984 (6.299%) | 1370.392 (6.281%) |
| Post-attention residual + RMSNorm | 373.905 (1.794%) | 369.460 (1.717%) | 373.524 (1.713%) | 373.623 (1.714%) | 373.582 (1.712%) |
| Gate/Up + SwiGLU + R4 | 6312.505 (30.284%) | 6923.884 (32.184%) | 7200.245 (33.024%) | 7201.745 (33.039%) | 7200.007 (33.000%) |
| Down projection | 3518.473 (16.880%) | 3543.210 (16.470%) | 3543.929 (16.254%) | 3545.507 (16.265%) | 3542.060 (16.235%) |
| Final residual | 1.776 (0.009%) | 1.782 (0.008%) | 1.788 (0.008%) | 1.790 (0.008%) | 1.800 (0.008%) |
| KV-cache carrier conversion | 164.515 (0.789%) | 164.683 (0.765%) | 164.751 (0.756%) | 164.799 (0.756%) | 164.768 (0.755%) |
| KV-cache append DMA | 109.912 (0.527%) | 111.639 (0.519%) | 111.492 (0.511%) | 111.775 (0.513%) | 111.571 (0.511%) |
| Block internal orchestration | 30.122 (0.145%) | 29.976 (0.139%) | 29.923 (0.137%) | 29.953 (0.137%) | 29.937 (0.137%) |
| Layer bookkeeping | 17.389 (0.083%) | 16.835 (0.078%) | 17.432 (0.080%) | 17.436 (0.080%) | 17.423 (0.080%) |
| Stage-boundary bookkeeping | 1.758 (0.008%) | 1.756 (0.008%) | 1.757 (0.008%) | 1.760 (0.008%) | 1.759 (0.008%) |
| DSP unattributed residual | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) |
| DSP runtime setup/teardown | 73.801 (0.354%) | 73.946 (0.344%) | 73.992 (0.339%) | 73.931 (0.339%) | 73.930 (0.339%) |
| Token embedding | 1.413 (0.007%) | 1.438 (0.007%) | 1.415 (0.006%) | 1.428 (0.007%) | 1.426 (0.007%) |
| Final model RMSNorm | 14.661 (0.070%) | 14.722 (0.068%) | 14.653 (0.067%) | 14.700 (0.067%) | 14.752 (0.068%) |
| LM head + greedy selection (excluding final norm) | 3225.164 (15.473%) | 3235.690 (15.040%) | 3239.260 (14.857%) | 3238.850 (14.859%) | 3234.963 (14.827%) |
| True Host-DSP boundary | 591.546 (2.838%) | 580.569 (2.699%) | 583.916 (2.678%) | 570.206 (2.616%) | 600.870 (2.754%) |
| Complete Host wall | 20844.346 (100.000%) | 21513.400 (100.000%) | 21803.208 (100.000%) | 21797.843 (100.000%) | 21817.904 (100.000%) |

## R4 phases

| Phase | Configuration | R4 all28 us | R4 per layer us | Prepare us | Layout / fused-worker us | HMX exposed interval us | Finish us |
|---|---|---|---|---|---|---|---|
| prefill | HVX butterfly OPT9 | 3858.207 | 137.793 | 0.000 | 3854.731 | 0.000 | 0.000 |
| prefill | Prior dense OPT7 | 6152.168 | 219.720 | 288.681 | 1768.817 | 717.708 | 3303.037 |
| prefill | Dense tiled bulk OPT9 | 2759.106 | 98.540 | 370.205 | 203.251 | 1169.116 | 1005.971 |
| prefill | Dense tiled bulk / 4 HVX OPT10 | 2530.819 | 90.386 | 370.106 | 203.311 | 1168.673 | 778.402 |
| decode | HVX butterfly OPT9 | 348.325 | 12.440 | 0.000 | 346.309 | 0.000 | 0.000 |
| decode | Prior dense OPT7 | 631.139 | 22.541 | 277.738 | 27.257 | 125.503 | 193.462 |
| decode | Dense tiled bulk OPT9 | 631.844 | 22.566 | 277.719 | 27.270 | 126.212 | 193.478 |
| decode | Dense tiled bulk / 4 HVX OPT10 | 631.485 | 22.553 | 277.736 | 27.270 | 125.840 | 193.481 |

All numeric projection, attention, MLP, worker/wait and physical counters are retained below. These counters overlap and must not be summed as engine utilization. The separate complete module ledger is additive. Decode columns are per token; complete42-step wall appears above. Primary throughput uses arithmetic mean complete Host wall; counter-median diagnostics do not replace it. Measured zeros remain zero. Auxiliary r1 has ten rotated rounds with one trajectory per arm, and is not a selection gate. Formal repeat10 has ten complete trajectories per arm per round, RPC repeat_count1.
See COUNTERS.md for every captured numeric field and repeat1/repeat10 median comparison.
