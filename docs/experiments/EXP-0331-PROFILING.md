# EXP0331 complete profiling

| Configuration | Prefill64 Host ms | Prefill token/s | Decode42 Host ms | Decode token/s | Prefill / decode wall overhead |
|---|---|---|---|---|---|
| No R4 | 34.918849 | 1832.82 | 883.905949 | 47.52 | Reference |
| Previous HVX butterfly OPT9 | 39.497256 | 1620.37 | 912.258789 | 46.04 | +13.112% / +3.208% |
| Optimized dense OPT10 | 38.179790 | 1676.28 | 926.405559 | 45.34 | +9.339% / +4.808% |
| Butterfly input/radix fusion OPT11 | 39.360892 | 1625.98 | 912.081465 | 46.05 | +12.721% / +3.188% |
| Butterfly input/radix + H12 quant fusion OPT12 | 39.243402 | 1630.85 | 913.523626 | 45.98 | +12.385% / +3.351% |

| Phase | Comparison | Wall delta | Paired95CI ratio |
|---|---|---|---|
| prefill | B11/B9 | -0.345% | [0.993819, 0.999395] |
| prefill | B12/B9 | -0.643% | [0.991081, 0.996047] |
| prefill | B12/B11 | -0.298% | [0.994486, 0.999695] |
| prefill | B11/D10 | +3.094% | [1.028239, 1.033397] |
| prefill | B12/D10 | +2.786% | [1.025253, 1.030269] |
| decode | B11/B9 | -0.019% | [0.994310, 1.005561] |
| decode | B12/B9 | +0.139% | [0.996958, 1.006210] |
| decode | B12/B11 | +0.158% | [0.996592, 1.006462] |
| decode | B11/D10 | -1.546% | [0.980845, 0.988106] |
| decode | B12/D10 | -1.391% | [0.982333, 0.989464] |

## Prefill

| Module (us; share of complete Host wall) | No R4 | Previous HVX butterfly OPT9 | Optimized dense OPT10 | Butterfly input/radix fusion OPT11 | Butterfly input/radix + H12 quant fusion OPT12 |
|---|---|---|---|---|---|
| I/O and metadata | 261.449 (0.749%) | 258.914 (0.656%) | 264.940 (0.694%) | 257.896 (0.655%) | 259.457 (0.661%) |
| Input RMSNorm | 2056.556 (5.890%) | 2134.522 (5.404%) | 2085.821 (5.463%) | 2135.966 (5.427%) | 2135.527 (5.442%) |
| QKV + Q/K Norm-RoPE | 7052.158 (20.196%) | 7045.261 (17.837%) | 7049.874 (18.465%) | 7047.291 (17.904%) | 7043.105 (17.947%) |
| QK-Softmax-AV | 3497.011 (10.015%) | 3493.634 (8.845%) | 3493.426 (9.150%) | 3491.428 (8.870%) | 3492.671 (8.900%) |
| O projection | 2096.741 (6.005%) | 2094.744 (5.304%) | 2095.607 (5.489%) | 2101.235 (5.338%) | 2094.568 (5.337%) |
| Post-attention residual + RMSNorm | 2152.059 (6.163%) | 2101.873 (5.322%) | 2107.200 (5.519%) | 2102.351 (5.341%) | 2102.513 (5.358%) |
| Gate/Up + SwiGLU + R4 | 7171.423 (20.537%) | 11737.084 (29.716%) | 10432.672 (27.325%) | 11615.126 (29.509%) | 11481.915 (29.258%) |
| Down projection | 3896.784 (11.160%) | 3905.776 (9.889%) | 3917.370 (10.260%) | 3898.105 (9.903%) | 3912.808 (9.971%) |
| Final residual | 2.598 (0.007%) | 2.544 (0.006%) | 2.485 (0.007%) | 2.604 (0.007%) | 2.539 (0.006%) |
| KV-cache carrier conversion | 139.356 (0.399%) | 137.851 (0.349%) | 138.542 (0.363%) | 138.529 (0.352%) | 139.521 (0.356%) |
| KV-cache append DMA | 281.430 (0.806%) | 282.598 (0.715%) | 280.006 (0.733%) | 282.487 (0.718%) | 281.564 (0.717%) |
| Block internal orchestration | 39.743 (0.114%) | 39.734 (0.101%) | 39.658 (0.104%) | 39.481 (0.100%) | 39.473 (0.101%) |
| Layer bookkeeping | 25.446 (0.073%) | 25.472 (0.064%) | 26.513 (0.069%) | 26.207 (0.067%) | 26.546 (0.068%) |
| Stage-boundary bookkeeping | 20.231 (0.058%) | 19.864 (0.050%) | 20.267 (0.053%) | 20.084 (0.051%) | 20.065 (0.051%) |
| DSP unattributed residual | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) |
| DSP runtime setup/teardown | 120.984 (0.346%) | 118.995 (0.301%) | 122.012 (0.320%) | 119.487 (0.304%) | 119.604 (0.305%) |
| Token embedding | 61.509 (0.176%) | 61.486 (0.156%) | 61.844 (0.162%) | 61.694 (0.157%) | 61.664 (0.157%) |
| Final model RMSNorm | 17.408 (0.050%) | 17.465 (0.044%) | 14.647 (0.038%) | 17.315 (0.044%) | 17.526 (0.045%) |
| LM head + greedy selection (excluding final norm) | 5282.566 (15.128%) | 5276.244 (13.359%) | 5290.363 (13.856%) | 5269.077 (13.387%) | 5280.629 (13.456%) |
| True Host-DSP boundary | 743.397 (2.129%) | 743.193 (1.882%) | 736.541 (1.929%) | 734.530 (1.866%) | 731.707 (1.865%) |
| Complete Host wall | 34918.849 (100.000%) | 39497.256 (100.000%) | 38179.790 (100.000%) | 39360.892 (100.000%) | 39243.402 (100.000%) |

## Decode per token

| Module (us; share of complete Host wall) | No R4 | Previous HVX butterfly OPT9 | Optimized dense OPT10 | Butterfly input/radix fusion OPT11 | Butterfly input/radix + H12 quant fusion OPT12 |
|---|---|---|---|---|---|
| I/O and metadata | 261.750 (1.244%) | 261.861 (1.206%) | 268.882 (1.219%) | 261.210 (1.203%) | 263.307 (1.211%) |
| Input RMSNorm | 366.260 (1.740%) | 366.150 (1.686%) | 365.161 (1.656%) | 366.128 (1.686%) | 366.156 (1.683%) |
| QKV + Q/K Norm-RoPE | 2532.828 (12.035%) | 2542.982 (11.708%) | 2552.622 (11.573%) | 2542.301 (11.707%) | 2547.436 (11.712%) |
| QK-Softmax-AV | 1907.512 (9.064%) | 1930.460 (8.888%) | 1923.985 (8.723%) | 1932.859 (8.901%) | 1927.041 (8.860%) |
| O projection | 1365.954 (6.491%) | 1374.827 (6.330%) | 1373.934 (6.229%) | 1375.420 (6.334%) | 1374.000 (6.317%) |
| Post-attention residual + RMSNorm | 373.328 (1.774%) | 369.124 (1.699%) | 373.204 (1.692%) | 369.154 (1.700%) | 369.065 (1.697%) |
| Gate/Up + SwiGLU + R4 | 6399.137 (30.406%) | 7006.314 (32.257%) | 7315.288 (33.165%) | 7011.520 (32.287%) | 7026.084 (32.303%) |
| Down projection | 3546.779 (16.853%) | 3567.709 (16.426%) | 3581.271 (16.236%) | 3570.128 (16.440%) | 3575.748 (16.440%) |
| Final residual | 2.013 (0.010%) | 2.010 (0.009%) | 2.028 (0.009%) | 2.014 (0.009%) | 2.009 (0.009%) |
| KV-cache carrier conversion | 164.339 (0.781%) | 164.777 (0.759%) | 164.754 (0.747%) | 164.740 (0.759%) | 164.724 (0.757%) |
| KV-cache append DMA | 138.995 (0.660%) | 140.233 (0.646%) | 148.043 (0.671%) | 139.230 (0.641%) | 144.552 (0.665%) |
| Block internal orchestration | 30.990 (0.147%) | 31.129 (0.143%) | 31.139 (0.141%) | 31.121 (0.143%) | 31.127 (0.143%) |
| Layer bookkeeping | 17.386 (0.083%) | 17.179 (0.079%) | 17.278 (0.078%) | 17.195 (0.079%) | 17.183 (0.079%) |
| Stage-boundary bookkeeping | 1.825 (0.009%) | 1.818 (0.008%) | 1.839 (0.008%) | 1.813 (0.008%) | 1.814 (0.008%) |
| DSP unattributed residual | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) | 0.000 (0.000%) |
| DSP runtime setup/teardown | 73.647 (0.350%) | 73.755 (0.340%) | 73.761 (0.334%) | 73.690 (0.339%) | 73.710 (0.339%) |
| Token embedding | 1.387 (0.007%) | 1.414 (0.007%) | 1.405 (0.006%) | 1.404 (0.006%) | 1.406 (0.006%) |
| Final model RMSNorm | 15.306 (0.073%) | 15.309 (0.070%) | 15.318 (0.069%) | 15.292 (0.070%) | 15.378 (0.071%) |
| LM head + greedy selection (excluding final norm) | 3272.289 (15.549%) | 3278.500 (15.094%) | 3290.790 (14.919%) | 3278.132 (15.095%) | 3289.667 (15.125%) |
| True Host-DSP boundary | 573.656 (2.726%) | 574.895 (2.647%) | 556.571 (2.523%) | 562.874 (2.592%) | 560.155 (2.575%) |
| Complete Host wall | 21045.380 (100.000%) | 21720.447 (100.000%) | 22057.275 (100.000%) | 21716.225 (100.000%) | 21750.563 (100.000%) |

## R4 phases

| Phase | Configuration | R4 all28 us | R4 per layer us | Prepare us | Layout / fused-worker us | HMX exposed interval us | Finish us |
|---|---|---|---|---|---|---|---|
| prefill | Previous HVX butterfly OPT9 | 3837.678 | 137.060 | 0.000 | 3834.340 | 0.000 | 0.000 |
| prefill | Optimized dense OPT10 | 2527.621 | 90.272 | 368.773 | 203.037 | 1167.760 | 777.461 |
| prefill | Butterfly input/radix fusion OPT11 | 3722.326 | 132.940 | 0.000 | 3719.174 | 0.000 | 0.000 |
| prefill | Butterfly input/radix + H12 quant fusion OPT12 | 3572.994 | 127.607 | 0.000 | 3569.820 | 0.000 | 0.000 |
| decode | Previous HVX butterfly OPT9 | 344.077 | 12.288 | 0.000 | 341.475 | 0.000 | 0.000 |
| decode | Optimized dense OPT10 | 629.901 | 22.496 | 277.547 | 27.351 | 124.419 | 193.352 |
| decode | Butterfly input/radix fusion OPT11 | 344.028 | 12.287 | 0.000 | 341.423 | 0.000 | 0.000 |
| decode | Butterfly input/radix + H12 quant fusion OPT12 | 344.046 | 12.287 | 0.000 | 341.444 | 0.000 | 0.000 |

All numeric projection, attention, MLP, worker/wait and physical counters are retained below. These counters overlap and must not be summed as engine utilization. The separate complete module ledger is additive. Decode columns are per token; complete42-step wall appears above. Primary throughput uses arithmetic mean complete Host wall; counter-median diagnostics do not replace it. Measured zeros remain zero. Auxiliary r1 has ten rotated rounds with one trajectory per arm, and is not a selection gate. Formal repeat10 has ten complete trajectories per arm per round, RPC repeat_count1.
See COUNTERS.md for complete repeat1/repeat10 counters.

## Interpretation and remaining boundary

Selected B12 reduces the prefill R4 wrapper from 137.059914 to 127.606938 us/layer (6.8970%); over28 layers this is 0.264683 ms, close to the observed complete Host-wall reduction 0.253854 ms. This scale comparison does not isolate hardware engine utilization or infer an unmeasured counter. The small E2E gain is consistent with the limited fraction of total wall occupied by the changed R4 work.

Complete prefill wall improves 0.6427% with paired95CI ratio [0.991081213180049, 0.9960469966575153], which excludes1 in this fixed ten-round run. Decode wall delta +0.1386% has CI [0.9969576752773651, 1.006210382459488] including1; no decode speedup/regression is established. Selected butterfly prefill overhead versus noR4 remains 12.3846% (95CI [1.121074501989653, 1.1270635436727117]), FAIL10percent; decode overhead 3.3508% PASS. DenseOPT10 prefill/decode gates both pass in this samebinary run, and remains faster prefill than B12. Numerical/physical validation passes despite the descriptive cost gate failure. No default changes or baseline adoption.

Further research would need to address the fullrotation completion dependency and output-consumption granularity, rather than assuming extra VTCM-pass removal must help. OPT13 is concrete counterevidence to that assumption. This round does not prove the necessity of vector butterfly over the strengthened dense comparator; their FP32/FP16 finite-precision contracts remain distinct.
