# EXP-0300 interim result: completed model only

Qwen1.7B and Llama1B completed five short and ten formal repeat10 rounds. All prefill and same-length decode confidence gates pass. Qwen0.6 and A8/SP2 unchanged. Fixed project trajectories, not named datasets.

Llama3B has passed all16 A16 numerical cases, A8/SP2 regressions and five short repeat10 rounds. Formal rounds1-9 completed; round10 is incomplete. ADB lost the device twice. On first reconnection uptime was below1 minute, battery2%, temperature51.6 C. The underlying cause is not proven. No more automatic hardware retries until power and temperature recover.

All five complete W4 B replays before the first disconnection were retained, including slow samples, and joined to five continuations. The partial replay remains archived. The second disconnection occurred at W4 C; final C candidate/control remain incomplete. No Llama3B formal result is promoted or entered in the workbook.

Workbook update: only12 completed A16 rows for Qwen1.7B and Llama1B across ABC. Preserve previous Llama3B values, Qwen0.6, A8/SP2, all D rows and historical quality caveats. Both experiments remain active for task recovery; no lock clearing or baseline promotion.

## Completed measurement

# EXP-0300 Qwen3-1.7B generic A16 long-context repair

Fixed token fixtures, not named datasets. Full Host wall includes embedding, all layers, final norm, LM head, greedy, FastRPC and long host staging. Cold loading and external tokenizer excluded. Five short and ten formal rotated paired rounds, ten retained replays each. Repeat1 auxiliary only.

| Configuration | Prompt | Decode | Prefill token/s | Decode token/s |
|---|---:|---:|---:|---:|
| A W16A16 | 64 | 42 | 778.91 | 12.50 |
| A W4A16 | 64 | 42 | 1038.81 | 15.32 |
| B W16A16 | 536 | 46 | 766.14 | 11.71 |
| B W4A16 | 536 | 46 | 954.48 | 14.19 |
| C W16A16 | 741 | 3 | 777.17 | 11.42 |
| C W4A16 | 741 | 3 | 948.60 | 13.78 |

## Paired confidence gates

Reference is the faster eligible same-campaign unchanged native M64 or long64 prefill. Prefill ratio lower95% >=0.90. Identical-long-shape candidate/control decode wall upper95% <=1.10. Bootstrap10000, fixed seeds. Cross-context decode is not this regression guard.

| Configuration | Shape | Reference | Prefill ratio [95%CI] | Decode wall ratio [95%CI] | Gate |
|---|---|---|---:|---:|---|
| w16 | B | A0 | 0.98372 [0.97686,0.99252] | 0.99352 [0.97837,1.01059] | pass |
| w16 | C | A0 | 0.99795 [0.98640,1.00952] | 1.00001 [0.98525,1.01540] | pass |
| w4 | B | A0 | 0.91883 [0.91703,0.92045] | 1.00175 [0.99823,1.00524] | pass |
| w4 | C | A0 | 0.91317 [0.91175,0.91469] | 1.00149 [0.99893,1.00411] | pass |

## Descriptive decode across contexts

Prompt lengths and decode counts both differ here. These are descriptive throughput ratios, not the same-length candidate/control regression gate above.

| Recipe | A64+42 token/s | B536+46 token/s | C741+3 token/s | B/A TPS | C/A TPS |
|---|---:|---:|---:|---:|---:|
| w16 | 12.50 | 11.71 | 11.42 | 93.68% | 91.37% |
| w4 | 15.32 | 14.19 | 13.78 | 92.62% | 89.96% |

C4 uses QBH_LONG_OPT=3 for A16 long runs, with exact original control0 and retained C2 option1. C4 retains C2 arithmetic and row-major probability layout. One bounded KV slot in existing phase-dead VTCM alternates asynchronous V and next-group K reads behind QK/softmax and AV, and clears only required cache padding before DMA overwrites valid rows. Cache DMA counters now measure exposed issue/join time. C2 also fuses conversion, scale and masking, and omits entirely masked tiles. All contributing maximum/sum operations and rounding retain their original order. Independent long-prefill rows use main plus three existing HVX workers. Per-row FP32 order and FP16 rounding unchanged. 13,312-byte private scratch uses phase-dead expanded-weight VTCM. One HMX owner; no new workers/DDR spill. Native M64, decode, A8/SP2 and excluded Qwen0.6 dispatch unchanged.

Implementation validation is not model-quality/PPL acceptance; historical floating alignment failures remain failures. No automatic baseline promotion.

## prefill additive modules

Microseconds; decode per token; parentheses are complete Host wall shares.
| Module | A W16 | A W4 | B old W16 | B new W16 | B old W4 | B new W4 | C old W16 | C new W16 | C old W4 | C new W4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I/O、metadata | 136.7 (0.17%) | 374.6 (0.61%) | 1290.7 (0.15%) | 1297.7 (0.19%) | 4058.7 (0.57%) | 4033.3 (0.72%) | 1684.3 (0.14%) | 1704.4 (0.18%) | 5623.6 (0.54%) | 5526.2 (0.71%) |
| Input RMSNorm | 490.9 (0.60%) | 492.4 (0.80%) | 4383.4 (0.52%) | 4384.4 (0.63%) | 4390.9 (0.62%) | 4390.9 (0.78%) | 5842.9 (0.48%) | 5845.1 (0.61%) | 5853.7 (0.56%) | 5853.4 (0.75%) |
| QKV＋RoPE | 11988.4 (14.59%) | 11752.7 (19.08%) | 110925.6 (13.07%) | 110400.2 (15.78%) | 109032.9 (15.35%) | 107986.3 (19.23%) | 147058.6 (12.06%) | 146993.4 (15.42%) | 147664.5 (14.09%) | 146163.0 (18.71%) |
| QK–Softmax–AV | 4016.8 (4.89%) | 3985.0 (6.47%) | 226737.0 (26.72%) | 81986.9 (11.72%) | 229767.6 (32.35%) | 84517.0 (15.05%) | 395065.6 (32.41%) | 134340.9 (14.09%) | 401038.8 (38.27%) | 139083.2 (17.80%) |
| O projection | 6017.5 (7.32%) | 5031.5 (8.17%) | 55673.0 (6.56%) | 55330.5 (7.91%) | 47031.5 (6.62%) | 46339.8 (8.25%) | 73848.2 (6.06%) | 73786.2 (7.74%) | 63797.9 (6.09%) | 62837.5 (8.04%) |
| Post-attention residual＋RMSNorm | 476.6 (0.58%) | 476.5 (0.77%) | 4250.1 (0.50%) | 4250.5 (0.61%) | 4246.7 (0.60%) | 4244.1 (0.76%) | 5666.1 (0.46%) | 5666.4 (0.59%) | 5660.3 (0.54%) | 5657.6 (0.72%) |
| Gate/Up＋SwiGLU | 31105.0 (37.86%) | 22418.7 (36.39%) | 282730.8 (33.31%) | 280880.1 (40.15%) | 202870.2 (28.56%) | 202755.6 (36.11%) | 374750.2 (30.74%) | 374736.2 (39.30%) | 272294.9 (25.98%) | 272045.1 (34.83%) |
| Down | 14293.9 (17.40%) | 8655.0 (14.05%) | 131403.9 (15.48%) | 130457.9 (18.65%) | 81391.4 (11.46%) | 80325.7 (14.30%) | 173930.3 (14.27%) | 173831.5 (18.23%) | 111010.1 (10.59%) | 109310.0 (13.99%) |
| Final residual | 141.1 (0.17%) | 140.5 (0.23%) | 1256.0 (0.15%) | 1256.2 (0.18%) | 1254.6 (0.18%) | 1254.6 (0.22%) | 1674.2 (0.14%) | 1674.1 (0.18%) | 1672.2 (0.16%) | 1672.1 (0.21%) |
| KV carrier conversion | 189.4 (0.23%) | 187.5 (0.30%) | 3694.9 (0.44%) | 3694.0 (0.53%) | 4573.9 (0.64%) | 4565.5 (0.81%) | 5092.8 (0.42%) | 5092.3 (0.53%) | 6270.1 (0.60%) | 6272.5 (0.80%) |
| KV append DMA | 446.9 (0.54%) | 434.9 (0.71%) | 5301.7 (0.62%) | 5358.9 (0.77%) | 5644.6 (0.79%) | 5463.0 (0.97%) | 7149.6 (0.59%) | 7178.6 (0.75%) | 8006.6 (0.76%) | 7850.8 (1.01%) |
| Block orchestration | 19.6 (0.02%) | 21.3 (0.03%) | 138.3 (0.02%) | 139.0 (0.02%) | 151.1 (0.02%) | 151.2 (0.03%) | 183.0 (0.02%) | 184.0 (0.02%) | 200.2 (0.02%) | 200.1 (0.03%) |
| Layer bookkeeping | 29.3 (0.04%) | 29.4 (0.05%) | 241.3 (0.03%) | 239.6 (0.03%) | 167.5 (0.02%) | 166.7 (0.03%) | 316.8 (0.03%) | 317.2 (0.03%) | 217.4 (0.02%) | 217.2 (0.03%) |
| Stage-boundary bookkeeping | 7.6 (0.01%) | 7.7 (0.01%) | 21.3 (0.00%) | 22.1 (0.00%) | 23.3 (0.00%) | 22.5 (0.00%) | 26.7 (0.00%) | 27.0 (0.00%) | 28.0 (0.00%) | 27.1 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 97.4 (0.12%) | 110.2 (0.18%) | 518.3 (0.06%) | 518.9 (0.07%) | 615.6 (0.09%) | 616.3 (0.11%) | 674.1 (0.06%) | 675.2 (0.07%) | 808.0 (0.08%) | 801.0 (0.10%) |
| Embedding | 51.3 (0.06%) | 49.5 (0.08%) | 521.3 (0.06%) | 508.1 (0.07%) | 505.2 (0.07%) | 505.7 (0.09%) | 690.1 (0.06%) | 693.3 (0.07%) | 724.0 (0.07%) | 713.4 (0.09%) |
| Final model RMSNorm | 5.3 (0.01%) | 48.6 (0.08%) | 6.1 (0.00%) | 6.2 (0.00%) | 22.0 (0.00%) | 21.9 (0.00%) | 5.9 (0.00%) | 6.1 (0.00%) | 31.5 (0.00%) | 31.0 (0.00%) |
| LM head＋greedy（不含 final norm） | 11899.2 (14.48%) | 6642.0 (10.78%) | 12050.8 (1.42%) | 11976.5 (1.71%) | 7691.8 (1.08%) | 7197.6 (1.28%) | 11985.4 (0.98%) | 11999.9 (1.26%) | 7756.5 (0.74%) | 7742.7 (0.99%) |
| Host staging | 0.0 (0.00%) | 0.0 (0.00%) | 133.0 (0.02%) | 137.9 (0.02%) | 140.8 (0.02%) | 169.1 (0.03%) | 164.7 (0.01%) | 209.4 (0.02%) | 179.1 (0.02%) | 177.4 (0.02%) |
| Host-DSP boundary | 752.8 (0.92%) | 750.7 (1.22%) | 7402.9 (0.87%) | 6761.1 (0.97%) | 6705.1 (0.94%) | 6833.3 (1.22%) | 13191.2 (1.08%) | 8502.6 (0.89%) | 9130.6 (0.87%) | 8966.2 (1.15%) |
| Host wall | 82165.6 (100.00%) | 61609.0 (100.00%) | 848680.5 (100.00%) | 699606.8 (100.00%) | 710285.5 (100.00%) | 561560.0 (100.00%) | 1219000.6 (100.00%) | 953463.8 (100.00%) | 1047968.2 (100.00%) | 781147.6 (100.00%) |

## decode additive modules

Microseconds; decode per token; parentheses are complete Host wall shares.
| Module | A W16 | A W4 | B old W16 | B new W16 | B old W4 | B new W4 | C old W16 | C new W16 | C old W4 | C new W4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I/O、metadata | 139.7 (0.17%) | 426.3 (0.65%) | 119.8 (0.14%) | 120.9 (0.14%) | 468.9 (0.67%) | 490.6 (0.70%) | 111.9 (0.13%) | 136.3 (0.16%) | 475.2 (0.66%) | 485.4 (0.67%) |
| Input RMSNorm | 485.9 (0.61%) | 487.1 (0.75%) | 486.5 (0.57%) | 486.4 (0.57%) | 487.4 (0.69%) | 487.4 (0.69%) | 486.4 (0.56%) | 486.4 (0.56%) | 487.5 (0.67%) | 487.6 (0.67%) |
| QKV＋RoPE | 11822.1 (14.77%) | 12635.9 (19.35%) | 12184.7 (14.17%) | 12115.0 (14.18%) | 12781.8 (18.16%) | 12777.6 (18.13%) | 12158.1 (13.88%) | 12154.0 (13.88%) | 12786.0 (17.64%) | 12791.4 (17.62%) |
| QK–Softmax–AV | 2908.3 (3.63%) | 3382.8 (5.18%) | 8196.9 (9.53%) | 8148.9 (9.54%) | 8987.8 (12.77%) | 9059.4 (12.85%) | 10091.0 (11.52%) | 10091.8 (11.52%) | 11075.6 (15.28%) | 11092.3 (15.28%) |
| O projection | 5939.4 (7.42%) | 5439.8 (8.33%) | 6113.7 (7.11%) | 6072.3 (7.11%) | 5541.2 (7.87%) | 5537.4 (7.85%) | 6108.8 (6.97%) | 6104.4 (6.97%) | 5544.5 (7.65%) | 5544.0 (7.64%) |
| Post-attention residual＋RMSNorm | 471.4 (0.59%) | 471.4 (0.72%) | 471.5 (0.55%) | 471.6 (0.55%) | 471.4 (0.67%) | 471.3 (0.67%) | 471.6 (0.54%) | 471.6 (0.54%) | 471.4 (0.65%) | 471.4 (0.65%) |
| Gate/Up＋SwiGLU | 30794.2 (38.48%) | 22888.5 (35.06%) | 30787.5 (35.80%) | 30571.7 (35.79%) | 22830.5 (32.44%) | 22839.3 (32.40%) | 30649.1 (34.98%) | 30644.5 (34.99%) | 22843.1 (31.52%) | 22846.1 (31.48%) |
| Down | 14114.1 (17.64%) | 9628.2 (14.75%) | 14414.3 (16.76%) | 14301.0 (16.74%) | 9658.8 (13.73%) | 9647.1 (13.68%) | 14359.3 (16.39%) | 14349.8 (16.38%) | 9664.4 (13.34%) | 9658.0 (13.31%) |
| Final residual | 139.4 (0.17%) | 139.2 (0.21%) | 139.4 (0.16%) | 139.4 (0.16%) | 139.2 (0.20%) | 139.2 (0.20%) | 139.4 (0.16%) | 139.4 (0.16%) | 139.2 (0.19%) | 139.2 (0.19%) |
| KV carrier conversion | 425.5 (0.53%) | 1145.1 (1.75%) | 30.3 (0.04%) | 30.3 (0.04%) | 163.0 (0.23%) | 163.2 (0.23%) | 30.8 (0.04%) | 30.8 (0.04%) | 162.5 (0.22%) | 162.7 (0.22%) |
| KV append DMA | 242.0 (0.30%) | 264.2 (0.40%) | 340.6 (0.40%) | 313.2 (0.37%) | 346.0 (0.49%) | 395.5 (0.56%) | 290.9 (0.33%) | 298.0 (0.34%) | 338.8 (0.47%) | 364.1 (0.50%) |
| Block orchestration | 14.6 (0.02%) | 16.2 (0.02%) | 14.7 (0.02%) | 14.7 (0.02%) | 16.3 (0.02%) | 16.3 (0.02%) | 14.8 (0.02%) | 14.8 (0.02%) | 16.3 (0.02%) | 16.3 (0.02%) |
| Layer bookkeeping | 16.8 (0.02%) | 16.4 (0.03%) | 24.0 (0.03%) | 24.0 (0.03%) | 16.4 (0.02%) | 16.4 (0.02%) | 25.1 (0.03%) | 25.0 (0.03%) | 16.4 (0.02%) | 16.4 (0.02%) |
| Stage-boundary bookkeeping | 1.7 (0.00%) | 1.7 (0.00%) | 1.7 (0.00%) | 1.7 (0.00%) | 1.7 (0.00%) | 1.8 (0.00%) | 1.8 (0.00%) | 1.9 (0.00%) | 1.7 (0.00%) | 1.8 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.7 (0.06%) | 61.8 (0.09%) | 51.7 (0.06%) | 51.7 (0.06%) | 62.9 (0.09%) | 62.9 (0.09%) | 52.0 (0.06%) | 52.0 (0.06%) | 62.8 (0.09%) | 62.9 (0.09%) |
| Embedding | 1.2 (0.00%) | 1.6 (0.00%) | 3.4 (0.00%) | 3.5 (0.00%) | 4.0 (0.01%) | 4.1 (0.01%) | 3.4 (0.00%) | 3.4 (0.00%) | 3.4 (0.00%) | 3.4 (0.00%) |
| Final model RMSNorm | 3.2 (0.00%) | 3.0 (0.00%) | 4.0 (0.00%) | 3.9 (0.00%) | 3.4 (0.00%) | 3.6 (0.01%) | 3.7 (0.00%) | 3.7 (0.00%) | 3.0 (0.00%) | 3.1 (0.00%) |
| LM head＋greedy（不含 final norm） | 11781.0 (14.72%) | 7581.5 (11.61%) | 11943.9 (13.89%) | 11865.5 (13.89%) | 7693.8 (10.93%) | 7675.0 (10.89%) | 11915.4 (13.60%) | 11911.5 (13.60%) | 7690.3 (10.61%) | 7686.2 (10.59%) |
| Host staging | 0.0 (0.00%) | 0.0 (0.00%) | 10.1 (0.01%) | 10.2 (0.01%) | 9.6 (0.01%) | 11.0 (0.02%) | 8.0 (0.01%) | 8.2 (0.01%) | 11.8 (0.02%) | 9.2 (0.01%) |
| Host-DSP boundary | 673.8 (0.84%) | 701.8 (1.07%) | 671.0 (0.78%) | 677.8 (0.79%) | 689.9 (0.98%) | 697.0 (0.99%) | 691.5 (0.79%) | 661.3 (0.75%) | 675.4 (0.93%) | 735.4 (1.01%) |
| Host wall | 80026.1 (100.00%) | 65292.6 (100.00%) | 86009.9 (100.00%) | 85423.7 (100.00%) | 70373.8 (100.00%) | 70496.1 (100.00%) | 87613.2 (100.00%) | 87588.8 (100.00%) | 72469.4 (100.00%) | 72576.9 (100.00%) |

