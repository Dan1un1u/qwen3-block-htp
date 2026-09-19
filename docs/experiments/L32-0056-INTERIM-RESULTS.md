# L32-0056 interim result: completed model only

Qwen1.7B and Llama1B completed five short and ten formal repeat10 rounds. All prefill and same-length decode confidence gates pass. Qwen0.6 and A8/SP2 unchanged. Fixed project trajectories, not named datasets.

Llama3B has passed all16 A16 numerical cases, A8/SP2 regressions and five short repeat10 rounds. Formal rounds1-9 completed; round10 is incomplete. ADB lost the device twice. On first reconnection uptime was below1 minute, battery2%, temperature51.6 C. The underlying cause is not proven. No more automatic hardware retries until power and temperature recover.

All five complete W4 B replays before the first disconnection were retained, including slow samples, and joined to five continuations. The partial replay remains archived. The second disconnection occurred at W4 C; final C candidate/control remain incomplete. No Llama3B formal result is promoted or entered in the workbook.

Workbook update: only12 completed A16 rows for Qwen1.7B and Llama1B across ABC. Preserve previous Llama3B values, Qwen0.6, A8/SP2, all D rows and historical quality caveats. Both experiments remain active for task recovery; no lock clearing or baseline promotion.

## Completed measurement

# L32-0056 Llama3.2-1B generic A16 long-context repair

Fixed token fixtures, not named datasets. Full Host wall includes embedding, all layers, final norm, LM head, greedy, FastRPC and long host staging. Cold loading and external tokenizer excluded. Five short and ten formal rotated paired rounds, ten retained replays each. Repeat1 auxiliary only.

| Configuration | Prompt | Decode | Prefill token/s | Decode token/s |
|---|---:|---:|---:|---:|
| A W16A16 | 64 | 42 | 962.35 | 17.88 |
| A W4A16 | 64 | 42 | 1254.24 | 23.94 |
| B W16A16 | 536 | 46 | 1094.36 | 17.05 |
| B W4A16 | 536 | 46 | 1380.14 | 22.31 |
| C W16A16 | 741 | 3 | 1100.74 | 16.51 |
| C W4A16 | 741 | 3 | 1378.00 | 21.65 |

## Paired confidence gates

Reference is the faster eligible same-campaign unchanged native M64 or long64 prefill. Prefill ratio lower95% >=0.90. Identical-long-shape candidate/control decode wall upper95% <=1.10. Bootstrap10000, fixed seeds. Cross-context decode is not this regression guard.

| Configuration | Shape | Reference | Prefill ratio [95%CI] | Decode wall ratio [95%CI] | Gate |
|---|---|---|---:|---:|---|
| w16 | B | A0 | 1.13752 [1.12113,1.15372] | 0.99488 [0.98510,1.00514] | pass |
| w16 | C | A0 | 1.14400 [1.13227,1.15558] | 1.00014 [0.98773,1.01314] | pass |
| w4 | B | A0 | 1.10038 [1.09926,1.10156] | 0.99705 [0.99367,1.00055] | pass |
| w4 | C | A0 | 1.09868 [1.09749,1.09983] | 0.98057 [0.97908,0.98214] | pass |

## Descriptive decode across contexts

Prompt lengths and decode counts both differ here. These are descriptive throughput ratios, not the same-length candidate/control regression gate above.

| Recipe | A64+42 token/s | B536+46 token/s | C741+3 token/s | B/A TPS | C/A TPS |
|---|---:|---:|---:|---:|---:|
| w16 | 17.88 | 17.05 | 16.51 | 95.32% | 92.34% |
| w4 | 23.94 | 22.31 | 21.65 | 93.16% | 90.43% |

C4 uses QBH_LONG_OPT=3 for A16 long runs, with exact original control0 and retained C2 option1. C4 retains C2 arithmetic and row-major probability layout. One bounded KV slot in existing phase-dead VTCM alternates asynchronous V and next-group K reads behind QK/softmax and AV, and clears only required cache padding before DMA overwrites valid rows. Cache DMA counters now measure exposed issue/join time. C2 also fuses conversion, scale and masking, and omits entirely masked tiles. All contributing maximum/sum operations and rounding retain their original order. Independent long-prefill rows use main plus three existing HVX workers. Per-row FP32 order and FP16 rounding unchanged. 13,312-byte private scratch uses phase-dead expanded-weight VTCM. One HMX owner; no new workers/DDR spill. Native M64, decode, A8/SP2 and excluded Qwen0.6 dispatch unchanged.

Implementation validation is not model-quality/PPL acceptance; historical floating alignment failures remain failures. No automatic baseline promotion.

## prefill additive modules

Microseconds; decode per token; parentheses are complete Host wall shares.
| Module | A W16 | A W4 | B old W16 | B new W16 | B old W4 | B new W4 | C old W16 | C new W16 | C old W4 | C new W4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I/O、metadata | 69.2 (0.10%) | 219.8 (0.43%) | 638.4 (0.10%) | 635.2 (0.13%) | 1975.6 (0.36%) | 1947.7 (0.50%) | 855.4 (0.09%) | 853.1 (0.13%) | 2650.3 (0.32%) | 2537.0 (0.47%) |
| Input RMSNorm | 281.8 (0.42%) | 282.3 (0.55%) | 2497.9 (0.38%) | 2498.7 (0.51%) | 2499.7 (0.45%) | 2500.1 (0.64%) | 3327.8 (0.34%) | 3329.6 (0.49%) | 3331.9 (0.40%) | 3331.3 (0.62%) |
| QKV＋RoPE | 5336.0 (8.02%) | 5754.9 (11.28%) | 48200.8 (7.34%) | 47956.0 (9.79%) | 51806.8 (9.36%) | 51690.4 (13.31%) | 64202.3 (6.62%) | 64148.8 (9.53%) | 69435.1 (8.30%) | 69049.6 (12.84%) |
| QK–Softmax–AV | 11600.1 (17.44%) | 11613.6 (22.76%) | 241613.7 (36.79%) | 77304.2 (15.78%) | 242032.3 (43.73%) | 77669.5 (20.00%) | 419327.1 (43.26%) | 123309.2 (18.32%) | 420444.0 (50.25%) | 124065.2 (23.07%) |
| O projection | 3428.3 (5.16%) | 2871.3 (5.63%) | 30881.3 (4.70%) | 30787.5 (6.29%) | 25843.9 (4.67%) | 25852.9 (6.66%) | 41139.2 (4.24%) | 41158.2 (6.11%) | 34720.4 (4.15%) | 34627.3 (6.44%) |
| Post-attention residual＋RMSNorm | 273.4 (0.41%) | 272.7 (0.53%) | 2422.4 (0.37%) | 2423.4 (0.49%) | 2417.9 (0.44%) | 2419.9 (0.62%) | 3227.7 (0.33%) | 3228.5 (0.48%) | 3222.9 (0.39%) | 3226.9 (0.60%) |
| Gate/Up＋SwiGLU | 23430.5 (35.23%) | 17321.1 (33.94%) | 211343.2 (32.18%) | 209847.8 (42.85%) | 156163.9 (28.22%) | 155932.6 (40.15%) | 281775.3 (29.07%) | 281642.6 (41.84%) | 209059.4 (24.99%) | 208241.4 (38.73%) |
| Down | 10740.8 (16.15%) | 5862.6 (11.49%) | 96960.7 (14.76%) | 96198.0 (19.64%) | 52723.6 (9.53%) | 52655.2 (13.56%) | 129221.0 (13.33%) | 129116.9 (19.18%) | 71061.4 (8.49%) | 70462.5 (13.10%) |
| Final residual | 81.1 (0.12%) | 81.2 (0.16%) | 720.0 (0.11%) | 720.1 (0.15%) | 719.4 (0.13%) | 719.2 (0.19%) | 959.4 (0.10%) | 959.4 (0.14%) | 958.5 (0.11%) | 958.5 (0.18%) |
| KV carrier conversion | 98.0 (0.15%) | 98.0 (0.19%) | 1904.1 (0.29%) | 1902.7 (0.39%) | 1902.7 (0.34%) | 1902.1 (0.49%) | 2626.9 (0.27%) | 2627.2 (0.39%) | 2625.8 (0.31%) | 2620.0 (0.49%) |
| KV append DMA | 194.1 (0.29%) | 191.5 (0.38%) | 2118.0 (0.32%) | 2123.2 (0.43%) | 2015.1 (0.36%) | 1986.0 (0.51%) | 2885.6 (0.30%) | 2892.2 (0.43%) | 2839.3 (0.34%) | 2728.0 (0.51%) |
| Block orchestration | 14.4 (0.02%) | 16.1 (0.03%) | 89.6 (0.01%) | 90.0 (0.02%) | 98.9 (0.02%) | 99.3 (0.03%) | 117.5 (0.01%) | 117.7 (0.02%) | 130.6 (0.02%) | 131.0 (0.02%) |
| Layer bookkeeping | 13.6 (0.02%) | 14.0 (0.03%) | 94.3 (0.01%) | 94.4 (0.02%) | 93.8 (0.02%) | 93.6 (0.02%) | 124.0 (0.01%) | 124.1 (0.02%) | 123.4 (0.01%) | 123.6 (0.02%) |
| Stage-boundary bookkeeping | 6.7 (0.01%) | 7.0 (0.01%) | 17.5 (0.00%) | 18.9 (0.00%) | 18.2 (0.00%) | 17.7 (0.00%) | 21.6 (0.00%) | 21.2 (0.00%) | 22.5 (0.00%) | 22.4 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 93.3 (0.14%) | 105.2 (0.21%) | 508.4 (0.08%) | 508.2 (0.10%) | 597.2 (0.11%) | 595.6 (0.15%) | 658.1 (0.07%) | 657.4 (0.10%) | 781.3 (0.09%) | 776.9 (0.14%) |
| Embedding | 60.4 (0.09%) | 59.6 (0.12%) | 513.5 (0.08%) | 492.3 (0.10%) | 468.1 (0.08%) | 467.5 (0.12%) | 690.6 (0.07%) | 689.8 (0.10%) | 625.3 (0.07%) | 595.2 (0.11%) |
| Final model RMSNorm | 4.5 (0.01%) | 48.3 (0.09%) | 5.1 (0.00%) | 5.1 (0.00%) | 21.3 (0.00%) | 20.7 (0.01%) | 5.0 (0.00%) | 5.2 (0.00%) | 30.5 (0.00%) | 30.4 (0.01%) |
| LM head＋greedy（不含 final norm） | 10078.0 (15.15%) | 5493.3 (10.77%) | 10108.6 (1.54%) | 10066.5 (2.06%) | 5885.8 (1.06%) | 5674.3 (1.46%) | 10122.8 (1.04%) | 10117.2 (1.50%) | 6339.9 (0.76%) | 5949.9 (1.11%) |
| Host staging | 0.0 (0.00%) | 0.0 (0.00%) | 104.3 (0.02%) | 81.1 (0.02%) | 87.0 (0.02%) | 99.4 (0.03%) | 111.9 (0.01%) | 129.9 (0.02%) | 131.7 (0.02%) | 128.8 (0.02%) |
| Host-DSP boundary | 699.6 (1.05%) | 714.4 (1.40%) | 6023.4 (0.92%) | 6029.9 (1.23%) | 6044.7 (1.09%) | 6023.9 (1.55%) | 7989.8 (0.82%) | 8056.1 (1.20%) | 8143.4 (0.97%) | 8129.4 (1.51%) |
| Host wall | 66503.9 (100.00%) | 51026.9 (100.00%) | 656765.2 (100.00%) | 489783.1 (100.00%) | 553415.7 (100.00%) | 388367.7 (100.00%) | 969389.0 (100.00%) | 673184.2 (100.00%) | 836677.7 (100.00%) | 537735.2 (100.00%) |

## decode additive modules

Microseconds; decode per token; parentheses are complete Host wall shares.
| Module | A W16 | A W4 | B old W16 | B new W16 | B old W4 | B new W4 | C old W16 | C new W16 | C old W4 | C new W4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I/O、metadata | 67.6 (0.12%) | 238.3 (0.57%) | 62.3 (0.11%) | 62.3 (0.11%) | 251.1 (0.56%) | 255.7 (0.57%) | 68.7 (0.11%) | 68.2 (0.11%) | 255.7 (0.54%) | 247.8 (0.54%) |
| Input RMSNorm | 276.8 (0.50%) | 277.2 (0.66%) | 276.9 (0.47%) | 276.9 (0.47%) | 277.2 (0.62%) | 277.2 (0.62%) | 276.9 (0.46%) | 276.9 (0.46%) | 277.2 (0.59%) | 277.2 (0.60%) |
| QKV＋RoPE | 5256.9 (9.40%) | 5787.1 (13.86%) | 5286.2 (8.96%) | 5263.8 (8.97%) | 5770.7 (12.83%) | 5762.6 (12.85%) | 5323.3 (8.79%) | 5317.7 (8.78%) | 5835.4 (12.39%) | 5759.3 (12.47%) |
| QK–Softmax–AV | 1901.4 (3.40%) | 2015.7 (4.83%) | 5141.1 (8.72%) | 5129.6 (8.74%) | 5304.0 (11.80%) | 5300.1 (11.82%) | 6424.0 (10.61%) | 6420.6 (10.60%) | 6678.9 (14.18%) | 6604.4 (14.30%) |
| O projection | 3381.0 (6.05%) | 2857.2 (6.84%) | 3391.1 (5.75%) | 3381.4 (5.76%) | 2872.8 (6.39%) | 2864.8 (6.39%) | 3413.9 (5.64%) | 3414.6 (5.64%) | 2920.9 (6.20%) | 2866.7 (6.21%) |
| Post-attention residual＋RMSNorm | 268.5 (0.48%) | 268.2 (0.64%) | 268.6 (0.46%) | 268.6 (0.46%) | 268.2 (0.60%) | 268.2 (0.60%) | 268.6 (0.44%) | 268.6 (0.44%) | 268.2 (0.57%) | 268.3 (0.58%) |
| Gate/Up＋SwiGLU | 23146.5 (41.39%) | 17435.3 (41.74%) | 22940.4 (38.90%) | 22798.1 (38.86%) | 17410.1 (38.72%) | 17383.0 (38.77%) | 23046.1 (38.05%) | 23042.2 (38.05%) | 17558.8 (37.28%) | 17380.7 (37.63%) |
| Down | 10602.2 (18.96%) | 5901.8 (14.13%) | 10604.4 (17.98%) | 10532.3 (17.95%) | 5782.4 (12.86%) | 5760.1 (12.85%) | 10675.1 (17.63%) | 10667.3 (17.62%) | 5944.0 (12.62%) | 5761.2 (12.47%) |
| Final residual | 79.8 (0.14%) | 79.8 (0.19%) | 79.9 (0.14%) | 79.9 (0.14%) | 79.8 (0.18%) | 79.8 (0.18%) | 79.9 (0.13%) | 79.8 (0.13%) | 79.7 (0.17%) | 79.8 (0.17%) |
| KV carrier conversion | 134.1 (0.24%) | 134.0 (0.32%) | 9.9 (0.02%) | 10.0 (0.02%) | 9.8 (0.02%) | 9.8 (0.02%) | 10.0 (0.02%) | 10.1 (0.02%) | 10.0 (0.02%) | 9.7 (0.02%) |
| KV append DMA | 108.1 (0.19%) | 120.5 (0.29%) | 189.8 (0.32%) | 177.3 (0.30%) | 176.7 (0.39%) | 189.2 (0.42%) | 191.3 (0.32%) | 188.6 (0.31%) | 158.4 (0.34%) | 156.6 (0.34%) |
| Block orchestration | 9.3 (0.02%) | 10.4 (0.03%) | 9.3 (0.02%) | 9.3 (0.02%) | 10.4 (0.02%) | 10.4 (0.02%) | 9.3 (0.02%) | 9.4 (0.02%) | 10.5 (0.02%) | 10.4 (0.02%) |
| Layer bookkeeping | 9.8 (0.02%) | 9.9 (0.02%) | 9.9 (0.02%) | 9.9 (0.02%) | 9.9 (0.02%) | 9.9 (0.02%) | 10.0 (0.02%) | 10.0 (0.02%) | 9.9 (0.02%) | 9.9 (0.02%) |
| Stage-boundary bookkeeping | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) | 1.3 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 49.2 (0.09%) | 60.9 (0.15%) | 50.7 (0.09%) | 50.7 (0.09%) | 60.8 (0.14%) | 60.9 (0.14%) | 50.8 (0.08%) | 50.9 (0.08%) | 60.8 (0.13%) | 60.7 (0.13%) |
| Embedding | 1.6 (0.00%) | 1.7 (0.00%) | 4.1 (0.01%) | 4.1 (0.01%) | 4.0 (0.01%) | 4.0 (0.01%) | 4.0 (0.01%) | 4.0 (0.01%) | 4.0 (0.01%) | 3.9 (0.01%) |
| Final model RMSNorm | 3.2 (0.01%) | 2.9 (0.01%) | 3.5 (0.01%) | 3.2 (0.01%) | 3.0 (0.01%) | 3.3 (0.01%) | 3.7 (0.01%) | 3.7 (0.01%) | 2.7 (0.01%) | 2.7 (0.01%) |
| LM head＋greedy（不含 final norm） | 9949.8 (17.79%) | 5869.7 (14.05%) | 9975.5 (16.92%) | 9937.8 (16.94%) | 5983.5 (13.31%) | 5904.8 (13.17%) | 10047.5 (16.59%) | 10035.7 (16.57%) | 6345.0 (13.47%) | 6010.3 (13.01%) |
| Host staging | 0.0 (0.00%) | 0.0 (0.00%) | 7.4 (0.01%) | 7.3 (0.01%) | 10.3 (0.02%) | 8.7 (0.02%) | 10.1 (0.02%) | 6.8 (0.01%) | 10.4 (0.02%) | 6.8 (0.01%) |
| Host-DSP boundary | 670.5 (1.20%) | 695.3 (1.66%) | 658.8 (1.12%) | 659.3 (1.12%) | 679.3 (1.51%) | 678.1 (1.51%) | 646.7 (1.07%) | 681.0 (1.12%) | 672.3 (1.43%) | 671.3 (1.45%) |
| Host wall | 55917.6 (100.00%) | 41767.1 (100.00%) | 58970.9 (100.00%) | 58662.9 (100.00%) | 44965.2 (100.00%) | 44831.6 (100.00%) | 60561.1 (100.00%) | 60557.6 (100.00%) | 47104.2 (100.00%) | 46188.9 (100.00%) |

