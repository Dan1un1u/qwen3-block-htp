# Long-prefill throughput gate: user-approved10% tolerance

This is a current-policy reassessment of frozen measurements, not a new hardware experiment. User explicitly changes the throughput-parity gate to10%. Define pass as the original paired95% CI lower bound of long-prefill TPS / same-experiment M64 TPS >=0.90. This is throughput loss, not a10% wall-time increase. Preserve original references, measurements, CI estimates, historical strict-parity classifications and all numerical/physical/decode guards. No automatic quality or baseline promotion.

| Model | Shape | Prefill TPS | M64 TPS | Throughput change |95% ratio CI | Current10% gate |
|---|---|---:|---:|---:|---|---|
| Qwen3-0.6B | 536+46 | 2603.42 | 2957.34 | -11.97% | [0.876750, 0.883092] | fail |
| Qwen3-0.6B | 741+3 | 2548.72 | 2957.34 | -13.82% | [0.858590, 0.864219] | fail |
| Qwen3-1.7B | 536+46 | 1826.92 | 1850.73 | -1.29% | [0.985157, 0.989076] | pass |
| Qwen3-1.7B | 741+3 | 1860.81 | 1850.73 | +0.54% | [1.002960, 1.008607] | pass |
| Llama3.2-1B | 536+46 | 2505.95 | 2305.96 | +8.67% | [1.081447, 1.091901] | pass |
| Llama3.2-1B | 741+3 | 2499.13 | 2305.96 | +8.38% | [1.078151, 1.089085] | pass |
| Llama3.2-3B | 536+46 | 1151.42 | 1182.00 | -2.59% | [0.971393, 0.976873] | pass |
| Llama3.2-3B | 741+3 | 1168.46 | 1182.00 | -1.15% | [0.985877, 0.991122] | pass |

Six of eight measured W4A8-SP2 long-shape rows pass. Qwen0.6B536+46 and741+3 remain failed. Untested recipes are pending, not passes. A/D rows are not long-prefill parity assessments. No TPS values changed. Workbook labels and maintenance policy use this reassessment; sealed experiment evidence remains immutable.

## Frozen sources

- /mnt/d/llm_exp/results/qwen3-block-htp/exp0296/0.6B/profiling_summary.json; SHA256 9fb8f5358535ab25a786ff36cd462524397a015a69feeaa1ddea46eb361d2f03
- /mnt/d/llm_exp/results/qwen3-block-htp/exp0297/profiling_summary.json; SHA256 c52f0ecb8f491d28424305c22738fd27403694bd6af9619f6103e3efcd4cd992
- /mnt/d/llm_exp/results/llama32-htp/l32-0051/profiling_summary.json; SHA256 9dccef7c43ea6138223fe70a630684a1209d37548412c971868022cebb809921
- /mnt/d/llm_exp/results/llama32-htp/l32-0053/profiling_summary.json; SHA256 f2f845a97cc461afb90fb4bf7c34e659d4f12bc6ef8724dc855dd64c84fc2cf2
