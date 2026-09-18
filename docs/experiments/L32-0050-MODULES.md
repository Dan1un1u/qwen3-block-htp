# L32-0050 long prefill profiling

Full Llama-3.2-1B-Instruct W4A8-SP2, no rotation; fixed trajectory.
Ten processes x ten replays per shape; five preceding auxiliary single-replay checks. Weights loaded and session prepared before timing. Every measured invocation, including each process first invocation, is retained.
Prefill columns are complete prompt microseconds; decode columns are microseconds per decode step. Parentheses give Host-wall shares. Host input/RoPE staging is included; file loading, external tokenization, audit copies and report formatting are excluded.

| Module | 536 prefill | 536+46 decode | 741 prefill | 741+3 decode |
|---|---:|---:|---:|---:|
| I/O、metadata | 1094.6 (0.13%) | 118.5 (0.27%) | 1438.7 (0.10%) | 113.9 (0.22%) |
| Input RMSNorm | 12706.8 (1.51%) | 871.3 (2.01%) | 18302.1 (1.28%) | 873.1 (1.69%) |
| QKV＋RoPE | 81064.3 (9.66%) | 1668.1 (3.84%) | 114038.5 (7.99%) | 1661.7 (3.22%) |
| QK–Softmax–AV | 623205.9 (74.29%) | 27435.6 (63.20%) | 1129762.7 (79.13%) | 35648.3 (69.11%) |
| O projection | 14198.1 (1.69%) | 888.8 (2.05%) | 18967.5 (1.33%) | 888.6 (1.72%) |
| Post-attention residual＋RMSNorm | 12619.7 (1.50%) | 876.7 (2.02%) | 18176.6 (1.27%) | 871.9 (1.69%) |
| Gate/Up＋SwiGLU | 49192.1 (5.86%) | 4792.0 (11.04%) | 65400.4 (4.58%) | 4771.8 (9.25%) |
| Down | 28161.4 (3.36%) | 2611.9 (6.02%) | 37685.7 (2.64%) | 2601.6 (5.04%) |
| Final residual | 10.5 (0.00%) | 0.9 (0.00%) | 13.4 (0.00%) | 0.9 (0.00%) |
| KV carrier conversion | 189.9 (0.02%) | 48.2 (0.11%) | 302.3 (0.02%) | 48.1 (0.09%) |
| KV append DMA | 1053.2 (0.13%) | 90.0 (0.21%) | 1417.9 (0.10%) | 90.6 (0.18%) |
| Block orchestration | 162.9 (0.02%) | 16.5 (0.04%) | 214.8 (0.02%) | 16.6 (0.03%) |
| Layer bookkeeping | 89.7 (0.01%) | 9.1 (0.02%) | 118.0 (0.01%) | 9.1 (0.02%) |
| Stage-boundary bookkeeping | 17.2 (0.00%) | 1.3 (0.00%) | 19.8 (0.00%) | 1.3 (0.00%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 493.4 (0.06%) | 49.2 (0.11%) | 636.9 (0.04%) | 49.3 (0.10%) |
| Embedding | 516.4 (0.06%) | 6.1 (0.01%) | 678.8 (0.05%) | 6.2 (0.01%) |
| Final model RMSNorm | 55.7 (0.01%) | 55.3 (0.13%) | 55.5 (0.00%) | 55.3 (0.11%) |
| LM head＋greedy（不含 final norm） | 4388.0 (0.52%) | 3202.4 (7.38%) | 4365.5 (0.31%) | 3172.9 (6.15%) |
| Host input/RoPE staging | 81.7 (0.01%) | 8.3 (0.02%) | 114.0 (0.01%) | 9.0 (0.02%) |
| Host–DSP boundary | 9564.5 (1.14%) | 660.4 (1.52%) | 15933.1 (1.12%) | 690.0 (1.34%) |
| Complete Host wall | 838865.8 (100.00%) | 43410.5 (100.00%) | 1427642.3 (100.00%) | 51580.2 (100.00%) |

| Shape | Prefill token/s | Decode token/s |
|---|---:|---:|
| 536+46 | 638.96 | 23.04 |
| 741+3 | 519.04 | 19.39 |

Decode token count denotes decode RPC steps; prefill also computes the first selected token. Fixed input trajectory is project-owned, not a named dataset or quality evaluation. No cross-shape 10% speed gate or automatic baseline promotion.
