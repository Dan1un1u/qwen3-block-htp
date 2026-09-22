# Llama3B AV-to-O matched ablation modules

Units: microseconds (share of complete Host wall). Prefill is64 tokens, decode is one token averaged over42 positions.

## prefill

| Module | Separate AV RQ | Folded into O |
|---|---:|---:|
| I/O、metadata | 246.838 (0.459%) | 247.668 (0.466%) |
| Input RMSNorm | 3576.617 (6.655%) | 3631.850 (6.832%) |
| QKV＋RoPE | 9089.181 (16.911%) | 9103.736 (17.126%) |
| QK–Softmax–AV | 6647.132 (12.368%) | 6018.143 (11.322%) |
| O projection | 4151.955 (7.725%) | 4142.954 (7.794%) |
| Post-attention residual＋RMSNorm | 3591.856 (6.683%) | 3563.454 (6.704%) |
| Gate/Up＋SwiGLU | 13651.676 (25.401%) | 13669.182 (25.715%) |
| Down | 7433.418 (13.831%) | 7427.791 (13.974%) |
| Final residual | 2.802 (0.005%) | 2.764 (0.005%) |
| KV carrier conversion | 80.152 (0.149%) | 80.855 (0.152%) |
| KV append DMA | 237.306 (0.442%) | 237.097 (0.446%) |
| Block orchestration | 37.878 (0.070%) | 38.178 (0.072%) |
| Layer bookkeeping | 23.780 (0.044%) | 24.038 (0.045%) |
| Stage-boundary bookkeeping | 7.472 (0.014%) | 7.569 (0.014%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 91.901 (0.171%) | 94.056 (0.177%) |
| Embedding | 77.661 (0.144%) | 78.448 (0.148%) |
| Final model RMSNorm | 82.306 (0.153%) | 82.424 (0.155%) |
| LM head＋greedy（不含 final norm） | 3851.238 (7.166%) | 3857.350 (7.257%) |
| Host-DSP boundary | 864.467 (1.608%) | 848.476 (1.596%) |
| Full Host wall | 53745.634 (100.000%) | 53156.032 (100.000%) |

E2E token/s: control=1190.794; folded=1204.003

## decode

| Module | Separate AV RQ | Folded into O |
|---|---:|---:|
| I/O、metadata | 246.629 (0.591%) | 246.207 (0.592%) |
| Input RMSNorm | 2310.061 (5.539%) | 2307.993 (5.552%) |
| QKV＋RoPE | 3830.808 (9.185%) | 3835.946 (9.227%) |
| QK–Softmax–AV | 6593.097 (15.808%) | 6428.470 (15.463%) |
| O projection | 2904.986 (6.965%) | 2903.419 (6.984%) |
| Post-attention residual＋RMSNorm | 2299.227 (5.513%) | 2307.500 (5.550%) |
| Gate/Up＋SwiGLU | 12406.966 (29.748%) | 12414.006 (29.861%) |
| Down | 6200.316 (14.867%) | 6205.786 (14.927%) |
| Final residual | 1.759 (0.004%) | 1.763 (0.004%) |
| KV carrier conversion | 12.482 (0.030%) | 12.514 (0.030%) |
| KV append DMA | 166.089 (0.398%) | 165.804 (0.399%) |
| Block orchestration | 30.114 (0.072%) | 30.096 (0.072%) |
| Layer bookkeeping | 16.493 (0.040%) | 16.486 (0.040%) |
| Stage-boundary bookkeeping | 1.834 (0.004%) | 1.834 (0.004%) |
| DSP unattributed | 0.000 (0.000%) | 0.000 (0.000%) |
| Runtime setup/teardown | 51.081 (0.122%) | 51.084 (0.123%) |
| Embedding | 2.115 (0.005%) | 2.109 (0.005%) |
| Final model RMSNorm | 83.215 (0.200%) | 83.034 (0.200%) |
| LM head＋greedy（不含 final norm） | 3858.165 (9.251%) | 3857.686 (9.279%) |
| Host-DSP boundary | 690.896 (1.657%) | 701.422 (1.687%) |
| Full Host wall | 41706.331 (100.000%) | 41573.158 (100.000%) |

E2E token/s: control=23.977; folded=24.054
