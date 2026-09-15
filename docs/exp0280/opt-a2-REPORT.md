# EXP-0280 C2 targeted native decode optimization

No rotation; native per-channel W4, SP2 mode8, FP32 residual; frozen weights/qparams. OPT computes only the32 valid channels per decode tile, masks off inactive gather lanes and writes deterministic padding. ORIGINAL retains sequential128-value decode gathers; NORM_COMPACT uses compact Norm plus C1 dual128 gather (mask17). Five-arm numerical controls additionally include C1 dual128 and both-compact. Prefill native arithmetic and scheduling unchanged. Fixed5short10formal cyclic pairedrepeat10 primary; repeat1 auxiliary. Full28 hidden bytes, generated token/logit codes identical to frozen baseline; selected0/14/27 andchain3 independent gates and actual finalNorm/head all16steps pass. No full28 CPU transformer equivalence or quality claim.

C1 dual-gather scheduling alone was inconclusive versus compact; C2 removes irrelevant-row table lookups. Norm padding differs in old compact/native paths and changes downstream gather address distribution even though row0 is exact. Layer14 retained captures show35.60 versus126.23 distinct indices per128-value tile. Hardware bank mapping is unproven. Historical factorial comparison therefore mixes layout, gather scheduling and padding effects; do not attribute it solely to physical format.

C2 full-model decode wall improves0.426% over original (paired95% CI0.013–0.803%); native C2 is0.168% faster than the previous compact-Norm control (CI0.053–0.294%). Prefill is unchanged within uncertainty. Keep optimized native decode; compact fallback is unnecessary. These are modest full-model gains, not a large speedup. All attempts retained; no selected-baseline promotion.

## prefill

| Module | OPT | ORIGINAL | NORM_COMPACT |
|---|---:|---:|---:|
| I/O、metadata | 245.15 (0.71%) | 244.39 (0.71%) | 244.53 (0.70%) |
| Input RMSNorm | 2021.31 (5.85%) | 2021.17 (5.84%) | 2108.83 (6.06%) |
| QKV＋Q/K Norm-RoPE | 7046.02 (20.38%) | 7045.47 (20.37%) | 7040.35 (20.22%) |
| QK–Softmax–AV | 3491.16 (10.10%) | 3491.02 (10.09%) | 3494.48 (10.03%) |
| O projection | 2085.41 (6.03%) | 2086.13 (6.03%) | 2089.58 (6.00%) |
| Post-attention residual＋RMSNorm | 2114.85 (6.12%) | 2114.29 (6.11%) | 2200.88 (6.32%) |
| Gate/Up＋SwiGLU | 7086.91 (20.50%) | 7085.60 (20.49%) | 7081.93 (20.34%) |
| Down | 3800.09 (10.99%) | 3798.62 (10.98%) | 3799.29 (10.91%) |
| Final residual | 3.27 (0.01%) | 3.20 (0.01%) | 3.16 (0.01%) |
| KV carrier conversion | 135.98 (0.39%) | 135.81 (0.39%) | 136.34 (0.39%) |
| KV append DMA | 294.10 (0.85%) | 293.47 (0.85%) | 293.61 (0.84%) |
| Block orchestration | 35.14 (0.10%) | 35.18 (0.10%) | 35.17 (0.10%) |
| Layer bookkeeping | 26.72 (0.08%) | 26.79 (0.08%) | 26.55 (0.08%) |
| Stage-boundary bookkeeping | 20.39 (0.06%) | 20.50 (0.06%) | 20.41 (0.06%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 117.03 (0.34%) | 116.85 (0.34%) | 116.59 (0.33%) |
| Embedding | 54.90 (0.16%) | 55.09 (0.16%) | 55.18 (0.16%) |
| Final model RMSNorm | 13.69 (0.04%) | 13.72 (0.04%) | 13.68 (0.04%) |
| LM head＋greedy，不含 final norm | 5169.48 (14.95%) | 5166.27 (14.94%) | 5171.00 (14.85%) |
| Host-DSP boundary | 812.65 (2.35%) | 829.52 (2.40%) | 894.27 (2.57%) |
| Complete Host wall | 34574.24 (100.00%) | 34583.10 (100.00%) | 34825.84 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.999744 | [0.9956377384338706, 1.00388479217203] |
| NORM_COMPACT_over_OPT | 1.007277 | [1.0054173743947366, 1.009295338122103] |
## decode

| Module | OPT | ORIGINAL | NORM_COMPACT |
|---|---:|---:|---:|
| I/O、metadata | 240.26 (1.14%) | 239.98 (1.13%) | 240.72 (1.14%) |
| Input RMSNorm | 368.93 (1.75%) | 369.00 (1.74%) | 371.82 (1.76%) |
| QKV＋Q/K Norm-RoPE | 2504.85 (11.87%) | 2506.71 (11.83%) | 2505.24 (11.85%) |
| QK–Softmax–AV | 1922.42 (9.11%) | 1923.28 (9.07%) | 1919.72 (9.08%) |
| O projection | 1346.30 (6.38%) | 1346.84 (6.35%) | 1344.21 (6.36%) |
| Post-attention residual＋RMSNorm | 368.52 (1.75%) | 368.38 (1.74%) | 371.43 (1.76%) |
| Gate/Up＋SwiGLU | 6352.93 (30.10%) | 6446.08 (30.41%) | 6364.72 (30.10%) |
| Down | 3507.98 (16.62%) | 3509.82 (16.56%) | 3509.19 (16.60%) |
| Final residual | 2.04 (0.01%) | 2.05 (0.01%) | 2.03 (0.01%) |
| KV carrier conversion | 291.82 (1.38%) | 291.91 (1.38%) | 291.92 (1.38%) |
| KV append DMA | 119.88 (0.57%) | 120.51 (0.57%) | 119.91 (0.57%) |
| Block orchestration | 27.87 (0.13%) | 27.81 (0.13%) | 27.83 (0.13%) |
| Layer bookkeeping | 16.67 (0.08%) | 16.66 (0.08%) | 16.69 (0.08%) |
| Stage-boundary bookkeeping | 1.78 (0.01%) | 1.77 (0.01%) | 1.77 (0.01%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 73.71 (0.35%) | 73.69 (0.35%) | 73.74 (0.35%) |
| Embedding | 1.73 (0.01%) | 1.76 (0.01%) | 1.75 (0.01%) |
| Final model RMSNorm | 14.67 (0.07%) | 14.70 (0.07%) | 14.67 (0.07%) |
| LM head＋greedy，不含 final norm | 3240.91 (15.35%) | 3241.53 (15.29%) | 3238.61 (15.32%) |
| Host-DSP boundary | 704.26 (3.34%) | 695.31 (3.28%) | 727.01 (3.44%) |
| Complete Host wall | 21107.52 (100.00%) | 21197.76 (100.00%) | 21142.99 (100.00%) |

| Paired Host wall ratio | Estimate | 95% CI |
|---|---:|---:|
| OPT_over_ORIGINAL | 0.995743 | [0.9919708703431102, 0.9998703158663232] |
| NORM_COMPACT_over_OPT | 1.001680 | [1.0005329479978924, 1.0029516089451553] |

## Warm complete-model E2E

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| OPT | 1851.089 | 47.376 |
| ORIGINAL | 1850.615 | 47.175 |
| NORM_COMPACT | 1837.716 | 47.297 |

Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.
