# EXP0329 full profiling

| Configuration | Prefill64 Host ms | Prefill token/s | Decode42 Host ms | Decode token/s | Prefill / decode wall overhead |
|---|---|---|---|---|---|
| No R4 | 34.631832 | 1848.01 | 878.719668 | 47.80 | Reference |
| HVX butterfly R4 (OPT9) | 39.400735 | 1624.34 | 904.782562 | 46.42 | +13.77% / +2.97% |
| Dense HMX factor R4 (OPT7) | 41.667766 | 1535.96 | 918.106069 | 45.75 | +20.32% / +4.48% |

## Prefill64

| Module (us; Host-wall share) | No R4 | HVX butterfly R4 (OPT9) | Dense HMX factor R4 (OPT7) |
|---|---|---|---|
| I/O and metadata | 243.6 (0.70%) | 243.6 (0.62%) | 245.6 (0.59%) |
| Input RMSNorm | 2055.7 (5.94%) | 2133.6 (5.42%) | 2085.1 (5.00%) |
| QKV + Q/K Norm-RoPE | 7035.6 (20.32%) | 7032.4 (17.85%) | 7035.1 (16.88%) |
| QK-Softmax-AV | 3492.9 (10.09%) | 3491.8 (8.86%) | 3497.3 (8.39%) |
| O projection | 2086.3 (6.02%) | 2086.8 (5.30%) | 2095.6 (5.03%) |
| Post-attention residual + RMSNorm | 2152.1 (6.21%) | 2102.1 (5.34%) | 2108.0 (5.06%) |
| Gate/Up + SwiGLU + R4 | 7019.9 (20.27%) | 11700.1 (29.70%) | 13994.8 (33.59%) |
| Down projection | 3801.2 (10.98%) | 3830.2 (9.72%) | 3814.6 (9.15%) |
| Final residual | 2.1 (0.01%) | 2.2 (0.01%) | 2.1 (0.01%) |
| KV-cache carrier conversion | 138.6 (0.40%) | 137.6 (0.35%) | 137.4 (0.33%) |
| KV-cache append DMA | 290.7 (0.84%) | 291.0 (0.74%) | 291.0 (0.70%) |
| Block internal orchestration | 38.3 (0.11%) | 38.9 (0.10%) | 39.1 (0.09%) |
| Layer bookkeeping | 25.5 (0.07%) | 25.7 (0.07%) | 26.1 (0.06%) |
| Stage-boundary bookkeeping | 20.0 (0.06%) | 20.1 (0.05%) | 20.3 (0.05%) |
| DSP unattributed residual | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| DSP runtime setup/teardown | 116.8 (0.34%) | 117.4 (0.30%) | 119.3 (0.29%) |
| Token embedding | 59.5 (0.17%) | 59.8 (0.15%) | 59.9 (0.14%) |
| Final model RMSNorm | 17.4 (0.05%) | 17.4 (0.04%) | 14.6 (0.04%) |
| LM head + greedy selection (excluding final norm) | 5189.0 (14.98%) | 5191.8 (13.18%) | 5189.2 (12.45%) |
| True Host-DSP boundary | 846.5 (2.44%) | 878.3 (2.23%) | 892.9 (2.14%) |
| Complete Host wall | 34631.8 (100.00%) | 39400.7 (100.00%) | 41667.8 (100.00%) |

## Decode mean per token

| Module (us; Host-wall share) | No R4 | HVX butterfly R4 (OPT9) | Dense HMX factor R4 (OPT7) |
|---|---|---|---|
| I/O and metadata | 240.0 (1.15%) | 241.6 (1.12%) | 243.6 (1.11%) |
| Input RMSNorm | 366.7 (1.75%) | 366.7 (1.70%) | 365.8 (1.67%) |
| QKV + Q/K Norm-RoPE | 2493.0 (11.92%) | 2500.5 (11.61%) | 2502.4 (11.45%) |
| QK-Softmax-AV | 1959.2 (9.36%) | 1967.5 (9.13%) | 1970.5 (9.01%) |
| O projection | 1361.5 (6.51%) | 1363.1 (6.33%) | 1369.2 (6.26%) |
| Post-attention residual + RMSNorm | 373.7 (1.79%) | 369.3 (1.71%) | 373.5 (1.71%) |
| Gate/Up + SwiGLU + R4 | 6284.5 (30.04%) | 6871.0 (31.90%) | 7155.5 (32.73%) |
| Down projection | 3509.6 (16.77%) | 3526.4 (16.37%) | 3527.6 (16.14%) |
| Final residual | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) |
| KV-cache carrier conversion | 165.1 (0.79%) | 165.1 (0.77%) | 165.3 (0.76%) |
| KV-cache append DMA | 111.4 (0.53%) | 112.7 (0.52%) | 112.6 (0.52%) |
| Block internal orchestration | 30.0 (0.14%) | 30.5 (0.14%) | 30.3 (0.14%) |
| Layer bookkeeping | 17.2 (0.08%) | 17.3 (0.08%) | 17.3 (0.08%) |
| Stage-boundary bookkeeping | 1.8 (0.01%) | 1.8 (0.01%) | 1.8 (0.01%) |
| DSP unattributed residual | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| DSP runtime setup/teardown | 73.8 (0.35%) | 73.7 (0.34%) | 73.8 (0.34%) |
| Token embedding | 1.5 (0.01%) | 1.5 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 14.9 (0.07%) | 15.0 (0.07%) | 15.0 (0.07%) |
| LM head + greedy selection (excluding final norm) | 3213.4 (15.36%) | 3218.2 (14.94%) | 3220.7 (14.73%) |
| True Host-DSP boundary | 702.8 (3.36%) | 698.9 (3.24%) | 711.5 (3.25%) |
| Complete Host wall | 20921.9 (100.00%) | 21542.4 (100.00%) | 21859.7 (100.00%) |

## R4 subinterval diagnostics

| Phase | Implementation | R4 wrapper all28 us | Mean per layer us | Prepare us | Layout / combined us | HMX exposed submit/wait us | Finish us |
|---|---|---|---|---|---|---|---|
| prefill | HVX butterfly R4 (OPT9) | 3858.095 | 137.789 | 0.000 | 3854.561 | 0.000 | 0.000 |
| prefill | Dense HMX factor R4 (OPT7) | 6154.248 | 219.795 | 287.425 | 1770.690 | 717.375 | 3306.128 |
| decode | HVX butterfly R4 (OPT9) | 348.275 | 12.438 | 0.000 | 345.955 | 0.000 | 0.000 |
| decode | Dense HMX factor R4 (OPT7) | 631.875 | 22.567 | 276.966 | 27.435 | 126.614 | 193.984 |

Formal arithmetic means;19.2ticks/us. C=noR4,B=HVX FP32,D=denseFP16. Subinterval counters may overlap; B layout field is combined worker interval. See REPORT.md for scope and contracts.
