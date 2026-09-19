# L32-0057 Llama1B long decode repair

Head64 decode was excluded from the parallel GQA path. Admit it with existing opt bit128, retaining control31 and selecting159. No weights, quantization, residual or SP2 changes. Engineering counters match:1589 HMX commands and1255936 tile pairs per complete decode token;256 full-prefix pack calls. Parallel overlay rises from596480 to2494464bytes within the existing8098272byte VTCM plan. One serialized HMX/DMA owner; four independent GQA preparation slots. Exact arithmetic validated against frozen0054 references.

Five short rounds then ten formal paired rounds, repeat10. Native64+42 is unchanged; long includes Host input/RoPE staging. Fixed length fixtures, not named datasets. All samples retained. Worker service counters overlap; module tables use enclosing attention wall.

| Recipe | Shape | Control P TPS | Candidate P TPS | Control D TPS | Candidate D TPS | Decode gain |
|---|---|---:|---:|---:|---:|---:|
| a8 | 64+42 | 2373.56 | 2373.56 | 44.75 | 44.75 | +0.00% |
| a8 | 536+46 | 2546.38 | 2539.90 | 37.67 | 50.85 | +35.00% |
| a8 | 741+3 | 2536.95 | 2546.64 | 34.16 | 49.54 | +45.03% |
| sp2 | 64+42 | 2344.20 | 2344.20 | 44.77 | 44.77 | +0.00% |
| sp2 | 536+46 | 2493.67 | 2493.71 | 37.53 | 50.83 | +35.44% |
| sp2 | 741+3 | 2489.58 | 2488.34 | 34.12 | 49.05 | +43.74% |

## Confidence gates

| Recipe | Length | Decode candidate/control wall95% CI | Long/native decode TPS95% CI | Long/native prefill TPS95% CI | Pass all |
|---|---:|---|---|---|---|
| a8 | 536 | [0.7360588520720595, 0.7460498363188104] | [1.130867782308078, 1.142082607010771] | [1.0661360533351862, 1.0739896918375353] | True |
| a8 | 741 | [0.6840642112534296, 0.6946238108861774] | [1.098485491585892, 1.1171715889306233] | [1.0674113338819022, 1.0791080365738475] | True |
| sp2 | 536 | [0.7333311341230405, 0.7433012429969796] | [1.1308351762009836, 1.1403479508092424] | [1.0612905925914766, 1.0665222069755265] | True |
| sp2 | 741 | [0.6920660228036061, 0.6990765269129979] | [1.090160977386763, 1.1016659520044647] | [1.0585548033795051, 1.064758280733351] | True |

Exact audits: 2992 layer outputs and 850345984 KV elements; head and padding verified. No PPL claim or automatic baseline promotion.
Source: 0e796912fa562bd5ab9797b40fec894d18aba862; selected QBH_LONG_OPT=159, prior31 retained. Evidence root: /mnt/d/llm_exp/results/llama32-htp/l32-0057.
