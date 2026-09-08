# After EXP0251

Wide-score repair maps correctly to the tested native-U8 hardware path with no demonstrated Host-wall penalty and passes the10% single-layer speed gate. Keep it as an experimental implementation; model-quality acceptance is unresolved.

Discuss next scope: validate longer cached contexts/all layers and device PPL before promotion; separately examine whether actual dense-HMX rounding plus current log2 exponent/SOLE quantization amplifies one-code Q/K changes. EXP0250 software residual precision ablations are the starting evidence. Any higher-precision exponent/reciprocal or R3 compensation needs a new explicit experiment and frozen comparisons. Do not silently reintroduce expensive mode5 refinement, butterfly/FWHT, grouped weights or mixed precision. No new algorithm or full-model experiment is authorized by this closure.
