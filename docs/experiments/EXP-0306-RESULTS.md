# EXP-0306: Qwen3-0.6B FP-intermediate motivation

M64 full28 prefill plus one fixed teacher-forced decode; native signed W4 matrices, FP32 residual/RMSNorm, vector FP Softmax/SwiGLU with fused boundary conversions, uniform INT16 Down, no rotations. Same operator-boundary attribution as EXP0305; four attention contexts, three SwiGLU workers, retain matrix-internal DMA/HMX. Model-specific original0.6 prefix mode0 retained (1.7 historical prefix patch is not transplanted). Native cache layout retained. Not an optimized-baseline speed comparison.

All1016 historical INT16 deployment payloads recovered byte-exact against EXP0304 manifest6f90b9ce5138b251f4ab732cb1ee3a12314b081a9073ea6ae2da802b40d26142. No new quantization or AV-O folding. Device path is a dedicated alias of verified source package; original weights/scales unchanged.

28 x65536 SwiGLU pairs exact against independent FP32 polynomial/rounding oracle, maximum one code against float64. Independent integer QK/AV, float64 continuous Softmax, actual vendor-QHL reciprocal sqrt reference retained for Q/K Norm, FP32 projection/residual/norm. Singlelayer, consecutive3, full28 prefill/decode layer hashes and head token/code exact; full56 boundaries. 8MiB requested/granted, peak5744384 bytes, zero timed intermediate DDR/spill. Additive ledgers close exactly; overlapping worker sums excluded.

Initial audit1 CLI rejected unsupported legacy prefill-cache override before device compute; restored existing native cache configuration. Failed evidence retained. No numerical tolerance or physical guard relaxed. Five short plus ten formal repeat10, formal200 token/pass profiles; all rounds retained. No other device work, payload hashing or build during formal timing.

Prefill Host wall 49438.94739 us, 1294.525943 token/s. One subsequent decode 31767.82662 us, 31.478389 token/s (diagnostic, not long decode benchmark).

- FP Softmax + QDQ: 2858.015625 us, 5.780899% complete Host wall.
- FP SwiGLU + QDQ: 11147.982813 us, 22.548989% complete Host wall.

Combined 28.329888%. FP arithmetic and fused QDQ cannot be separated here. Fractions describe an explicit operator-boundary diagnostic, not the critical path of overlapping production execution. No quality or baseline promotion.

Figure b workbook G extended with Llama3B L32-0064 and Qwen0.6 EXP0306, percentage values only, four bars sum100%. All A-F archive parts and existing G cells outside D41:E65/A68 preserved. Figure a unchanged pending external historical-code Qwen0.6 measurements. Workbook checksum0fafa21de314c24ce60120b64c15b587dd207936543e90599397200aabaa25cd. Full additive modules in MODULES.md.
