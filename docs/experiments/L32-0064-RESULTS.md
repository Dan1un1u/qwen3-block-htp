# L32-0064: Llama3.2-3B FP-intermediate motivation

M64 full28 prefill plus one fixed teacher-forced decode; native signed W4 matrices, FP32 residual/RMSNorm, vector FP Softmax/SwiGLU with fused boundary conversions, uniform INT16 Down, no rotations. Operator-boundary diagnostic, four attention contexts and three SwiGLU workers; retain GEMM DMA/HMX overlap. Not an optimized-baseline speed comparison.

904 historical deployment payloads recovered byte-exact against L32-0044 manifest. Only 56 per-layer INT16 qparams/LUT files reused from verified L32-0058 metadata; no new weight quantization or AV-O folding. New package manifest and deployment SHA checks retained.

28 x 65536 SwiGLU pairs exact against independent FP32 polynomial/rounding oracle, float64 reference maximum one output code. Independent integer QK/AV, float64 continuous Softmax, FP32 projection/residual/norm reference. Singlelayer, consecutive3 and full28 each match prefill/decode layer hashes and head token/code exactly; fullmodel56 exact boundaries. 8MiB grant, peak 8360416 bytes, no timed intermediate DDR/spill, one HMX owner. Exclusive ledgers close exactly; overlapping worker sums excluded from stacked fractions.

Initial singlelayer audit rejected by 3B production-only pipeline guard before compute. Dedicated FP-island builds now allow only the exact phase3/wide7/W4U8/INT16 combination; production guard unchanged. Failed audit and initial staging/preflight race retained. No numerical threshold changes.

Five short rounds plus ten formal rounds, each repeat10; formal200 token/pass records. No concurrent device hashes/builds during timing. All rounds retained.

Prefill Host wall 85207.07025 us, 751.111379 token/s. One subsequent decode 43117.25678 us, 23.192570 token/s (diagnostic, not long decode benchmark).

- FP Softmax + QDQ: 4192.936458 us, 4.920879% of complete Host wall.
- FP SwiGLU + QDQ: 29822.657292 us, 35.000214% of complete Host wall.

Combined 39.921093%. These include floating arithmetic and fused QDQ; QDQ alone is not identifiable. These are exclusive diagnostic-schedule fractions, not production critical-path percentages. No quality or baseline promotion. Figure a unchanged. Full additive table in MODULES.md; plotting data in figure-b-llama3b.csv.
