# L32-0059: SP2 HVX shift-add versus native HMX Down

Llama3.2-1B-Instruct layer7, full8192->2048 projection. Frozen SP2/W4/scales from validated FP32-residual package, confirmed equal to latest model weights. Prefill64 rows and decode1 row; no fullmodel extrapolation.

The final candidate uses four HVX workers, vector K shifts/adds, native-W4 gather unpack once per tile and reuse across rows, plus double-buffer weight DMA. HMX control uses the existing native W4 radix256 arithmetic, three raw digits, two prefill passes / decode low-high row co-pack and the same double-buffer weight DMA. Same binary and input tensors. No multiplication in the HVX integer dot core; common FP32 scale/residual epilogue retains multiplication.

Timing contract: independent complete component from logical SP2/residual input copy through preparation, DDR weight DMA, matrix/accumulator recovery, FP32 scale/residual epilogue and output publication. It includes raw-accumulator diagnostic publication for both arms. Worker creation and HMX acquisition precede DSP timing and are included only in RPC wall. The deployed fused model already carries intermediate tensors in VTCM and emits native SP2 planes, so these component totals are NOT the production model Down ledger and NOT E2E timing. Core stage excludes input/output copies and input-plane preparation, but includes HVX weight unpack / HMX raw extraction and merge, plus worker completion waits.

| Phase | HMX complete us | HVX4 complete us | HVX/HMX | Paired95% CI | HMX core us | HVX4 core us |
|---|---:|---:|---:|---|---:|---:|
| prefill | 7797.336 | 49404.308 | 6.336x | [6.314, 6.360] | 205.374 | 44673.495 |
| decode | 455.997 | 2215.833 | 4.859x | [4.843, 4.876] | 91.057 | 2122.269 |

All241 SP2 levels x signedW4[-7,7] device tests, random full matrices and full8192->2048 model-operand matrices match independent INT64 accumulators and FP32 outputs exactly. Every timed output hash remains exact. Both arms stay within8MiB; no vector tensor spill in the accepted dot kernel disassembly. Expanded HVX weights stay in VTCM and their cost is included.

The first direct-native-N-vector HVX version was correct but slower (exploratory repeat1: prefill290590.52us, decode5084.27us). The final K-vector version is substantially better; no claim of a hardware-optimal HVX kernel. An intermediate build spilled vector data and was rejected at disassembly inspection before device execution. Build/include repairs are retained.

Conclusion: removing multipliers at the arithmetic level does not make this HVX mapping competitive with HMX. Matrix-engine throughput, vector shifts/masks/reductions, operand preparation and scheduling must be compared together. Keep the current HMX implementation; no baseline promotion, PPL or model-quality inference.
