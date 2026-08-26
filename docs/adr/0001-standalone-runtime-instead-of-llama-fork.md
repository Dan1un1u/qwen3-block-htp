# Build a standalone block runtime instead of forking llama.cpp-npu

The project uses `llama.cpp-npu` and `htp-ops-lib` as implementation references
but builds an independent fixed-scope Host/FastRPC/DSP runtime. Forking the full
llama.cpp stack would retain ggml's per-node hybrid execution and make it easy to
lose the cross-kernel scheduling boundary that motivated leaving QNN; a small
runtime gives the project ownership of VTCM, DMA, HVX, HMX, and the single
Execution Unit boundary at the cost of implementing its own model kernels.

