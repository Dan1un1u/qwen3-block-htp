# qwen3-block-htp

Standalone research runtime for executing one Qwen3 transformer block on a
Qualcomm Hexagon HTP without QNN. The project owns the block schedule, VTCM
layout, DMA transfers, HVX work, and HMX kernels.

The first experiment is a minimal FastRPC bring-up. It intentionally does not
implement a graph runtime or full-model inference.

Project constraints and experiment state live on the independent
`codex/qwen3-block-project-memory` branch.

