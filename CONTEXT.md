# Standalone HTP Block Laboratory

This language describes experiments that execute fixed Qwen3 computation scopes
through a project-owned Host-to-DSP runtime without QNN.

## Language

**Standalone HTP Runtime**:
The project-owned Host and DSP software stack that reaches HTP without a QNN
execution graph.
_Avoid_: QNN backend, custom-op backend

**Execution Unit**:
The complete functional scope entered by one run invocation and evaluated as
one latency boundary.
_Avoid_: graph, individual callback

**Bring-up Probe**:
A minimal Execution Unit whose purpose is to prove communication, placement,
resource access, timing, and deterministic output before model kernels exist.
_Avoid_: model baseline, performance candidate

**Integer HMX Tile Probe**:
A substrate Execution Unit that proves direct V79 integer-HMX instructions,
project-owned memory and device lifecycle, and exact integer output for one
architectural tile.
_Avoid_: optimized MatMul, projection baseline

**Projection Probe**:
A substrate Execution Unit that composes multiple architectural integer-HMX
tiles into one matrix projection validated by an Implementation Reference.
_Avoid_: optimized Linear, Qwen3 block baseline

**Block Package**:
A self-describing offline artifact containing one Qwen3 block's packed weights,
quantization metadata, workload inputs, and reference identity.
_Avoid_: model, QNN context, GGUF

**Physical Plan**:
The selected tile sizes, physical layouts, VTCM regions, buffer count,
prefetch distance, workers, and dependency order for an Execution Unit.
_Avoid_: QNN schedule, graph optimization

**DMA Publication Barrier**:
The explicit ordering point after User DMA reports idle and before a VTCM
buffer is published to an HMX consumer. It makes completed DMA writes visible
to the matrix engine before the producer signals buffer readiness.
_Avoid_: diagnostic delay, implicit DMA completion

**Native Boundary**:
The declared input, output, persistent state, and weight placement used when an
Execution Unit is timed without compatibility conversions.
_Avoid_: hidden conversion boundary

**Project Variant**:
One numerical and physical implementation evaluated within the Standalone HTP
Runtime, such as F16F16, W4F16, or W4U8.
_Avoid_: Qualcomm baseline, compiler specialization

**Implementation Reference**:
An independent software formulation of a Project Variant's declared arithmetic
and indexing contract.
_Avoid_: teacher accuracy, self-consistency test

**Selected Baseline**:
A Project Variant explicitly approved by the user to represent the current
project outcome.
_Avoid_: fastest observation, local pass, candidate
