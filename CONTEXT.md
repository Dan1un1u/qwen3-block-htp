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

**Qwen3 Middle-Block Projection**:
A model-shaped Linear from a Qwen3 middle block whose K and N dimensions match
the block architecture, while M is the declared execution-row scope.
_Avoid_: full Qwen3 block, arbitrary projection probe

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

**W4U8 Projection Substrate**:
A model-shaped Projection Probe that proves the canonical W4U8 arithmetic and
weight-storage boundary for the gate/up and down shapes.
_Avoid_: complete W4U8 block, optimized W4U8 baseline

**Expanded-S8 Control**:
The strict A/B Project Variant whose Native Boundary contains exactly the S8
carrier produced by its paired packed-W4 variant, while every other declared
contract remains equal.
_Avoid_: independent W8 quantization variant, performance baseline

**Prepared Runtime Session**:
A Standalone HTP Runtime session that retains declared accelerator resources
across multiple Execution Unit invocations while reporting prepare and release
costs separately from hot invocation latency.
_Avoid_: free setup, kernel-only speedup, persistent kernel

**Fixed Full-VTCM Session**:
A Prepared Runtime Session that requires the architecture-defined total VTCM
as one common experiment budget while each Physical Plan reports only its own
peak live allocation.
_Avoid_: current-available best effort, per-operator VTCM request, assumed grant

**HMX Crouton Address Contract**:
The architecture-required base alignment and address encoding for tensors
consumed or produced by HMX. On the current V79 FP16 path, Crouton activation,
weight, and output tiles require 2 KiB alignment, and bias/scale loads require
256-byte alignment. A logically in-range but under-aligned VTCM pointer is an
invalid physical contract and may produce non-finite or unstable results.
_Avoid_: optional arena padding, numerical tolerance, compiler alignment hint

**Zero-Count Worker Handoff**:
A producer/consumer semaphore contract whose first `down` must block until the
peer publishes work or completion. QuRT `qurt_sem_init()` defaults to count
one, so HMX command-ready, command-done, and worker-started handoffs must use
`qurt_sem_init_val(..., 0)` explicitly.
_Avoid_: default semaphore initialization, timing-dependent retry

**Qwen3 Middle Block**:
One complete Qwen3 transformer-layer Execution Unit from residual input to
residual output, containing both self-attention and gated MLP computation.
_Avoid_: MLP block, projection group, full model

**Paired Gate/Up Projection**:
One Execution Unit that evaluates two independent `2048 -> 6144` projections
sharing the same `[64, 2048]` activation by representing their concatenated
weights and outputs as one `2048 -> 12288` physical projection. It does not
apply SiLU, multiply gate and up outputs, or execute the down projection.
_Avoid_: fused MLP, gate/up arithmetic fusion, complete Qwen3 block

**VTCM-Resident MLP Intermediate**:
The Gate/Up tile values, activated Gate-by-Up product, and Down activation that
exist only in HMX accumulator state, HVX registers, or declared VTCM regions
between the legal DDR input/weight boundary and final-output boundary.
_Avoid_: shared-rpcmem workspace, cached DDR intermediate, hidden tensor output

**W4 Scale Placement**:
The physical stage at which a per-output-channel W4 scale is applied. In
HVX-prescale, HVX expands each signed nibble and multiplies it by the channel
scale before integer HMX. In HMX-postscale, HVX expands the raw signed nibble
and HMX applies the channel scale during accumulator conversion. These are
mathematically equivalent under the experiment's exact integer scale domain,
but they have different HVX/HMX work and scheduling costs.
_Avoid_: different quantization algorithm, different W4 weights

**Interference-Aware Phase Pipeline**:
A Physical Plan that continues to overlap DDR User DMA with accelerator work,
but prevents HVX weight expansion and HMX matrix consumption from executing at
the same instant. Packed bundles are expanded in a bounded output-channel
group, all HVX workers reach a group barrier, and HMX then consumes that group
while User DMA prefetches the next packed group. It tests whether avoiding
HVX/HMX VTCM and execution-resource interference is faster than maximizing
nominal overlap.
_Avoid_: serialized pipeline, disabled prefetch, reduced-HVX fallback

**Streaming Microchunk Handoff**:
A Physical Plan that retains the current 64- or 96-K-tile VTCM allocation and
DMA transaction boundary, while dividing each resident expanded slot into
ordered 32-K-tile readiness regions. HVX publishes each completed region with
a generation-tagged release; one persistent HMX consumer begins after the
first required region and accumulates subsequent streams as they become ready.
The slot is released only after its complete superchunk is consumed. Producer
lookahead is bounded so HVX cannot run an entire group ahead of HMX.
_Avoid_: strict phase separation, full-group barrier, per-microchunk FastRPC,
per-microchunk expanded-slot recycle
