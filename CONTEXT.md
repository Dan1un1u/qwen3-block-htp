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

**Provisional Source Parent**:
An accepted experiment explicitly chosen by the user as the implementation
starting point for the next experiment, without promoting it to Selected
Baseline. EXP-0023 currently has this role.
_Avoid_: final baseline, automatic promotion, fastest observation

**HVX FP16 Common-Operator Suite**:
The shared FP16 implementations of RMSNorm, RoPE, stable causal Softmax, and
SiLU-by-Up used identically by F16F16 and W4F16. Their hot elementwise and
reduction loops use HVX SIMD; they do not change either projection variant.
_Avoid_: projection optimization, A8 operator suite, scalar reference path

**Standard Stable FP16 Softmax**:
The conventional causal Softmax computation that applies the existing score
scale and mask, subtracts the row maximum, evaluates a direct approximation to
the natural exponential, sums probabilities, and normalizes. It excludes the
previous project's log2-quantized Softmax algorithm.
_Avoid_: log2 Softmax, LUT-quantized probability algorithm, changed mask rule

**Block Timing Ledger**:
The ordered set of mutually exclusive DSP wall-time intervals that spans one
complete Qwen3 Middle Block envelope exactly once. Its entries may be summed;
overlapping engine-work counters may not be inserted into this ledger.
_Avoid_: sum of stage work counters, profiling total, accelerator work sum

**Attributed Interval**:
One exclusive entry in the Block Timing Ledger, bounded by adjacent timestamps
and assigned to exactly one block activity or explicitly named handoff gap.
_Avoid_: overlapping stage counter, estimated operator share

**Unattributed Gap**:
The portion of the Block Timing Ledger that remains only because an interval
has not yet been assigned a meaningful activity name. It is a measurement
defect to close, not an operator or an optimization target.
_Avoid_: other compute, runtime overhead, unexplained bottleneck

**Engine Work Counter**:
A diagnostic duration or count for DMA, HVX, or HMX work that may overlap other
engines and therefore explains a Physical Plan without contributing an
additive entry to the Block Timing Ledger.
_Avoid_: exclusive wall time, block share

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

**Pipelined W4F16 Projection**:
A W4F16 Physical Plan that moves packed W4, expands and scales bounded FP16
Crouton tiles, and consumes them on FP16 HMX through overlapping DMA, HVX, and
HMX stages while expanded weights remain temporary VTCM state.
_Avoid_: scalar W4F16 expansion, pre-expanded FP16 weight storage, W4U8 pipeline

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

**GQA-Group Attention Pipeline**:
A complete Attention Physical Plan with eight independent tasks for Qwen3's
sixteen Q heads and eight KV heads. Each task owns two Q heads and one K/V head,
performs QK Norm and RoPE, submits unchanged QK and AV commands to one serialized
HMX owner, and runs standard Softmax on an HVX context while other groups make
progress. Scores, probabilities, worker scratch, and output remain VTCM state.
_Avoid_: concurrent HMX ownership, online FlashAttention, changed Softmax
mathematics, per-head FastRPC

**Fair Gate/Up Control**:
The fastest valid F16F16 Gate/Up schedule under the same complete-block
contract as a W4F16 candidate. For EXP-0032 this is the batch-four F16F16
schedule, not the inherited batch-two schedule.
_Avoid_: inherited control by default, unoptimized F16 comparator, micrograph
control

**Gate/Up DMA-8 Bundle**:
A packed-W4 prefetch boundary containing eight Gate/Up output bundles. Its
resident compressed data feeds two consecutive four-output-tile HMX commands
while preserving the fixed expanded-weight plan and zero-intermediate-DDR
contract.
_Avoid_: eight-output HMX command, larger expanded-weight arena, DDR cache

**Pre-encoded HMX Scale Cache**:
A read-only VTCM copy of the exact FP16 HMX scale blocks deterministically
derived from the Project Variant's existing FP32 per-channel scales before the
timed RPC. It removes repeated hot-path representation conversion without
changing quantization parameters, scale mathematics, or rounding.
_Avoid_: new calibration, approximate scale, persistent DDR intermediate,
changed numerical recipe

**Crouton-Native SwiGLU Handoff**:
A Gate/Up-to-Down MLP schedule in which matching Gate and Up HMX output tiles
remain in Crouton tile order, HVX applies the unchanged SiLU-by-Up arithmetic
directly to corresponding tile elements, and the result is written into its
final Down-activation Crouton position. It avoids full row-major Gate, Up, and
Middle materialization as well as output unpack and activation repack.
_Avoid_: changed SwiGLU mathematics, hidden DDR intermediate, flat-tensor
streaming, Down accumulation overlap

**Gate/Up HMX Batch-8**:
A complete-block Gate/Up Physical Plan that consumes eight output tiles in one
HMX command. For W4F16 it is distinct from a Gate/Up DMA-8 Bundle: DMA-8 only
prefetches eight packed bundles but feeds two batch-four HMX commands, whereas
HMX Batch-8 reduces the Gate/Up command count itself from 96 to 48.
_Avoid_: DMA-8 synonym, eight independent commands, changed projection math

**GQA Scratch Reservation**:
The aligned 64 KiB VTCM reservation retained for four complete-block GQA HMX
worker contexts after the MLP row-major Middle materialization is removed. The
current scheduler's historical arena aliases are an implicit Attention
contract; reclaiming the full Middle arena without this reservation corrupts
the QK/AV path.
_Avoid_: unused Middle storage, optional padding, MLP intermediate

**Producer-Backpressure Ring**:
A bounded VTCM producer/consumer ring whose slot count controls how far HMX may
advance before HVX post-processing drains completed output. A ring may be
mathematically correct and zero-DDR yet slow when too few slots force the
producer behind one consumer; capacity and worker exposure are scheduling
parameters that require performance evidence.
_Avoid_: storage-only ring, zero-DDR performance proof, unbounded queue

**Head-Aligned QKV Batch-4**:
A Q/K/V projection boundary in which one packed-weight DMA bundle, one HVX W4
expansion group, one HMX output command, and one Q/K readiness publication all
cover four 32-channel tiles, exactly one 128-channel Qwen3 head. Stage A may
apply it to Q alone; extending it to K/V requires the preceding gate to pass.
_Avoid_: QKV weight fusion, changed head layout, repeated first-Q prefetch

**Crouton-Native QKV Handoff**:
A producer-consumer contract in which Q/K/V projection output remains in HMX
Crouton tile order, Q/K Norm-RoPE consumes those tiles without a complete
row-major Q/K tensor, and Q/K/V are transformed directly into the Crouton
operands consumed by QK and AV. It preserves projection arithmetic, head
semantics, standard Attention mathematics, and one HMX owner.
_Avoid_: Head-Aligned QKV Batch-4, row-major QKV cache, changed Attention math

**Crouton-Native AV-to-O Handoff**:
An Attention boundary in which each AV HMX result is written into the final
Crouton activation position consumed by the O projection, eliminating the
complete row-major Attention-concat materialization and O activation repack.
_Avoid_: fused AV-O matrix multiplication, second HMX owner, DDR workspace

**Crouton-Native Norm-to-Projection Handoff**:
An FP16 RMSNorm or fused Residual-RMSNorm store path that writes its unchanged
normalized values directly into the HMX Crouton activation layout consumed by
the following projection. Input RMSNorm-to-QKV and Post-Attention
Residual-RMSNorm-to-Gate/Up are separate measurable milestones.
_Avoid_: changed RMSNorm reduction, approximate normalization, hidden row-major
copy, quantized activation

**Boundary-Local Speed Gain**:
A reduction in the complete additive ledger interval declared for one
producer-consumer boundary. It is evidence that the physical handoff removed
work, but it is not a Project Variant speed pass unless complete Host wall also
meets the same repeat-one, repeat-ten and fair-control contract. EXP-0038
demonstrated this distinction for QKV, AV-to-O and Input-Norm-to-QKV.
_Avoid_: complete-block speedup, adopted optimization, hidden overlap gain

**Combined Norm-to-Projection Candidate**:
The EXP-0038 Physical Plan that enables both Input-RMSNorm-to-QKV and
Post-Attention-Residual-RMSNorm-to-Gate/Up Crouton-native stores while leaving
QKV and AV-to-O in the EXP-0036 control form. It is byte-exact, zero
intermediate DDR, and locally passes both repeat counts for F16F16 and W4F16,
including an eleven-round paired W4F16 confirmation. Its Adoption Status is
pending and it is not a Selected Baseline.
_Avoid_: EXP-0038 overall pass, QKV-native candidate, adopted baseline
