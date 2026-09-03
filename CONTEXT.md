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

**Recipe-Specialized Baseline**:
A Selected Baseline approved for one numerical recipe after the Public Common
Layer is frozen; it may diverge from the fair cross-recipe configuration.
_Avoid_: Public Common Baseline, fair comparator, local candidate

**Recipe-Fastest Promotion Score**:
The formal rotated-pair repeat-ten complete-block Host wall latency used as the
only performance-ranking metric among candidates that already satisfy the
mathematical, physical, scope and evidence eligibility contract. Repeat-one,
module timings and engine counters remain required diagnostics, not promotion
vetoes. Near-noise results require additional paired rounds and otherwise stay
tied.
_Avoid_: one lucky run, module-local gate, work-cycle ranking, correctness gate

**Enabling Candidate**:
An evidence-valid structural implementation retained as a possible parent even
when its standalone repeat-ten wall is slower than the Selected recipe-fastest
baseline. It exists to keep the search from becoming single-path greedy, but it
cannot be called a baseline and a later combination must still beat the current
baseline before promotion.
_Avoid_: selected baseline, permission to ignore final wall time, automatic adoption

**Provisional Source Parent**:
An accepted experiment explicitly chosen by the user as the implementation
starting point for the next experiment, without promoting it to Selected
Baseline. EXP-0038 currently has this role and is also the source experiment
for the first Selected Baseline; the two roles remain conceptually distinct.
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

**GQA-Group-Major QKV Super-Projection**:
A Q/K/V scheduling and storage plan whose outer unit is one Qwen3 GQA group:
two Q heads, one K head, and one V head. It reuses one published activation
carrier and presents the unchanged Q/K/V weight segments in group-major order,
while publishing each head as soon as its own projection segment completes.
_Avoid_: changed Q/K/V mathematics, concatenated logical Linear, delayed
whole-group readiness, concurrent HMX owners, fused Attention

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

**Combined Norm-to-Projection Baseline Mode**:
The EXP-0038 Physical Plan that enables both Input-RMSNorm-to-QKV and
Post-Attention-Residual-RMSNorm-to-Gate/Up Crouton-native stores while leaving
QKV and AV-to-O in the EXP-0036 control form. It is byte-exact, zero
intermediate DDR, and locally passes both repeat counts for F16F16 and W4F16,
including an eleven-round paired W4F16 confirmation. In the W4F16 variant it is
the user-promoted `W4F16-EXP0038-NORMS` Selected Baseline. This promotion does
not change EXP-0038's overall failed local gate or adopt its QKV/AV candidates.
_Avoid_: EXP-0038 overall pass, QKV-native baseline, all-mode baseline

**Integer Log2 GQA Attention Pipeline**:
The EXP-0042 complete Attention replacement that consumes native-tile
asymmetric-U8 Q/K/V, runs U8-by-S8 QK on integer HMX, produces a four-bit log2
Softmax probability carrier on HVX, runs U8-by-S8 AV on integer HMX, and hands
native U8 output tiles directly to the O projection. Eight GQA groups share
one serialized HMX owner while four HVX contexts perform group-local work.
_Avoid_: isolated custom Softmax, FP16 Attention island, concurrent HMX owners

**Actual-Boundary Attention Audit**:
The EXP-0042 correctness method that captures actual device Q/K/V at the
declared integer-Attention input boundary only in audit-on mode, independently
recomputes QK, log2 Softmax and AV on the Host, and requires byte-exact stage
agreement. Its explicit diagnostic DDR export is not a zero-DDR performance
run; the physical gate is measured separately with audit disabled.
_Avoid_: circular device reference, offline-stage hash gate, audit DDR hidden
as core traffic

**Generation-Safe Q/K Head-Ready Overlap**:
The EXP-0044 Stage-B schedule in which each completed 128-channel Q or K head
is published with a generation tag during QKV projection. Dedicated HVX
workers consume only complete, disjoint heads for the unchanged Q Norm-RoPE
and fused K Norm-RoPE/carrier preparation, while the main thread continues W4
unpack and synchronous projection HMX. Every prep worker joins before QK HMX
starts, preserving one HMX owner and zero intermediate DDR.
_Avoid_: partial-head consumption, concurrent HMX owners, Attention HMX during
projection, changed qparams or arithmetic

**W4U8 QKVO Batch-4 Pipeline**:
The EXP-0045 Physical Plan that retains Generation-Safe Q/K Head-Ready
Overlap while batching four adjacent 32-channel output tiles for Q, K, V, and
O. Each batch uses one linked packed-weight-plus-bias DMA publication, one HVX
W4-to-S8 expansion group, and one integer-HMX command. Two compressed and two
expanded VTCM slots allow the next batch's DMA and expansion to overlap the
previous HMX command. Q/K publication remains aligned to one complete
128-channel head, and O continues to consume the native U8 Attention carrier.
_Avoid_: fused QKV weights, changed arithmetic, concurrent HMX owners, DDR
workspace, QNN lowering

**Shared Native U8 QKV Input Carrier**:
The EXP-0048 Stage-A boundary in which input RMSNorm scatters its unchanged
asymmetric-U8 result once into the exact integer-HMX activation layout shared
by Q, K, and V. It removes three repeated row-major-to-HMX carrier packs while
leaving Q/K/V weights, qparams, commands, and arithmetic unchanged.
_Avoid_: fused QKV projection, shared QKV weights, changed RMSNorm arithmetic,
row-major QKV input

**Native U8 O-to-Residual Handoff**:
The EXP-0048 Stage-B boundary in which integer HMX writes O-projection output
in native tile order and the post-Attention residual gathers those tiles
directly before producing the unchanged native Gate/Up carrier. The target
interval is O projection plus post-Attention residual, because gather work is
moved across that boundary rather than eliminated entirely.
_Avoid_: fused O-residual arithmetic, hidden gather cost, row-major O output,
changed qparams

**Batched W4U8 Gate/Up HMX Consumer**:
The EXP-0049 schedule in which the single integer-HMX owner consumes eight
consecutive Gate/Up output tiles during one worker command. Each tile still
uses its original streamed W4-to-S8 expanded regions, bias/qparam encoding and
U8 output code. The final output store executes inside the same command as the
accumulation. Eight expanded VTCM slots prevent in-batch slot reuse; HVX
expansion, DMA publication and SwiGLU post-processing remain concurrent around
the serialized HMX consumer. This changes command cadence, not HMX tile math.
_Avoid_: eight-output matrix fusion, changed Gate/Up weights, ring-depth-only
optimization, eight independent HMX commands

**Accumulate-Store Command Fusion**:
A scheduling transformation that performs the final integer-HMX U8 output
store before returning from the streaming accumulation worker command. It
removes a second command wake-up that previously contained no matrix
accumulation. The accumulator arithmetic and output bytes are unchanged.
_Avoid_: fused projection arithmetic, fused SwiGLU, omitted output store,
operator fusion

**Complete Profiling Comparison**:
The mandatory experiment-closure report containing the primary Host and DSP
latencies, every mutually exclusive Block Timing Ledger interval, relevant
overlapping engine-work diagnostics, physical traffic and resource counters,
correctness results, and evidence provenance for both repeat-one and repeat-ten
scopes. Absolute control and candidate values and their deltas are shown even
when a field is unchanged or zero. Additive ledger intervals and overlapping
HMX/HVX/DMA work are always placed in separate tables.
_Avoid_: gate-only summary, selected-stage-only table, link-only handoff, sum of
overlapping engine counters

**Generation-Safe Two-Chunk Down Command**:
The EXP-0050 W4U8 Down schedule in which one integer-HMX worker command begins
after the first 96-K-tile expanded chunk is ready, accumulates it, waits inside
the same command for the second chunk's generation tag while HVX produces that
chunk, accumulates the second chunk, and stores the unchanged U8 output. It
reduces worker-command cadence without delaying HMX until both chunks are
pre-ready and without changing tile arithmetic, weights, qparams, or traffic.
_Avoid_: pre-ready serialization, fused Down arithmetic, two concurrent HMX
owners, changed K blocking

**Generation-Safe Q Head-Pair Command**:
The rejected EXP-0051 Stage-A W4U8 schedule in which one integer-HMX worker
command computes two consecutive 128-channel Q heads, publishes the first head
before waiting for the second head's expanded-weight generation, then computes
and publishes the second. It reduces Q commands from 16 to 8 without changing
tile arithmetic or per-head publication granularity, but does not shorten the
QKVO physical HMX lifetime and fails the repeat-one complete Host-wall gate.
_Avoid_: accepted QKV optimization, permission to extend the schedule to K/V,
plain batch eight, proof that command count is the main QKV bottleneck

**Independent Ordered Q-Then-K Head Preparation**:
The EXP-0052 W4U8 Q/K Norm-RoPE scheduler that exposes sixteen Q-head tasks
before eight K-head tasks to three HVX workers. It replaces the former
group-owned task, in which a worker could finish two Q heads and then occupy a
worker while waiting for that group's unavailable K head. Projection commands,
head publication, arithmetic, qparams, tensor layouts and total logical work
remain unchanged; only task claim order changes. The accepted local evidence
is a shorter Q/K preparation join tail and complete QKV interval, not a faster
QKV matrix multiplication.
_Avoid_: reordered Attention semantics, QKV projection fusion, more HMX
parallelism, reduced Q/K Norm-RoPE work, Selected Baseline without promotion

**Paired Native U8 Q/K Norm-RoPE Task**:
The EXP-0055 W4U8 preparation schedule that replaces sixteen independent Q
head tasks and eight independent K head tasks with eight adjacent-Q-head pairs
and four adjacent-K-head pairs. Each pair converts the invariant gamma once
and each row's shared RoPE cosine/sine table once, then executes the original
per-head RMS reduction, reciprocal square root, rotation, requantization and K
carrier construction independently. It is byte-exact, halves task cadence,
and locally reduces Q/K Norm-RoPE work, preparation join, QKV ledger time and
complete Host wall at repeat one and repeat ten.
_Avoid_: paired-head arithmetic, shared RMS statistics, changed qparams,
Attention fusion, Selected Baseline without promotion

**Paired Full-Width QK Requant-Softmax Load**:
The EXP-0063 W4U8 Attention boundary that packs the corresponding 64-byte
score rows from both Q heads of one GQA group into one 128-byte HVX vector,
applies the unchanged QK requantization once at full SIMD occupancy, and then
computes two mathematically independent log2 Softmax rows. Performance mode
does not materialize the requantized score carrier; audit mode writes each
64-byte half back to its original score tiles solely to retain the authoritative
QK hash. It removes a separate full-carrier pass without merging probability
mass, changing qparams, changing mask semantics, or changing QK/AV HMX work.
_Avoid_: shared two-head Softmax denominator, changed Attention arithmetic,
ordinary 64-byte row fusion, Selected Baseline without promotion

**Per-Head QK/AV HMX Command Batch**:
The EXP-0064 W4U8 Attention schedule that submits both QK output tiles and all
four AV output tiles of one Q head through the integer-HMX worker's existing
multi-output loop. It removes one command handoff per extra output tile while
preserving every tile pair, weight, bias, qparam, score, probability, and AV
byte. Attention HMX command count falls from 96 to 32 and total block HMX
command count from 256 to 192 without changing one-owner ordering or creating
an intermediate DDR boundary.
_Avoid_: fused QK/AV arithmetic, concurrent HMX owners, changed Attention
math, fewer HMX tile pairs, Selected Baseline without promotion

**Per-Group SOLE Probability-LUT Templates**:
The EXP-0065 W4U8 Softmax implementation that recognizes the SOLE division
approximation depends on a row sum only through its highest set bit and the
next bit. It builds the fourteen reachable 32-byte LUT-bank templates once in
each GQA group's reused VTCM scratch and selects two banks per paired-head row,
instead of running two sixteen-entry scalar builders on every row. It preserves
QK codes, sums, mask semantics, probability bytes, and all HMX work exactly.
_Avoid_: changed Softmax approximation, shared head denominator, DDR lookup
table, major block-level speedup, Selected Baseline without promotion

**Expanded Persistent Attention HVX Domain**:
The EXP-0068 W4U8 schedule that increases the persistent Attention execution
domain from the main DSP thread plus three HVX workers to the main thread plus
five workers. The larger domain consumes the same twenty-four paired Q/K
Norm-RoPE tasks and eight GQA groups, preserves one serialized HMX owner and
all numerical bytes, and shortens the Q/K preparation join tail. It is a
parallel-domain occupancy change, not reduced work: aggregate Q/K Norm-RoPE,
Softmax, and Attention wait counters may rise even while the additive
QKV-plus-Attention critical interval falls.
_Avoid_: concurrent HMX owners, changed Attention arithmetic, six worker
threads, reduced Softmax work, Selected Baseline without promotion

**Split Attention Context Domain**:
The rejected EXP-0069 schedule that keeps EXP-0068's six-context persistent
pool and uses all six contexts for Q/K preparation, but allows only five total
contexts to claim GQA Attention tasks. It reduces Softmax aggregate work,
Attention queue wait, and the Attention ledger, yet the complete-block gain is
too small and the repeat-one paired Host wall regresses. It is not an adopted
source parent.
_Avoid_: accepted Attention optimization, fewer persistent workers, changed
Attention arithmetic, permission to retune four/five active contexts without
a new block-level hypothesis

**Per-GQA-Group QK/AV HMX Command Batch**:
The EXP-0070 W4U8 Attention schedule that treats the two contiguous Q heads of
one GQA group as two HMX M tiles. It submits one worker command for the four QK
output tiles and one for the eight AV output tiles, instead of one QK2 and one
AV4 command per Q head. Attention commands fall from 32 to 16 and total block
commands from 192 to 176 while all 49,408 tile pairs, numerical bytes, DMA
traffic, VTCM residency, context counts, and one-owner ordering remain
unchanged. It passes repeat-one and repeat-ten ordinary and paired Host-wall
and Attention gates and is the latest locally eligible W4U8 candidate.
_Avoid_: one physical HMX instruction for two heads, concurrent HMX owners,
changed QK/AV arithmetic, changed tile-pair count, Selected Baseline without
promotion

**Audit-Only Probability Row-Sum Reduction**:
The rejected EXP-0066 Softmax schedule that skips probability row-sum and
min/max reductions when numerical telemetry is absent, while retaining them
and their exact values in audit mode. It reliably reduces Softmax and Attention
work and improves repeat-ten wall latency, but fails the repeat-one paired
complete-wall gate and is not an adopted source parent.
_Avoid_: changed probability bytes, missing audit telemetry, accepted W4U8
candidate, permission to stack the rejected change without a new hypothesis

**Parallel Native U8 Input RMSNorm**:
The EXP-0071 schedule that splits the 64 independent Input RMSNorm rows into
sixteen four-row tasks consumed by the main DSP thread and five already-
persistent HVX workers. Each task calls the unchanged per-row centered-square
reduction, scalar square root, gamma, requantization, and direct native HMX
activation store. It changes execution occupancy only and is byte-exact at the
native activation boundary.
_Avoid_: batched reciprocal-square-root approximation, changed RMSNorm math,
new worker creation, row-major handoff, Selected Baseline without promotion

**Stage Timing Boundary Interval**:
The additive profiling interval that explicitly accounts for qtimer reads and
adjacent stage-boundary bookkeeping not enclosed by an operator stage timer.
It closes the complete invocation ledger without assigning instrumentation
overhead to an operator and is not an optimization target.
_Avoid_: unattributed operator work, hidden runtime cost, relaxed ledger gate

**Six-Context Native U8 Residual**:
The EXP-0072 schedule that lets the main DSP context and all five already-
persistent HVX workers claim the same sixteen four-row native U8 residual
tasks at both block residual boundaries. It changes only occupancy; task
granularity, Q14 residual arithmetic, fused RMSNorm, qparams, layouts, buffers,
DMA, HMX, and VTCM are unchanged. The measured combined residual ledger falls
about 21.5% and complete repeat-ten Host wall about 0.63%.
_Avoid_: new worker creation, changed residual math, changed task size,
additional buffers, Selected Baseline without promotion

**Shared SOLE LUT-Template Bank**:
The rejected EXP-0073 Softmax storage experiment that replaces eight
context-local builds of the same fourteen-template, 448-byte SOLE probability
bank with one read-only VTCM bank shared by all GQA tasks. It is byte-exact and
zero-DDR, but increases Softmax aggregate work by about 8.2%, Attention wall by
about 4.4%, and repeat-ten Host wall by about 0.55%. Under the current six-
context schedule, local redundant construction is cheaper than cross-context
shared reads, most plausibly because of VTCM locality or bank contention.
_Avoid_: reusable shared table is automatically faster, template-build count
as a latency proxy, stacking the shared bank onto later Attention candidates

**Globally Staged Row-Sliced U8 Attention**:
The rejected EXP-0074 schedule that replaces eight coarse full-GQA tasks with
three global stages: eight paired-head QK commands, sixteen independent 32-row
Softmax tasks, and eight paired-head AV commands. It is byte-exact, retains
zero intermediate DDR and the same HMX work, and shortens the Attention ledger
about 4.6%, but increases aggregate Softmax work about 7.8% and Attention pool
wait about 134%. Repeat-ten Host wall improves about 0.7%, while repeat-one
ordinary and paired Host medians are marginally worse, so the strict complete-
block gate fails. The result supports finer Softmax task granularity but rejects
full QK/Softmax/AV stage barriers as the implementation boundary.
_Avoid_: accepted W4U8 candidate, free stage barriers, lower aggregate Softmax
work, permission to stack global staging onto a later candidate

**Dependency-Driven Row-Sliced U8 Attention**:
The EXP-0075 W4U8 schedule that starts the six-context Attention pool once and
lets every context move independently through the shared QK queue, sixteen
32-row Softmax tasks, and the AV queue. Per-group generation tags publish QK
scores and the two completed probability slices; no full-pool QK-to-Softmax or
Softmax-to-AV barrier exists. It keeps byte-identical arithmetic, 176 HMX
commands, 49,408 tile pairs, zero intermediate DDR and one HMX owner. Relative
to EXP-0072 it reduces the Attention ledger about 19.6% and repeat-ten Host wall
about 1.9%, and is the latest locally eligible W4U8 candidate.
_Avoid_: full FlashAttention fusion, concurrent HMX owners, changed Softmax
math, Selected Baseline without user promotion

**Two-Stage AV Requantization Boundary**:
The accepted W4U8 Attention output mapping first rounds and saturates the HMX
accumulator into a U8 carrier centred at 128, then applies the group multiplier
and final output zero point in HVX. Rejected EXP-0076 proved that folding a
multiplier of nine or eleven into one affine HMX conversion changes rounding
order: on the real layer-14 boundary it changes 19,713 of 131,072 bytes, with a
5 LSB maximum error. Only the multiplier-one group is exactly fusible under the
current qparams. EXP-0075 remains the source parent.
_Avoid_: calling the post-HMX pass redundant, direct affine fusion under the
same byte-exact contract, performance testing after this correctness failure

**Gate/Up Cross-Phase Initial DMA Prefetch**:
The rejected EXP-0077 schedule starts the same first four linked Gate/Up weight
descriptors after O projection and overlaps them with the native U8
post-attention residual/RMSNorm. The existing Gate/Up pipeline later adopts and
publishes that chain without resubmission. It is byte-exact and preserves all
weight bytes, descriptor counts, HMX commands, tile pairs, VTCM residency and
zero-intermediate-DDR constraints. Repeat ten improves Gate/Up about 1.05% and
complete Host wall about 0.74%, but repeat one Gate/Up and the combined interval
regress, so the strict gate fails. Advancing only the first DMA chain does not
remove per-block MLP pipeline startup and teardown cost.
_Avoid_: accepted W4U8 candidate, permission to stack this mode, proof that
Gate/Up weight prefetch is useless, or a substitute for persistent MLP workers

**Q-Prefix-4 GQA QKV Schedule**:
The rejected EXP-0085 common schedule that emits the first four GQA groups of
Q, then all K groups, then the remaining four Q groups before V. It was the
best bounded compromise between the current all-Q-first control and pure
group-major order: pure group-major exposes K early but moves the critical tail
to late Q heads. Seven-round formal evidence shows Q-Prefix-4 improves repeat-
ten QKV by about 2.08% for F16F16 and 4.50% for W4F16, but regresses W4U8 QKV
by about 0.84% and complete W4U8 Host wall by about 0.52%. The W4U8 paired-head
preparation and projection pipeline is already balanced around its current
Q-major order, so this is not a common cross-recipe optimization.
_Avoid_: shared baseline promotion, assuming earlier K publication always
shortens the critical path, recipe-specific adoption without a new approved
experiment, or repeating the same bounded group/window/prefix search

**Rejected Down First-Chunk Prestage**:
The EXP-0089 Stage-B schedule that uses a phase-overlay VTCM ring to move and
expand the first four Down output bundles while Gate/Up is still producing the
middle activation. It preserves all arithmetic and physical-work counts and
reduces the Down interval, but the borrowed DMA/HVX work increases Gate/Up
ready and producer-slot waits. Gate/Up regresses at repeat one and repeat ten,
and ordinary repeat-ten complete Host wall also regresses, so Stage C HMX
interleaving was not run and EXP-0084 remains the selected W4U8 baseline.
_Avoid_: accepted cross-phase MLP pipeline, permission to interleave Down HMX,
proof that Down prestaging is free, or stacking this schedule onto a candidate

**Rejected Gate/Up Activation-Priority Queue**:
The EXP-0090 audit-only hypothesis that ready Gate/Up SwiGLU tasks might spend
a material interval behind W4-to-S8 expansion tasks in one FIFO. Seven device
runs observe all 192 activation tasks, but their median mean residence is only
28.7 ticks versus about 40.7 ticks of work per task; only 1.29 older tasks are
ahead on average and the typical maximum is three. The serial gate therefore
stops before priority-queue implementation. EXP-0084 remains the selected
W4U8 baseline.
_Avoid_: hidden long activation backlog, priority scheduling as an established
bottleneck, implementing the rejected dual-class queue without a new hypothesis

**Four-Row Native U8 Residual Shuffle**:
The locally eligible EXP-0093 residual boundary plan that reconstructs four
contiguous activation rows from native integer-HMX tiles in registers while
preserving byte-exact residual semantics and VTCM residency.
_Avoid_: accepted baseline, changed residual arithmetic, residual approximation

**Prepared-Session Attention Metadata Cache**:
A VTCM-resident set of immutable, group-specific Attention lookup and bias
objects built during the warmup call of one Prepared Runtime Session and reused
by later measured block calls. It excludes mutable per-context gather scratch.
EXP-0094 demonstrated that this can remove V-pack setup work without changing
the mathematical or physical HMX contract, but a local work reduction alone is
not sufficient for baseline adoption when complete Host wall does not pass its
predeclared gate.
_Avoid_: accepted baseline, complete Attention speedup, free VTCM capacity

**Four-Row Native Softmax Carrier Shuffle**:
The locally eligible EXP-0095 layout transform that reconstructs four logical
QK score rows from four aligned physical HMX tiles with two levels of HVX
`vshuff`, applies the unchanged integer Softmax one row at a time, and
inverse-transposes four probability rows back to native AV input tiles. Every
HVX worker owns a private 1 KiB carrier in phase-retired VTCM; sharing that
carrier by GQA group is invalid because the two row slices of one group can run
concurrently. The accepted evidence is byte-exact and improves complete Host
wall, but it is not a Selected Baseline until explicit user promotion.
_Avoid_: accepted baseline, changed Softmax approximation, shared per-group
carrier, additional VTCM allocation

**Public Common Layer**:
The user-governed source, physical contracts and profiling primitives eligible
for the fair three-recipe comparison. Shared eligibility does not require
identical implementation code, but recipe-specific advantages must not be
hidden inside this layer.
_Avoid_: fastest-per-recipe result, identical code-path requirement, permission
to merge a specialization silently

**Frozen Public Baseline**:
A user-approved consolidated source revision that fixes the Public Common
Layer and its fair comparison configurations. Future specializations branch
from it and cannot silently redefine it.
_Avoid_: immutable repository history, prohibition on new experiments,
automatic promotion of a specialized result

**Recipe Specialization**:
An explicitly recipe-scoped optimization evaluated separately from the fair
common baseline. It may update the fastest result for that recipe but does not
retroactively change the Public Common Layer.
_Avoid_: public/common optimization, fair cross-recipe comparison, permission
to alter another recipe

**Documented Gate Exception**:
An explicit user adoption decision that preserves the experiment's original
failed local gate while recording why a bounded component is nevertheless
accepted. It never converts the historical gate result into a pass.
_Avoid_: relaxed gate after observing data, automatic exception, erased
negative evidence

**Rejected Clipped-Q7 Gate/Up Arithmetic**:
The EXP-0096 candidate that replaces the formal 128 KiB float-SiLU-derived
Gate/Up LUT with the repository's older direct clipped-Q7 SiLU and Q5 product
helper. Although disassembly proves a zero-`vgather` HVX multiply/shift/pack
path and activation work falls in a diagnostic run, the helper is not the same
mathematical mapping: exhaustive U8-by-U8 comparison finds 57,270 mismatches
with a 65-LSB maximum, and the real block changes 6,169 output bytes. The
experiment stops before performance testing.
_Avoid_: byte-exact arithmetic replacement, same formula as the formal LUT,
accepted optimization, permission to resume the old clipped-Q7 helper

**Rejected Exact Affine Gate/Up Carrier**:
The EXP-0097 representation that proves each formal Gate/Up LUT row can be
encoded exactly by one unsigned 16-bit magnitude, coefficient sign, and a Q15
or Q14 rounded shift. It reduces the hot-loop gathers from four to two and is
byte-exact over all 65,536 codes and the real block, but the added vector
multiply, dual shifts and selection increase activation work by about 32%.
Longer HVX tasks increase pair-slot and HMX ready waits, so Gate/Up and complete
Host wall regress.
_Avoid_: smaller table implies faster kernel, gather count as the sole latency
proxy, accepted optimization, reopening without lower total HVX latency

**Selected-Baseline Shape and KV-Cache Characterization**:
EXP-0147's completed comparison of selected F16F16 EXP-0109, W4F16 EXP-0140
and W4U8 EXP-0144 across prefill `M={16,32,64,128}` and decode `M=1` with
`past={64,256,1024,4096}`.  It preserves the selected M64 kernel identities and
adds only shape mechanics plus an explicit persistent cache boundary.  All 24
formal cells pass reproducibility and the physical contract; the aggregate
local gate remains failed because several large-shape W4F16 independent audits
reach one FP16 ULP (`max_abs=0.125`) above the immutable 0.0625 threshold.
_Avoid_: new baseline, recipe retuning, proof of teacher-level quantization
accuracy, or describing formal device reproducibility as an independent
mathematical proof

**Persistent KV-Cache Native Boundary**:
The declared K/V state that survives one block invocation and may reside in
DDR.  Its traffic is reported separately from forbidden intermediate DDR.
EXP-0147 uses a logical head-major representation for characterization, but
shows that serial export from HMX-native K and grouped-HMX V is especially
expensive for W4F16.  A future cache-native representation must be directly
consumable by generalized Attention rather than repeatedly repacked.
_Avoid_: relabeling scratch as cache, assuming half-sized U8 cache traffic is
sufficient for speedup, or treating the current head-major carrier as fixed

**Generalized Attention Cliff**:
The EXP-0147 transition outside the selected 64-token Attention schedule.  At
W4U8 M128, generalized Attention consumes about 90.5% of block wall time; in
decode its share grows from about 88.8% at L64 to 99.8% at L4096.  The cliff is
caused by cache-carrier conversion, repeated scans, synchronization and a
padded 64-row HMX query carrier with one useful decode row, not by projection
math or KV-cache byte volume alone.  The next approved hypothesis has not yet
been registered: first compare cache-native M1 HVX dot-product and padded-HMX
paths with online tile-wise Softmax, then reuse the winning substrate for M128.
_Avoid_: projection-tail optimization as the immediate priority, M64 schedule
retuning, full score/probability DDR tensors, or mixing quantization-fidelity
repair into the physical generalization experiment

**Real Layer Replay**:
A single-layer execution driven in token order by hidden states and position
metadata captured from a declared full-model teacher; the tested layer still
computes its own K/V, cache updates, Attention and block output.
EXP-0148 completed this gate for layer 14 over positions 0-71 with all three
recipes, a versioned 28-layer state ABI and cache lengths 0→64→72.  Its W4U8
decode result exposes a cache-carrier packing cliff rather than a failure of
state ownership or intermediate residency.
_Avoid_: synthetic cache snapshot, imported runtime K/V, autonomous text
generation, or feeding one layer's output back as its next-token input

**Prepared Decode Session**:
The persistent runtime state that owns versioned per-layer cache descriptors,
valid lengths and storage across one prefill followed by consecutive decode
steps.
_Avoid_: one frozen timing snapshot, cache files reloaded for every token, or
Host-owned intermediate state

**Cache-Native Attention**:
Attention whose persistent K/V representation is directly consumable by its
QK and AV paths without reconstructing a separate logical cache carrier on
every step.
_Avoid_: cache-aware Attention, head-major cache as a required format, or zero
intermediate DDR alone

**Full-Stack Cache-Native Scaling**:
Evidence that Cache-Native Attention retains its local benefit when the same
persistent state ABI is composed across every transformer layer.
_Avoid_: single-layer extrapolation, Softmax speedup, or baseline promotion

**Consecutive-Layer Vertical Slice**:
Two to three adjacent Qwen3 transformer layers executed as one DSP unit with
independent per-layer caches and no Host boundary for their intermediate hidden
states.
EXP-0149 fixes this slice to layers 13, 14 and 15.  The teacher supplies only
the layer-13 input; both internal hidden handoffs remain VTCM-resident, while
the three cache descriptors persist independently in the existing 28-layer ABI.
_Avoid_: duplicated isolated block calls, full transformer stack, or layer
outputs replayed from files between the selected layers

**Full Transformer Stack**:
All Qwen3 transformer layers executed by the project runtime after the real
single-layer replay and Consecutive-Layer Vertical Slice gates pass.
_Avoid_: tokenizer, sampler, general graph compiler, or a collection of
Host-separated block invocations

**Single-Buffer rpcmem_alloc2 Mapping Probe**:
The EXP-0151 capacity Execution Unit that allocates one exactly 2,900,000,000
byte uncached Host buffer with `rpcmem_alloc2`, maps that same file descriptor
into cDSP with `fastrpc_mmap`, obtains the DSP virtual address with
`HAP_mmap_get`, and validates Host/DSP sentinel exchange near the beginning,
middle and end of the mapping before releasing it. It establishes only whether
the current device and FastRPC stack can represent one full-stack-sized shared
arena; it does not prove that a generated 28-layer package fits, remains
resident under workload pressure, or executes correctly.
_Avoid_: segmented mapping, delayed mapping, successful full-stack package,
performance baseline, Host refill

EXP-0151 passed on PJZ110 without a reboot. The runtime-resolved symbol,
allocation, shared fd, `fastrpc_mmap`, DSP `HAP_mmap_get`, aligned
begin/middle/end bidirectional sentinel exchange, `HAP_mmap_put`, Host unmap and
release all succeeded. This removes the legacy signed-int allocation limit as
the immediate full-stack blocker, but a separately registered experiment must
still prove the generated package size, runtime residency, correctness and
profiling under the real 28-layer workload.

**Single-Arena Full-Stack Resume**:
The EXP-0152 continuation of the 28-layer integration contract after EXP-0151.
Each recipe owns one exact-size `rpcmem_alloc2` package arena, one shared fd and
one persistent cDSP mapping for a Prepared Runtime Session. Layer descriptors
retain 32-bit offsets within that arena. No Host refill, segmented mapping,
per-layer FastRPC call, delayed mapping, new recipe kernel or new numerical
contract is permitted. Static exact-size audit and real package mapping precede
all model execution.
_Avoid_: reopening EXP-0150 in place, mapping probe as full-model correctness,
multi-arena weights, package streaming

**Full-Stack FP16 Composition Gate**:
The user-approved EXP-0152 correctness rule for comparing F16F16 and W4F16
hardware execution with their independent software implementations across 28
layers. Every conditional layer output and conditional FP16 cache comparison must contain
no non-finite values, retain cosine of at least 0.99999, and place at most one
percent of elements outside the legacy mixed tolerance `0.0625 +
0.002*abs(reference)`. The final composed output must additionally retain
cosine of at least 0.99999 and NRMSE of at most 0.003. The legacy elementwise
violation count remains visible as a diagnostic rather than being erased. A
cache receives the one-percent local rule only when its software reference
consumes the same actual incoming hidden state. A formal full-stack cache
comparison includes upstream composed drift and therefore requires valid
structure, byte-stable old prefixes and no non-finite values; its cosine,
NRMSE and legacy mixed-bound fraction remain diagnostic. Do not invent an
additional composed-cache cosine gate beyond the user-approved scheme 1.
W4U8 remains governed by its zero-LSB implementation-reference gate.
_Avoid_: exact FP16 byte agreement, teacher-accuracy gate, hiding legacy
violations, applying the relaxed FP16 rule to W4U8

**Full-Stack Hidden-Trajectory Diagnostic**:
The isolated EXP-0152 mode that copies each layer's final residual from VTCM
into one explicit 28-layer DDR capture solely to distinguish conditional local
error from composed drift. Its 7,340,032-byte diagnostic write is separately
reported, and the mode is never formal zero-intermediate-DDR or performance
evidence. The ordinary RUN ABI keeps the capture offset and byte count at zero.
_Avoid_: formal runtime, hidden intermediate DDR, performance result, replacing
the ordinary zero-DDR physical gate

**Completed Full Transformer Stack Integration**:
EXP-0152's completed execution of real Qwen3 transformer layers 0-27 for one
M64 prefill followed by eight continuous decode steps, using one exact-size
`rpcmem_alloc2` arena and one FastRPC call per stack step. It ends at layer 27
before final model RMSNorm or LM head and therefore is not a complete text
generation model. Both FP16 recipes pass the user-approved scheme-1 local and
composed-output gates. The W4U8 implementation matches an independent integer
reference exactly for all nine outputs and all 56 final K/V caches. Its prefill
preserves the low-bit projection benefit, while its decode is dominated by
repacking each persistent U8 K/V cache into HMX carriers on every token.
_Avoid_: full tokenizer-to-sampler model, teacher-accuracy proof, promoted
baseline, cache-native U8 Attention

**Prefill HMX Carrier Reuse**:
The EXP-0157 W4U8 cache-initialization path that retains the HMX-native K and V
carriers already constructed for prefill QK and AV, copies their populated
tiles with aligned HVX vectors, initializes only the padded capacity tail and
then performs the declared persistent-cache DMA.  It eliminates a second full
carrier construction while keeping the versioned persistent cache ABI and
decode consumer unchanged.  This is a cache-lifetime and handoff optimization,
not a change to Attention mathematics, quantization parameters or the HMX
compute workload.
_Avoid_: row-major cache, duplicate prefill pack, zero-copy claim, decode
speedup, baseline promotion

**Selected Full-Stack W4U8 Cache-Native Baseline**:
The user-promoted EXP-0162 implementation for the real 28-layer M64 prefill and
continuous-decode execution scope. It extends the EXP-0160 two-slot bounded
Attention consumer with immutable 32-token HMX-native K/V segments and one
mutable tail whose sealed/tail state is derived from runtime valid length.
Its formal M64 prefill is 43.596 ms; over positions 64-103 it averages 49.715
ms/token, and after the first seal it averages 49.325 ms/token. All 41 outputs
and all 28-layer cache lifecycle references are byte exact. It is
separate from both the frozen Public Common Baseline and the single-block
recipe-fastest W4U8 baseline because those have different execution scopes.
_Avoid_: complete text-generation model, common three-recipe cache contract,
teacher-accuracy baseline, replacement for the single-block baseline,
EXP-0157, EXP-0159 or EXP-0160 as the current full-stack W4U8 baseline

**Selected Full-Stack A16 Cache-Native Baselines**:
The user-promoted EXP-0158 F16F16 and W4F16 implementations for the real
28-layer M64 prefill and continuous-decode execution scope. Both apply one
cache-lifetime principle: persist the exact M64 prefill QK/AV HMX carriers,
append each decode token to a bounded contiguous journal, and patch only the
journal tail into the padded carrier in VTCM with HVX. They remove duplicate
prefill carrier conversion while keeping QK, Softmax, AV, projections and
model values unchanged. Ten rotated formal runs pass correctness, physical,
structural, fairness and strict prefill/decode speed gates. These two selected
full-stack baselines are separate from the frozen Public Common Baseline and
the single-block recipe-fastest baselines because those have different
execution scopes.
_Avoid_: row-major cache, full-prefix decode pack, scalar tail patch, changed
Attention arithmetic, replacement for a single-block baseline, arbitrary
cache length beyond the bounded M64-plus-eight contract

**W4U8 Compact Delta Journal Cache Contract**:
The accepted EXP-0159 baseline contract that keeps the selected W4U8 prefill HMX
base carriers immutable after initialization, appends each decode token's
quantized K/V rows to one contiguous per-layer journal, and patches only the
bounded tail into HMX K/V tiles in VTCM immediately before QK and AV. It tests
and proves that the A16 journal principle removes W4U8's per-token DDR tile
read-modify-write, halves cache DMA descriptors and reduces cache-write bytes
from 1,892,352 to 57,344 per full-stack decode token without changing
Attention mathematics, qparams, projection schedules or the accepted prefill
carrier-reuse path. The candidate is 2.979% faster end to end, while some saved
write-side time reappears as VTCM tail-reconstruction work inside Attention.
_Avoid_: row-major cache, complete-prefix repack, DDR HMX tile read-modify-write,
changed Softmax approximation, changed projection schedule, A16 modification,
automatic extrapolation beyond positions 64-71

**Group-Pipelined Delta-Tail Reconstruction Baseline**:
The accepted EXP-0160 W4U8-only implementation that retains the accepted EXP-0159
persistent cache ABI and arithmetic while changing only its decode Attention
consumer. It tests direct compact-base DMA into final HMX locations, clearing
only the unpopulated tail, and a two-slot GQA group pipeline so DMA/HVX work for
group g+1 overlaps HMX work for group g. The target is to recover the 2.859 ms
Attention tax observed when EXP-0159 moved cache work from persistent DDR
read-modify-write to VTCM reconstruction. Ten rotated same-binary replays show
QK-Softmax-AV improving from 10.129 to 8.962 ms and complete decode improving
from 50.293 to 49.042 ms/token, with exact outputs/cache, unchanged HMX work,
8 MiB VTCM and zero intermediate DDR/spill. It is now the selected full-stack
W4U8 cache-native baseline.
_Avoid_: changed cache contents, qparams, Softmax approximation, HMX command
count, projection scheduling, A16 changes, intermediate DDR, claimed overlap
without complete Host-wall improvement

**Selected W4U8 Long-KV Segmented-Cache Baseline**:
The user-promoted EXP-0161 layer-14 baseline. It stores immutable 32-token
HMX-native K/V segments plus one compact mutable tail and consumes long context
with a fixed-size Attention overlay instead of reconstructing a total-length
carrier. Its candidate is byte exact at L64, L256, L1024 and L4096 and reduces
L4096 wall time from 27.280 ms to 9.291 ms. It remains a synthetic snapshot
baseline and is distinct from the Selected Full-Stack W4U8 baseline.
_Avoid_: continuous-decode claim, full-stack baseline, imported-cache
real-replay claim, total-length VTCM carrier, A16 modification

**Dynamic Segmented Cache Lifecycle**:
The accepted EXP-0162 predecessor to the current full-stack W4U8 cache baseline. One persistent per-layer W4U8 cache
derives its sealed-segment count and active-tail length from runtime valid
length, appends each token once, seals a complete 32-token tail exactly once,
and subsequently treats that segment as immutable. The first test runs one
real M64 prefill and forty continuous full-stack decode steps so all 28 layers
cross the 96-token sealing boundary. It compares against an equal-capacity
EXP-0160-compatible monolithic delta control and does not change model math.
Ten rotated sessions pass exact correctness and the physical contract; post-seal
decode improves 7.500% and the complete 40-token decode average improves
1.261%. The user promoted it on 2026-09-03, superseding EXP-0160 for this
full-stack scope; EXP-0163 later superseded it after six-seal validation.
EXP-0161 remains the separate synthetic long-KV snapshot baseline.
_Avoid_: capacity-derived fixed sealed count, per-snapshot cache package,
repacking a sealed prefix, mutable sealed segment, automatic baseline promotion

**Selected Multi-Seal Full-Stack Cache Baseline**:
The accepted EXP-0163 W4U8 baseline that extends the
Dynamic Segmented Cache Lifecycle from one seal to six. One M64 prefill and
192 continuous decode steps reach valid length 256 while publishing 28 layer
segments at each of positions 95, 127, 159, 191, 223 and 255. Ten rotated
sessions are byte exact and preserve the 8 MiB VTCM, zero-intermediate-DDR,
one-FastRPC and no-QNN contracts. Against the equal-capacity monolithic delta
control, complete decode improves 16.598% and positions 224-255 improve
30.273%; the first seal's one-step cost is amortized by later reads. The user
promoted it on 2026-09-03, superseding EXP-0162 for the full-stack scope.
_Avoid_: fixed imported snapshot, unbounded VTCM growth,
full-prefix reconstruction, changed Attention arithmetic, full-model accuracy claim

**Completed Deterministic W4F16 Token-Generation Boundary**:
The completed EXP-0164 standalone path from real token IDs through embedding,
Qwen3 layers 0-27 with persistent cache, final RMSNorm, a streamed
per-output-channel W4-to-FP16 HMX LM head, greedy argmax and selected-token
feedback. Ten independent sessions each run one M64 prefill and fifteen decode
calls. All 160 selected tokens exactly match an independent W4F16 reference and
all sessions decode to the same readable prefix,
`低比特量化（Low-Bitwidth Quantization）在大模型推理中`. The timed path returns
only the selected token and never materializes full logits in DDR. Exact 8 MiB
VTCM, one FastRPC call per pass, no QNN, zero inter-layer hidden DDR and zero
spill/fill are preserved. This closes implementation correctness and semantic
usability for W4F16, but it is not a broad model-quality claim or a speed
baseline: the LM head is correctness-first and no earlier evidence has the same
token-to-token boundary. W4U8 may now be considered only after explicit user
adoption of this result.
_Avoid_: offline hidden injection described as inference, claiming BF16 token
identity, timed full-logit DDR materialization, stochastic sampling, broad
quality proof from one prompt, or automatic baseline promotion

**Completed W4F16 Prefill Generation-Boundary Candidate**:
The evidence-valid EXP-0165 candidate that keeps EXP-0164's transformer,
persistent cache, model values, prompt, tokenizer and generated sequence fixed,
while optimizing only the newly added token-to-token boundary. It uses an HVX
FP16 maximum reduction with stable first-lane tie resolution and a phase-overlaid
batch-eight streamed W4 LM head. The overlay reuses VTCM regions whose
transformer lifetime has ended; it neither increases the 8 MiB request nor
materializes logits in DDR. Ten rotated formal pairs reduce complete M64
prefill from 82.847 to 65.670 ms and the LM-head-plus-argmax interval from
26.484 to 9.397 ms. All 160 tokens and selected FP16 logit bits match the
control and independent W4F16 reference. This is a passing candidate awaiting
user adoption, not an automatically promoted baseline.
_Avoid_: changed transformer or cache, changed W4 values or scales, full-logit
DDR output, attributing the gain to reduced model bytes, broad accuracy claim,
or baseline promotion without user approval
