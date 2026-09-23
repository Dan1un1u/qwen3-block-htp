# L32-0069: HVX butterfly R4 on ordinary W4A8 boundaries

**Result:** The measured vector butterfly does NOT take longer than historical complete inference. A four-worker full8192 transform costs 165.773 us per M64 layer; sixteen successive transforms take 2.64931 ms, about 9.93% of historical 26.67730 ms full16 ordinary-A8 prefill. Single-row decode takes 10.035 us per layer, 0.160664 ms for sixteen transforms, about 0.744% of historical per-step decode wall. These percentages are nonpaired cost references, not measured incremental E2E regressions.

## Contract and placement

Llama-3.2-1B, intermediate8192, normalized Sylvester H8192. Boundary is Gate/Up U8 -> pre-A8 SwiGLU FP16 values -> FP32 HVX butterfly -> fixed ordinary U8 native Down input. No SP2/INT16 Down carrier; FP32 arithmetic scratch is not an activation quantization recipe. No R3, HMX Hadamard or projection execution. This is an isolated boundary component, not integration of a rotated full-model checkpoint. Down weights were not folded because no Down projection or model-quality inference is performed.

Input values are deterministic synthetic Gate/Up U8 codes evaluated with original L32-0068 layer0 qparams into prequantization SwiGLU FP16. They are NOT captured model trajectories. Timing is data-independent; separate sentinel cases include zero, constant, impulses, alternating signs, random and extreme outliers. Numerical reference and input provenance are in FIXTURES.json and HOST_REFERENCE.json.

The full dimension is mixed after SwiGLU, not a local block rotation and not rotation of already-quantized middle A8. The original runtime R4 site is qbh_r4_prepare_tile in src/dsp/llama_dense_r4.inc. Normalized H8192 is mathematically the same Sylvester transform as H16 tensor H512; its declared FP32 butterfly rounding differs from the historical two-stage FP16 HMX implementation.

## Timings

| Phase | Workers | Single transform core (us) | Directly timed 16-transform core (ms) | Historical full16 wall (ms) | Core / historical wall |
|---|---:|---:|---:|---:|---:|
| M64 prefill | 1 | 642.443 | 10.274923 | 26.677299 | 38.516% |
| M64 prefill | 4 | 165.773 | 2.649311 | 26.677299 | 9.931% |
| M1 decode | 1 | 10.035 | 0.160664 | 21.597517 | 0.744% |

Historical reference is L32-0068 ordinary A8: prefill64 Host26.67729899ms; continuous42decode Host907.09569526ms, or21.59751655ms averaged per decode step. It includes embedding/all16blocks/finalnorm/head/greedy/FastRPC. The reference is neither same-session nor paired with this component measurement. No new E2E tokens/s is reported.

The16-transform experiment actually executes sixteen transforms successively inside each qtimer window on one VTCM tensor (with worker dispatch/join per transform); it is NOT single-layer latency multiplied by16, and NOT a real16-layer graph. Preparation happens once before and quantization once after this window. Thus its boundary costs must not be described as full-stack costs. H applied an even number of times approaches identity and is independently checked.

| Phase | FP16 -> FP32 prepare (us) | Ordinary A8 quantize + native store (us) | Single-transform core, practical workers (us) | Whole measured boundary stages (us) |
|---|---:|---:|---:|---:|
| prefill | 46.547 | 248.661 | 165.773 | 460.982 |
| decode | 0.727 | 3.885 | 10.035 | 14.648 |

Preparation and final A8/native stores are single-worker vector code in both arms; they are measured separately, not attributed to the butterfly. The no-rotation boundary control performs the same preparation/quantization, not the production fused SwiGLU LUT. Hence boundary-stage differences isolate the inserted core, but this control cannot quantify the cost relative to the optimized production LUT or lost Gate/Up/Down overlap.

## Numerical and physical validation

- 24 device audit cases: FP32 results bit-exact to independent staged FP32 reference; every live A8 byte and decode padding byte exact. Maximum relative L2 versus Float64 reference is 4.0827051087190494e-07. Independent direct Walsh dot-products check11 columns across64 rows; they agree with the reference to the declared tolerance. No model-quality or PPL claim.
- Five short and ten formal rounds;10 in-RPC repetitions per sample, one warmup RPC excluded. Alternating worker/control order. Formal80 timedRPCs, short40; each CLI has one additional untimed warmup. Core times are elapsed DSP wall, not summed worker busy time. No early stopping.
-8MiB VTCM granted, prefill peak3670016B and decode573440B. No HMX commands. Core disassembly has no calls or vector stack spills; first five butterfly stages stay in vector registers, subsequent stages use VTCM. Wrapper spills only invariant conversion constants across calls, not activation tensors. Input loading and audit publication are separately timed and excluded from core.
- Four worker prefill partitions16 independent token rows per worker; decode uses one worker to avoid per-stage synchronization. No scalar activation loop in transform.
- Battery29.0C before and after. No other model process was running at start. No device configuration changes were needed.

## Interpretation

This does not support a claim that HVX butterfly R4 is intrinsically slower than complete inference. Moving an entire Down dense projection to HVX in prior SP2 shift/add experiments and executing O(K log K) Hadamard are different workloads. The core is cheap enough that boundary preparation, quantization/native stores and loss of producer-consumer overlap may matter more. This diagnostic does not measure those integration effects or establish that online R4 is free. Further full-model integration would require folded Down weights, recalibration and an explicitly scoped paired E2E experiment.

## Reproduction and provenance

- Source: a43398ba64760da3756a006ba2dde99638873cd1; opt-in probe modes16(oneworker),17(fourworkersprefill),18(no-rotation boundary control). Production paths/defaults unchanged.
- Driver: tools/probe_llama32_hvx_r4.py; use /home/daniuniu/.cache/qwen3-block-htp-py/bin/python after project-memory registration/preflight.
- Firmware/runtime deployment and all binary hashes: DEPLOYMENT.json. Raw logs: device/audit, device/short, device/formal. Numerical outputs retained only for audits.
- Reported95% intervals bootstrap ten round means; they reflect this stable session only, not device-to-device, thermal or historical-comparison uncertainty.
- Environment recovery: initial proxy CONNECT failure resolved by bootstrap-only proxy bypass; missing system numpy resolved by existing project venv; WSLInterop binfmt registration restored before ADB deployment. No numerical/build failures or timing rounds discarded.
