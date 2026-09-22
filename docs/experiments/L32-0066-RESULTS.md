# L32-0066: Llama3.2-3B AV-to-O scale folding

M64 prefill +42 continuous fixed-token decode; full28layers, per-output-channel W4, uniform INT16 Down, FP32 residual and Norm, no rotation. Same binary, production scheduling, native layout and four-row vector-granule AV conversion control for one valid decode row. Only AV multiplier/zero metadata and corresponding O input scale differ. No FP nonlinear arm in this supplement. No model-quality/PPL or baseline promotion.

Five short and ten formal AB/BA paired rounds, repeat10; complete warm Host wall includes embedding, all blocks, final norm, head/greedy and FastRPC; excludes loading/tokenizer/ADB. CI:20,000 paired bootstrap draws of ten round means; all formal samples retained.

| Configuration | Prefill64 ms | Prefill token/s | Decode42 ms | Decode token/s |
|---|---:|---:|---:|---:|
| control | 53.745634 | 1190.794 | 1751.665906 | 23.977 |
| folded | 53.156032 | 1204.003 | 1746.072636 | 24.054 |

| Phase | Wall reduction | Folded/control wall ratio,95%CI |
|---|---:|---|
| prefill | 1.0970% | 0.989030 [0.987598, 0.990735] |
| decode | 0.3193% | 0.996807 [0.995780, 0.997983] |

## Correctness and limits

Both complete reference trajectories contain0 saturated AV elements out of18,235,392. Each arm matches its own independent actual-arithmetic reference at1,3 and28 layers, including1,204 full-model layer hashes and43 head token/code pairs. Both use the same four-row vector granule for one valid decode row; no deliberately inflated64-row control. Signed24 partial and signed32 reconstruction bounds pass. VTCM peak8,360,416/8,388,608 bytes; zero timed intermediate DDR/spill and exact additive ledgers across12,900 short/formal invocations (8,600 formal). Shared matrix work/transfer counters and protocol settings match.

Scale reassociation still changes FP32 rounding: final hidden relativeL2 difference=0.365847, head token changes=7/43 on fixed inputs. No cross-arm bitwise/model-quality equivalence is claimed. Zero clipping is restricted to these checked trajectories, not an all-input proof. This is a performance replication of the conditional Llama1B fold, not a repair of the rejected Qwen saturation case.

INT16 payloads are verified L32-0064 package files; original frozen L32-0058 token continuation. No re-quantization, new calibration, weight or runtime kernel change. Only experiment harness/model metadata change. Payload and binary hashes rechecked locally/device after measurement. E/workbook and previous result records remain unchanged.

## AV conversion attribution

Counters overlap with other work and are not additive critical-path shares. Decode counters below are normalized per token; prefill per64tokens.

| Phase | AV RQ control µs | AV RQ folded µs |
|---|---:|---:|
| prefill | 2474.094 | 8.401 |
| decode | 164.095 | 11.377 |

Full additive latency/share table: MODULES.md. Raw commands, logs, references and seals remain alongside.
