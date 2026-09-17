# L32-0041: Llama 3.2 3B W4A8-SP2 adaptation completed

Fresh original-derived3B GPTQ per-output-channel signedW4, staticA8 calibration, fused SP2 mode8, FP32 residual, no R3/R4. Original BF16 shards and tokenizer are read-only and hash-pinned.28 layers/hidden3072/FFN8192/head128/GQA3. No1B weights/scales reused; frozen tokenizer-identical calibration token corpus reused.

## Validation

- Selected source layers0/13/27: prefill/decode hidden and KV exactly match independent oracle.
- Consecutive3 and full28: same exact gates. Full28 verifies199680 FP32 output values acrossM64+M1, plus all KV caches.
- Full embedding→28blocks→final norm→W4 head→greedy: all16 IDs and selected logit codes exactly match own independent software contract.
- Five short and ten formal rounds, repeat10; all1600 formal token-boundary ledgers reconcile; no timed intermediate DDR spill;28 blocks per invocation, oneRPC/token boundary.
- Fixed8MiB VTCM; observed peak8360416bytes.
- Original1B frozen layer7 prefill/decode bounded regression passes exact hidden/KV; no baseline or quality promotion.

## E2E

M64+15decode, KV capacity80. Complete Host wall includes embedding/all28blocks/final norm/head/greedy/FastRPC; excludes cold loading/session setup/external tokenizer. Fixed trajectory from independent greedy oracle, with a separate native-greedy check. Each formal round contains10 trajectories; report reciprocal of mean Host wall.

| Model/config | Prefill token/s | Decode token/s |
|---|---:|---:|
| Llama-3.2-3B-Instruct W4A8-SP2 + FP32 residual, no rotations | 512.14 | 18.64 |

Full additive module table: MODULES.md. Cross-size historical1B timings are not paired comparisons. No10percent cross-size gate applies. No memory-peak result or memory optimization claim.

Example generated text: `The capital of France is Paris.assistant  That's correct!`. The fixed16-token measurement continues after EOS. The answer before the first EOS is correct for this prompt, but noPPL or general text-quality acceptance is claimed.

## Repairs / preserved unsuccessful attempts

The initial copied3B arena did not fit8MiB. Removed unused legacy MLP packed copies (shared32-bit offsets would otherwise exceed4GiB), reused dead VTCM slots and QKV activation storage for SP2. Only the guarded native direct-W4 recipe is supported.

The initial head64-specific K native fallback overgrew the stack forhead128. Reused the existing full-head128 vector packing. Both prefill and decode require odd GQA head tails; dynamic K scatter must use the full-vector predicate. Expanded FP32 Norm worker stride must include reduction/repair slots; a missing512-byte tail caused inter-worker overlap. Finally, an unaligned SP2 plane reproducibly caused a HMX exception as a stream crossed4MiB;64KiB plane alignment fixed it within the same8MiB arena. This is an observed placement constraint, not an assertion about undocumented HMX internals.

Failed runs/builds remain in the result tree. No gate was relaxed. No R3/R4, W16A16/W4A16 adaptation, PPL or new accuracy method was included.

Measured binary source: 6f911883fa98a426a5d8482bbaeceebc6a2d7d40. Source closure: c990326d949d56778cf5990d5b4b246f1a247710. Model artifacts: MODEL_ARTIFACTS.json. Reproduction entry: docs/LLAMA32_3B_PORT.md and tools/execute_llama32_3b.py.
