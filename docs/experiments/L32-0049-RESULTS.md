# L32-0049 Llama3B W16A16 segmented resident implementation

Fresh verified original BF16 checkpoint converted to FP16. Seven 768 MiB backbone shards (four layers each), resident FP16 embedding and full tied vocabulary head. Single FastRPC session; host delayed FD registration and DSP HAP_mmap2/HAP_munmap2 within each token-level RPC. No disk/host weight copying during inference, no multi-session or CPU fallback. Mapping and unmapping are included in the full DSP/Host ledger.

Existing generic FP16 HMX/HVX, basic double-buffered DMA and Gate/Up producer/consumer pipeline. Capacity adaptation reuses dead O/Gate and norm/Down storage, allocates the existing 16-slot FP16 ring rather than full intermediate arrays, and removes unused compressed-weight storage. A real inherited FP16 producer addressing omission was repaired: producer now uses the same modulo ring slots as its SwiGLU consumer. Failed arena and numeric attempts remain archived. No shape-specific performance search.

Independent selected layers0/13/27 and chain3 pass unchanged composition_v2. Segmented versus monolithic outputs/KV are byte-exact across20 files. Untimed28 single-layer monolithic RPCs compose to the same full-stack M64+1 final hidden values and all56 prefill K/V tensors. This crosscheck establishes the transport/composition contract for these steps; it is not a 43-step bit-exact software oracle. Existing1B W4A16 and3B A8/SP2 regressions pass byte-exact checks.

Full43-step mathematical FP16 alignment remains FAILED: maximum hidden/norm NRMSE 0.004798236 versus unchanged0.003; prefill KV minimum cosine 0.999990956, no mixed-tolerance violations. Preserve this failure separately from implementation checks and speed. No PPL/model-quality acceptance and no automatic baseline promotion.

Five short and ten formal repeat10 processes; one repeat1 auxiliary. Fixed common prompt/input trajectory M64+42/cache128. Complete Host wall includes embedding/all28layers/finalnorm/full FP16 head/greedy/FastRPC and mapping; cold model load/tokenizer/audit writes excluded. Project-owned workload, not a named-dataset measurement.

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| Llama3B W16A16 | 409.770413 | 6.785250 |

Mapping timing is nested in layer bookkeeping/runtime teardown; do not add it again to the module sum.
Prefill map+unmap: 723.701 us. Decode map+unmap: 699.055 us.


Additional independent cache-layout check: full43 steps using row-major cache reconstruction reproduce all264192 hidden/norm FP16 elements and all43 selected IDs/logit bits from the native-cache production path exactly. This checks long-KV layout correctness, not mathematical FP16 or PPL acceptance.
