# Llama 3.2 3B W4A8-SP2 port (L32-0041)

The 3B build is an explicit shape specialization of the latest no-rotation 1B runtime. Default builds remain 1B. It supports only the tested native packed per-output-channel W4, static A8, SP2 mode8, FP32-residual path without R3/R4. W16A16 and W4A16 for3B are not covered by this migration.

Original checkpoint: `/mnt/d/llm_exp/models/llama3.2-3B-Instruct-origin` (read-only, two BF16 safetensor shards, tied embedding/head). Provenance and all original file hashes: `/mnt/d/llm_exp/results/llama32-htp/l32-0041/original-provenance.json`.

Shapes: 28 layers, hidden3072, FFN8192, 24 query heads, 8 KV heads, head dimension128, GQA3, vocabulary128256. RoPE uses the original llama3 scaling configuration, factor32. There is no Q/K RMSNorm.

## Build and data

Under a registered project-memory experiment and passing preflight:

```
QBH_LLAMA_MODEL_SIZE=3B bash scripts/build_llama32.sh 28
```

The build seal records source commit, model size, layer count and all host/DSP binary hashes. Staged counts1/3/28 are supported. For the preserved1B build, use `QBH_LLAMA_MODEL_SIZE=1B` and counts1/3/16.

Fresh conversion tools:

- `tools/prepare_llama32_3b.py`: original-derived per-channel GPTQ and fresh static A8 calibration; reuses only tokenizer-identical frozen calibration token IDs, never1B weights or scales.
- `tools/llama32_3b_reference.py`: exact independent selected-layer/chain references and SP2 two-plane integer bounds.
- `tools/prepare_llama32_3b_frontend.py`: original FP16 embedding, original final norm, fresh quantized LM head, all28 blocks and independent16-token greedy trajectory.
- `tools/execute_llama32_3b.py`: immutable deployment, selected/chain/full gates and complete Host-wall profiling.
- `tools/report_llama32_3b.py`: additive module accounting and formal aggregate.

Artifacts live under `/mnt/d/llm_exp/models/llama32-htp/l32-0041`; results under `/mnt/d/llm_exp/results/llama32-htp/l32-0041`. Attempts are immutable; do not overwrite packages. Existing published models/weights are untouched.

## Native changes and limits

- 128-dimensional RoPE applies the existing exact64-dimensional SIMD arithmetic to paired halves; head tile pack/unpack uses all four32-byte tiles.
- GQA3 handles odd query-head tails in both prefill and decode softmax, without reading a fourth head. A128-byte K vector uses full-vector scatter rather than the head64 partial predicate.
- FP32 Norm scratch stride includes gamma conversion, reduction and rare-boundary repair storage, with disjoint worker slots.
- 3B omits unused legacy MLP weight bundles to fit the32-bit shared-buffer offset ABI. Only the guarded direct-W4 execution contract is supported.
- VTCM reuses idle legacy compressed slots and the completed QKV activation region. The SP2 low plane is aligned to64KiB: the unaligned original3B layout caused a reproducible HMX exception when a stream crossed the4MiB address boundary; the aligned path passes independent numerical gates. This is an observed placement constraint, not a claim about undocumented HMX internals.
- No timed intermediate activation DDR spill, one HMX owner and one RPC per token boundary. Fixed8MiB VTCM request.

This work establishes implementation correctness and performance. It does not establish usable text or PPL quality for static A8; no accuracy-baseline promotion is implied. DRAM peak optimization and memory paper results remain deferred.