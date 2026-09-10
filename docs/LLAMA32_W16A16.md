# L32-0001 W16A16 model adapter

Checkpoint: LLM-Research/Llama-3.2-1B-Instruct, original BF16 Transformers files.
Original input remains /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin. The six
SHA256 values are pinned in tools/llama_reference.py; rehash before export.

## Implementation and validation boundary

Build QBH_MODEL_LLAMA32=ON uses model ABI128, 16 layers, hidden2048, FFN8192,
Q32/KV8/head64, no Q/K RMSNorm, epsilon1e-5, attention scale1/8. RoPE tables use
checkpoint llama3 frequency interpolation, theta500000 and factor32. Original
BF16 tensors are numerically converted to FP16. The shared embedding tensor
provides both row-major embedding and HMX-packed LM head carriers. Native F16
RoPE currently uses a scalar DSP helper, not a host fallback. Optimized head128
GQA/QKV carriers and all A8 block launches are explicitly rejected in this build.

Stage validation: independent Transformers oracle in FP32/FP16 at positions
0/8192/32768; single layer0/7/15 prefill/decode; continuous layers0-2 then0-15;
then token embedding/final norm/head/greedy text and lightweight heldout NLL.
Current device contexts are M64 plus bounded decode. Long positions are checked
in the host oracle; full 128K device context is not implemented or validated.

Model-specific entrypoints (all require the Llama project-memory preflight):
- tools/llama_reference.py: independent host math/reference and original hashes.
- tools/export_llama32.py, tools/run_llama32_layer.py: standalone layer fixtures.
- tools/export_llama32_stack.py, tools/run_llama32_stack.py: continuous slices.
- tools/prepare_llama32_frontend.py, tools/run_llama32_frontend.py: tied-head
  full token boundary, text decoding/EOS handling and frozen NLL comparison.
- scripts/build_llama32.sh [3|16]: owning worktree build using isolated SDK6.6,
  Tools19, NDKr26c. No original Qwen experiment launcher is run for Llama.

The native RPC/CLI symbols retain qwen3 names for transport compatibility.
Raw legacy record labels remain available; protocol.json records L32-0001,
actual source/build/package hashes and complete commands. They must not be
mistaken for historical Qwen3 experimental measurements.

## Numerical gate and evidence

Single-layer Llama scan compares an independent floating reference using the
existing FP16 composition_v2 constants, NRMSE<=0.003 and cosine>=0.99999. Cache
mixed-bound fraction<=0.01 at atol0.0625/rtol0.002; unchanged prefix and unwritten
padding remain byte-exact. Byte differences to the floating reference are
retained as diagnostic counts. Continuous replay uses its existing FP16 gate.
Hardware keeps exact8MiB VTCM and zero timed intermediate DDR/spill.

The eight-document EN/ZH NLL fixture scores16 continuation tokens after64 context
tokens per document (128 scored tokens). Retained project-authored CC0 raw texts
are re-tokenized from scratch with Llama; no Qwen token IDs, cache, weights or
calibration are reused. The fixture is frozen before teacher scoring; it is a
small heldout implementation diagnostic, not a general pretrained-model quality
benchmark or sufficient standalone quantization acceptance. No fitting uses it.
Full heldout quantization acceptance remains a later protocol before W4 approval.

Generation arithmetic replay runs a fixed16 steps, including any tokens after
EOS, to compare exact token sequences. User-facing text is truncated at the first
of Llama stop IDs128001/128008/128009. Timings from a functional run are auxiliary,
not formal performance measurements or extrapolated layer throughput.

Results and retained failures: /mnt/d/llm_exp/results/llama32-htp/l32-0001.
Generated packages: /mnt/d/llm_exp/models/llama32-htp/l32-0001.
