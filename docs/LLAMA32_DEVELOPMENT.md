# Current Llama status (L32-0001)

Llama-3.2-1B-Instruct W16A16 functional adaptation is validated. See
LLAMA32_W16A16.md and models/llama32/validation.json for exact evidence and limits.
The organization-time notes below describe the starting boundary, not current
implementation status. Next: a separately registered per-channel W4A16 port,
using fresh Llama calibration and a larger independent PPL acceptance set.

# Llama 3.2 development boundary

Organization EXP-0266 freezes Qwen3 and prepares two development branches. It
does not port the model or run another Qwen3 experiment.

## Source and memory

- Frozen Qwen3: `/home/daniuniu/work/qwen3-block-htp`, original EXP-0265 branch.
- No rotation: `/home/daniuniu/work/llama32-htp`, `codex/llama32-no-rotation`.
- Rotation: `/home/daniuniu/work/llama32-htp-rotation`, `codex/llama32-rotation`.
- Llama authority: `/home/daniuniu/work/llama32-htp-project-memory`,
  `codex/llama32-project-memory`, separate from the Qwen3 authority.

Both source worktrees share the original Git object store and common runtime
history. New build directories belong to their respective WSL worktrees. New
Llama artifacts use `/mnt/d/llm_exp/models/llama32-htp` and
`/mnt/d/llm_exp/results/llama32-htp`, separated by recipe and experiment. No large
payloads are copied, removed or committed during organization.

## Model boundary to implement next

The current native ABI still contains Qwen-specific compile-time dimensions,
normalization, RoPE, vocabulary, token-generation modes and file conventions.
The new model JSON records this boundary; it is not a general graph runtime.
Select the exact Llama checkpoint/revision before filling dimensions or choosing
its tokenizer. Derive model behavior from that checkpoint and verify against an
independent floating implementation. Preserve tested HMX packing/DMA/worker
primitives; adapt model semantics explicitly.

Start with W16A16 layer correctness, then consecutive layers and complete
embedding/transformer/final norm/head/greedy text. Bring W4A16 per-output-channel
weights into the same boundary next, and only then A8. Quantization exports
must start from original Llama tensors, with Llama-specific calibration and
fresh prefix/cache and rotation artifacts. The selected Qwen3 C64 quantizer
policy is a starting method, not a reusable weight payload. Rotation matrices
and R4 factor shapes must be derived for the selected Llama dimensions.

## Baseline and measurement rules

Keep source commit, runtime build, package manifest, arithmetic/rotation flags
and actual measured scope together. Migration selection does not retroactively
promote Qwen3 Selected records or clear failed ideal-R3/quality gates. R4 weights
are a fresh RTN speed fixture, with no R4-specific calibration or PPL acceptance.
Preserve repeat1 as auxiliary; formal paired repeat10 Host wall controls the
10% slowdown decision, both phases with CI upper <=1.10. Quality remains PPL
primary and short answers auxiliary. New models need independent measurements;
Qwen3 throughput is historical provenance only.

Use `tools/recipe.py` for inspection. Existing `scripts/device_exp*.py` and
exporters refer to frozen Qwen3 paths and experiment locks. Do not run them as
Llama tools or change them to silently point at a different model.
