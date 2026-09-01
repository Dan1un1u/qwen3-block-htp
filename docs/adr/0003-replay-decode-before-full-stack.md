---
status: accepted
---

# Establish real replay decode before expanding the full transformer stack

The user approved evolving the standalone block laboratory into a staged Qwen3
runtime: first build a full-stack-compatible Prepared Decode Session and prove
real single-layer replay prefill plus consecutive decode, then execute two to
three adjacent layers without a Host activation boundary, and only then expand
to every transformer layer. Expanding the current cache-free block first was
rejected because it would freeze per-layer RPC and DDR boundaries that decode
would later need to replace; optimizing a synthetic single-layer decode to
completion was also rejected because it could overfit a physical schedule that
does not survive multi-layer weight and cache pressure.
