# Session Handoff — EXP-0216 completed

This file is a navigation aid. Bootstrap and read the four authority files first.

Source: `/home/daniuniu/work/qwen3-block-htp` on
`codex/exp-0216-w4u8-prefill-direct-w4-qkvo` at `25b2fdfead7546a981ccb457d9106e081bb6cb34`.
EXP-0216 completed its final five-pair short gate and ten-pair formal profile.
The experiment was finalized normally after report/evidence retention; no
running experiment remains. Next available ID is EXP-0217. No new experiment
or baseline promotion was performed; the user's current task was to complete
EXP-0216 and report. Selected W4U8 decode baseline remains EXP-0211.

Final evidence: `D:/llm_exp/results/qwen3-block-htp/exp0216/20260905_formal_25b2fdf`.
Short/cache/hidden audit: `D:/llm_exp/results/qwen3-block-htp/exp0216/20260905_short_gate_25b2fdf`.
Full tables: `docs/experiments/EXP-0216-PROFILE.md`.
Artifact directory: `D:/llm_exp/models/qwen3-block-htp/exp0216/artifacts/25b2fdfead7546a981ccb457d9106e081bb6cb34`.

Prefill: 1463.150767 -> 1628.645016 tok/s,
10/10 pair wins; median paired speed +11.171191%.
Decode: 45.963969 -> 45.941593 tok/s, preserved.
All 60 audit files, 112 layer hidden hash pairs, 1930 selected-token/code pairs,
3860 invocation ledgers and 108080 layer ledgers pass. Exact 8 MiB VTCM,
zero intermediate DDR/spill, one FastRPC/pass, one HMX owner and no QNN remain.

The initial f724a22 collection lacked required generation_profile fields.
d134995 repaired those fields and passed the old script gates, but closure
review found the cache-audit label overstated four hidden-only binary files.
25b2fdf adds all 56 full prefill K/V cache exports after audit RPC and checks
all nonzero per-layer hidden hashes. Both older collections are preserved as
supplementary/incomplete evidence; do not use them as the final gate.

Only the user may promote a baseline. A later experiment must be registered
and pass preflight before stateful work. Do not repeat completed exploration.
