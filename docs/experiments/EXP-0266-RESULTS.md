# EXP0266 repository organization closure

Qwen3 frozen at 48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5. Both new Llama branches and their worktrees exist, committed and synchronized. Only config/branch.json differs between branches. 50 native/build files remain byte-identical to the Qwen3 freeze, and existing experiment runner source paths are retained. Original README preserved verbatim.

No-rotation source: e0e3bc77704a9369693fe48a5cc5fed3005ac36c. Rotation source: cd2360f31c1b1a151db7ecf30acfbe73bded9979. No-rotation default is W4A8 OFF; rotation default is dense R3 OPT2 with R3+R4 OPT6 optional. Both keep W16A16 and per-channel C64 W4A16 OPT2.

Five historical baseline reports/build provenance/package manifests and launch references verified against authority. No large weight payload rehash, creation or deletion. 16 model/recipe plans checked across both branches, with incompatible rotation/model and corrupted R3-flag rejection checks. Python AST and shell syntax pass. Independent Llama bootstrap succeeds on both owning worktrees; idle and wrong-owner stateful preflight reject as expected.

Separate Llama memory at /home/daniuniu/work/llama32-htp-project-memory, branch codex/llama32-project-memory. New model checkpoint and native adapter remain unimplemented. No Qwen or Llama hardware run, performance measurement or PPL in this organization task: profiling fields and E2E are N/A. Historical Qwen speeds are retained in baselines/qwen3-frozen/manifest.json, with protocol/denominator distinctions; they are not Llama measurements.

Runtime build was unnecessary because all 50 native/build files are unchanged. Old tools remain cataloged, not relocated, preserving imports and sealed evidence references. No baseline-quality promotion or failed-gate change.

Evidence ledger SHA256: 052b816c686b11b0bc57908b9f86147047c1ecc427af7d8ee54a985dc5e2c093
