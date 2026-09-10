# L32-0003 closed: quantized functional chains

No active experiment or background device workload. Bootstrap either registered
Llama worktree, then read authority files. Both branches are clean, synchronized,
and share all common code; config/branch.json is the sole difference. Heads are
in status. Qwen3 frozen branch and original model inputs remain unchanged.

L32-0002 W4A16 functional pass, text usable in tested prompt, PPL31.039101 vs
BF16 teacher26.697999 (+16.26%): retained quality gate FAILED. L32-0003 W4A8 OFF
functional pass with exact1/3/16-layer and16-step greedy oracle; text repeats
Sleep, PPL1206603.740108. A8 has no model-quality threshold. R3/R4 unsupported.
Do not rerun completed gates or confuse functional pass with quality acceptance.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0003/validation_summary.json,
80-file evidence_sha256.json (hash and copy in authority). W4A16 sealed L32-0002.
No baseline promotion or formal profiling. User deferred optimization. Report
results and discuss W4A16 quality gap before any new experiment (next L32-0004).
