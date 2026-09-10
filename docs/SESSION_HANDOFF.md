# L32-0005 verified, propagation pending

No hardware/background jobs. Active owner transfers to rotation for common-code
propagation only. No-rotation source 42a783a6a8ba1b1d7dc145abf1c0535af36480c5 clean/pushed; native profiled
be389d2, profiling source a6f5e61. Single/3/16 A8 exact and W4 byte-identical,
padding/softmax audit and generation8/full16 passed. Fixed10 rotated three arms,
240 additive ledgers checked. A8 prefill1431.835/decode39.55584 vs matched W4
1260.21580/23.59130tok/s. Both10% gates pass. Full details in source
LLAMA32_A8_RELATIVE_SPEED.md and results l32-0005/PROFILE.md.
Next preflight rotation, cherry-pick ee8d122..42a783a common commits, verify only
config/branch.json differs; seal evidence and both heads. No weights/PPL/quality
promotion, no Llama rotation support. Preserve all failed attempts.
