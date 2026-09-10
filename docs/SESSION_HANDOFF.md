# L32-0005: correctness passed through3 layers

Active owner no-rotation, source a6f5e61; native latest be389d2. All pushed/clean.
Original model/weights/qparams unchanged. L32-0004 candidate16 archived as baseline.
A8 stack1-a07 exactpass after native tile gather mask correction; a06 rejected
incorrect vsetq endpoint mask, preserved. Explicit rotated byte mask fixes it.
CPU SDK libnative layout probe passed original mask but device failed: use device
arithmetic gate as authority. Do not infer device correctness from libnative alone.
Single-layer a8-stack1-audit-a02 exactpass with common/SwiGLU padding poison;
8 HVX softmax calls,0 mismatches to scalar;1 common and256 SwiGLU poison events.
Audit a01 rejected by CLI because generic numerical audit incompatible with replay;
fixed scoped padding-triggered comparison, no generic gate weakening.
Stack3-a01 output/KV exactpass. Build16-a01 in progress.
Next16-layer A8 exact and W4 regression, deploy/gate/long/formal using new
run_llama32_a8_relative_profile.py. Fixed10 rotated3-arm M64+7, no optional stop.
No new PPL, A8 still unusable text/no quality gate; W4 prior PPL gate remainsfailed.
After report, propagate common code under owner transfer and seal both heads/ledger.
