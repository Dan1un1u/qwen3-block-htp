# L32-0005 in progress

Active owner no-rotation. Source HEAD 4cf30fe110ec417033dc5946cee31aaae69b4bb9. Prior evidence immutable.
L32-0004 candidate16 binaries archived as L32-0005 baseline-build16 with verified hashes.
A8 native per-head V LUT reuse, row4 common/SwiGLU, HVX softmax, integer RMS tree,
pooled residuals, RoPE repair early-out and native tile gather implemented.
Stack1 a02/a04/a05 exact output and KV pass. a01 remote namespace collision;
a03 CLI positional option error; build1-a02 unused gamma compile error, all retained.
Latest aligned HVX RoPE gather build1-a05 passed; stack1-a06 running.
System python invocation lacked numpy before any output/hardware; use spinquant venv.
Next: inspect a06, then padding/softmax audit,3/16 gates, full generation, fixed10
rotated three-arm M64+7 profiling. A8 has no quality gate; weights/qparams unchanged.
No full PPL or new model generation. Must propagate common source and seal both branches.
