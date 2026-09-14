# L32-0016 active: high-precision residual before rotations
User approves no-rotation implementation from L32-0012 SP2 mode8. Read
docs/experiments/L32-0016.md. L32-0015 rotations deferred; failed evidence retained.
Native W4/SP2/attention/KV/GateUp/LMhead contracts frozen except O/Down float output.
FP16 original embedding -> FP32 residual; FP32 add/norm -> A8 after norm only.
Do not take rotated Down weights from L32-0015 or L32-0013/14.
