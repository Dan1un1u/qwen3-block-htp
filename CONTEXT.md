# L32-0018 active: speed optimization of FP32 Norm/O
User authorizes continuing L32-0017. Read docs/experiments/L32-0018.md.
Starting source59348e01bb10fc38c7974550345377b5449ac9c1.
L32-0017 fixed10:FP32 1639.74/42.26tok/s,control1996.44/45.73.
Prefill+21.75%fails,decode+8.21%passes. Dominant extra FP32 Norm8.21ms/O1.21ms.
Preserve exact arithmetic and physical checks,weights/scales/SP2, no rotations
or recalibration/PPL. Diagnose substage cost then optimize and fullmodel pair.
