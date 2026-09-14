# L32-0016 active: FP32 residual frontend validation
Read docs/experiments/L32-0016.md and PROJECT_STATUS.yaml.
Native explicit-rounding source e3cf749, no rotations; baseline0012 mode8
weights/SP2/alpha/A8 scales/head frozen. Layer0/7/15,3layer,full16 prefill+decode
exact; VTCM8098272,no spill. Frontend initial FMA/reference mismatch isolated;
firstprefill repaired exactly, remaining frontend-a02 oracle and fulldevice
generation/PPL pending. Do not report quality/speed acceptance.
Models/results llama32-htp/l32-0016; frontend-a01 failed evidence retained.
L32-0015 rotations deferred until residual implementation evaluated.
