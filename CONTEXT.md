# L32-0017 completed: FP32 vector pipeline recovery, speed first
Read docs/LLAMA32_FP32_PIPELINE.md. No active experiment.
Source59348e01bb10fc38c7974550345377b5449ac9c1, no-rotation branch, clean/pushed.
HVX FP32 O/Down, overlapping Down epilogue,4-context exact prenorm,decode
vector broadcast and embedding conversion implemented. Fixed weights/scales/SP2.
Layer0/7/15,3layer,16layer and fullgeneration exact;320 formal boundaries pass.
VTCM8098272,no intermediateDDR/spill. No PPL or rotations.
Fixed10 paired M64+15: control1996.44/45.73tok/s,FP32 1639.74/42.26tok/s.
Prefill+21.75%fails (95%upper22.79%);decode+8.21%passes(upper8.84%).
No overall performance acceptance or baseline promotion;0012 remains frozen.
Next discuss FP32 Norm (prefill10.22ms vs2.01ms) and O;Down pipeline recovered.
User explicitly prioritizes speed: do not auto-resume calibration/PPL or rotations.
L32-0015 deferred. Evidence l32-0017,ledger
ae1572aca736152bbf59a9a5e445b979a1549d3ce25d880a78bfcef051c19dcf.
