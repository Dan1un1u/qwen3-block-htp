# L32-0016 completed: FP32 residual implemented, quality still unusable
Read docs/LLAMA32_FP32_RESIDUAL.md. No active experiment.
Native source e3cf74916aebbc9254615d68b1c4c135b43fee03, clean/pushed.
No rotations;0012 mode8 weights/SP2/alpha/A8scales/head frozen. OriginalFP16
embedding; raw nativeW4 O/Down->FP32 residual;FP32Norm->A8. Layer0/7/15,
3layer,full16 replay and full16-token generation arithmetic exact. VTCM8098272,
no intermediateDDR/spill; explicit FP32 rounding fix, prior failures retained.
Matched128-doc/2048-target devicePPL: BF16 26.6980, original0012SP2 130712.2867,
FP32 1766.7649 (EN645.9578,ZH4832.2942). Text still unusable. No performance
acceptance; original0012 speed conclusion remains historical frozen baseline.
Next discussion: train-only recalibration on newFP32 dataflow;FP16storage and
R3/R4/performance separately. Do not autoresume deferred L32-0015 rotations.
Evidence/results llama32-htp/l32-0016; ledger SHA256
69b591d67d1ce3f45a0e2cc5e467f7b2d90ccddff854a581d79637769ec77de8.
