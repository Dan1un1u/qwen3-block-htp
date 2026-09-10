# L32-0006 in progress

Owner no-rotation at fe28d5cbe8a2492ae4d93d1c82e6c6c1b9ff2581, clean/pushed. L32-0005 candidate16 archived as
l32-0006/baseline-build16. Round1 paired exact SOLE templates and dead-K scratch
shuffle4 softmax,3-way prefill SwiGLU passed r1-stack1-a01/a02. Round2 aligned
four-row head64 native KV transpose passed r2-stack1-a01. No weights/qparams
change. Round3 batch32 GateUp continuous+prefetch+stream (six -> model-sized
eight groups),QKV16,O16singleDMA,Down8singleDMA now building1 (build1-r3-a01).
Next inspect build and run r3-stack1; audit padding,then3/16 exact,W4 regression,
fullgeneration and frozen10 three-arm M64+7 plus10pairedM64+15 A8. All earlier
sealed results immutable. Qwen frozen manifestOFF target1705.318/48.357tok/s,
different model/historical, no matched claim. No rotation/quality promotion.
