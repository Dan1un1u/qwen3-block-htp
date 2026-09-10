# Current Llama context

Authority: read contract,status,context,index after bootstrap. Qwen3 frozen.
L32-0001 W16A16 complete; L32-0002 W4A16 functional complete, quality failed.
W4A16 closure: no-rotation990bf61 and rotatione91ff0f shared all common code.
Current L32-0003 source owner/head is in PROJECT_STATUS.yaml; A8 changes have
not yet propagated to rotation.

W4A16 original-derived C64 signed per-channel GPTQ weights, one FP32 output
scale, no groups. Original tensors read-only. Layer0/7/15 prefill/decode,
continuous3/16 pass; text to EOS8tokens exact: The capital of France is Paris.
16-token forced-after-EOS replay remains failed at step12. FP16 full-context
predicts device token There, while cached HF generation predicted I. Never
promote failed replay. Independent2048-target PPL device31.039101/BF16
26.697999 (+16.26%); EN+15.84%,ZH+16.69%. Quantized software31.027335, so main
loss is quantization. Original overall5%/language10% criteria failed.
No baseline promotion. Functional speedM64+7 is437.3534/6.3121tok/s auxiliary.

W4 K8192 runtime fixes: DMA2 (GateUp8), exactly dividing64-K-tile Down regions,
phase-disjoint O/Gate and norm/Down reuse; full M64 per-head scores retained;
LM-head ping-pong slots distinct and histogram uses free Up tail. Peak8330752B.
All failed attempts retained; latest successful frontend device-frontend-a03.
Evidence results/llama32-htp/l32-0002/validation_summary.json.
Models root /mnt/d/llm_exp/models/llama32-htp/l32-0002: quant-a01, layers-a02,
stack3-a01,stack16-a01,frontend-a01. Data/results l32-0002/data: fresh Llama
calibration65536tokens, heldout2048targets128documents, EN/ZH Wikipedia,
source/window/32gram disjoint audit passed. Public raw corpus cache reuse only.

Active L32-0003 owns no-rotation for W4A8 OFF chain, optimization deferred.
A8 model-quality threshold explicitly absent; arithmetic/physical gates retained.
W4A8 OFF now passes exact1/3/16-layer prefill+decode output and KV checks.
NativeW4 all7 projections/layer, GQA4/head64/noQKnorm/RoPE-before-KV, FFN8192
bundle counts and GQA4 decode scratch fixed. Exact8MiB acquired, peak7668960B,
zero timed intermediate DDR/spill, no W4->S8 weight expansion.
Fresh65536-token Llama calibration-a01 static minmax; no heldout fitting.
Results l32-0003/device-stack1-a05,device-stack3-a01,device-stack16-a01.
A8 frontend-a01 independent oracle16 IDs all42845 (Sleep), selected U8 codes
[157,158,156,158,156,156,156,157,156,156,158,156,156,158,155,156].
Device-frontend-a02 matches all IDs AND codes; independent2048-target PPL is
currently running in its runner. Auxiliary217.3520 prefill/2.57149 decode tok/s
(M64+15); optimization deferred. R3/R4 and native segmented Llama KV remain
unsupported and explicitly rejected. W4A16 full16 regression passes unchanged.
NLL arithmetic spot oracle recorded for EN id256 and ZH id576 first target.
Original /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin, six hashes in
tools/llama_reference.py. Toolchain/device unchanged, see contract and docs.
