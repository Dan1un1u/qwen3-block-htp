# Current Llama context

Authority: read contract,status,context,index after bootstrap. Qwen3 frozen.
L32-0001 W16A16 complete; L32-0002 W4A16 functional complete, quality failed.
Source no-rotation990bf61 and rotatione91ff0f share all common code.

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
W4A8 code/export not yet implemented/validated. Fresh A8 qparams/calibration
required. R3/R4 remain unsupported Llama modes, never silently fallback.
Original /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin, six hashes in
tools/llama_reference.py. Toolchain/device unchanged, see contract and docs.
