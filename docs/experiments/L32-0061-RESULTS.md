# L32-0061: AV-to-O multiplier folding feasibility

## Outcome and scope
One Llama3.2-1B-Instruct layer0, no rotation, FP32 residual, uniform INT16 Down. Prefill64, then one independent fixed-input decode with64 historical KV tokens; capacity80. Same binary and packed W4 weights in both arms. No model frontend/head, no full-model E2E/PPL and no baseline promotion.

The existing runtime permits this experiment through package metadata alone. Keep AV HMX shift unchanged. Original AV multiplier9 and target zero142 are replaced by multiplier1/zero128; O consumes the raw carrier with zero128 and input scale multiplied by9. Confirmed all8 attention groups use identical factors. Native src/include unchanged from4806798f979834f6913cd97f09c401e476ad709d. Existing AV identity path skips the vector pass. Host preparation already folds the new scale into the O FP32 epilogue metadata and updates its input-zero compensation.

## Numerical evidence
Both hardware arms match their own independent whole-block FP32 output references bit-for-bit. All actual AV carriers captured via the existing no-rotation chain audit match the corresponding reference. No model-quality claim follows.

| Check | Prefill | Decode |
|---|---:|---:|
| Actual AV elements |131072|2048|
| AV post-HMX saturation |0|0|
| Raw AV carrier range |120..135|126..131|
| O integer max absolute accumulator, original |2493|1305|
| O integer max absolute accumulator, folded |277|145|
| Final block max absolute difference between arms |5.960464477539063e-8|1.4901161193847656e-8|
| Final block relative L2 difference |1.2497232281855044e-8|2.7557486569283753e-8|

Integer equality before floating conversion is exact on all sampled rows. For saturated inputs in general, the candidate changes the contract: the removed clipping contribution is explicitly represented by the oracle identity. Current sample has no such saturation; it does not prove all layers or arbitrary sequences are saturation-free. FP32 scale reassociation is not bitwise-equivalent between arms. Integer-to-float conversion and scaling order explain the small differences; the new-contract hardware/reference checks remain exact. Universal O raw signed24 bound540090 and centered signed32 bound532224 pass. Uniform INT16 Down exhaustive LUT/byte reconstruction and accumulator bounds also pass.

## Paired timing
Five short rounds then ten formal AB/BA paired rounds, each process one priming replay plus ten measured replays. Bootstrap20000, seed610061. Complete single-block Host wall includes FastRPC and necessary boundary work; loading and untimed captures excluded. These are not full-model token/s measurements.

| Quantity (microseconds) | Original | Folded | Folded/original |
|---|---:|---:|---:|
| Prefill complete block Host wall |1791.901|1772.029|0.988910|
| Decode complete block Host wall |1430.573|1352.054|0.945114|
| Prefill DSP invocation |1504.183|1502.189|0.998675|
| Decode DSP invocation |1153.608|1096.426|0.950432|
| Prefill AV RQ accumulated stage time |61.856|0.333|0.005380|
| Decode AV combined stage time |78.692|19.979|0.253892|
| Prefill O projection |98.569|98.236|0.996629|
| Decode O projection |55.732|56.599|1.015569|

Prefill Host wall change -1.108951%,95% ratio CI[0.967173,1.006351]: no significant improvement established. Decode Host wall -5.488619%,95% ratio CI[0.930557,0.959497]: improvement in this tested implementation. Both satisfy unchanged10% slowdown confidence gate. Do not equate accumulated worker RQ time with removable wall time.

## Attribution limitation discovered
In current head64 short-decode branch, block_imp.c calls full-M64 qbh_attention_u8_requant_av; the row4 conditional is guarded by QBH_LLAMA_3B. Thus setting QBH_W4U8_DECODE_AV_REQUANT_ROWS=4 does not restrict this particular1B branch. Its RQ is included inside u8_attention_av_hmx_ticks, while the separately named AV requant counter is zero. The measured decode improvement includes elimination of this redundant full-row processing. It is not a comparison against a minimal row4 RQ control, nor evidence that HMX matrix arithmetic itself sped up.

## Physical, provenance and recovery
600 measured token boundaries plus60 per-process priming boundaries passed exact output/KV checks. Full additive invocation ledgers reconcile;8MiB requested/acquired, peak plan8098272bytes; timed intermediate DDR read/write/spill zero. Warmup and capture samples are separate. Phone battery85%, temperature33.6C before and33.8C after; no clock/frequency-control claim.

D-drive historical weights were intentionally retired. The device retained the old layer0 package: manifest and all79 payload hashes match the preserved L32-0016/layer0-a02 manifest. It was recovered to a NEW L32-0061 directory, then INT16 and folded arms were created and independently sealed. Historical records unchanged. Source tools remain in tools/probe_llama32_av_o_fold.py. Measured runtime source f99ad7a203025f5c631665e3d04df4eca86eee1c; subsequent commits fix only probe parsing/capture and archive verification. Closure source820254505b42f6da6cddf821ac0809fa831ebaae.

Initial generic audit flag was incompatible with replay and exited2 before DSP; switched to supported no-rotation chain capture. The first successful control capture exposed duplicate JSON record parsing and an old VTCM field name; original raw output was retained and reparsed. No numerical gate weakened, no failed runtime evidence overwritten. No source DSP algorithm edits. Device released.

## Next discussion
Local feasibility established. Before generalization, audit additional layers' saturation and distinguish scale folding from fixing full-row1B decode RQ work. No additional configurations or fullmodel runs performed under this bounded probe.

Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0061;234 retained files; EVIDENCE_SHA256.json SHA256 fe3b3aa087607c42c85acdc5bf41f726ce05394bfc691a46b3d69d4ced0ffc39.
