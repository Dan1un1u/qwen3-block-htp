# EXP0248 — R3 numerical repair and latency rejection

The fixed layer0 replay numerical mismatch is repaired: HMX dense R3 plus guarded HVX SF32 direct-dot refinement matches the independent scalar Float64 oracle Q/K codes and complete layer outputs exactly across M64 and eight M1 steps. This is a single-layer numerical repair, not full-model quality acceptance. No butterfly/FWHT, weight regrouping, re-quantization, or scalar Float64 candidate fallback.

The candidate fails the unchanged10percent speed gate. Five short and ten alternating formal paired rounds (each repeat1/repeat10) completed. No full-model run or PPL evaluation was started; other recipes and selected baselines remain frozen. Experimental mode5 is retained for diagnosis, not promoted.

## Root cause and boundary evidence

Independent NumPy int64 QK, log2 Softmax and AV reproduce actual captured device inputs and outputs exactly. The template kernel retains raw HMX score bytes; its score multiplier5 maps a raw one-code change to five1/8-log2 score codes. The exponent approximation then rounds to integer powers of2, and SOLE normalization changes coefficients according to the top bits of the weight sum. These discretizations amplify tiny R3 rounding changes. They are existing arithmetic semantics, not a demonstrated wrong matrix layout, cache append or HMX state transition.

Concrete example: head9, query9, key8 raw QK changes150→151. Weight sum59296→46352 crosses a SOLE coefficient boundary145→209, and U8 probability changes144→208 (64codes). Exact division on those same integer exponential weights would produce141→180: SOLE adds error, while coarse exponent/score discretization also remains. This diagnostic calculation is not a new Softmax implementation or quality evaluation.

|Prefill boundary|Original HMX versus scalar max code difference|After refinement|
|---|---:|---:|
|Q/K codes|1|0|
|Raw QK scores|1|0|
|Probability|64|0|
|AV|35|0|
|O|9|0|
|Midpoint residual|9|0|
|Post RMSNorm|6|0|
|Gate / Up|17 / 20|0 / 0|
|SwiGLU|7|0|
|Down|10|0|
|Final output|7|0|

All nine final outputs now have zero code difference and dequantized cosine1 (floating computation within rounding of1), passing the unchanged <=2code/cosine>=0.999 gate. Raw pre-R3 carriers match, HMX identity is exact, independent dense component bound passes, native prefill K is independently repacked exactly, V cache matches, and outputs reproduce across builds. Guarded values are16811/196608 (8.55%) for prefill; decode mean285.375/3072 (9.29%). The guard uses two local FP16 spacings plus minimum-normal tolerance. Its validation is limited to this frozen replay and depends on the existing component error bound; it is not a universal HMX error theorem.

The first new checker omitted K's actual signed-S8 recenter saturation, producing24 raw-score mismatches. Native K packing clips centered values to[-128,127]; an asymmetric K zero point125 can otherwise produce130. The corrected independent oracle implements that frozen operation and all raw-score/probability/AV checks pass. Original checker and recovery record are retained. This additional representational clipping deserves future quality attribution; it was not changed here. Legacy zero mismatch fields are never treated as independent evidence.

## Performance

|Scope|Control us|Refined R3 us|Paired latency regression|95% ratio interval|
|---|---:|---:|---:|---|
|repeat1_prefill|1721.901|36840.391|+2038.50%|[20.1026,22.1125]|
|repeat1_decode|974.932|1966.875|+102.70%|[1.8697,2.1086]|
|repeat10_prefill|1537.276|36447.513|+2274.75%|[23.5115,24.3226]|
|repeat10_decode|951.845|1908.047|+100.12%|[1.9854,2.0116]|

Full paired profiling and module/counter tables: FULL_PROFILE.md. No outlier deletion. All2970 timed RPC outputs match independently hashed audited outputs. Every timed run requests/grants8MiB VTCM, peak6682752bytes, zero intermediate DDR/spill, one RPC per step and seven native W4 projections. The64KiB dense-sign mask reuses dead expanded-weight VTCM; no expanded W4 weight or intermediate DDR is introduced. Six separate audit replays/54RPCs export diagnostic tensors; their latencies are not performance evidence.

The costly refinement and surrounding R3 preparation/quantization make this implementation unacceptable for deployment. These results do not establish a lower bound for optimized dense R3 or show that rotation's software PPL improvement disappears. EXP0246 software PPL remains separate evidence. Real-device full-model PPL and E2E token/s are N/A because the speed gate stopped escalation.

## Recommended discussion

Prioritize the integer attention numerical contract: quantify the independent loss from coarse QK conversion, exponent rounding, SOLE normalization and K recenter saturation. Then evaluate a smoother/more precise attention implementation with the same native-W4 linear projections and frozen weights. This targets the observed amplification rather than spending tens of milliseconds forcing R3 to match ideal rounding. First validate attention boundaries independently, then use a fresh frozen PPL panel once hardware prerequisites pass. Any new attention algorithm requires its own registered experiment; no additional method started here.

Runtime source: bf84233fe77202f0bd201d706d9af69c70d0fc8c. Reporting source: 3587939fe7a727153a19f080a9d0d1c219f66083. Results: /mnt/d/llm_exp/results/qwen3-block-htp/exp0248. Parent EXP0247 evidence remains immutable. Build caches remain EXP0247_DENSE_R3=ON (single experimental layer0), ABI114; not full-model deployment binaries.
