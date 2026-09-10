# EXP0265 optimized dense R4 full-model diagnostic E2E

User-approved PC079 changes performance gating to formal paired repeat10 only; repeat1 stays auxiliary and cannot veto. Sealed EXP0264 is eligible under this new rule, its original recorded classification is unchanged. Three-layer gates completed before this28-layer integration.

Frozen R3 OPT2 + wideNR64/native per-channel W4 control versus same path with full6144 dense R4 OPT6. Fresh original Down W R quantized RTN[-7,7]/singleFP32scale peroutput for28layers; otherC64 weights, head/embedding/norms, staticA8 ranges and fixed rotatedEOSseed unchanged. Full Hadamard is H12 tensor H512 via two dense HMX factors, explicit FP16 normalization and intermediate rounding; no butterfly or groupquantization. Prefix is the existing frozen R3 diagnosticseed, not recomputed through R4. This costfixture has no R4 recalibration or quality-optimized Down quantizer.

Export:28 originalFP64 xW invariance checks, independentW4unpack, layer0 byteexact toEXP0261,65536SwiGLU LUT entries perlayer Decimal80/independentmpmath100. Changed87 files; all906 packagefiles verified locally/remotely and parent/original hashes retained. Threeconsecutive layers,9steps: noR4 reproduces sealedEXP0259; OPT3,OPT6 and repeat exact. Exposed lastlayer actualR4 factors compareFloat64; SDKDown/residual exact; conditionalidealR4 outputmax1LSB/cos>=.999. All finiteFP16quantization checked perlayer inuntimed audits. KnownidealR3 wholelayerfailure unchanged, no PPL or model-quality acceptance.

Full28layers OPT3/OPT6/audit/repeat produce identical16feedbacktokens and selectedlogitcodes. Fullaudit exercises all63488finiteFP16values on all28layers eachstep. Only selected maxima codes are compared in generation, not fullvocabulary logits. Five short and ten alternatingformal pairs repeat1/10,5280fullmodelRPCs and147840layerledgers verified. Everysequence starts fresh then appends persistentKV continuously. OneFastRPC perfullmodelstep;8MiB requested/granted,zero timedintermediateDDR/spill/audit, no interlayerhiddenDDR and a singleHMXowner.

Primary values are medians of ten round means. Effect is median within-round time ratio with10000pairedbootstrap seed265. No outlier deletion, optional resampling or post-result tuning. Each module table uses the identical two median-Host-ranked rounds for allfields within eacharm; rows close toHostwall. Historical F16 EXP0218 and Selected W4A16 EXP0166 are nonpaired matchingfullmodel M64 references, not currentpairedcontrols.

| Scope | R3 control us | R3+R4 us | Paired change | Ratio95%CI | Role |
|---|---|---|---|---|---|
| r1_prefill_ns | 39029.219 | 41944.558 | +7.58% | [1.067870,1.081887] | auxiliary |
| r1_decode_ns | 21961.052 | 22550.965 | +3.01% | [1.007420,1.059689] | auxiliary |
| r10_prefill_ns | 37433.521 | 40478.950 | +8.02% | [1.071033,1.086132] | gate |
| r10_decode_ns | 20539.982 | 21709.915 | +4.43% | [1.033043,1.059265] | gate |

Repeat10 speedeligible: True. Repeat1 cannot veto. Numerical/physical/evidence rules remain unchanged; no baseline promotion.

| 模块 | F16A16 EXP0218 | W4A16 EXP0166 | W4A8＋R3＋R4 EXP0265 | A8相对W4A16增速 |
|---|---|---|---|---|
| I/O、metadata | 99.0 (0.12%) | 357.4 (0.57%) | 260.6 (0.64%) | +37.18% |
| Input RMSNorm | 490.2 (0.61%) | 495.1 (0.79%) | 562.9 (1.39%) | -12.06% |
| QKV＋Q/K Norm-RoPE（含R3） | 11470.6 (14.22%) | 11767.3 (18.69%) | 7041.0 (17.39%) | +67.13% |
| QK–Softmax–AV | 3987.8 (4.94%) | 3964.3 (6.30%) | 3445.2 (8.51%) | +15.07% |
| O projection | 5765.5 (7.15%) | 5044.3 (8.01%) | 1261.3 (3.12%) | +299.94% |
| Post-attention residual＋RMSNorm | 473.8 (0.59%) | 472.8 (0.75%) | 658.5 (1.63%) | -28.20% |
| Gate/Up＋SwiGLU（含R4） | 29714.1 (36.82%) | 22434.4 (35.62%) | 17083.3 (42.20%) | +31.32% |
| Down | 13438.5 (16.65%) | 8506.4 (13.51%) | 3376.2 (8.34%) | +151.95% |
| Final residual | 140.0 (0.17%) | 140.8 (0.22%) | 184.3 (0.46%) | -23.65% |
| KV carrier conversion | 173.2 (0.21%) | 147.2 (0.23%) | 136.2 (0.34%) | +8.09% |
| KV append DMA | 343.6 (0.43%) | 336.8 (0.53%) | 280.0 (0.69%) | +20.30% |
| Block orchestration | 15.9 (0.02%) | 18.3 (0.03%) | 34.2 (0.08%) | -46.47% |
| Layer bookkeeping | 24.2 (0.03%) | 22.8 (0.04%) | 27.4 (0.07%) | -16.86% |
| Stage-boundary bookkeeping | 8.2 (0.01%) | 5.1 (0.01%) | 18.9 (0.05%) | -73.10% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A：零分母 |
| Runtime setup/teardown | 82.1 (0.10%) | 100.7 (0.16%) | 116.6 (0.29%) | -13.68% |
| Embedding | 68.4 (0.08%) | 81.2 (0.13%) | 39.9 (0.10%) | +103.37% |
| Final model RMSNorm | 49.7 (0.06%) | 48.6 (0.08%) | 4.2 (0.01%) | +1046.90% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6720.8 (10.67%) | 5256.9 (12.99%) | +27.85% |
| Host–DSP 边界 | 2353.5 (2.92%) | 2310.6 (3.67%) | 691.4 (1.71%) | +234.17% |
| 完整 Host wall | 80692.2 (100.00%) | 62974.7 (100.00%) | 40479.0 (100.00%) | +55.57% |

Text sample, before first EOS (timing deliberately continues16outputs afterEOS for fixedworkload, so later fragments are not scored):

Prompt: What is the capital of France? Answer briefly.

R3: The capital of France is Paris.

R3+R4: 予以简明直接的答复如下：

R4 produces decodable text but does not answer this question beforeEOS. This is a semantic failure for this sample, not a formatting failure. No broader quality or PPL conclusion from one speedfixture prompt.

R4 phase wall telemetry is included in Gate/Up+SwiGLU. Matmul is exposedsubmit/wait, not totalengine compute. Aggregateparallelwork overlapswall and mustnotbeadded; no component extrapolation toE2E.

| R4 phase us | r1 prefill | r1 decode |
|---|---|---|
| prepare | 4377.839 | 270.694 |
| matmul | 717.214 | 124.326 |
| layout | 1768.828 | 27.170 |
| finish | 3328.828 | 209.337 |
| parallel_work | 20161.458 | 0.000 |
| parallel_join | 563.385 | 0.000 |

| R4 phase us | r10 prefill | r10 decode |
|---|---|---|
| prepare | 4382.031 | 270.818 |
| matmul | 718.073 | 124.651 |
| layout | 1768.997 | 27.267 |
| finish | 3324.859 | 209.393 |
| parallel_work | 20170.266 | 0.000 |
| parallel_join | 563.115 | 0.000 |

Actual E2E (completeHost;64prefill includes fixedEOS,15continuousdecode afterfirst output):

| arm/repeat | prefill64 tok/s | decode15 tok/s | 16outputs/modelHost tok/s | 16outputs/generationloop tok/s |
|---|---|---|---|---|
| a0_r1 | 1639.797 | 45.535 | 43.442 | 42.768 |
| a1_r1 | 1525.824 | 44.344 | 42.085 | 41.387 |
| a0_r10 | 1709.698 | 48.686 | 46.307 | 44.662 |
| a1_r10 | 1581.069 | 46.062 | 43.689 | 42.287 |

Allthroughput uses measured complete embedding/28layers/finalnorm/nativeW4head/greedy/FastRPC. Hostdenominators and token counts in summary.json; generationloop additionally includes ondeviceHostloop and profilingserialization. Hot inference excludes tokenizer/detokenizer,ADB and modelload; coldstartup separately recorded. No perlayer extrapolation or PPL.
