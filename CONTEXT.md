# L32-0008 closed: exact native W4 SP2 Down component

No active experiment or jobs. Next L32-0009 awaits user direction. Owner was
no-rotation; source final 0a00e0a28d7961846648922c47e4be264e6b78df; tested native4ed16cf. Rotation252aee5 and
frozen Qwen48eb1ea unchanged. Original weights, qparams, model runtime kernels,
existing recipe defaults unchanged. Prototype is a separate RPC/CLI only.

241-level signed SP2 codebook (zero,one/two distinct powers0..14),8-bit index,
shared scale from frozen Llama middle calibration maxabs/24576. Prototype
input is pre-reconstructed signed16; NOT a fused SwiGLU producer. Preserve
signed[-7,7] per-output W4. Device packs radix256 low/high U8, four retained
non-saturating conversions exponent24/16/8/0 read exact accumulator bytes,
HVX reconstructs L+256H-32768*sum(W). Decode uses spare64-row spatial positions;
prefill two passes. Exact int32 result only; final scale/requant/residual not
integrated. Do not label this full-layer or full-model SP2 acceptance.

Final20 functional probes934912 exact outputs,including399360 actual Llama
SP2 outputs on layer0/7/15 prefill64 anddecode1,K8192,N2048. Signed extrema,
carry,cancellation,longK,and both row packing/two passes pass. Emulator22049
samples passes. Local Down NRMSE U8->SP2: layer0 prefill19.406->6.688%,decode
70.434->5.287%;layer7 prefill40.951->3.426%,decode100->5.852%;layer15 prefill
14.386->2.946%,decode100->4.093%. Reference is same W4 and A8 Gate/Up trajectory
before middle quantization, NOT FP16 teacher or independent model PPL.

Fixed10 ABC/CBA component cycles perphase,1warmup+1measuredRPC perprocess:
Host us prefill U8single5212.57/U8exact5284.60/SP26779.78;
decode962.26/963.55/987.42. SP2/single ratio1.30066 CI[1.29327,1.30813] prefill,
1.02615 [.99155,1.05209] decode. SP2/exact1.28293/1.02477. U8single is identity
sat conversion timing control with sharedi32 return buffers, NOT optimized
full-block baseline. Isolated explicit packing dominates prefill increment.
No full-block10% gate decision,E2E orPPL. No model baseline promotion.

Acquire8MiB VTCM,peak2797568B,oneHMX owner,no weight expansion,no intermediate
TENSOR DDR or tensor spill;64B DDR DMA descriptor and128B compiler constant
signmask stack recorded separately.60 additive ledgers reconcile;HMX issue
counters asynchronous,conversion timing includes completionwait.

101 valid processes/182 valid probeRPC calls;2 DMA failures and1 known stale
DSP deployment excluded/preserved. Fixed descriptor placement in DDR,alignment,
RT,HVX type include. Build-a05 failed but orchestration incorrectly deployed
oldDSP with newhost; invalidattempt signed-small-rows-a02. Build seal and
deployment HEAD/hash guard now prevent this. See recovery_notes for details.

Recommended next: fuse SP2 quantizer/carrier generation into SwiGLU,connect
Down scale/residual,then singleblock actual-arithmetic validation and fixed10
completeHost10%gate. Keep PPL primary but do not infer model recovery from local
error gains. Existing full-model speed baseline remains L32-0006 and W16/W4
PPL remains L32-0007 (26.691749/31.039101 vsBF16teacher26.697999). A8 prior text
unusable/PPL1206603.74 unchanged; no new quality threshold or rotation support.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0008, SUMMARY.md/summary.json/timing-ledger.json and 606-file
ledger docs/experiments/L32-0008-evidence-sha256.json sha256 087f79a79669b658a1eccedde28b73e4cf71444405a755a64443bdcca850a50c.
