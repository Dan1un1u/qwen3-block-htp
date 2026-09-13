# L32-0009 closed: fused SP2 Down full-block and single-layer speed passed

No active experiment or jobs. Final no-rotation source 625073a5b680c9a115b694eb73d4c07286052212, clean and pushed.
Formal measured source a4a5882596d4e75be0e47bebe3c17e568caf077c; optional host-only audit export457e2a6
uses identical DSP SHA. Current build is layer1, seal457e2a6; source625073a adds
only report documentation. Future deployment must rebuild/reseal after the next
approved experiment, not relabel an old build. Rotation252aee5 and Qwen48eb1ea
remain frozen/clean/synchronized. Defaults and existing W16/W4A16 recipes unchanged.

Fixed L32-0008 241-level signed SP2 codebook and calibration scale. FP32 carrier
scale has zero code differences from all six prior SP2 activation fixtures.
Gate/Up65536-entry LUT stores v+32768; HVX gathers and directly packs low=l and
high=h+128 into native64x32 tiles. Decode lowrows0..3/highrows4..7 share one HMX
stream; prefill uses two streams. Per-output W4 scales and weights unchanged.
Exact L+256H-32768sumW, then declared Q31 positive multiplier and signed rounded
requantization to original DownU8, followed by original Q14 residual.
Host proves for every channel and every possible U8 input each raw partial dot
fits signed24; otherwise reject. Three non-saturating retain conversions suffice.
Final mode4 register-only signed24 unpack/merge/Q31/pack and one HVX ownership
transfer for entireDown; main retains scalar DMA submission with original
double buffering and a singleHMX owner. Error paths explicitly release ownership.
Mode3 retains perbatch lock control; old1/2 rejected. SP2 explicit opt-in,
requires dedicated LUT/qparams package; default remains U8.

Correctness: final SP2 layer0/7/15 prefill64/decode1 outputs/KV exact,399360 unique
block outputs. Separate actual Down carrier audits399360 codes exact before
residual. LUT65536 pairs/layer exact byte decomposition; original L32-0008 signed
carry stress remains valid evidence. Producer carrier side audit limited to
first6144/8192 channels by old capture size, not an all-channel producer dump.
Down and final outputs cover all2048 output channels. Independent Q31 vs exact
scale gave zero output-code differences on six fixtures.

Final ten AB/BA process pairs,each1warmup+10measured identical-cache M64+1
replays. Repeat1 auxiliary excluded. Singlelayer0 Host us:
prefill U8 2015.28022 / SP2 2135.93487,ratio1.059869912,CI[1.026326658,1.088838579];
decode U8 1426.11088 / SP2 1441.63639,ratio1.010886608,CI[.964618192,1.056183229].
Both95% upper<=1.10 PASS. 400timedRPC+40warmup;440 additive ledgers reconcile.
VTCM8MiB acquired,SP2peak8212960B,U8peak7668960B. No weight expansion or timed
intermediate tensorDDR/spill. Final producer no stackspill; Down256B constant
0/255 mask stack access only, no tensor spills. Generic HMX owner no added HVX
instructions outside its lock. No fullmodel inference/PPL/E2E, no baseline promotion.

Previous scalar/mode2 and inlinedmode3 retain exact-output diagnostic evidence
but fail physical acceptance: mode2 local array stackstores; inlined mode3
HVX constants hoisted before lock. Fixed final function isolation and ownership;
mode3 formal prefill upper1.10190 failed before final iteration, not rerun to
pick a favorable number. Final third bounded optimization passed its new tenpairs.
One standalone audit CLI rejected beforeDSP; one localPython numpy import failed
beforedevice. Host-only replay diagnostic export fixed access; three diagnostic
DDR capture processes excluded from allspeed. Historical audit naming R3 does
not enable rotation (mode0). 73successfulDSPprocesses/1186RPC,plusoneCLI rejection.
Finalnative454RPC across correctness,formal,anddiagnostic.

Next discuss/register fullmodel SP2 validation separately: generate all16layers
from original-derived frozen Llama weights/calibration,actualhardware PPL/text,
then E2E. Do not infer quality recovery from local Down or layer-speed gate.
Wholemodel speed baseline stays L32-0006; W16/W4 PPL L32-0007
26.691749/31.039101 vsBF16teacher26.697999. PriorA8 unusable; unchanged evidence.

Source docs/LLAMA32_SP2_DOWN_FUSED.md. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0009,
SUMMARY.md/summary.json/contract-audit.json/run_inventory.json. Build binaries
for allfive tested heads archived and hashesverified. package_manifests.json
records all frozen packages. Ledger docs/experiments/L32-0009-evidence-sha256.json,
415files,SHA256 9a1beb6e33de90527a753bb9272833f9a187bde7e091a631a28be4ee0dd956ac.
