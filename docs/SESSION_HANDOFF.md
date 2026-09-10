# L32-0004 pipeline speed candidate

Active no-rotation source2aa7831 (full hash in status). Baseline16 binaries saved
with hashes under results/llama32-htp/l32-0004/baseline-build16. Candidate HVX head64
RoPE, A8 valid decode row, sparse exact rounding repair, existing head worker pool,
K masked scatter and V LUT/native patch packing. W4 OPT2 generalized GQA2 to GQA4.
W4 layer0 prefill/decode and3-layer exact replay passed; A8 single latesta03 exact
output/KV passes. A8 three-layera02 running session93787 afterbuild3-a02.
Then build16, run both recipe16-layer replays, deploy/gate/formal via
 tools/run_llama32_pipeline_profile.py. Formal fixed10AB/BA pairs per recipe,
M64+7 W4 /M64+15 A8. All original weights/calibration unchanged, quality unaccepted.
No formal results yet. Failed build1-a03 retained; fixed local C identifier scope.
