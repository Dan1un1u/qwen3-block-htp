# L32-0020 optimization checkpoint
Authority closure source `d3ba5e680e1f6bfebee5d981fa53357c19eac489`, profile/native validation `0eb7615c43bb6e36c402f3a5d12d7162c32722dd`.
Read `docs/LLAMA32_ROTATION_PIPELINE.md` and docs/experiments/L32-0020.md. No active experiment.
Candidate1 retained: H64 native constant copies/four-row A8 packing and R4
four-way gather/LUT pipeline using phase-dead Down scratch. Candidate2 register
transpose correct but slower; reverted. Producer overlap already existed.
Layers0/7/15 OFF/R3/R4/both numerical and physical gates pass. R4 ideal-exact
CLI failure preserved; conditional tail exact and independent cosine>=.999.
Fixed10 warmed four-arm cycles: prefill OFF1942.87,R31868.04,R42207.37,
both2186.07us;decode OFF1455.23,R31402.32,R41499.09,both1524.38us.
R3 passes both10%gates. R4 prefill+13.61%,both+12.52% fail; both decode+4.75% passes.
No continuous3/16/frontend/PPL/E2E rotation expansion. No default promotion.
Next discussion: R4 layout77.29us and SP2 finish94.24us vs HMX33.29us;
focus native stage2 consumption/worker overlap, not failed register transpose.
Do not selectively rerun failed formal gate. A next candidate needs a new
registered bounded experiment; do not repeat completed exploration.
0019 cold results remain historical and incomparable as paired gain.
Historical0018 no-rotation FP32/SP2 E2E2069.70prefill/42.51decode tok/s.
Ledger `/mnt/d/llm_exp/results/llama32-htp/l32-0020/evidence-ledger-checkpoint-a01.json`
SHA256 `06e74b1fd182511052387ba0f2d36fb653a2168081a92a517f3ac4495d233d91`;371files,12frozen package manifests,64CLI,944boundaries.
Prior0019 ledger and all files reverified unchanged. Qwen and other branch frozen.
