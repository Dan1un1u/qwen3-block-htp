# L32-0027 closure: paper revision; original fast path accepted with known rounding

User accepts the original fast HMX implementation for paper preparation and
explicitly pauses further software/hardware numerical alignment. This replaces
the prior next-step suggestion of H512 integer-plane repair. Do not resume it,
run new PPL, build models, or profile based on the old0026 handoff.

Source /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation,
HEAD 0f7d083dc99af7e2983ecd426fca03e9d57a0a07. Native src/include unchanged from cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7.
Only docs/PAPER_STORY_RUNTIME_V2.md and its EVIDENCE.json were added.
Active none; next L32-0028. No hardware/model/build/native work in0027.

Read the V2 report for the current paper story: consumer-native physical
interfaces and bounded-VTCM scheduling; exact SP2 integer decomposition to
native packed-W4; phase-specific decode row packing/prefill overlap; paired
cross-model attribution plus FP32 residual and dense rotation extensions.

Qwen EXP0268 fair optimized-U8 control gives SP2 fullmodel latency +2.7099%
prefill / +0.8185% decode. Llama0012 gives +1.8851% / -0.0234% (decode CI
crosses zero). Llama0018 FP32 residual without rotations fullmodel is
2069.70/42.51token/s, paired latency -3.2309%/+8.1282%. Llama0022 fastR3R4
on FP32-residual/SP2 has only singlelayer speed +6.2074%/-0.9298% (decode
CI crosses zero); no combined fullfrontend E2E or PPL. Do not compose these
separate rates into an unmeasured combined result.

Original fastfull160023 independent cosine .5051155954/.7882501259 FAIL
remains preserved. User acceptance is paper/performance working scope only,
not numerical gate/quality pass. Later0026 rejected candidates remain rejected;
mode3 exact dense is diagnostic only. Do not relabel failures, measured source
heads, binaries, or historical quality/default flags. Broad PPL usable-quality
claims and all-A8 floating-boundary claims remain unsupported.

Report SHA256 02f0b458fd2d79e73c711bfa16b741c40fb151217548da0baa6a8baad91b6904.
Evidence index SHA256 b2444718b14fffcbf57475a5bced441ca3ae2624b8df19030bfcba3069093379.
Checked all local document links, five Qwen frozen report hashes, existing0018
and0026 ledger hashes, and0018 speed calculations. Index records hashes of
consumed authority at registration commit47c5df3; those snapshots are historical,
not assertions that mutable authority files stay byte-identical after closure.
No new PPL/performance result. All frozen source branches remain unchanged.

Next work is paper drafting/figures from existing evidence. Further native or
hardware work requires a newly authorized bounded experiment and preflight.
