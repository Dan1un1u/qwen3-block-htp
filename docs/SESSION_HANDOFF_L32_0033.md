# L32-0033 complete; A8 remains

Source 0dfebbd6f72b145574c66d22d7dd13dfb3175883, measured 78560f4c12c63d2d1ac07405baca84c4ed275b8d; branch codex/llama32-no-rotation clean/pushed. All Llama A9 device work finished. Qwen EXP0278/vsum owns device: do not deploy/hash/profile Llama until release. Ledger c647b984f5cc875a366f6829589c66a6b5428961b867ff1159cc4f9813866904, 1497 files / 300912966 bytes under /mnt/d/llm_exp/results/llama32-htp/l32-0033; immutable.

FP vector softmax full fixed wall vs retained log2: prefill .94964867 CI[.94791243,.95164266], decode .98853313 [.98712262,.98995377]; no universal log2 advantage. Llama retains original per-layer division, selected7 EXACT; Qwen NR64 differs. Probability/AV approximation changes documented, all own selected/chain3/full16/frontend references pass, no quality acceptance/promotion. No vector stack spills, 8MiB, no tensor DDR spill.

Remaining approved A8: one already supported longer KV all-on vs representative FFN disable2, same SP2/FP32 no rotations. Register L32-0034 then preflight; CPU preparation allowed while Qwen owns device. Propose M64+33 (34 outputs, cache128), crossing KV96 tile boundary, unchanged original log2 wide0. Freeze fixtures and reference before timing. Numerical/physical gates unchanged, no runtime redesign for sweeping. A1/A2/A3/A5/A6 complete, A7 citeQwen272.
