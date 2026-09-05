# EXP-0222 completed — authoritative handoff

Bootstrap/read authority before future project work. No active experiment or running jobs. Next223. Source codex/exp-0222-w4f16-lm-head-ablation HEAD 6c65970b523a80b9a9343f0b246451690332ded5. Complete host-only ablation; do not rerun/overwrite outputs.

A: NLL 3.874188 -> 3.853870, tasks 9 -> 8; B: NLL 4.600959 -> 4.413664, tasks 0 -> 0; C: NLL 4.323169 -> 4.293221, tasks 2 -> 2. Values are software-versus-software, never pair these FP16-head results with DSP W4 values. Frozen EXP0221 A/B/C transformers, norms and embedding unchanged; only head replaced with fresh W, W*gamma, or (W*gamma)*H2048 respectively. W4 controls and bilingual repeats pass. No calibration, clipping, learned rotation, DSP changes or other-recipe changes.

Evidence D:/llm_exp/results/qwen3-block-htp/exp0222; report docs/experiments/EXP-0222-RESULTS.md; 27 evidence files and ledger SHA256 54dbdc87eb1b8df4fc5e7baf2c0f0e385d41e583c0caec5975b3d5c5c5556c78. Source archive D:/llm_exp/models/qwen3-block-htp/exp0222/artifacts/6c65970b523a80b9a9343f0b246451690332ded5. Fresh FP16 head artifacts and original package identities retained in closure/provenance. Profiling/E2E N/A under PC042; do not invent hybrid-head DSP speeds from historical results.

User explicitly requests discussion before deciding next direction. No further experiment authorized by completion. No baseline promoted.
