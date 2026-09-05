# EXP-0223 completed — authoritative handoff

Bootstrap/read authority before future project work. No active experiment or running jobs. Next224. Source codex/exp-0223-w4f16-gptq-channel-clipping HEAD c9c2491ff70875068f9a043a4305d07892cf5271. All package/quality/profiling/archive work complete. Do not repeat or overwrite retained outputs.

A: DSP NLL 3.873038 -> 3.969016, tasks 8 -> 4; B: DSP NLL 4.601481 -> 3.911864, tasks 0 -> 2; C: DSP NLL 4.320988 -> 3.956556, tasks 2 -> 9. All are actual DSP scores with frozen EXP0221 absmax controls; software results separate. Per-variant gates {"A": false, "B": true, "C": true}; incremental clipped C versus clipped A True. No baseline promotion.

Fixed80ratio row L2.4 range selection precedes GPTQ; only196transformer projections per variant changed. Original Qwen3-origin only, same8192calibration tokens and true-sequential GPTQ, no LPBQ/group scale/learned rotation/evaluation tuning. Heads, embeddings and norms byte-identical to corresponding EXP0221 package. Oracles,588packing/168forward checks, quick/full/repeat consistency, independent16speed tokens and physical contract pass.6way1warmup/5short/10formal,960invocation/26880layer ledgers.

Evidence D:/llm_exp/results/qwen3-block-htp/exp0223; report docs/experiments/EXP-0223-RESULTS.md and EXP-0223-PROFILE.md; 456 evidence files and ledger SHA256 7299b9e9ac5c462f5b144ba6dc1ddfb574323006d80fffca73f4c18e3c5ad413. Source/archive D:/llm_exp/models/qwen3-block-htp/exp0223/artifacts/c9c2491ff70875068f9a043a4305d07892cf5271.588clip NPZ and84calibration hidden checkpoints hashed in intermediate_sha256.json. Runtime frozen EXP0218 ABI108, no frozen recipe changes. Discuss results before another direction; no automatic further optimization authorized.
