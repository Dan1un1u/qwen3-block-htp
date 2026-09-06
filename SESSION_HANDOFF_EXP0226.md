# EXP0226 completed - discuss mixed result

Source branch codex/exp-0226-w4f16-rotation-data-sampling, HEADb90d9a7309fe470fb7dc7b8de83b483621ad481c. No active experiment or running/paused jobs. Next available experiment227; a new direction requires user approval. Baselines unchanged.

100step exact-Cayley sampling ablation and exports/validation completed; selected step100 before qbh. English training200documents(two random128-token windows each), Chinese same400documents(one random body window); other training/quantizer parameters unchanged. Validation64Wiki/news balanced en/zh and independent from training/GPTQ/qbh. Data SHA2567126e60d0dd74508982082fdab4cd2e2ecb8cb85375511b0f2fe04a5d0d573c6.

DSP PPL40.44461662608407/9of24 versus old22510050.22449369429062/20of24; English46.24650852->29.51483975. UnrotatedA remains38.87362375/19of24. Both content errors and format-following failures count under unchanged strict scoring. Independent actual validation3.561694003186048->3.5737880625555354; STE3.6923705115914345->3.6601522751152515. No consistent quality recovery; local effectiveness fail, evidence valid, adoption pending.

1warmup5short10three-way formal complete;480 invocation/13440 layer ledgers pass. Selected prefill64tokens/63222.422us=1012.29909tok/s; decode15tokens/1390274.374us=10.789237tok/s. Runtime/other recipes frozen.

Results D:/llm_exp/results/qwen3-block-htp/exp0226; models/checkpoints/artifacts D:/llm_exp/models/qwen3-block-htp/exp0226.321 evidence files and467 intermediates. Evidence ledger SHA256fcc3c22ed7b437be85552ca40ce953a891ded9ebc4eb8e05387210724f88a797; closure SHA25621db2339632d67b76778ed649f01499c165cecab085c7fa2574df8fb9786e724. Reports docs/experiments/EXP-0226-RESULTS.md and EXP-0226-PROFILE.md. Source/env archives include export source c729dedad45bee3899f34ae2dbed3ee9a042cae0 and final b90d9a7309fe470fb7dc7b8de83b483621ad481c.

Canonical reload repair model.half() restores FP16 RoPE buffers; prior initial scores preserved under validation_failed_fp32_rope_reload.json. All32Wiki control scores exactly reproduce original exports;8vs16CPU scoring oracle matches. No threshold changes or overwritten historical artifacts.

Suggested next discussion: freeze rotations and test same8192-token GPTQ calibration with broader English document coverage (currently32windows from3documents). Training/export objective mismatch remains a hypothesis. Do not launch another experiment or promote automatically.
