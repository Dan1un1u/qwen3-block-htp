# EXP0233 independent AutoRound reference

Software final: {'AR-P': 'fail', 'AR-G': 'fail'}; 1024 documents / 16384 targets. Reserve used: True. No promotion.

|Stratum|F16|C64|AR-P|AR-G|AR-P/F16 [95%CI]|AR-G/F16 [95%CI]|
|---|---:|---:|---:|---:|---|---|
|overall|25.060797|27.154144|27.480036|26.613251|1.096535 [1.0819081866681948, 1.1119950954819637]|1.061947 [1.0518851037235766, 1.0719623625052006]|
|en|20.634639|22.083722|22.475545|21.579593|1.089214 [1.0698004399791754, 1.1095406512401584]|1.045795 [1.0324145096551265, 1.059786104805987]|
|zh|30.436372|33.388735|33.598844|32.821060|1.103904 [1.0821504478125727, 1.1270741904563089]|1.078350 [1.0635187268974786, 1.0928904229351606]|
|wiki|20.790832|22.889492|23.124193|22.247155|1.112230 [1.0894238955435056, 1.1358111177756351]|1.070046 [1.05484645800745, 1.0857101976193742]|
|news|30.207717|32.213365|32.656377|31.836210|1.081061 [1.0629534165791101, 1.0999391284986737]|1.053910 [1.040800455170742, 1.0670897261189296]|
|en_wiki|18.287364|19.785471|20.061216|19.116040|1.096999 [1.0672081220378438, 1.1285804365259084]|1.045314 [1.0257876875638232, 1.0656782123513218]|
|zh_wiki|23.637014|26.480484|26.654830|25.891132|1.127673 [1.093834087655415, 1.1634952573782553]|1.095364 [1.0715890261296446, 1.1194192641754612]|
|en_news|23.283200|24.648936|25.180436|24.360633|1.081485 [1.0581078182027552, 1.1062832498992037]|1.046275 [1.0275350104634522, 1.0648121309981007]|
|zh_news|39.191614|42.099216|42.351887|41.605827|1.080636 [1.0530919599577404, 1.109276140351574]|1.061600 [1.043155718569809, 1.0805836527151358]|

## Matched-control interpretation

Overall versus C64: {'AR-P': {'ppl_ratio': 1.0120015303230723, 'ratio_ci95': [0.9974795271170259, 1.0272107419065681]}, 'AR-G': {'ppl_ratio': 0.9800806238684353, 'ratio_ci95': [0.9672250606656897, 0.993201535075956]}}

AR-P is official AutoRound blockwise learned rounding with a project-format[-7,7] quantization adapter, oneFP32scale peroutputrow; AR-G uses unchanged official int_sym full-range[-8,7], signed FP32scales andgroup128. Both use the exact existingC64 512x128calibration positions,200iterations/block,batch8,seed233,official best training-MSE parameter choice. C64 uses GPTQ and originalFP32solver weights; AutoRound AMP rounds originalweights toFP16 first and uses iterative optimization. AR-P isolates format compatibility, not solver alone; AR-G is a multifactored native-tool reference. Both freeze originalgamma/embedding/norms and the inheritedEXP221head. This is not the best unconstrained full-tool recipe or a causal estimate for grouping alone.

Official intel/auto-round v0.5.1 commit73669aa50871bca9f244ad99faa9979790f7c729 was archived with Apache2license; every installed source module matches the pinned checkout. Isolated dependency installation reused frozen torch2.8.0 andtransformers4.51.0 paths without changing old environments. All effective arguments/source archives are retained. Setup dependency and JSONdtype serialization failures remain in commands/recovery, without training-score tuning.

## Correctness and data

Independent NumPy tests cover project/native grids, zero/one-sided rows,FP16/FP32 and finite nonzero STE gradients. All392projection nibble/scalereconstructions exactly match exported effectiveFP16weight hashes;56blocks contain512finite teacher/student calibration outputs, first4stored. Frozen nontransformer parameters/buffers match before/after training and when reloaded for scoring. Canonical evaluation repeat/causal/CE checks pass; F/C64development reproduces all prior pertokenNLL/top1 exactly. Independent complete-model calibration forwards match all84 stored teacher/student layer states exactly(maxNRMSE0), and the two tool branches share exactFP16teacher states. Every36 final PPLaggregate independently reduced from rawtoken NLL.

Development is the reusedEXP230128document panel, descriptive only. Final1024documents were frozen before training/scoring and independently reconstructed, excluding every prior calibration/training/evaluation role throughEXP232 using document/text/32grams. Same tokenizer64prompt+16targets, masks, eagerFP16backend for all. Gates overall5% and all language/domain/cell10% use paired stratified documentbootstrap5000seed233. No candidate selected from finalPPL; reserve rule fixed. The unused inherited dataset scale_search_reference metadata is explicitly corrected in recovery/dataset_metadata_erratum.json without changing dataset/freeze hashes; actual calibration is512x128 for both candidates. The same PC052panel may be reused by the next two diagnostics with explicit disclosure, never to select a deployed model.

Development PPL: {'F': 25.337473379675465, 'C64': 27.150223991918875, 'AR-P': 27.747103852723924, 'AR-G': 27.368874681631816}

## Execution boundary and continuation

Host-only quality experiment; no DSP code/build/device/profiling changes. DequantizedPyTorch speed is not W4runtime throughput. Otherrecipes frozen. Next authorized phaseEXP234 group12864K, thenEXP235 C64sensitivity. No automaticpromotion or newoptimization direction.
