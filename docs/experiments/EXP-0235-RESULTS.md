# EXP0235 C64 precision sensitivity

Completed diagnostic:61fixed development variants on128documents/2048targets, then allfive predeclared families on the full shared PC0521024documents/16384targets. No hybrid selection, quantization, deployment or baseline promotion.

## Fixed full-panel family confirmation

|Variant|Restored weights (millions)|PPL|PPL vs F16|PPL ratio vs C64 [95% CI]|C64 excess NLL removed|Diagnostic thresholds|
|---|---:|---:|---:|---|---:|---|
|F|1720.451072|25.060797|+0.000000%|0.922909 [0.910573,0.934761]|100.000%|pass|
|C64|0.000000|27.154144|+8.353074%|1.000000 [1.000000,1.000000]|-0.000%|fail|
|ATT_ALL|352.321536|26.892944|+7.310809%|0.990381 [0.983124,0.997648]|12.048%|fail|
|MLP_ALL|1056.964608|25.492364|+1.722079%|0.938802 [0.928917,0.948389]|78.717%|pass|
|HEAD|311.164928|26.457783|+5.574386%|0.974355 [0.970391,0.978238]|32.383%|fail|

Restored counts cover the196transformer projection matrices plus LMhead, excluding the fixed FP16 embedding and norms. F is the full floating control; C64 restores zero matrices. ATT_ALL restores q/k/v/o across28layers; MLP_ALL restores gate/up/down across28layers; HEAD restores only LMhead. All other parameter and buffer bytes remain C64-identical.

|Stratum|F|C64|ATT_ALL|MLP_ALL|HEAD|
|---|---:|---:|---:|---:|---:|
|overall|25.060797|27.154144|26.892944|25.492364|26.457783|
|en|20.634639|22.083722|21.984094|20.916864|21.566990|
|zh|30.436372|33.388735|32.897896|31.068741|32.457671|
|wiki|20.790832|22.889492|22.528798|21.078134|22.358047|
|news|30.207717|32.213365|32.102487|30.831032|31.309277|
|en_wiki|18.287364|19.785471|19.643870|18.539658|19.370098|
|zh_wiki|23.637014|26.480484|25.837412|23.964181|25.806904|
|en_news|23.283200|24.648936|24.603116|23.598882|24.013047|
|zh_news|39.191614|42.099216|41.887770|40.279559|40.822425|

All per-stratum paired NLL differences, PPL ratios and95%intervals are retained in summary.json. Reference thresholds remain overall5% and language/domain/cell10%, but passing a diagnostic restoration does not accept a deployable mixed-precision recipe. Both panel halves were fixed and unconditional for every family, regardless of development ranking. F/C64 full-panel scores are verified byte-for-byte reuse of EXP233, not new measurements.

## Development family controls

|Variant|PPL|C64 excess NLL removed|
|---|---:|---:|
|F|25.337473|100.000%|
|C64|27.150224|-0.000%|
|ATT_ALL|27.056457|5.007%|
|MLP_ALL|25.550471|87.885%|
|HEAD|26.383564|41.453%|

## Fixed single-layer localization (development only)

|Rank|Restore set (zero-based layer)|Weights (millions)|PPL|Delta NLL vs C64 [95% CI]|C64 excess NLL removed|
|---:|---|---:|---:|---|---:|
|1|ATT_L00|12.582912|26.857458|-0.0108417 [-0.0186393,-0.0029444]|15.690%|
|2|MLP_L27|37.748736|26.913075|-0.0087731 [-0.0134628,-0.0039602]|12.696%|
|3|MLP_L04|37.748736|26.913313|-0.0087642 [-0.0179922,-0.0001869]|12.683%|
|4|MLP_L05|37.748736|26.941876|-0.0077035 [-0.0144896,-0.0007288]|11.148%|
|5|MLP_L24|37.748736|26.962127|-0.0069521 [-0.0114903,-0.0024272]|10.061%|
|6|MLP_L22|37.748736|26.988591|-0.0059711 [-0.0111232,-0.0008715]|8.641%|
|7|MLP_L20|37.748736|26.996665|-0.0056719 [-0.0097519,-0.0016836]|8.208%|
|8|MLP_L10|37.748736|27.019900|-0.0048117 [-0.0090812,-0.0004232]|6.963%|
|9|ATT_L05|12.582912|27.027550|-0.0045286 [-0.0092887,+0.0002725]|6.554%|
|10|ATT_L04|12.582912|27.029915|-0.0044411 [-0.0095718,+0.0008388]|6.427%|
|11|MLP_L02|37.748736|27.037707|-0.0041529 [-0.0175774,+0.0088108]|6.010%|
|12|MLP_L08|37.748736|27.053135|-0.0035824 [-0.0082342,+0.0009866]|5.184%|
|13|MLP_L25|37.748736|27.064629|-0.0031576 [-0.0073203,+0.0009167]|4.570%|
|14|MLP_L09|37.748736|27.065072|-0.0031413 [-0.0074311,+0.0009995]|4.546%|
|15|ATT_L22|12.582912|27.072089|-0.0028820 [-0.0056726,-0.0000618]|4.171%|
|16|MLP_L03|37.748736|27.073486|-0.0028304 [-0.0104704,+0.0044152]|4.096%|
|17|MLP_L06|37.748736|27.076255|-0.0027281 [-0.0086703,+0.0028764]|3.948%|
|18|MLP_L15|37.748736|27.077151|-0.0026951 [-0.0056979,+0.0004078]|3.900%|
|19|MLP_L26|37.748736|27.078918|-0.0026298 [-0.0086508,+0.0030829]|3.806%|
|20|ATT_L27|12.582912|27.079030|-0.0026257 [-0.0050543,-0.0003592]|3.800%|
|21|MLP_L11|37.748736|27.089782|-0.0022287 [-0.0060607,+0.0016835]|3.225%|
|22|MLP_L21|37.748736|27.093299|-0.0020989 [-0.0069093,+0.0029386]|3.037%|
|23|ATT_L08|12.582912|27.093909|-0.0020764 [-0.0083445,+0.0038777]|3.005%|
|24|MLP_L23|37.748736|27.105652|-0.0016430 [-0.0051144,+0.0018028]|2.378%|
|25|ATT_L21|12.582912|27.125083|-0.0009264 [-0.0060782,+0.0044559]|1.341%|
|26|MLP_L16|37.748736|27.125909|-0.0008960 [-0.0052742,+0.0035868]|1.297%|
|27|MLP_L17|37.748736|27.128335|-0.0008066 [-0.0043331,+0.0026024]|1.167%|
|28|ATT_L11|12.582912|27.129328|-0.0007699 [-0.0054491,+0.0037607]|1.114%|
|29|ATT_L12|12.582912|27.129561|-0.0007613 [-0.0039415,+0.0023857]|1.102%|
|30|ATT_L13|12.582912|27.136064|-0.0005217 [-0.0032880,+0.0021979]|0.755%|
|31|MLP_L12|37.748736|27.136539|-0.0005042 [-0.0040308,+0.0031918]|0.730%|
|32|MLP_L00|37.748736|27.142158|-0.0002971 [-0.0062510,+0.0053869]|0.430%|
|33|MLP_L13|37.748736|27.143379|-0.0002521 [-0.0034789,+0.0029576]|0.365%|
|34|ATT_L26|12.582912|27.150921|+0.0000257 [-0.0026149,+0.0027409]|-0.037%|
|35|ATT_L23|12.582912|27.157774|+0.0002780 [-0.0017413,+0.0023370]|-0.402%|
|36|ATT_L09|12.582912|27.161496|+0.0004151 [-0.0041716,+0.0054427]|-0.601%|
|37|ATT_L01|12.582912|27.163377|+0.0004843 [-0.0030554,+0.0040617]|-0.701%|
|38|MLP_L14|37.748736|27.164985|+0.0005435 [-0.0026098,+0.0035845]|-0.787%|
|39|ATT_L03|12.582912|27.175190|+0.0009191 [-0.0048452,+0.0067154]|-1.330%|
|40|ATT_L17|12.582912|27.177506|+0.0010044 [-0.0025703,+0.0044920]|-1.453%|
|41|ATT_L25|12.582912|27.185648|+0.0013039 [-0.0010607,+0.0038282]|-1.887%|
|42|MLP_L18|37.748736|27.185999|+0.0013168 [-0.0025257,+0.0051658]|-1.906%|
|43|ATT_L14|12.582912|27.186416|+0.0013322 [-0.0017335,+0.0045054]|-1.928%|
|44|ATT_L06|12.582912|27.189243|+0.0014361 [-0.0037083,+0.0066597]|-2.078%|
|45|ATT_L24|12.582912|27.194770|+0.0016394 [-0.0009999,+0.0042876]|-2.372%|
|46|ATT_L19|12.582912|27.197709|+0.0017475 [-0.0014989,+0.0050857]|-2.529%|
|47|ATT_L18|12.582912|27.210088|+0.0022025 [-0.0011698,+0.0055296]|-3.187%|
|48|MLP_L19|37.748736|27.221058|+0.0026056 [-0.0020202,+0.0077291]|-3.771%|
|49|MLP_L01|37.748736|27.225202|+0.0027578 [-0.0068589,+0.0117068]|-3.991%|
|50|MLP_L07|37.748736|27.239793|+0.0032936 [-0.0012918,+0.0078432]|-4.766%|
|51|ATT_L20|12.582912|27.246127|+0.0035261 [-0.0000199,+0.0072649]|-5.103%|
|52|ATT_L02|12.582912|27.247212|+0.0035659 [-0.0015425,+0.0106826]|-5.160%|
|53|ATT_L15|12.582912|27.259165|+0.0040045 [+0.0006835,+0.0073792]|-5.795%|
|54|ATT_L16|12.582912|27.259633|+0.0040217 [+0.0010289,+0.0068626]|-5.820%|
|55|ATT_L10|12.582912|27.265625|+0.0042414 [+0.0011658,+0.0075112]|-6.138%|
|56|ATT_L07|12.582912|27.281412|+0.0048203 [+0.0003692,+0.0092118]|-6.976%|

Each single-layer result is conditional on all other C64 weights. Effects are nonadditive and can compensate or amplify one another; never sum recovered fractions or extrapolate a top-k combination. The56 exploratory intervals are pointwise, not simultaneous multiple-comparison significance. No ranked layer combination was constructed or scored. The128development documents are exposed diagnostic data.

## Correctness and provenance

Fresh original checkpoint files and tokenizer match the verified EXP218 ledger. Full FP16 restoration and empty restoration exactly reproduce historical F/C64 per-token NLL and top1. Every new score passes exact repeat, future-mask invariance, all-logit finiteness and independent CE difference below5e-6. All65 model states match the complete expected parameter/buffer SHA256 map. Original/C64 snapshots remain immutable; the final128document C64 sentinel matches exactly. All594 reported stratum PPL values independently reduced from raw token NLL via math.fsum.

Actual evaluation source a5e5b565e34cde9068cd7e6633e1fbb13647ec1b; report source 6658f53550cafae5a85a2488f1f69673ea1e4fb1. InputfreezeSHA256 2decbae51004dccd6f06d66c103d1664fc3f8de6af079b2575a76e1b18002d58; restoration manifestSHA256 98cc2bbbcbfa27686093ab590e5a6ca67c3847dc7603ce863b1b99e6e813de46. Retained stage commands and source archives establish execution provenance. No original, F16F16, W4U8, C64 package or DSP runtime changed.

The evaluation panel is the shared PC052 panel, independently reconstructed and frozen before EXP233 scoring, and independent of calibration/training. It is exposed paired reuse here, not a newly independent test perphase. No selection or training used these scores.

## Completion boundary

All three PC052 diagnostics are complete. Results support discussion of the next precision-recovery step only. No automatic next optimization or baseline promotion. Device profiling/E2E for restoration hybrids is N/A; full_profiling_report.md retains complete unavailable sections and verified historical references.
