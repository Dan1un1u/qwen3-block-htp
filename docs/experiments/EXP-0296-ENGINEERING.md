# EXP0296 engineering loop
1. opt159 adds parallel long decode. Qwen0.6 audit741 passes420 exact layer hashes; matched exploratory repeat3 decode21.303->51.929TPS, prefill2390.229->2394.957.
2. opt415 adds VTCM K transpose and vector bias sums. First compile unsupported intrinsic, then partial-decode arithmetic failure from unreplicated scalar halfwords; repaired20422b6. audit64 passes112hashes. Exploration159->415 prefill2378.777->2489.028,decode51.555->58.590.
3. opt927 adds exact common-numerator NR64 and causal tile bounds.16000000 independent host identity cases pass. Audits65/129 pass140/168hashes. Exploration415->927 prefill2495.754->2511.786,decode57.035->59.049; marginal effect not a formal significance claim.
4. opt1951 attempted matching V LUT banks; audit129 secondchunk mismatch and timed head mismatch. REJECTED; native implementation restored fromd69d9ec by a normal newcommit, no reset/history rewrite. All failed logs retained. Its exploratory timing is ineligible and not used.
Freeze finalopt927, oldopt31 samebinary control and native64/long64 references. No weights/math/tolerance changes. Five-short/ten-formal still required. No quality or baseline promotion.
