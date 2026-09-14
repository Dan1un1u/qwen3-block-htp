# L32-0015 diagnosis complete; discussion before gate/speed progression
Read docs/experiments/L32-0015.md, docs/LLAMA32_R3_R4_SP2_CHECKPOINT.md,
then docs/LLAMA32_R4_SP2_DIAGNOSIS.md.
Frozen195rows: all399360 actual-R4 integer-tail outputs exact;37 output codes differ
from ideal-factor reference, max2LSB. Six low-energy layer0 rows fail cosine;57
exact zero rows have undefined cosine (also zero in old baseline). One decode
difference traced through SP2 threshold -> Q31 crossing0.5 -> residual oneLSB.
Local software controls identify coarse output/residual U8 as much larger loss than
HMX rounding. No wholemodel quality/PPL result. Original failures retained.
Runner zero cosine now nullable, never auto-pass. No threshold or runtime change.
Native build64dbdff; newer source contains Python-only diagnostics. Rebuild/seal if
future runner requires exact current HEAD. No formal speed, consecutive/fullmodel.
Active lock retained for next decision; R3 fullblock remains pending.
