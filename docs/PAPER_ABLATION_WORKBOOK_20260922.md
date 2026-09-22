# Hardware workbook update, 2026-09-22

User approved updating only already measured, length-matched main-table results; B/C retain their original numeric values and are marked pending refresh.

A64+42 final rows now use actual uniform INT16 Down results: Qwen0.6 EXP-0304 (2993.3312 / 93.3147 TPS), Qwen1.7 EXP-0307 I (1837.2715 / 47.2117), Llama1B L32-0065 M (2315.3542 / 46.3253), Llama3B L32-0066 folded (1204.0026 / 24.0540). No unconditional Qwen AV fold: retain its original independent RQ. Other A recipes unchanged. Do not reinterpret unpaired A8 versus newer INT16 timings as isolated quantization cost.

F now leads with the fair valid-row Llama1B/3B AV comparisons. Original L32-0062 values and raw round observations are retained at row51 onward as historical, not a fair ~4% decode claim.

H contains138 paired phase comparisons; I contains89 configurations; J contains3570 module latency/share rows. Sources: optimized EXP-0282/L32-0038 pipeline ablations, EXP-0284/L32-0040 three-way A8/SP2/INT16 tests, EXP-0304/L32-0058 replication, EXP-0307/L32-0065 integer/fusion tests, and L32-0066 3B fold. History keeps original SP2 measurement identity. Local1/3layer replay has no extrapolated E2E TPS. B/C retain historical SP2 labels and values. The separate L32-0058 INT16 long result is archived in H/I, not substituted into C.

D/E/G worksheet XML is byte-identical. All unrelated values/formulas/styles preserved; new formula caches checked, no cell errors. Artifact Tool authored all cells; selective OOXML merge preserved native workbook features. Desktop save verified by SHA256. No hardware rerun, new quality claim, runtime change, or model-artifact baseline mutation. Before/after snapshots, source hashes and reproducible authoring scripts retained here.
