# Reproduce derived paper tables and figure

No device run is performed by these commands. Existing stage ledgers remain immutable. Use a fresh output directory through QBH_PAPER_REPORT_ROOT; outputs refuse to replace retained tables.

```bash
export QBH_PAPER_REPORT_ROOT=/tmp/htp-paper-report-new
/home/daniuniu/.cache/qwen3-block-htp-analysis-py/bin/python /home/daniuniu/work/qwen3-block-htp/scripts/plot_paper_representation_costs.py
python3 /home/daniuniu/work/qwen3-block-htp/scripts/consolidate_paper_ablations.py
```

The consolidation validates the frozen stage ledgers and the exact SUMMARY/REPORT hashes. Each stage's source scripts regenerate tables from all fixed rounds (exclusive output paths); their device measurements must not be rerun into sealed result namespaces.
