#!/usr/bin/env python3
"""Replay three sessions per EXP-0310 formal case after authorized preflight/deploy."""
import json
from pathlib import Path
from exp0310_bandwidth_sweep import out, run
cases=json.loads((Path(__file__).resolve().parents[1]/'experiments/exp0310_bandwidth_formal_cases.json').read_text())
(out/'formal_cases.json').write_text(json.dumps(cases,indent=2))
for session in range(3):
    keys=list(cases)
    keys=keys[session*5:]+keys[:session*5]
    for key in keys:
        run(cases[key],f'formal_{session}_{key}')
