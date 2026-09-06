#!/usr/bin/env python3
"""Run the remaining frozen EXP234 scores after retained export/control commands close."""
import json,subprocess,time
from pathlib import Path
from data_exp0234 import RESULT,SOURCE,preflight

def completed(script,tail):
    matches=[]
    for p in (RESULT/'commands').glob('*.json'):
        c=json.loads(p.read_text());cmd=c['command']
        if any(str(x).endswith('/'+script) for x in cmd) and (not tail or cmd[-len(tail):]==tail):matches.append(c)
    if not matches:return False
    assert len(matches)==1,('unexpected multiple attempts',script,tail)
    assert matches[0]['returncode']==0,('preserve failed stage',script,tail)
    return True

def run(stage,*args):
    subprocess.run(['python3',str(SOURCE/'scripts/run_exp0234_stage.py'),stage,*args],check=True)

def main():
    preflight()
    while not (completed('export_exp0234.py',[]) and completed('evaluate_exp0234.py',['primary','G8'])):
        time.sleep(30)
    for phase in ['development','primary']:run('evaluate',phase,'G64')
    run('summary')
    if json.loads((RESULT/'summary_primary.json').read_text())['reserve_trigger']:
        for v in ['G8','G64']:run('evaluate','reserve',v)
        run('summary','--combined')
    print('EXP234_SCORING_FINISHED_READY_FOR_REPORT',flush=True)

if __name__=='__main__':main()
