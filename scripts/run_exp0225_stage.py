#!/usr/bin/env python3
"""Retained command logs and fail-closed stage dispatch; no background scheduler."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0225')
CPU='/home/daniuniu/.cache/qwen3-block-htp-py/bin/python'
GPU='/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python'
COMMANDS={
 'smoke':[GPU,str(ROOT/'scripts/learned_rotation_exp0225.py'),'smoke'],
 'train':[GPU,str(ROOT/'scripts/learned_rotation_exp0225.py'),'train'],
 **{s:[CPU,str(ROOT/'scripts/export_exp0225.py'),s] for s in ['step000','step050','step100','control','select','quality']},
 **{s:[CPU,str(ROOT/'scripts/measure_exp0225.py'),s] for s in ['quick','full','repeat','warmup','short','formal']},
 'quality-report':[CPU,str(ROOT/'scripts/summarize_exp0225.py'),'--quality-only'],
 'speed-report':[CPU,str(ROOT/'scripts/summarize_exp0225.py'),'--speed-only'],
}
def run(stage):
    subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    command=COMMANDS[stage] if stage!='deploy' else [CPU,str(ROOT/'scripts/measure_exp0225.py'),'deploy','--variant',json.loads((RESULT/'selection.json').read_text())['selected']]
    record=RESULT/'commands'/f'{stage}_{time.time_ns()}';record.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    with record.with_suffix('.log').open('x') as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        code=proc.wait()
    record.with_suffix('.json').write_text(json.dumps(dict(stage=stage,command=command,source_head=head,returncode=code,elapsed_s=time.monotonic()-started,log_sha256=hashlib.sha256(record.with_suffix('.log').read_bytes()).hexdigest()),indent=2)+'\n')
    if code:raise SystemExit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=list(COMMANDS)+['deploy']);a=p.parse_args();run(a.stage)
