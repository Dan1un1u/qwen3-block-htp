#!/usr/bin/env python3
"""Retained stage logs and source snapshots; manually dispatched, no scheduler."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0227')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0227')
CPU='/home/daniuniu/.cache/qwen3-block-htp-py/bin/python'
GPU='/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python'

def run(stage):
    subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    if stage in ['init','unit']:
        command=[GPU,str(ROOT/'scripts/block_reconstruction_exp0227.py'),stage]
    elif stage.startswith(('smoke-','train-')):
        phase,v=stage.split('-');command=[GPU,str(ROOT/'scripts/block_reconstruction_exp0227.py'),phase,'--variant',v]
    elif stage.startswith(('export-','validate-','quality-')):
        phase,v=stage.split('-');command=[CPU,str(ROOT/'scripts/export_exp0227.py'),phase,v]
    elif stage=='verify-controls':
        command=[CPU,str(ROOT/'scripts/verify_controls_exp0227.py')]
    elif stage.startswith('deploy-'):
        command=[CPU,str(ROOT/'scripts/measure_exp0227.py'),'deploy','--variant',stage.split('-')[1]]
    else:
        command=[CPU,str(ROOT/'scripts/measure_exp0227.py'),stage]
    record=RESULT/'commands'/f'{stage}_{time.time_ns()}';record.parent.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive=OUTPUT/'artifacts'/head/'source.tar';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():subprocess.run(['git','archive','--format=tar','-o',str(archive),head],cwd=ROOT,check=True)
    started=time.monotonic()
    with record.with_suffix('.log').open('x') as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        code=proc.wait()
    record.with_suffix('.json').write_text(json.dumps(dict(stage=stage,command=command,source_head=head,returncode=code,elapsed_s=time.monotonic()-started,
        source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),log_sha256=hashlib.sha256(record.with_suffix('.log').read_bytes()).hexdigest()),indent=2)+'\n')
    if code:raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args();run(a.stage)
