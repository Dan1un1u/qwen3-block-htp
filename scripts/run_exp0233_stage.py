#!/usr/bin/env python3
"""Retain exact commands, source archives and stage logs."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0233')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py',
        'preflight','--source-worktree',str(S)],check=True)
    modules={'setup':('autoround_exp0233.py',['setup']), 'data':('data_exp0233.py',['prepare']),
             'audit':('data_exp0233.py',['audit']), 'oracle':('autoround_exp0233.py',['oracle']),
             'export':('autoround_exp0233.py',['export']), 'evaluate':('evaluate_exp0233.py',[]),
             'summary':('summarize_exp0233.py',[]),'model-audit':('audit_exp0233.py',[])}
    script,args=modules[a.stage]
    python=('/home/daniuniu/.cache/qwen3-block-htp-autoround-py/bin/python' if a.stage in ['oracle','export']
            else '/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python')
    command=[python,str(S/'scripts'/script),*args,*a.args]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
    archive=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0233/artifacts')/head/'source.tar'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():subprocess.run(['git','archive','--format=tar','-o',str(archive),head],cwd=S,check=True)
    R.mkdir(parents=True,exist_ok=True);root=R/'commands'/(a.stage+'_'+time.strftime('%Y%m%dT%H%M%S')+'_'+str(time.time_ns()))
    root.parent.mkdir(exist_ok=True);started=time.monotonic()
    with root.with_suffix('.log').open('x') as f:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=S)
        for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        code=proc.wait()
    root.with_suffix('.json').write_text(json.dumps(dict(command=command,source_head=head,
        elapsed_s=time.monotonic()-started,returncode=code,
        log_sha256=hashlib.sha256(root.with_suffix('.log').read_bytes()).hexdigest(),
        source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2)+'\n')
    if code:raise SystemExit(code)

if __name__=='__main__':main()
