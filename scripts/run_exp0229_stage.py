#!/usr/bin/env python3
"""Retained stage command/source/log evidence; explicit invocation only."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0229')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args()
    subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True)
    python='/home/daniuniu/.cache/qwen3-block-htp-'+('spinquant-' if a.stage=='teacher' else '')+'py/bin/python'
    if a.stage=='teacher':cmd=[python,str(S/'scripts/teacher_exp0229.py')]
    elif a.stage in ['statistics-test','primary-summary','combined-summary']:
        cmd=[python,str(S/'scripts/summarize_exp0229.py')]+(['--test'] if a.stage=='statistics-test' else ['--combined'] if a.stage=='combined-summary' else [])
    else:cmd=[python,str(S/'scripts/measure_exp0229.py'),a.stage]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
    artifacts=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0229/artifacts')/head;artifacts.mkdir(parents=True,exist_ok=True)
    archive=artifacts/'source.tar'
    if not archive.exists():subprocess.run(['git','archive','--format=tar','-o',str(archive),head],cwd=S,check=True)
    root=R/'commands'/f'{a.stage}_{time.time_ns()}';root.parent.mkdir(exist_ok=True)
    started=time.monotonic()
    with root.with_suffix('.log').open('x') as f:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=S)
        for line in process.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        code=process.wait()
    root.with_suffix('.json').write_text(json.dumps(dict(stage=a.stage,command=cmd,source_head=head,
        returncode=code,elapsed_s=time.monotonic()-started,source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256(root.with_suffix('.log').read_bytes()).hexdigest()),indent=2)+'\n')
    if code:raise SystemExit(code)
if __name__=='__main__':main()
