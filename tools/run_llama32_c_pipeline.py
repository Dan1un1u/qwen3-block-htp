#!/usr/bin/env python3
"""Durable sequential continuation after the already-running initial training.

Every stage keeps its own log and requires successful preflight. Stops at the
first failure; never retries, overwrites artifacts or promotes a baseline.
"""
import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013')
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013')
PYTHON='/home/daniuniu/work/rotation-quant/.venv/bin/python'
OUT=RESULTS/'pipeline-a01'
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def main():
    OUT.mkdir(exist_ok=False)
    save(OUT/'owner.json',dict(pid=os.getpid(),source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),scope='L32-0013 sequential authorized training/export/device continuation'))
    while not (MODELS/'initial/complete.json').exists():
        alive=False
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:args=(p/'cmdline').read_bytes().decode().split('\0')
            except (OSError,UnicodeError):continue
            if str(ROOT/'tools/llama32_c_rtn_train.py') in args and '--worker' in args and 'initial' in args:alive=True
        if not alive:raise RuntimeError('Initial worker stopped without completion; preserve attempt and investigate')
        time.sleep(15)
    print('INITIAL_TRAINING_COMPLETE',flush=True)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='4',HF_HOME='/home/daniuniu/work/rotation-quant/cache/huggingface',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def script(name,*args):return [PYTHON,'-B',str(ROOT/'tools'/name),*map(str,args)]
    steps=[('b_init',script('llama32_c_rtn_train.py','b_init')),('c_training',script('llama32_c_rtn_train.py','c')),('export',script('export_llama32_c_rtn.py')),('calibrate',script('calibrate_llama32_c_rtn.py','calibrate')),('package',script('prepare_llama32_c_package.py','prepare')),('oracle',script('prepare_llama32_c_package.py','oracle')),('fixtures',script('prepare_llama32_c_package.py','fixtures')),('teacher',script('evaluate_llama32_c.py','teacher')),('integer_ppl',script('evaluate_llama32_c.py','integer'))]
    for count in [1,3,16]:
        steps.append((f'build{count}',['bash',str(ROOT/'scripts/build_llama32.sh'),str(count)]))
        if count==1:
            for i in [0,7,15]:steps.append((f'layer{i}',script('run_llama32_stack.py','--package',MODELS/f'gates-a01/layer{i}-count1','--output',RESULTS/f'device-c-layer{i}-a01','--sp2-mode','9')))
        if count==3:steps.append(('stack3',script('run_llama32_stack.py','--package',MODELS/'gates-a01/layer0-count3','--output',RESULTS/'device-c-stack3-a01','--sp2-mode','9')))
    for stage in ['deploy','generate','ppl','compare']:steps.append((stage,script('run_llama32_c_device.py',stage)))
    completed=[]
    save(OUT/'plan.json',dict(steps=[dict(name=n,command=c) for n,c in steps]))
    for i,(name,command) in enumerate(steps):
        print('PIPELINE_STAGE_BEGIN',name,flush=True)
        started=time.time()
        with (OUT/f'{i:02d}-{name}.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(name=name,command=command,returncode=result.returncode,started_unix=started,ended_unix=time.time())
        save(OUT/f'{i:02d}-{name}.json',record)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {OUT}/{i:02d}-{name}.log')
        completed.append(record)
        if name.startswith('build'):
            archive=RESULTS/f'c-{name}-seal';archive.mkdir()
            seal=json.loads((ROOT/'build/llama-build-seal.json').read_text())
            for path in seal['files']:shutil.copy2(path,archive/Path(path).name)
            shutil.copy2(ROOT/'build/llama-build-seal.json',archive/'build-seal.json')
        print('PIPELINE_STAGE_COMPLETE',name,flush=True)
    save(OUT/'complete.json',dict(steps=completed));print('C_PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
