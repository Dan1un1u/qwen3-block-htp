#!/usr/bin/env python3
"""Durable sequential precision experiment; preserve every stage result."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0014')
steps=[('collect',['llama32_smooth_accuracy.py','collect']),('fit',['llama32_smooth_accuracy.py','fit']),('carriers',['llama32_smooth_accuracy.py','carriers']),('rtn-selection',['llama32_smooth_accuracy.py','eval','--weights','rtn','--scope','selection']),('gptq-selection',['llama32_smooth_accuracy.py','eval','--scope','selection']),('input-bridge',['llama32_smooth_accuracy.py','eval']),('input-full',['llama32_smooth_accuracy.py','eval','--scope','full']),('residual-bridge',['llama32_smooth_accuracy.py','eval','--mode','residual_only']),('down-bridge',['llama32_smooth_accuracy.py','eval','--mode','down_only']),('carriers-bridge',['llama32_smooth_accuracy.py','eval','--mode','all']),('native-prepare',['llama32_smooth_native.py','prepare']),('native-bridge',['llama32_smooth_native.py','evaluate'])]
if __name__=='__main__':
    assert (OUT/'prepare.json').is_file()
    dest=OUT/'pipeline-a01';dest.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='8')
    for name,argv in steps:
        cmd=[sys.executable,'-u','-B',str(ROOT/'tools'/argv[0]),*argv[1:]]
        start=time.time();print('STAGE_START',name,flush=True)
        with (dest/(name+'.log')).open('x') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        record=dict(stage=name,command=cmd,started=start,seconds=time.time()-start,returncode=result.returncode,source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
        (dest/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');print('STAGE_DONE',name,result.returncode,flush=True)
        if result.returncode:sys.exit(result.returncode)
    (dest/'complete.json').write_text(json.dumps(dict(stages=len(steps),completed=time.time()))+'\n')
