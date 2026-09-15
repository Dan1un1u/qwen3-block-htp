"""EXP0271 provenance-sealed no-rotation FP32 residual device checks."""
import argparse,subprocess,shutil,shlex,statistics,json,struct,os
import numpy as np
from common_exp0279 import *
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
REMOTE='/data/local/tmp/qwen3-block-htp/exp0279'
def adb(*args,check=True):return subprocess.run([old.ADB,*args],check=check,capture_output=True,text=True,timeout=600)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def records(p):
 out=[]
 for line in Path(p).read_text().splitlines():
  try:out.append(json.loads(line))
  except ValueError:pass
 return out
def stage(count,trace=False):
 preflight();seal=read(S/'build/qwen3-sp2-build-seal.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();assert seal['source_head']==head;assert seal['paper_trace']==trace
 for n,h in seal['files'].items():assert sha(n)==h
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  c=(S/b/'CMakeCache.txt').read_text();assert ('QBH_PAPER_TRACE:BOOL='+('ON' if trace else 'OFF')) in c;assert 'QBH_MODEL_LLAMA32:BOOL=OFF' in c and f'QBH_EXP0257_LAYER_COUNT:STRING={count}\n' in c
 a=1
 while (R/'binaries'/f'l{count}-a{a}').exists():a+=1
 d=R/'binaries'/f'l{count}-a{a}';d.mkdir(parents=True,exist_ok=False);remote=REMOTE+f'-l{count}-a{a}';assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 shutil.copy2(S/'build/qwen3-sp2-build-seal.json',d/'build-seal.json');adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli')
 (R/f'runtime-l{count}.json').write_text(json.dumps(dict(remote=remote,seal=seal,archive=str(d)),indent=2)+'\n');print('STAGED',count,flush=True)

