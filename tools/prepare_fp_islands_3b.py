#!/usr/bin/env python3
"""Restore short-context INT16 contract without changing any weight codes/scales."""
import json,os,shutil,shlex
from pathlib import Path
import numpy as np
from profile_fp_islands_3b import R,P,REMOTE,save,read,preflight,adb,windows
from llama_reference import sha256

def main():
 preflight();parent=P.parent/'recovered-sp2';old=read(parent/'manifest.json');recovery=read(R/'package-recovery.json')
 src=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0058/int16');im=read(src/'manifest.json')
 assert sha256(src/'manifest.json')=='371e47910744a9b5e91b8d91d318e6a082feb3277a27dbc2a320d61b30263ab1'
 P.mkdir(exist_ok=False);files={};changes=[]
 for f in parent.rglob('*'):
  if not f.is_file() or f.name=='manifest.json':continue
  n=str(f.relative_to(parent));expected=old['files'][n]['sha256'];assert sha256(f)==expected,n
  out=P/n;out.parent.mkdir(parents=True,exist_ok=True)
  replace=n.startswith('layer') and n.endswith(('/qparams_u8.bin','/silu_up_lut_u16.bin'))
  if replace:
   assert sha256(src/n)==im['files'][n]['sha256'],n
   shutil.copy2(src/n,out);changes.append(n)
  else:
   if 'weight_' in n and n in im['files']:assert expected==im['files'][n]['sha256'],n
   os.link(f,out)
  files[n]=dict(bytes=out.stat().st_size,sha256=sha256(out))
 assert len(changes)==56
 save(P/'manifest.json',dict(experiment='L32-0064',model='Llama3.2-3B',layers=28,recipe='W4A8-INT16',parent=str(parent),parent_manifest_sha256=sha256(parent/'manifest.json'),int16_metadata_manifest_sha256=sha256(src/'manifest.json'),changed_files=changes,files=files))
 base=read('/mnt/d/llm_exp/results/llama32-htp/l32-0058/base.json')['parent'];save(R/'base.json',base)
 prior=read('/mnt/d/llm_exp/results/llama32-htp/l32-0058/fixture.json')
 ids=np.fromfile(P/'generation_prompt_token_ids_u32.bin','<u4').tolist();assert len(ids)==64 and ids==prior['prompt_ids'][:64]
 save(R/'fixture.json',dict(prompt_ids=ids,fixed=prior['fixed'][:2],source='L32-0058 prefix and forced continuation; short-context RoPE from L32-0044'))
 remote=REMOTE+'/int16';assert adb('shell','test ! -e '+remote,check=False).returncode==0
 adb('shell','mkdir -p '+remote)
 commands=[]
 for n in files:
  dst=remote+'/'+n;commands.append('mkdir -p '+shlex.quote(str(Path(dst).parent)))
  if n not in changes:commands.append('ln -s '+shlex.quote(base['remote']+'/'+n)+' '+shlex.quote(dst))
 script=R/'deploy.sh';script.write_text('set -e\n'+'\n'.join(commands)+'\n');adb('push',windows(script),REMOTE+'/deploy.sh');adb('shell','sh '+REMOTE+'/deploy.sh')
 for n in changes:adb('push',windows(P/n),remote+'/'+n)
 adb('push',windows(P/'manifest.json'),remote+'/manifest.json')
 # Device independently verifies every payload, including symlink targets.
 checks=R/'device-checks.sha256';checks.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in files.items()));adb('push',windows(checks),remote+'/checks.sha256')
 z=adb('shell','cd '+remote+' && sha256sum -c checks.sha256');(R/'device-checks.txt').write_text(z.stdout)
 print('PACKAGE VERIFIED',len(files),flush=True)
if __name__=='__main__':main()
