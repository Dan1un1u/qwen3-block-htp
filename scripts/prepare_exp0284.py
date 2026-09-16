"""Restore only original ordinary-A8 middle contract; never requantize weights."""
import sys,os,json,hashlib,subprocess
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/qwen3-block-htp');sys.path.insert(0,str(S/'scripts'))
from reference_exp0277 import layer
from reference_w4u8_hmx import load_qparams_bin
M=Path('/mnt/d/llm_exp/models/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0284');O=M/'exp0284'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2)
def verify(p,expected):
 assert sha(p/'manifest.json')==expected
 m=read(p/'manifest.json')
 for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
 return m
def main():
 out=subprocess.check_output(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'EXPERIMENT=EXP-0284' in out
 base=M/'exp0257/package';bm=verify(base,'ab2442027f315797dc7ff8f9c49b0d6bf27a3de3c958e831eddf4e1953f915bb')
 O.mkdir(exist_ok=False);R.mkdir(exist_ok=True)
 for key,src,count,origlayer in [('layer14-int16',M/'exp0269/layer14-fp32-a01',1,14),('full-int16',M/'exp0271/sp2-fp32',28,0)]:
  cfg=read(R.parent/'exp0282/A5-fair-a02'/('deployment-layer14-fp32-a01.json' if count==1 else 'deployment-sp2-fp32.json'));m=verify(src,cfg['manifest_sha256']);dst=O/key;dst.mkdir();changed=[]
  for n,v in m['files'].items():
   name=Path(n).name
   if name.endswith('.npy') or name in ['reference_w4u8_block_output_f32.bin','replay_decode_reference_00_f32.bin']:continue
   source=src/n
   if name in ['qparams_u8.bin','silu_up_lut_u16.bin']:
    changed.append(n);continue
   target=dst/n;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
  from uniform_int16_down import make
  audits={}
  for root in sorted(src.glob('layer*')):
   if root.is_dir() and (root/'qparams_u8.bin').exists():audits[root.name]=make(root,dst/root.name)
  write(R/(key+'-lut-audit.json'),audits)
  past=[None]*(1 if count==1 else 3)
  for step in range(2):
   inp='reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin';x=np.fromfile(dst/inp,'<f4').reshape(64,2048)[:64 if not step else 1]
   cos=np.fromfile(dst/('rope_cos_f16.bin' if not step else 'replay_decode_rope_cos_00_f16.bin'),'<f2').reshape(64,128);sin=np.fromfile(dst/('rope_sin_f16.bin' if not step else 'replay_decode_rope_sin_00_f16.bin'),'<f2').reshape(64,128)
   for i in range(len(past)):
    x,past[i],d=layer(x,dst/f'layer{i}',cos,sin,past[i],sp2=True)
    if count>1:np.save(dst/f'chain_step{step}_layer{i}_output.npy',x)
    print('REFERENCE',key,step,i,flush=True)
   pad=np.zeros((64,2048),'<f4');pad[:len(x)]=x;pad.tofile(dst/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'))
   prefix=f'fp32_step{step:02d}_' if count==1 else f'chain_step{step}_last_'
   for name,v in d.items():np.save(dst/(prefix+name+'.npy'),v)
   for name,v in zip(['cache_k','cache_v'],past[-1]):np.save(dst/(prefix+name+'.npy'),v)
  files={str(p.relative_to(dst)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dst.rglob('*')) if p.is_file()}
  for n,v in files.items():
   if 'weight' in Path(n).name:assert v['sha256']==m['files'][n]['sha256'],n
  write(dst/'manifest.json',dict(experiment='EXP-0284',recipe='nativeW4 uniformINT16 Down FP32residual2',parent=str(src),parent_sha256=sha(src/'manifest.json'),ordinary_middle_parent=str(base),ordinary_parent_sha256=sha(base/'manifest.json'),changed_metadata=changed,new_weight_codes=False,reference_scope='selectedlayer14' if count==1 else 'first3 layers only; full frontend separately checked',files=files))
  write(R/(key+'-package.json'),dict(path=str(dst),manifest_sha256=sha(dst/'manifest.json'),files=len(files)))
  print('PREPARED',key,flush=True)
if __name__=='__main__':main()
