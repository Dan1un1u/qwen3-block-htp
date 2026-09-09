from pathlib import Path
import sys,json,hashlib,numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from audit_exp0261 import CAP,BASE
from reference_w4u8_hmx import unpack_u8_hmx_activation
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0264');old=R.parent/'exp0263'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old/'EVIDENCE_SHA256.json')=='1404333eb0e87d0442d86c05be20df1e66d35c332373c58f6640c3ab10409b09'
seal=json.loads((old/'EVIDENCE_SHA256.json').read_text())['files']
def check(tag,ref):
 checked=[]
 for step in range(9):
  rows=64 if step==0 else 1
  for name in [f'step{step:02d}_r4.bin',f'step{step:02d}_r3.bin',f'step{step:02d}_output.bin']:
   a=R/tag/name;b=ref/name
   if str(b).startswith(str(old)+'/'):assert sha(b)==seal[str(b.relative_to(old))]['sha256']
   if name.endswith('_r4.bin'):
    x=np.fromfile(a,'u2').reshape(3,CAP)[:,:rows*6144];y=np.fromfile(b,'u2').reshape(3,CAP)[:,:rows*6144]
    assert np.array_equal(x,y),(tag,name,'R4',int(np.sum(x!=y)))
   elif name.endswith('_r3.bin'):
    x=np.fromfile(a,'u1');y=np.fromfile(b,'u1')
    # R3 component audit plus slots0..4 are full captured earlier data.
    assert np.array_equal(x[:BASE+5*CAP],y[:BASE+5*CAP]),(tag,name,'beforeMLP')
    for j,k in [(5,2048),(6,6144),(7,6144),(8,6144),(9,2048)]:
     xx=unpack_u8_hmx_activation(x[BASE+j*CAP:BASE+j*CAP+64*k],k)[:rows];yy=unpack_u8_hmx_activation(y[BASE+j*CAP:BASE+j*CAP+64*k],k)[:rows]
     assert np.array_equal(xx,yy),(tag,name,j,int(np.sum(xx!=yy)))
   else:assert a.read_bytes()==b.read_bytes(),(tag,name)
   checked.append(name)
 for b in ref.glob('*cache.bin'):
  a=R/tag/b.name;assert a.read_bytes()==b.read_bytes(),b.name;checked.append(b.name)
 print('EXACT',tag,len(checked),flush=True);return dict(tag=tag,reference=str(ref),comparisons=checked,pass_all=True)
def main():
 import audit_exp0261 as original
 from export_exp0261 import preflight
 preflight()
 results=[check(tag,old/'audit_a1') for tag in ['audit_previous','audit_constants','audit_a1']]
 (R/'exact_native.json').write_text(json.dumps(dict(pass_all=True,results=results),indent=2))
 original.R=R;saved=original.capture
 def capture(tag,step):
  original.R=R.parent/'exp0261' if tag=='audit_a2' else R
  value=saved(tag,step);original.R=R;return value
 original.capture=capture
 original.main()
 gate=json.loads((R/'numerical_gate.json').read_text());gate['exact_native_optimization']=True
 gate['parent_numerical_seal']='1404333eb0e87d0442d86c05be20df1e66d35c332373c58f6640c3ab10409b09'
 gate['exact_file_comparisons']=sum(len(z['comparisons']) for z in results)
 (R/'numerical_gate.json').write_text(json.dumps(gate,indent=2))
if __name__=='__main__':main()
