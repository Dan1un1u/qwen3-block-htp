"""EXP0294 Down-only ordinary A8 completion, frozen non-Down contract."""
import os,sys,json,subprocess,shutil,struct,shlex
from pathlib import Path
import numpy as np
import common_exp0288 as c
import device_exp0288 as d
import full_exp0294 as f
import reference_exp0288 as ref
import reference_math_exp0288 as mathref
from export_llama32_u8 import write_qparams
from reference_w4u8_hmx import load_qparams_bin,unpack_u8_hmx_activation
S=c.S;OLD_R=c.R;OLD_O=c.O;R=OLD_R.parent/'exp0294';O=OLD_O.parent/'exp0294'
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'EXPERIMENT=EXP-0294' in z.splitlines()
for m in [c,d,f]:m.R=R;m.O=O;m.preflight=preflight
d.REMOTE='/data/local/tmp/qwen3-block-htp/exp0294'
d.package_path=lambda name:O/name
read=c.read;write=c.write;sha=c.sha
REV=os.environ.get('QBH_REFERENCE_REVISION','')
if REV:
 assert REV=='-sdk'
 import reference_qhl_exp0294 as sdk_reference
 mathref.exact_qk_norm_rope_u8=sdk_reference.exact_qk_norm_rope_u8
os.environ.setdefault('QBH_SP2','0');os.environ.setdefault('QBH_U8_PREFILL_OPT','3')
def clone(src,dst):
 mf=read(src/'manifest.json');dst.mkdir(parents=True,exist_ok=False)
 for n,v in mf['files'].items():
  p=src/n;assert sha(p)==v['sha256'],p
  target=dst/n;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
 return mf
def replace_bytes(path,data):
 # Break our new hardlink before replacing; immutable parent never written.
 if path.exists():path.unlink()
 path.write_bytes(data)
def manifest(dst,parent):
 write(dst/'manifest.json',dict(experiment='EXP-0294',parent=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),model='Qwen3-0.6B',layers=len(list(dst.glob('layer[0-9]*'))),recipe='W4A8',fp32_residual=2,rotation='OFF',files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dst.rglob('*') if p.is_file()}))
def change_middle(dst,source_layer):
 q=load_qparams_bin(dst/'qparams_u8.bin');old=q['middle'].copy()
 calibrated=read(OLD_O/'calibration.json')['qparams'][source_layer]['middle'];q['middle']=dict(calibrated)
 g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale']
 u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
 z=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:]
 encoded=np.clip(np.floor(z/q['middle']['scale']+q['middle']['zero_point']+.5),0,255).astype('<u2')
 replace_bytes(dst/'silu_up_lut_u16.bin',encoded.tobytes())
 (dst/'qparams_u8.bin').unlink();write_qparams(dst/'qparams_u8.bin',q)
 fresh=load_qparams_bin(dst/'qparams_u8.bin')
 return dict(source_layer=source_layer,old=old,new=fresh['middle'],lut_min=int(encoded.min()),lut_max=int(encoded.max()))
def init():
 preflight();R.mkdir(exist_ok=False);O.mkdir(exist_ok=False)
 # Verify sealed ledger identity and consumed packages; original results immutable.
 expected=read(Path(str(S)+'-project-memory')/'docs/PAPER_HARDWARE_MAIN_TABLE_20260918.json')
 write(R/'parent-ledger.json',dict(path=str(OLD_R/'EVIDENCE_SHA256.json'),sha256=sha(OLD_R/'EVIDENCE_SHA256.json')))
 shutil.copy2(OLD_R/'deployment-frontend64-a03.json',R/'deployment-frontend64-a03.json')
 shutil.copy2(OLD_R/'frontend64-a03-teacher.json',R/'frontend64-a03-teacher.json')
 print('INIT',flush=True)
def selected(first,count,name,parent):
 preflight();src=OLD_O/parent;dst=O/name;clone(src,dst);changes=[]
 for j in range(count):changes.append(change_middle(dst/f'layer{j}',first+j))
 x=np.fromfile(src/'reference_w4u8_block_input_f32.bin','<f4').reshape(64,1024).copy()
 dx=np.fromfile(src/'replay_decode_input_00_f32.bin','<f4').reshape(64,1024)[:1].copy()
 cos=np.fromfile(src/'rope_cos_f16.bin','<f2').reshape(64,128);sin=np.fromfile(src/'rope_sin_f16.bin','<f2').reshape(64,128)
 dc=np.fromfile(src/'replay_decode_rope_cos_00_f16.bin','<f2').reshape(64,128);ds=np.fromfile(src/'replay_decode_rope_sin_00_f16.bin','<f2').reshape(64,128)
 for j in range(count):
  pp=dst/f'layer{j}';q=load_qparams_bin(pp/'qparams_u8.bin')
  x,kv,_=mathref.layer(x,pp,cos,sin,sp2=False);dx,dv,_=mathref.layer(dx,pp,dc,ds,kv,sp2=False)
  # Native-cache helper writes outputs; detach linked cache files beforehand.
  for p in pp.glob('*hmx_u8_segmented*.bin'):p.unlink()
  ref.native_cache(pp,kv,q)
  for k,n in enumerate(['k','v']):
   v=np.full((8,128,128),q['k_rope' if n=='k' else 'v']['zero_point'],'u1');v[:,:65]=dv[k]
   replace_bytes(pp/f'reference_kv_cache_{n}_u8.bin',v.tobytes())
 for n,v in [('reference_w4u8_block_output_f32.bin',x),('replay_decode_reference_00_f32.bin',dx)]:
  padded=np.zeros((64,1024),'<f4');padded[:len(v)]=v;replace_bytes(dst/n,padded.tobytes())
 manifest(dst,src);write(R/(name+'-changes.json'),changes);print('PREPARED',name,flush=True)
def full_package(mode):
 preflight();src=OLD_O/'frontend64-a03';dst=O/('full-a8-'+mode+REV);clone(src,dst)
 changes=[change_middle(dst/f'layer{i}',i) for i in range(28)]
 ids=np.fromfile(dst/'generation_prompt_token_ids_u32.bin','<u4')
 embed=np.memmap(dst/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(151936,1024))
 g=np.fromfile(dst/'generation_final_norm_weight_f16.bin','<f2');q=load_qparams_bin(dst/'generation_qparams_u8.bin')
 fixed=read(R/'frontend64-a03-teacher.json')['u8_generated_ids'];tokens=[];codes=[];caches=[None]*28
 for step in range(43):
  token=(fixed[step-1] if mode=='fixed' else tokens[-1]) if step else None
  x=np.array(embed[ids if not step else [token]],dtype='f4')
  if not step:
   co=np.fromfile(dst/'rope_cos_f16.bin','<f2').reshape(64,128);si=np.fromfile(dst/'rope_sin_f16.bin','<f2').reshape(64,128)
  else:
   co=np.fromfile(dst/f'generation_decode_rope_cos_{step-1:02d}_f16.bin','<f2').reshape(64,128);si=np.fromfile(dst/f'generation_decode_rope_sin_{step-1:02d}_f16.bin','<f2').reshape(64,128)
  for i in range(28):
   pp=dst/f'layer{i}';x,caches[i],_=mathref.layer(x,pp,co,si,caches[i],sp2=False)
   if not step:
    qq=load_qparams_bin(pp/'qparams_u8.bin')
    for p in pp.glob('*hmx_u8_segmented*.bin'):p.unlink()
    ref.native_cache(pp,caches[i],qq)
    for j,n in enumerate(['k','v']):
     v=np.full((8,128,128),qq['k_rope' if n=='k' else 'v']['zero_point'],'u1');v[:,:64]=caches[i][j];replace_bytes(pp/f'reference_kv_cache_{n}_u8.bin',v.tobytes())
  replace_bytes(dst/f'audit_hidden_{step:02d}_f32.bin',np.asarray(x,'<f4').tobytes())
  if not step:replace_bytes(dst/'reference_w4u8_block_output_f32.bin',np.asarray(x,'<f4').tobytes())
  a=mathref.norm(x[-1:],g,q['generation_final_norm_output']);logits=mathref.project(a,dst,'generation_lm_head',151936,q['generation_final_norm_output'],q['generation_lm_head_output'])[0]
  tok=int(logits.argmax());tokens.append(tok);codes.append(int(logits[tok]));print('REFERENCE',mode,step,tok,flush=True)
 replace_bytes(dst/'generation_expected_token_ids_u32.bin',np.asarray(tokens+[0]*21,'<u4').tobytes())
 manifest(dst,src);write(R/('a8-'+mode+REV+'-teacher.json'),dict(u8_generated_ids=tokens,u8_selected_codes=codes,prompt_ids=ids.tolist(),quality_accepted=False));write(R/('a8-'+mode+REV+'-changes.json'),changes)
def stage(count):d.stage(count)
def deploy(name):d.deploy(name)
def layer_run(name,tag,count):
 d.run(name,tag,count=count,fp32=2,dump=True)
def full_run(arm,tag,repeat=1,audit=False,greedy=False):
 os.environ['QBH_SP2']='8' if arm=='SP2' else '0';os.environ['QBH_U8_PREFILL_OPT']='0' if arm=='SP2' else '3'
 os.environ['QBH_PACKAGE']='frontend64-a03' if arm=='SP2' else 'full-a8-'+('greedy' if greedy else 'fixed')+REV
 os.environ['QBH_PAPER_FIXED_TOKENS']='0' if greedy else '1'
 f.O=OLD_O if arm=='SP2' else O
 z=f.full(2,repeat,tag,audit=audit,steps=43)
 teacher=read(R/('frontend64-a03-teacher.json' if arm=='SP2' else 'a8-'+('greedy' if greedy else 'fixed')+REV+'-teacher.json'))
 assert z['selected_codes']==[list(v) for v in zip(teacher['u8_generated_ids'][:43],teacher['u8_selected_codes'][:43])]
 if audit:
  p=(OLD_O/'frontend64-a03') if arm=='SP2' else O/os.environ['QBH_PACKAGE'];root=R/tag
  q=load_qparams_bin(p/'generation_qparams_u8.bin');g=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2')
  for step in range(43):
   a=np.fromfile(root/f'generation_hidden_step{step:02d}_f32.bin','<f4');b=np.fromfile(p/f'audit_hidden_{step:02d}_f32.bin','<f4').reshape(-1,1024)[-1]
   assert np.array_equal(a,b),(arm,step,'hidden')
   a=unpack_u8_hmx_activation(np.fromfile(root/f'generation_norm_step{step:02d}_u8_native.bin','u1'),1024)[:1];assert np.array_equal(a,mathref.norm(b[None],g,q['generation_final_norm_output'])),(arm,step,'norm')
  for i in range(28):
   for n in ['k','v']:assert (root/f'generation_prefill_layer{i:02d}_{n}_cache_u8.bin').read_bytes()==(p/f'layer{i}/reference_kv_cache_{n}_hmx_u8_segmented_step00.bin').read_bytes(),(i,n)
  write(root/'independent_gate.json',dict(pass_all=True,steps=43,hidden_norm_kv_ids_codes_exact=True))
 return dict(arm=arm,tag=tag,**z)
def campaign():
 for tag in ['audit-SP2','audit-A8-sdk','greedy-A8-sdk']:assert read(R/tag/'independent_gate.json')['pass_all']
 write(R/'auxiliary.json',[full_run(a,'aux-'+a) for a in ['SP2','A8']])
 for phase,n in [('short',5),('formal',10)]:
  rows=[]
  for i in range(n):
   for arm in (['SP2','A8'] if i%2==0 else ['A8','SP2']):rows.append(dict(cycle=i,**full_run(arm,f'{phase}/{i:02d}-{arm}',10)))
   write(R/f'{phase}-{i:02d}.json',rows[-2:])
  write(R/(phase+'.json'),dict(pass_all=True,rounds=n,repeat=10,runs=rows))
if __name__=='__main__':
 a=sys.argv
 if a[1]=='init':init()
 elif a[1]=='prepare':
  for v in [(0,1,'a8-layer0','l0-a02'),(14,1,'a8-layer14','l14-a01'),(27,1,'a8-layer27','l27-a01'),(0,3,'a8-chain3','chain3-a01')]:selected(*v)
 elif a[1]=='frontend':full_package(a[2])
 elif a[1]=='stage':stage(int(a[2]))
 elif a[1]=='deploy':deploy(a[2])
 elif a[1]=='layer':layer_run(a[2],a[3],int(a[4]))
 elif a[1]=='full':full_run(a[2],a[3],audit=True,greedy='greedy' in a)
 elif a[1]=='campaign':campaign()
