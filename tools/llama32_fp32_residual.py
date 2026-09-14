#!/usr/bin/env python3
"""L32-0016 immutable fixed-weight FP32 residual reference and device replay."""
import argparse,json,os,shlex,shutil,subprocess,re
from pathlib import Path
import numpy as np
from export_llama32_u8 import configs,divide,ROOT
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin,project_w4u8,exact_qk_norm_rope_u8,exact_attention_dynamic,HmxU8Converter,unpack_w4_codes
from llama_sp2_reference import table
from prototype_llama32_sp2 import oracle,preflight
from run_llama32_layer import adb,windows
M=Path('/mnt/d/llm_exp/models/llama32-htp');R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0016')
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(p):
 m=json.loads((p/'manifest.json').read_text())
 for n,v in m['files'].items():assert sha256(p/n)==v['sha256'],n
 return m

def norm(x,gamma,q):
 x=np.asarray(x,dtype='f4');s=np.cumsum(x*x,axis=1,dtype='f4')[:,-1:]
 inv=np.float32(1)/np.sqrt(s/np.float32(x.shape[1])+np.float32(1e-5))
 v=(x*inv)*gamma.astype('f4')
 code=v/np.float32(q['scale'])+np.float32(q['zero_point'])
 return np.clip(np.copysign(np.floor(np.abs(code)+np.float32(.5)),code),0,255).astype('u1')
def layer(x,p,q,cos,sin,past=None):
 cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
 def project(a,name,iq,oq):
  n={'q':2048,'k':512,'v':512,'gate':8192,'up':8192}[name]
  return project_w4u8(a,p,name,n,a.shape[1],q[iq],q[oq],cv)
 def raw(a,name,scale,zero=0):
  w=unpack_w4_codes(p,name,2048,a.shape[1]);ws=np.fromfile(p/(name+'_weight_w4_scale_f32.bin'),dtype='<f4')
  acc=oracle(a.astype('i4')-zero,w)
  assert np.max(np.abs(acc))<2**31
  return acc.astype('f4')*(np.float32(scale)*ws)
 a=norm(x,np.fromfile(p/'input_norm_weight_f16.bin',dtype='<f2'),q['input_norm'])
 qr=project(a,'q','input_norm','q_projection');kr=project(a,'k','input_norm','k_projection');v=project(a,'v','input_norm','v')
 qr=exact_qk_norm_rope_u8(qr,32,q['q_projection'],q['q_rope'],None,cos,sin).reshape(-1,32,64)
 kr=exact_qk_norm_rope_u8(kr,8,q['k_projection'],q['k_rope'],None,cos,sin).reshape(-1,8,64)
 k=kr.transpose(1,0,2);v=v.reshape(-1,8,64).transpose(1,0,2);count=0
 if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
 av,score,prob=exact_attention_dynamic(qr,k,v,count,configs(q),cv,divide);av=av.reshape(len(x),2048)
 o=raw(av,'o',q['attention_concat']['scale'],q['attention_concat']['zero_point']);res=x+o
 post=norm(res,np.fromfile(p/'post_norm_weight_f16.bin',dtype='<f2'),q['post_attention_norm'])
 g=project(post,'gate','post_attention_norm','gate');u=project(post,'up','post_attention_norm','up')
 mid=table(str(p/'silu_up_lut_u16.bin'))[g,u];down=raw(mid,'down',q['middle']['scale']);y=res+down
 return y,(k,v),dict(input_norm=a,q=qr,k=kr,attention=av,o=o,residual=res,post=post,gate=g,up=u,middle=mid,down=down,score=score,probability=prob)

def prepare(a):
 preflight();old=M/f'l32-0009/packages-a01/layer{a.layer}-sp2';verify(old)
 out=M/'l32-0016'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 # Fresh copies for references; preserve frozen package hashes, never edit hard links.
 shutil.copytree(old,out,dirs_exist_ok=True);(out/'manifest.json').unlink()
 q=load_qparams_bin(out/'layer0/qparams_u8.bin');past=None
 for step,phase in enumerate(['prefill','decode']):
  rows=64 if step==0 else 1;src=M/f'l32-0002/layers-a02/layer{a.layer}-{phase}';verify(src)
  x=np.fromfile(src/'block_input_f16.bin',dtype='<f2').reshape(64,2048)[:rows].astype('f4')
  cos=np.fromfile(src/'rope_cos_f16.bin',dtype='<f2').reshape(64,64);sin=np.fromfile(src/'rope_sin_f16.bin',dtype='<f2').reshape(64,64)
  y,cache,diag=layer(x,out/'layer0',q,cos,sin,past)
  for name,v in [('reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin',x),('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin',y)]:
   padded=np.zeros((64,2048),dtype='<f4');padded[:rows]=v;padded.tofile(out/name)
  for name,v in diag.items():np.save(out/f'fp32_{phase}_{name}.npy',v)
  if step==0:past=cache
  for j,n in enumerate(['k','v']):
   ref=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:cache[j].shape[1]]=cache[j];ref.tofile(out/f'layer0/reference_kv_cache_{n}_u8.bin')
 save(out/'manifest.json',dict(experiment='L32-0016',source_layer=a.layer,baseline=str(old),baseline_sha256=sha256(old/'manifest.json'),contract='fixed W4/SP2/scales; FP32 O,Down,hidden,norm; native integer attention and unchanged A8 head',files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()}));print(out,flush=True)

def run(a):
 preflight();p=M/('l32-0016/'+a.package if a.arm=='fp32' else f'l32-0009/packages-a01/layer{a.layer}-sp2');m=verify(p)
 head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());assert seal['source_head']==head
 assert f'QBH_LLAMA_LAYER_COUNT:STRING={a.layers}' in (ROOT/'android_ReleaseG_aarch64/CMakeCache.txt').read_text()
 d=R/a.attempt;d.mkdir(parents=True,exist_ok=False);remote='/data/local/tmp/llama32-htp/l32-0016/'+a.attempt
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,b in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  f=ROOT/b/'ship'/n;assert sha256(f)==seal['files'][str(f)];shutil.copy2(f,d/n);adb('push',windows(f),remote+'/'+n);assert adb('shell','sha256sum '+remote+'/'+n).stdout.split()[0]==sha256(f)
 adb('push',windows(p),remote+'/package');adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 for start in range(0,len(m['files']),32):
  names=list(m['files'])[start:start+32];checks=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/package/'+n) for n in names)).stdout
  for line in checks.splitlines():h,n=line.split(maxsplit=1);assert h==m['files'][n.removeprefix(remote+'/package/')]['sha256']
 old=json.loads((R.parent/f'l32-0012/layer{a.layer}-m8-a01/protocol.json').read_text())['command'];oldremote=old.split(' && ')[0].removeprefix('cd ');command=old.replace(oldremote,remote)
 command=command.replace(' ./qwen3_block_cli ',f' QBH_LLAMA_FP32_RESIDUAL={int(a.arm=="fp32")} ./qwen3_block_cli ')
 save(d/'protocol.json',dict(experiment='L32-0016',source_head=head,seal=seal,package=str(p),package_manifest_sha256=sha256(p/'manifest.json'),command=command,arm=a.arm))
 run=adb('shell',command,check=False);(d/'stdout.txt').write_text(run.stdout);(d/'stderr.txt').write_text(run.stderr)
 records=[]
 for line in run.stdout.splitlines():
  try:records.append(json.loads(re.sub(r':-?(?:nan|inf)([,}])',r':null\1',line)))
  except ValueError:pass
 steps=[];suffix='f32' if a.arm=='fp32' else 'u8'
 for step in range(2):
  name=f'actual_replay_output_{step:02d}_{suffix}.bin';adb('pull',remote+'/'+name,windows(d/name),check=False)
  if not (d/name).exists():continue
  rows=64 if step==0 else 1;dtype='<f4' if a.arm=='fp32' else 'u1'
  ref=('reference_w4u8_block_output_f32.bin' if a.arm=='fp32' else 'reference_w4u8_integer_attention_block_output_u8.bin') if step==0 else f'replay_decode_reference_00_{suffix}.bin'
  x=np.fromfile(d/name,dtype=dtype).reshape(64,2048)[:rows].astype('f8');y=np.fromfile(p/ref,dtype=dtype).reshape(64,2048)[:rows].astype('f8');delta=x-y;den=float(np.linalg.norm(x)*np.linalg.norm(y))
  steps.append(dict(step=step,finite=bool(np.isfinite(x).all()),max_abs=float(np.abs(delta).max()),nrmse=float(np.linalg.norm(delta)/max(np.linalg.norm(y),1e-30)),cosine=float(np.sum(x*y)/den) if den else None,mismatches=int(np.count_nonzero(delta))))
 save(d/'result.json',dict(process_exit=run.returncode,steps=steps,records=records,scope='functional, repeat1 auxiliary; no performance acceptance'))
 print(json.dumps(dict(process_exit=run.returncode,steps=steps)),flush=True)
 if len(steps)!=2:print(run.stdout[-3500:]);print(run.stderr[-1500:]);raise SystemExit(1)
def chain(a):
 preflight();old=M/'l32-0010/frontend-a01';om=verify(old)
 out=M/'l32-0016'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 for name in om['files']:
  rel=Path(name)
  if rel.parts[0].startswith('layer'):
   if int(rel.parts[0][5:])>=a.layers:continue
  elif name not in ['rope_cos_f16.bin','rope_sin_f16.bin']:continue
  dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,dst)
 past=[None]*a.layers
 for step,phase in enumerate(['prefill','decode']):
  rows=64 if step==0 else 1;src=M/f'l32-0002/layers-a02/layer0-{phase}';verify(src)
  x=np.fromfile(src/'block_input_f16.bin',dtype='<f2').reshape(64,2048)[:rows].astype('f4')
  padded=np.zeros((64,2048),dtype='<f4');padded[:rows]=x;padded.tofile(out/('reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin'))
  cos=np.fromfile(src/'rope_cos_f16.bin',dtype='<f2').reshape(64,64);sin=np.fromfile(src/'rope_sin_f16.bin',dtype='<f2').reshape(64,64)
  if step:
   shutil.copy2(src/'rope_cos_f16.bin',out/'replay_decode_rope_cos_00_f16.bin');shutil.copy2(src/'rope_sin_f16.bin',out/'replay_decode_rope_sin_00_f16.bin')
  for index in range(a.layers):
   root=out/f'layer{index}';q=load_qparams_bin(root/'qparams_u8.bin');x,cache,diag=layer(x,root,q,cos,sin,past[index])
   if not step:past[index]=cache
   np.save(out/f'fp32_{phase}_hidden{index}.npy',x)
   for j,n in enumerate(['k','v']):
    ref=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:cache[j].shape[1]]=cache[j];ref.tofile(root/f'reference_kv_cache_{n}_u8.bin')
   print(phase,index,flush=True)
  padded=np.zeros((64,2048),dtype='<f4');padded[:rows]=x;padded.tofile(out/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'))
 save(out/'manifest.json',dict(experiment='L32-0016',layers=a.layers,baseline=str(old),baseline_sha256=sha256(old/'manifest.json'),files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()}));print(out,flush=True)
def frontend(a):
 preflight();old=M/'l32-0010/frontend-a01';om=verify(old)
 original=M/'l32-0002/frontend-a01';verify(original)
 out=M/'l32-0016'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 for name in om['files']:
  if Path(name).name.startswith('reference_') or name in ['generation_embedding_weight_u8.bin','generation_expected_token_ids_u32.bin']:continue
  dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,dst)
 os.link(original/'generation_embedding_weight_f16.bin',out/'generation_embedding_weight_f16.bin')
 embed=np.memmap(out/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,2048));ids=np.fromfile(out/'generation_prompt_token_ids_u32.bin',dtype='<u4').tolist()
 qs=[load_qparams_bin(out/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(out/'generation_qparams_u8.bin');gamma=np.fromfile(out/'generation_final_norm_weight_f16.bin',dtype='<f2');cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');caches=[None]*16;tokens=[];codes=[]
 for step in range(16):
  x=np.array(embed[ids if not step else [tokens[-1]]],dtype='f4')
  if not step:x.tofile(out/'reference_w4u8_block_input_f32.bin')
  name='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';cos=np.fromfile(out/name,dtype='<f2').reshape(64,64);sin=np.fromfile(out/name.replace('cos','sin'),dtype='<f2').reshape(64,64)
  for index in range(16):
   x,caches[index],_=layer(x,out/f'layer{index}',qs[index],cos,sin,caches[index])
   if not step:
    for j,n in enumerate(['k','v']):
     ref=np.full((8,80,64),qs[index]['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:64]=caches[index][j];ref.tofile(out/f'layer{index}/reference_kv_cache_{n}_u8.bin')
  if not step:x.tofile(out/'reference_w4u8_block_output_f32.bin')
  act=norm(x[-1:],gamma,gq['generation_final_norm_output']);logits=project_w4u8(act,out,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0]
  token=int(logits.argmax());tokens.append(token);codes.append(int(logits[token]));print('FRONTEND_ORACLE',step,token,codes[-1],flush=True)
 np.array(tokens,dtype='<u4').tofile(out/'generation_expected_token_ids_u32.bin')
 reference=R/(a.attempt+'-reference');reference.mkdir(parents=True,exist_ok=False)
 prior=R.parent/'l32-0003/frontend-reference-a01'
 for name,h in json.loads((prior/'freeze.json').read_text()).items():assert sha256(prior/name)==h
 for name in ['dataset.json','heldout.bin']:os.link(prior/name,reference/name)
 teacher=json.loads((prior/'teacher.json').read_text());teacher.update(u8_generated_ids=tokens,u8_selected_codes=codes,fp32_residual=True,weights='frozen L32-0012 SP2 mode8')
 save(reference/'teacher.json',teacher);save(reference/'freeze.json',{f.name:sha256(f) for f in reference.iterdir() if f.is_file()})
 save(out/'manifest.json',dict(experiment='L32-0016',recipe='W4A8',layers=16,original=om['original'],fp32_residual=True,rotation='OFF',baseline_manifest_sha256=sha256(old/'manifest.json'),original_f16_embedding_manifest_sha256=sha256(original/'manifest.json'),frontend_teacher_sha256=sha256(reference/'teacher.json'),files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()}));print(out,reference,flush=True)
def basefront(a):
 preflight();old=M/'l32-0010/frontend-a01';om=verify(old)
 out=M/'l32-0016'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 for name in om['files']:
  dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,dst)
 reference=R/(a.attempt+'-reference');reference.mkdir(parents=True,exist_ok=False);prior=R.parent/'l32-0003/frontend-reference-a01'
 for name,h in json.loads((prior/'freeze.json').read_text()).items():assert sha256(prior/name)==h
 for name in ['dataset.json','heldout.bin']:os.link(prior/name,reference/name)
 oracle=R.parent/'l32-0010/sp2-oracle.json';assert sha256(oracle)==om['frontend_teacher_sha256']
 teacher=json.loads((prior/'teacher.json').read_text());teacher.update(json.loads(oracle.read_text()));save(reference/'teacher.json',teacher);save(reference/'freeze.json',{f.name:sha256(f) for f in reference.iterdir() if f.is_file()})
 om.update(experiment='L32-0016',baseline_manifest_sha256=sha256(old/'manifest.json'),frontend_teacher_sha256=sha256(reference/'teacher.json'))
 save(out/'manifest.json',om);print(out,reference,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','chain','frontend','basefront','run']);ap.add_argument('--layer',type=int,default=0);ap.add_argument('--layers',type=int,default=1);ap.add_argument('--attempt',required=True);ap.add_argument('--arm',choices=['base','fp32'],default='fp32');ap.add_argument('--package');a=ap.parse_args();globals()[a.action](a)
