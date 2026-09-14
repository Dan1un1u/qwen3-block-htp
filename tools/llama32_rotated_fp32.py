#!/usr/bin/env python3
"""L32-0019 independent dense-factor reference and conditional native audits."""
import argparse,json,shlex,shutil,subprocess,re
from pathlib import Path
import numpy as np
import llama32_fp32_residual as f
from llama_u8_reference import *
from probe_llama32_rotations import had,r4,err
from prototype_llama32_sp2 import preflight
from run_llama32_layer import adb,windows
from llama_reference import sha256
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0019');M=Path('/mnt/d/llm_exp/models/llama32-htp');ROOT=f.ROOT
CARRIER=64*40*64*2

def save(p,d):
 assert not p.exists(),p
 p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def quant(x,q):
 return np.floor(x.astype('f4')*np.float32(1/q['scale'])+np.float32(q['zero_point']+.5)).clip(0,255).astype('u1')
def rope(x,heads,q,cos,sin):
 x=(x.reshape(-1,heads,64).astype('f4')-np.float32(q['zero_point']))*np.float32(q['scale'])
 c=cos[:len(x)].astype('f4')[:,None,:];s=sin[:len(x)].astype('f4')[:,None,:]
 y=np.concatenate([-x[:,:,32:],x[:,:,:32]],2)
 return (x*c+y*s).astype('f2')
def layer(x,p,q,cos,sin,past=None,arm='off',native=None):
 cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');d={}
 def proj(a,n,iq,oq):return project_w4u8(a,p,n,{'q':2048,'k':512,'v':512,'gate':8192,'up':8192}[n],a.shape[1],q[iq],q[oq],cv)
 def raw(a,n,scale,zero=0):
  w=unpack_w4_codes(p,n,2048,a.shape[1]);ws=np.fromfile(p/(n+'_weight_w4_scale_f32.bin'),dtype='<f4');acc=f.oracle(a.astype('i4')-zero,w);assert np.abs(acc).max()<2**31
  return acc.astype('f4')*(np.float32(scale)*ws)
 a=f.norm(x,np.fromfile(p/'input_norm_weight_f16.bin',dtype='<f2'),q['input_norm']);qr=proj(a,'q','input_norm','q_projection');kr=proj(a,'k','input_norm','k_projection');v=proj(a,'v','input_norm','v')
 if arm in ['r3','both']:
  rp=[rope(qr,32,q['q_projection'],cos,sin),rope(kr,8,q['k_projection'],cos,sin)]
  d['r3_raw']=np.concatenate([z.transpose(1,0,2).reshape(-1,64) for z in rp]);d['r3_ideal']=(d['r3_raw'].astype('f8')@had(64)*.125).astype('f2')
  rotated=native['r3'] if native is not None and 'r3' in native else d['r3_ideal'];n=len(x)*32
  qr=quant(rotated[:n].reshape(32,len(x),64).transpose(1,0,2),q['q_rope']);kr=quant(rotated[n:].reshape(8,len(x),64).transpose(1,0,2),q['k_rope'])
 else:
  qr=exact_qk_norm_rope_u8(qr,32,q['q_projection'],q['q_rope'],None,cos,sin).reshape(-1,32,64);kr=exact_qk_norm_rope_u8(kr,8,q['k_projection'],q['k_rope'],None,cos,sin).reshape(-1,8,64)
 d['q']=qr;d['k']=kr;k=kr.transpose(1,0,2);v=v.reshape(-1,8,64).transpose(1,0,2);count=0
 if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
 av,_,_=exact_attention_dynamic(qr,k,v,count,f.configs(q),cv,f.divide);av=av.reshape(len(x),2048);o=raw(av,'o',q['attention_concat']['scale'],q['attention_concat']['zero_point']);res=x+o
 post=f.norm(res,np.fromfile(p/'post_norm_weight_f16.bin',dtype='<f2'),q['post_attention_norm']);g=proj(post,'gate','post_attention_norm','gate');u=proj(post,'up','post_attention_norm','up')
 if arm in ['r4','both']:
  lut=np.fromfile(p/'silu_up_lut_u16.bin',dtype='<u2');d['swiglu']=lut[:65536].view('f2').reshape(256,256)[g,u];s,z=r4(d['swiglu']);d['r4_stage1']=s;d['r4_ideal']=z
  z=native['r4'] if native is not None and 'r4' in native else z;mid=lut[65536:][z.view('u2')].astype('i4')-32768
 else:mid=f.table(str(p/'silu_up_lut_u16.bin'))[g,u]
 down=raw(mid,'down',q['middle']['scale']);d.update(middle=mid,down=down,residual=res,gate=g,up=u);return res+down,(k,v),d

def prepare(a):
 preflight();old=M/f'l32-0016/layer{a.layer}-a02';bm=f.verify(old);out=M/'l32-0019'/a.attempt;out.mkdir(parents=True,exist_ok=False);shutil.copytree(old,out,dirs_exist_ok=True);(out/'manifest.json').unlink()
 if a.arm in ['r4','both']:
  fresh=M/f'l32-0015/rotated-down-a01/layer{a.layer}';fm=json.loads((fresh/'manifest.json').read_text())
  for n,h in fm['files'].items():assert sha256(fresh/n)==h
  for n in ['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin','qparams_u8.bin']:shutil.copy2(fresh/n,out/'layer0'/n)
 q=load_qparams_bin(out/'layer0/qparams_u8.bin');past=None
 for step in range(2):
  rows=1 if step else 64
  name='replay_decode_input_00_f32.bin' if step else 'reference_w4u8_block_input_f32.bin';x=np.fromfile(out/name,dtype='<f4').reshape(64,2048)[:rows]
  name='replay_decode_rope_cos_00_f16.bin' if step else 'rope_cos_f16.bin';cos=np.fromfile(out/name,dtype='<f2').reshape(64,64);sin=np.fromfile(out/name.replace('cos','sin'),dtype='<f2').reshape(64,64)
  y,cache,d=layer(x,out/'layer0',q,cos,sin,past,a.arm)
  if not step:past=cache
  pad=np.zeros((64,2048),dtype='<f4');pad[:rows]=y;pad.tofile(out/('replay_decode_reference_00_f32.bin' if step else 'reference_w4u8_block_output_f32.bin'))
  for n,v in d.items():np.save(out/f'rotation_{step}_{n}.npy',v)
  for j,n in enumerate(['k','v']):
   ref=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:cache[j].shape[1]]=cache[j];ref.tofile(out/f'layer0/reference_kv_cache_{n}_u8.bin')
 save(out/'manifest.json',dict(experiment='L32-0019',source_layer=a.layer,arm=a.arm,baseline_manifest_sha256=sha256(old/'manifest.json'),rotated_down_manifest_sha256=sha256(fresh/'manifest.json') if a.arm in ['r4','both'] else None,reference='independent FP64 dense factors rounded FP16, FP32 tail; separate conditional hardware audit',files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in out.rglob('*') if p.is_file()}));print(out,flush=True)

def run(a):
 preflight();p=M/'l32-0019'/a.package;m=f.verify(p);assert m['arm']==a.arm
 head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());assert seal['source_head']==head
 d=R/a.attempt;d.mkdir(parents=True,exist_ok=False);remote='/data/local/tmp/llama32-htp/'+R.name+'/'+a.attempt
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,b in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  src=ROOT/b/'ship'/n;assert sha256(src)==seal['files'][str(src)];shutil.copy2(src,d/n);adb('push',windows(src),remote+'/'+n);assert adb('shell','sha256sum '+remote+'/'+n).stdout.split()[0]==sha256(src)
 adb('push',windows(p),remote+'/package');adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 names=list(m['files'])
 for i in range(0,len(names),32):
  for line in adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/package/'+n) for n in names[i:i+32])).stdout.splitlines():
   h,n=line.split(maxsplit=1);assert h==m['files'][n.removeprefix(remote+'/package/')]['sha256']
 old=json.loads((R.parent/f'l32-0018/layer0-a03/protocol.json').read_text())['command'];oldremote=old.split(' && ')[0].removeprefix('cd ');command=old.replace(oldremote,remote);prefix,argv=command.split(' ./qwen3_block_cli ',1);wd,e=prefix.split(' && ',1);env=dict(t.split('=',1) for t in shlex.split(e));r3=a.arm in ['r3','both'];r4on=a.arm in ['r4','both']
 env.update(QBH_DENSE_R3=str(int(r3)),QBH_R3_OPT='2' if r3 else'0',QBH_DENSE_R4=str(a.r4_mode if r4on else 0),QBH_R4_OPT='6' if r4on else'0')
 if r3:env['QBH_W4U8_DECODE_DIRECT_N_Q_BATCH_N_TILES']='32'
 if a.audit:
  if r3:env['QBH_DENSE_R3_AUDIT']='1'
  if r4on:env['QBH_DENSE_R4_AUDIT']='1'
 command=wd+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+argv
 save(d/'protocol.json',dict(experiment=R.name.upper(),source_head=head,build_seal=seal,package=str(p),package_manifest_sha256=sha256(p/'manifest.json'),command=command,arm=a.arm,audit=a.audit))
 z=adb('shell',command,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr)
 records=[]
 for line in z.stdout.splitlines():
  try:records.append(json.loads(re.sub(r':-?(?:nan|inf)([,}])',r':null\1',line)))
  except ValueError:pass
 steps=[];past=None;q=load_qparams_bin(p/'layer0/qparams_u8.bin')
 for step in range(2):
  rows=1 if step else 64;n=f'actual_replay_output_{step:02d}_f32.bin';adb('pull',remote+'/'+n,windows(d/n),check=False)
  if not(d/n).exists():continue
  actual=np.fromfile(d/n,dtype='<f4').reshape(64,2048)[:rows];ideal=np.fromfile(p/('replay_decode_reference_00_f32.bin' if step else 'reference_w4u8_block_output_f32.bin'),dtype='<f4').reshape(64,2048)[:rows]
  den=np.linalg.norm(actual.astype('f8'),axis=1)*np.linalg.norm(ideal.astype('f8'),axis=1);cs=np.sum(actual.astype('f8')*ideal,axis=1)/np.maximum(den,1e-300)
  row=dict(step=step,finite=bool(np.isfinite(actual).all()),ideal_min_row_cosine=float(cs.min()),ideal_nrmse=float(np.linalg.norm(actual-ideal)/max(np.linalg.norm(ideal),1e-30)),ideal_mismatches=int(np.count_nonzero(actual!=ideal)),pass_ideal_cosine=bool((cs[den>0]>=.999).all()))
  native={};captures={}
  if a.audit and r3:
   n=f'actual_replay_chain_{step:02d}.bin';adb('pull',remote+'/'+n,windows(d/n));raw=(d/n).read_bytes();captures['r3raw']=np.frombuffer(raw,dtype='<f2',count=rows*40*64).reshape(-1,64);native['r3']=np.frombuffer(raw,dtype='<f2',count=rows*40*64,offset=CARRIER).reshape(-1,64)
   for key,offset,heads in [('q',2*CARRIER,32),('k',2*CARRIER+64*2048,8)]:
    a8=np.frombuffer(raw,dtype='u1',count=heads*4096,offset=offset).reshape(heads,2,64,32);captures[key]=a8.transpose(2,0,1,3).reshape(64,heads,64)[:rows]
  if a.audit and r4on:
   n=f'actual_replay_r4_{step:02d}.bin';adb('pull',remote+'/'+n,windows(d/n));raw=np.fromfile(d/n,dtype='<f2');count=rows*8192
   captures['swiglu']=raw[:count].reshape(16,rows,512).transpose(1,0,2).reshape(rows,8192);captures['r4s']=raw[524288:524288+count].reshape(16,rows,512).transpose(1,0,2).reshape(rows,8192);native['r4']=raw[1048576:1048576+count].reshape(rows,8192)
  if a.audit:
   x=np.fromfile(p/('replay_decode_input_00_f32.bin' if step else 'reference_w4u8_block_input_f32.bin'),dtype='<f4').reshape(64,2048)[:rows];n='replay_decode_rope_cos_00_f16.bin' if step else'rope_cos_f16.bin';cos=np.fromfile(p/n,dtype='<f2').reshape(64,64);sin=np.fromfile(p/n.replace('cos','sin'),dtype='<f2').reshape(64,64)
   y,cache,diag=layer(x,p/'layer0',q,cos,sin,past,a.arm,native)
   if not step:past=cache
   row['conditional_tail_mismatches']=int(np.count_nonzero(y!=actual));row['conditional_max_abs']=float(np.max(np.abs(y-actual)));parts=[]
   if r3:
    row['r3_prepare']=err(captures['r3raw'],diag['r3_raw']);row['r3_matmul']=err(native['r3'],(captures['r3raw'].astype('f8')@had(64)*.125).astype('f2'));row['r3_quant_mismatches']=sum(int(np.count_nonzero(captures[k]!=diag[k])) for k in ['q','k']);parts+=['r3_prepare','r3_matmul']
   if r4on:
    row['swiglu_bits_equal']=bool(np.array_equal(captures['swiglu'].view('u2'),diag['swiglu'].view('u2')));row['r4_stage1']=err(captures['r4s'],r4(captures['swiglu'])[0]);row['r4_stage2']=err(native['r4'],(np.einsum('ag,tgc->tac',had(16),captures['r4s'].astype('f8').reshape(rows,16,512))*.25).astype('f2').reshape(rows,8192));parts+=['r4_stage1','r4_stage2']
   row['pass_actual_arithmetic']=row['conditional_tail_mismatches']==0 and row.get('r3_quant_mismatches',0)==0 and row.get('swiglu_bits_equal',True) and all(row[k]['finite'] and row[k]['outside_1ulp_plus_minnormal']==0 for k in parts)
  steps.append(row)
 save(d/'result.json',dict(process_exit=z.returncode,steps=steps,records=records,scope='conditional actual arithmetic and independent ideal cosine; repeat1 auxiliary'))
 print(json.dumps(dict(process_exit=z.returncode,steps=steps)),flush=True)
 if len(steps)!=2:print(z.stdout[-3000:]);print(z.stderr[-1500:]);raise SystemExit(1)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--layer',type=int,default=0);ap.add_argument('--arm',choices=['off','r3','r4','both'],required=True);ap.add_argument('--attempt',required=True);ap.add_argument('--package');ap.add_argument('--audit',action='store_true');ap.add_argument('--r4-mode',type=int,choices=[1,3],default=1);ap.add_argument('--result-experiment',default='l32-0019');a=ap.parse_args();assert a.result_experiment in ['l32-0019','l32-0020','l32-0021','l32-0022','l32-0023','l32-0024'];R=R.parent/a.result_experiment;globals()[a.action](a)
