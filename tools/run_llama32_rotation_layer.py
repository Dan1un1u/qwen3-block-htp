#!/usr/bin/env python3
"""L32-0015 same-build single-layer replay and conditional R4 arithmetic audit."""
import argparse,json,shlex,subprocess,shutil,re
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin
from prototype_llama32_sp2 import preflight
from probe_llama32_rotations import save,err,had
from prepare_llama32_rotation_layers import tail,M
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0015')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);ap.add_argument('--arm',choices=['base','r4'],required=True);ap.add_argument('--attempt',required=True);ap.add_argument('--audit',action='store_true');a=ap.parse_args();preflight()
 package=M/(f'l32-0009/packages-a01/layer{a.layer}-sp2' if a.arm=='base' else f'l32-0015/layers-a01/layer{a.layer}')
 m=json.loads((package/'manifest.json').read_text())
 for n,h in m['files'].items():assert sha256(package/n)==h['sha256'],n
 head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());assert seal['source_head']==head
 d=OUT/a.attempt;d.mkdir(exist_ok=False);remote='/data/local/tmp/llama32-htp/l32-0015/'+a.attempt;assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,b in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  p=ROOT/b/'ship'/n;assert sha256(p)==seal['files'][str(p)];shutil.copy2(p,d/n);adb('push',windows(p),remote+'/'+n);assert adb('shell','sha256sum '+remote+'/'+n).stdout.split()[0]==sha256(p)
 adb('push',windows(package),remote+'/package');adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 names=list(m['files'])
 for first in range(0,len(names),32):
  out=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/package/'+n) for n in names[first:first+32])).stdout
  for line in out.splitlines():h,n=line.split(maxsplit=1);assert h==m['files'][n.removeprefix(remote+'/package/')]['sha256']
 old=json.loads((OUT.parent/f'l32-0012/layer{a.layer}-m8-a01/protocol.json').read_text())['command'];oldremote=old.split(' && ')[0].removeprefix('cd ');command=old.replace(oldremote,remote)
 prefix,argv=command.split(' ./qwen3_block_cli ',1);wd,e=prefix.split(' && ',1);env=dict(t.split('=',1) for t in shlex.split(e));env.update(QBH_DENSE_R4='1' if a.arm=='r4' else '0',QBH_R4_OPT='6' if a.arm=='r4' else '0')
 if a.audit:env['QBH_DENSE_R4_AUDIT']='1'
 command=wd+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+argv
 save(d/'protocol.json',dict(experiment='L32-0015',source_head=head,build_seal=seal,command=command,package=str(package),package_manifest_sha256=sha256(package/'manifest.json'),audit=a.audit,arm=a.arm,source_layer=a.layer))
 run=adb('shell',command,check=False);(d/'stdout.txt').write_text(run.stdout);(d/'stderr.txt').write_text(run.stderr)
 records=[]
 for l in run.stdout.splitlines():
  try:records.append(json.loads(re.sub(r':-?(?:nan|inf)([,}])',r':null\1',l)))
  except ValueError:pass
 profiles=[x for x in records if isinstance(x,dict) and x.get('record')=='replay_profile'];steps=[]
 for step,phase in enumerate(['prefill','decode']):
  name=f'actual_replay_output_{step:02d}_u8.bin';adb('pull',remote+'/'+name,windows(d/name),check=False)
  if not (d/name).exists():continue
  rows=64 if step==0 else 1;actual=np.fromfile(d/name,dtype='u1').reshape(-1,2048)[:rows];refname='reference_w4u8_integer_attention_block_output_u8.bin' if not step else 'replay_decode_reference_00_u8.bin';ideal=np.fromfile(package/refname,dtype='u1').reshape(-1,2048)[:rows]
  delta=actual.astype('i4')-ideal;row=dict(step=step,ideal_max_lsb=int(np.abs(delta).max()),ideal_mismatches=int(np.count_nonzero(delta)))
  q=load_qparams_bin(package/'layer0/qparams_u8.bin');aa=actual.astype('f8').reshape(-1)-q['block_output']['zero_point'];bb=ideal.astype('f8').reshape(-1)-q['block_output']['zero_point'];den=float(np.linalg.norm(aa)*np.linalg.norm(bb));row['ideal_cosine']=float(aa@bb/den) if den else None;row['ideal_cosine_defined']=bool(den)
  if a.audit and a.arm=='r4':
   name=f'actual_replay_r4_{step:02d}.bin';adb('pull',remote+'/'+name,windows(d/name));raw=np.fromfile(d/name,dtype='<f2');count=rows*8192
   z=raw[:count].reshape(16,rows,512).transpose(1,0,2).copy();s=raw[524288:524288+count].reshape(16,rows,512).transpose(1,0,2).copy();y=raw[1048576:1048576+count].reshape(rows,8192).copy()
   ref=M/f'l32-0003/layers-a01/layer{a.layer}-{phase}';lut=np.fromfile(package/'layer0/silu_up_lut_u16.bin',dtype='<f2',count=65536).reshape(256,256);expectz=lut[np.load(ref/'reference_gate.npy'),np.load(ref/'reference_up.npy')].reshape(z.shape)
   row['swiglu_bits_equal']=bool(np.array_equal(z.view('u2'),expectz.view('u2')))
   es=(z.astype('f8')@had(512)*float(np.float16(1/np.sqrt(512)))).astype('f2');ey=(np.einsum('ag,tgc->tac',had(16),s.astype('f8'))*.25).astype('f2').reshape(y.shape)
   row['stage1']=err(s,es);row['stage2']=err(y,ey)
   conditional,down=tail(y,package/'layer0',q,np.load(ref/'reference_residual.npy'));row['conditional_tail_mismatches']=int(np.count_nonzero(actual!=conditional));row['conditional_tail_max_lsb']=int(np.abs(actual.astype('i4')-conditional).max());np.save(d/f'conditional_down_{step:02d}.npy',down)
   row['pass_arithmetic']=row['swiglu_bits_equal'] and row['conditional_tail_mismatches']==0 and all(row[k]['finite'] and row[k]['outside_1ulp_plus_minnormal']==0 for k in ['stage1','stage2'])
   row['pass_ideal_gate']=(row['ideal_max_lsb']<=2 and row['ideal_cosine']>=.999) if row['ideal_cosine_defined'] else None
   row['ideal_gate_status']='undefined_cosine_exact_output' if row['ideal_mismatches']==0 and not row['ideal_cosine_defined'] else ('undefined_cosine' if not row['ideal_cosine_defined'] else ('pass' if row['pass_ideal_gate'] else 'fail'))
  else:row['pass_exact']=row['ideal_mismatches']==0
  steps.append(row)
 result=dict(process_exit=run.returncode,records=records,steps=steps,profiles=profiles,scope='diagnostic' if a.audit else 'functional; repeat1 auxiliary',fullmodel=False)
 save(d/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','profiles']}),flush=True)
 if len(steps)!=2:print(run.stdout[-3000:]);print(run.stderr[-2000:]);raise SystemExit(1)
if __name__=='__main__':main()
