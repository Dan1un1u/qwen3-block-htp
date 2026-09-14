#!/usr/bin/env python3
"""L32-0021 fixed ten five-arm warmed single-layer Host-wall comparison.
One warmup and ten measured M64+decode replays per process, ten cycles.
No inner repeat1 decision. Retains native CLI ideal-exact failures separately;
requires the registered component/conditional-tail/ideal-cosine gate beforehand.
"""
import json,shlex,re,subprocess,argparse
from pathlib import Path
import numpy as np
from llama32_rotated_fp32 import R as OLD_R,M,ROOT,save
R=OLD_R.parent/"l32-0021"
CONTROL=R.parent/"l32-0020"
from prototype_llama32_sp2 import preflight
from run_llama32_layer import adb
from llama_reference import sha256
from report_llama32_pipeline_profile import MODULES

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);ap.add_argument('--short',action='store_true');a=ap.parse_args();preflight()
 cycles_n=1 if a.short else 10;replays=3 if a.short else 11
 out=R/a.attempt;out.mkdir(exist_ok=False)
 head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 cfg={}
 for arm in ['off','r3','r4','both','oldboth']:
  base='both' if arm=='oldboth' else arm;origin=CONTROL if arm=='oldboth' else R
  expected_head='0eb7615c43bb6e36c402f3a5d12d7162c32722dd' if arm=='oldboth' else head
  d=origin/('timing-'+base+'-a01');p=json.loads((d/'protocol.json').read_text());res=json.loads((d/'result.json').read_text());assert p['source_head']==expected_head and len(res['steps'])==2 and all(x['pass_ideal_cosine'] and x['finite'] for x in res['steps'])
  # Separate audited validation, no captured hardware values become model goldens.
  audit=origin/('validate-layer0-'+base+'-a01')
  assert json.loads((audit/'protocol.json').read_text())['source_head']==expected_head
  ar=json.loads((audit/'result.json').read_text());assert all(x['pass_actual_arithmetic'] and x['pass_ideal_cosine'] for x in ar['steps'])
  if arm=='oldboth':
   ledger=CONTROL/'evidence-ledger-checkpoint-a01.json';assert sha256(ledger)=='06e74b1fd182511052387ba0f2d36fb653a2168081a92a517f3ac4495d233d91'
   frozen={q['path']:q['sha256'] for q in json.loads(ledger.read_text())['files']}
   for path in [d/'protocol.json',d/'result.json',audit/'protocol.json',audit/'result.json']:
    assert sha256(path)==frozen[str(path.relative_to(CONTROL))]
  remote=p['command'].split(' && ')[0].removeprefix('cd ')
  for file,h in p['build_seal']['files'].items():
   if Path(file).name=='llama_sp2_cli':continue
   assert adb('shell','sha256sum '+remote+'/'+Path(file).name).stdout.split()[0]==h
  manifest=json.loads((Path(p['package'])/'manifest.json').read_text());assert sha256(Path(p['package'])/'manifest.json')==p['package_manifest_sha256']
  names=list(manifest['files'])
  for j in range(0,len(names),32):
   for line in adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/package/'+n) for n in names[j:j+32])).stdout.splitlines():
    h,n=line.split(maxsplit=1);assert h==manifest['files'][n.removeprefix(remote+'/package/')]['sha256']
  prefix,argv=p['command'].split(' ./qwen3_block_cli ',1);wd,e=prefix.split(' && ',1);env=dict(t.split('=',1) for t in shlex.split(e));assert not any('AUDIT' in k for k in env)
  env.pop('QBH_REPLAY_DUMP_DIR',None)
  env['QBH_LLAMA_REPLAY_REPEATS']=str(replays);env['QBH_LLAMA_ROTATION_REPLAY_PROFILE']='1'
  command=wd+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+argv
  cfg[arm]=dict(command=command,source_head=expected_head,protocol_sha256=sha256(d/'protocol.json'),audit_sha256=sha256(audit/'result.json'),expected_hashes=[q['output_hash'] for q in res['records'] if q.get('record')=='replay_profile'])
 save(out/'protocol.json',dict(experiment='L32-0021',source_head=head,cycles=cycles_n,auxiliary=a.short,arms=cfg,order='cyclic shift of OFF/R3/R4/both/oldboth',scope='single layer0, M64 prefill + one decode past64; complete Host wall, no audit dumps; loading outside timer; no E2E extrapolation',inner_replays=replays,warmup_replays=1,seed=21021,bootstrap_samples=20000,gate='paired bootstrap95 upper<=1.10; actual-tail/component audits required; ideal-exact CLI failures retained'))
 cycles=[];arms=list(cfg)
 for i in range(cycles_n):
  cycle={}
  for arm in arms[i%len(arms):]+arms[:i%len(arms)]:
   d=out/f'cycle-{i:02d}-{arm}';d.mkdir();z=adb('shell',cfg[arm]['command'],check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr)
   records=[]
   for line in z.stdout.splitlines():
    try:records.append(json.loads(re.sub(r':-?(?:nan|inf)([,}])',r':null\1',line)))
    except ValueError:pass
   qs=[q for q in records if q.get('record')=='replay_profile'];checks=[]
   assert len(qs)==2*replays and records[-1].get('completed_steps')==2,(d,z.returncode)
   reps=[q for q in records if q.get('record')=='llama_replay_repeat'];assert [q['repeat'] for q in reps]==list(range(replays))
   assert [q['warmup'] for q in reps]==[True]+[False]*(replays-1)
   assert z.returncode==(1 if arm in ['r4','both','oldboth'] else 0),(d,z.returncode)
   for sequence,q in enumerate(qs):
    step=sequence%2
    assert q['output_hash']==cfg[arm]['expected_hashes'][step]
    assert q['dsp_status']==3 and q['numerical_status']==1 and q['output_nonfinite_count']==0
    assert q['vtcm_requested_bytes']==q['vtcm_acquired_bytes']==8388608 and q['vtcm_peak_plan_bytes']<=8388608
    for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','projection_failure_result','w4u8_mlp_weight_expand_ticks','w4u8_qkvo_weight_expand_ticks','w4f16_expand_ticks','cache_prefix_mismatches','cache_mismatches','cache_structure_mismatches','dense_r4_audit_bytes']:
     assert q[k]==0,(d,k,q[k])
    r3=int(arm in ['r3','both','oldboth']);r4=int(arm in ['r4','both','oldboth'])
    assert q['dense_r3_total_calls']==r3 and q['dense_r4_calls']==r4 and q['llama_fp32_residual']==1
    expected=(320 if step==0 else 8)*r3+(9216 if step==0 else 272)*r4
    assert q['hmx_fp16_tile_pair_count']==expected,(d,step,q['hmx_fp16_tile_pair_count'],expected)
    ticks=sum(sum(q[k] for k in fields) for _,fields in MODULES)-q['generation_final_norm_ticks'];assert ticks==q['invocation_ticks'];assert q['host_wall_ns']/1000>=ticks/19.2
   save(d/'result.json',dict(process_exit=z.returncode,profiles=qs,physical_and_repeatability_pass=True,ideal_exact_cli_pass=z.returncode==0));cycle[arm]=[{k:float(np.mean([q[k] for q in qs[2+step::2]])) for k in qs[step] if isinstance(qs[step][k],(int,float))} for step in range(2)]
  cycles.append(cycle);print('FORMAL_CYCLE',i+1,flush=True)
 rng=np.random.default_rng(21021);idx=rng.integers(0,cycles_n,(20000,cycles_n));speed={};modules={}
 for step,mode in enumerate(['prefill','decode']):
  speed[mode]={};modules[mode]={};x=np.array([c['off'][step]['host_wall_ns'] for c in cycles],float)
  for arm in arms:
   qs=[c[arm][step] for c in cycles];y=np.array([q['host_wall_ns'] for q in qs],float);ci=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]);ratio=float(y.mean()/x.mean())
   speed[mode][arm]=dict(mean_us=float(y.mean()/1000),latency_ratio=ratio,increase_percent=(ratio-1)*100,ci95=ci.tolist(),gate_pass=bool(not a.short and ci[1]<=1.1))
   host=float(y.mean()/1000);rows=[]
   for n,fields in MODULES:
    us=float(np.mean([sum(q[k] for k in fields)-(q['generation_final_norm_ticks'] if fields==['generation_lm_head_ticks'] else 0) for q in qs])/19.2);rows.append(dict(module=n,us=us,share=us/host*100))
   rows.extend([dict(module='Host–DSP 边界',us=host-sum(r['us'] for r in rows),share=(host-sum(r['us'] for r in rows))/host*100),dict(module='完整 Host wall',us=host,share=100.)]);modules[mode][arm]=rows
 improvements={}
 for step,mode in enumerate(['prefill','decode']):
  x=np.array([c['oldboth'][step]['host_wall_ns'] for c in cycles]);y=np.array([c['both'][step]['host_wall_ns'] for c in cycles]);ci=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]);improvements[mode]=dict(latency_ratio=float(y.mean()/x.mean()),ci95=ci.tolist())
 save(out/'summary.json',dict(improvement_vs_sealed0020=improvements,speed=speed,modules=modules,formal_cycles=0 if a.short else cycles_n,auxiliary=a.short,successful_rpc_boundaries=cycles_n*len(arms)*2*replays,e2e=None));print(json.dumps(speed),flush=True)
if __name__=='__main__':main()
