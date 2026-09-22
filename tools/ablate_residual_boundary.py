#!/usr/bin/env python3
"""L32-0067 exact residual-boundary four-arm ablation, same binary."""
import argparse,json,shlex,shutil,struct,subprocess
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from prototype_llama32_sp2 import preflight
from llama_reference import sha256
from run_llama32_frontend import records
from report_llama32_pipeline_profile import MODULES
from probe_llama32_av_o_fullmodel import M,BASE,FIXTURE
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0067')
OLD=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0065')
REMOTE='/data/local/tmp/llama32-htp/l32-0067'
PACKAGE_REMOTE='/data/local/tmp/llama32-htp/l32-0062'
MASKS={'F':128,'N':160,'R':192,'NR':224,'PARENT':0}
MODES=['F','N','R','NR']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify():
 preflight();R.mkdir(parents=True,exist_ok=True)
 import ablate_integer_fusion as previous
 previous.R=R;previous.verify()
 assert sha256(OLD/'oracle-M'/'summary.json')
 save(R/'reference-provenance.json',dict(source=str(OLD/'oracle-M'/'summary.json'),sha256=sha256(OLD/'oracle-M'/'summary.json'),package_manifest=sha256(M/'folded'/'manifest.json'),same_arithmetic_all_arms=True))
def stage():
 preflight();seal=read(ROOT/'build/llama-build-seal.json');assert not seal['fp_islands'] and seal['model_size']=='1B';nl=seal['layer_count']
 assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 d=R/f'binaries-I-{nl}';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name
 adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');save(d/'seal.json',seal)
def execute(mode,tag,nl=16,repeat=1,audit=False,poison=False):
    arm="folded"
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='I';root=REMOTE+f'/binaries-{binary_mode}-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='8',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE=str(MASKS[mode]),QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
    if poison:env['QBH_W4U8_DECODE_AV_PADDING_POISON']='1'
    fixture=read(FIXTURE);words=[0x51424556,2,repeat,110]
    for j in range(repeat):words += [j,3,43]+fixture['prompt_ids'][:64]+fixture['fixed'][:43]
    ef=d/'fixed-trajectory.bin'
    if not ef.exists():ef.write_bytes(struct.pack('<'+'I'*len(words),*words))
    er=root+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
    if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
    cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
    if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,arm=arm,layers=nl,repeat=repeat,audit=audit,seal_sha256=sha256(R/f'binaries-{binary_mode}-{nl}/seal.json'),package_manifest_sha256=sha256(M/arm/'manifest.json')))
    if not (d/'exit.json').exists():
        adb('push',windows(ef),er)
        if audit:adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
        z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
        if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'))
    assert read(d/'exit.json')['returncode']==0,(tag,(d/'stderr.txt').read_text()[-1000:])
    rr=records((d/'stdout.txt').read_text())
    if not (d/'records.json').exists():save(d/'records.json',rr)
    pp=[x for x in rr if x.get('record')=='generation_profile'];ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x]
    assert len(pp)==len(ss)==43*repeat,(tag,len(pp),len(ss))
    gold=read(OLD/'oracle-M'/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','w4u8_gate_up_swiglu_worker_ticks','u8_attention_softmax_ticks']} for k in times}
    for rep in range(repeat):
        for phase,lo,hi in [('prefill',0,1),('decode',1,43)]:
            wall=0;totals={k:0 for k in fields[phase]}
            for j in range(lo,hi):
                x,st=pp[rep*43+j],ss[rep*43+j];g=gold['heads'][str(nl)][j]
                assert (st['selected_token_id'],st['selected_logit_half_bits'])==(g['token'],g['code']),(tag,j,'head',st['selected_token_id'],g)
                assert x['dsp_status']==3 and x['numerical_status']==1
                assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']<=8388608
                assert x['intermediate_spill_fill_count']==x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==0
                for i in range(nl):
                    l=x[f'slice_layer_{i}'];assert l['status']==3
                    assert l['hidden_ddr_read_bytes']==l['hidden_ddr_write_bytes']==l['layer_unattributed_ticks']==0
                    if audit:
                        assert l['output_hash']==gold['layer_hashes'][i][j],(tag,j,i,l['output_hash'],gold['layer_hashes'][i][j])
                        exact+=1
                total=sum(sum(x[k] for k in f) for _,f in MODULES)-x['generation_final_norm_ticks'];assert total==x['invocation_ticks']
                assert x['host_wall_ns']==st['host_wall_ns'] and x['host_wall_ns']/1000>=total/19.2
                wall+=x['host_wall_ns'];peak=max(peak,x['vtcm_peak_plan_bytes'])
                for k in totals:totals[k]+=x[k]
            times[phase].append(wall)
            for k,v in totals.items():fields[phase][k].append(v)
    result=dict(mode=mode,arm=arm,boundary_mask=MASKS[mode],layers=nl,repeat=repeat,exact_layer_outputs=exact,boundaries=len(pp),peak=peak,times=times,fields=fields)
    save(d/'validated.json',result);print('PASS',tag,{k:float(np.mean(v)) for k,v in times.items()},flush=True);return result

def run():
 preflight()
 for mode in MODES:execute(mode,'warmup-'+mode)
 # Balanced four-arm Williams order; fixed in advance, no result-based ordering.
 orders=[['F','N','NR','R'],['N','R','F','NR'],['R','NR','N','F'],['NR','F','R','N']]
 for stage_name,n in [('short',5),('formal',10)]:
  rounds=[]
  for i in range(n):
   rounds.append({m:execute(m,f'{stage_name}-{i:02d}-{m}',repeat=10) for m in orders[i%4]})
  rng=np.random.default_rng(670067);ix=rng.integers(0,n,(20000,n));summary={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(r[m]['times'][phase])/1e6 for r in rounds]) for m in MODES}
   summary[phase]={m:dict(wall_ms=float(x.mean()),tps=float(tokens*1000/x.mean())) for m,x in a.items()}
   for ctl,opt in [('F','N'),('F','R'),('F','NR'),('R','NR'),('N','NR')]:
    x,y=a[ctl],a[opt];summary[phase][opt+'/'+ctl]=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist())
   ratios=(a['NR'][ix].mean(1)*a['F'][ix].mean(1))/(a['N'][ix].mean(1)*a['R'][ix].mean(1))
   summary[phase]['interaction']=dict(wall_ratio=float(a['NR'].mean()*a['F'].mean()/a['N'].mean()/a['R'].mean()),ci95=np.quantile(ratios,[.025,.975]).tolist())
  f=R/(stage_name+'-summary.json')
  if not f.exists():save(f,summary)
  print(stage_name,json.dumps(summary),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=16);ap.add_argument('--mode',default='F');ap.add_argument('--tag');a=ap.parse_args()
 if a.action=='audit':
  preflight();execute(a.mode,a.tag or f'audit{a.layers}-{a.mode}',nl=a.layers,audit=True)
 else:globals()[a.action]()
