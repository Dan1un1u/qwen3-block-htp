#!/usr/bin/env python3
"""PC070 fixed candidate; no fitting or candidate selection on new data."""
import argparse,json,subprocess,shutil
import numpy as np
import torch
import data_exp0256 as data
import residual_exp0255 as u
r=u.r;a=u.a
ROOT=data.RESULT
NAMES=['F','C64','B__wide_nr64','R_MSE__wide_nr64']
def read(n):return json.loads((ROOT/n).read_text())
def write(n,d):data.write(n,d)
def preflight():
 data.preflight();assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0256-integer-rmse-confirmation'
def bind():u.ROOT=ROOT;u.data=data;u.bind()
def copy_parent(n):
 p=data.verified('exp0255',n);out=ROOT/n;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:f.write(p.read_bytes())
 assert data.sha(out)==data.sha(p);return p

def inputs():
 ROOT.mkdir(parents=True,exist_ok=True);refs=[copy_parent(n) for n in ['inputs.json','parameters/R_MSE.json']]
 provenance=data.verified('exp0255','ARTIFACT_PROVENANCE.json');pv=json.loads(provenance.read_text());refs.append(provenance)
 for n in ['residual_exp0255.py','attention_exp0253.py','integer_attention_exp0252.py','floating_attention_exp0253.py','r3_exp0246.py','ablate_exp0244.py','prefix_exp0243.py','a8_exp0242.py']:
  p=data.SOURCE/'scripts'/n;assert data.sha(p)==pv['script_sha256']['scripts/'+n];refs.append(p)
 p=data.verified('exp0255','checks/numerical.json');assert json.loads(p.read_text())['pass_all'];refs.append(p)
 regression={}
 for n in NAMES:
  phase='reproduction' if n in ['F','C64'] else 'development';p=data.verified('exp0255',f'scores/{phase}_{n}.json');d=json.loads(p.read_text());refs.append(p)
  rows=read('inputs.json')['development'] if phase=='reproduction' else json.loads(data.verified('exp0255','dataset.json').read_text())['samples']
  if phase=='development':rows=[x for x in rows if x['split']=='development']
  assert [(x['id'],x['cell']) for x in rows[:4]]==[(x['id'],x['cell']) for x in d['samples'][:4]]
  regression[n]=dict(rows=rows[:4],reference=d['samples'][:4],reference_path=str(p),reference_sha256=data.sha(p))
 refs.append(data.verified('exp0255','dataset.json'));write('regression_inputs.json',regression)
 write('parameters_freeze.json',dict(files={'parameters/R_MSE.json':data.sha(ROOT/'parameters/R_MSE.json')},byte_identical_parent=True,no_fitting=True))
 write('selection.json',dict(candidate='R_MSE',rule='fixed_by_user_before_new_dataset',parameters_sha256=data.sha(ROOT/'parameters_freeze.json'),no_new_selection=True))
 write('inputs_freeze.json',dict(files={n:data.sha(ROOT/n) for n in ['inputs.json','regression_inputs.json','parameters_freeze.json','selection.json']},references={str(p):data.sha(p) for p in refs},protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0256.md'),candidate_frozen_before_new_dataset=True))
 print('INPUTS_FROZEN',data.sha(ROOT/'inputs_freeze.json'),flush=True)

def frozen_inputs():
 f=read('inputs_freeze.json');assert f['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0256.md')
 for n,h in f['files'].items():assert data.sha(ROOT/n)==h
 for n,h in f['references'].items():assert data.sha(n)==h
 u.parameter_check()

def freeze():
 frozen_inputs();data.frozen();assert (ROOT/'inputs_freeze.json').stat().st_mtime<(ROOT/'dataset.json').stat().st_mtime
 write('freeze.json',dict(files={n:data.sha(ROOT/n) for n in ['inputs_freeze.json','dataset_freeze.json','dataset.json','selection.json','parameters_freeze.json']},references=read('inputs_freeze.json')['references'],protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0256.md'),names=NAMES,before_scoring=True))
def frozen():
 frozen_inputs();data.frozen();f=read('freeze.json');assert f['names']==NAMES
 for n,h in f['files'].items():assert data.sha(ROOT/n)==h

def numerical():
 base=read('inputs.json')['parameters']['R3_ALL'];candidate=read('parameters/R_MSE.json');p=candidate['parameters'];changed=[n for n in base if base[n]!=p[n]]
 assert changed==candidate['changed_sites'] and len(changed)==55 and all(n.endswith(('residual_mid','residual_out')) for n in changed)
 cases=[]
 for recipe,params in [('B',base),('R_MSE',p)]:
  for name,p in params.items():
   if not name.endswith(('residual_mid','residual_out')):continue
   v=p['mse'];h=((np.arange(256,dtype=np.float32)-v['zero']+.5)*np.float32(v['scale'])).astype(np.float16)
   x=np.concatenate([h,np.nextafter(h,np.float16(-np.inf)),np.nextafter(h,np.float16(np.inf)),np.array([-65504,0,65504],np.float16)])
   code=np.clip(np.floor(x.astype(np.float32)*np.float32(v['inv_scale'])+np.float32(v['zero'])+np.float32(.5)),0,255)
   oracle=((code-np.float32(v['zero']))*np.float32(v['scale'])).astype(np.float16)
   got=a.qdq(torch.from_numpy(x).cuda(),v).cpu().numpy();assert np.array_equal(got,oracle) and a.qdq(torch.zeros(1,device='cuda',dtype=torch.float16),v).item()==0
   cases.append(dict(recipe=recipe,site=name,values=len(x),exact=True))
 write('checks/numerical.json',dict(pass_all=True,cases=cases,changed_sites=changed,other_parameters_exact=True,parent_payload_exact=True,unchanged_arithmetic_proofs=read('inputs_freeze.json')['references']))
 print('NUMERICAL_PASS',len(cases),flush=True)

def run(phase):
 assert read('checks/numerical.json')['pass_all']
 if phase=='final':assert read('reproduction_complete.json')['pass_all']
 for recipe in ['C64','F']:
  proof=f'checks/{phase}_{recipe}_weights.json'
  if (ROOT/proof).exists():continue
  model,before,mh=r.b.model_session(recipe);ins=u.Instrument(model);ins.build_prefix([151645])
  for n in NAMES:
   if (n=='F')!=(recipe=='F'):continue
   rows=read('regression_inputs.json')[n]['rows'] if phase=='reproduction' else read('dataset.json')['samples']
   result=u.score(model,ins,n,rows,phase)
   if phase=='reproduction':
    ref=read('regression_inputs.json')[n];assert result['samples']==ref['reference'],('exact regression',n)
    write(f'checks/reproduction_{n}.json',dict(exact=True,reference_sha256=ref['reference_sha256'],four_documents=True))
  assert before==a.state_digest(model);write(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=r.b.head()));ins.close();del ins,model;torch.cuda.empty_cache()
 write(f'{phase}_complete.json',dict(pass_all=True,names=NAMES));print('PHASE_COMPLETE',phase,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['inputs','freeze','numerical','reproduction','final']);args=p.parse_args();preflight();a.settings();bind()
 with torch.inference_mode():
  if args.phase=='inputs':inputs()
  elif args.phase=='freeze':freeze()
  else:
   frozen()
   if args.phase=='numerical':numerical()
   else:run(args.phase)
if __name__=='__main__':main()
