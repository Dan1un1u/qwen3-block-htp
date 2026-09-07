#!/usr/bin/env python3
"""Standard-library independent ledger coverage, state and raw-token PPL audit."""
import hashlib,json,math
from pathlib import Path
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0236')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0236')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 e=read(R/'evidence_sha256.json');actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='evidence_sha256.json'};assert actual==set(e)
 for n,h in e.items():assert sha(R/n)==h,n
 a=read(R/'artifacts_sha256.json');assert Path(a['root'])==O
 actual={str(p.relative_to(O)) for p in O.rglob('*') if p.is_file()};assert actual==set(a['files'])
 for n,h in a['files'].items():assert sha(O/n)==h['sha256'] and (O/n).stat().st_size==h['bytes'],n
 freeze=read(R/'dataset_freeze.json')
 for n,h in freeze['files'].items():assert sha(R/n)==h,n
 for n,h in freeze['references'].items():assert sha(Path(n))==h,n
 assert read(R/'independent_data_audit.json')['pass_all']
 baseline=read(R/'G64_state_manifest.json')['state_hashes'];audits=read(R/'head_state_audits.json');assert audits['pass_all'] and audits['only_head_changed'] and len(audits['audits'])==4
 for audit in audits['audits']:
  expected=dict(baseline);v=audit['variant']
  if v!='G64':
   m=read(R/v/'package.json');assert m['only_changed_tensor']=='lm_head.weight' and m['W4'] and not m['mixed_precision'];expected['lm_head.weight']=m['dequant_FP16_sha256'];assert sha(O/v/'manifest.json')==m['manifest_sha256']
  assert expected==audit['state_hashes'] and audit['all_parameter_and_buffer_hashes_match']
 assert read(R/'G64_final_sentinel.json')['samples']==read(R/'software/development_G64.json')['samples']
 reductions=0
 for phase in ['development','final','PC052']:
  summary=read(R/f'summary_{phase}.json');raw={v:read(R/f'software/{phase}_{v}.json') for v in ['F','G64','P64','H64']}
  assert all(len(r['samples'])==summary['documents'] for r in raw.values()) and summary['targets']==summary['documents']*16
  for n,h in summary['inputs'].items():assert sha(R/n)==h
  for name,g in summary['statistics'].items():
   def match(r):
    if name=='overall':return True
    if name in ['en','zh']:return r['cell'].startswith(name+'_')
    if name in ['wiki','news']:return r['cell'].endswith('_'+name)
    return r['cell']==name
   nll={}
   for v,r in raw.items():
    selected=[row for row in r['samples'] if match(row)];values=[x for row in selected for x in row['nll']];assert len(values)==16*len(selected) and all(math.isfinite(x) for x in values)
    nll[v]=math.fsum(values)/len(values);assert abs(nll[v]-g['nll'][v])<1e-12 and abs(math.exp(nll[v])-g['ppl'][v])<1e-10;reductions+=1
   for v in ['G64','P64','H64']:
    q=g['vs_F16'][v];ratio=math.exp(nll[v]-nll['F']);assert abs(q['ppl_ratio']-ratio)<1e-12;assert q['limit']==(1.05 if name=='overall' else 1.1)
    assert q['point_pass']==(ratio<=q['limit']);assert q['confident_pass']==(q['ratio_ci95'][1]<=q['limit'])
   for v in ['P64','H64']:assert abs(math.exp(nll[v]-nll['G64'])-g['vs_G64'][v]['ppl_ratio'])<1e-12
  for v in ['G64','P64','H64']:
   gates=[r['vs_F16'][v] for r in summary['statistics'].values()];status='fail' if not all(g['point_pass'] for g in gates) else 'pass' if all(g['confident_pass'] for g in gates) else 'inconclusive';assert status==summary['status'][v]
 for v in ['F','G64']:
  r=read(R/f'software/PC052_{v}.json');original=[]
  for path,h in r['reused_verified_controls'].items():assert sha(Path(path))==h;original+=read(Path(path))['samples']
  assert original==r['samples']
  path=R.parent/('exp0230' if v=='F' else 'exp0234')/f'software/development_{v}.json';assert read(path)['samples']==read(R/f'software/development_{v}.json')['samples']
 assert reductions==108
 c=read(R/'closure.json');assert c['artifact_ledger_sha256']==sha(R/'artifacts_sha256.json') and c['artifact_files']==len(a['files'])
 assert c['report_sha256']==sha(R/'REPORT.md') and c['full_profiling_report_sha256']==sha(R/'full_profiling_report.md')
 assert not c['mixed_precision'] and not c['baseline_promoted'] and c['only_head_changed']
 print(json.dumps(dict(pass_all=True,result_files=len(e),artifact_files=len(a['files']),whole_state_audits=4,independent_raw_token_PPL_reductions=reductions,evidence_ledger_sha256=sha(R/'evidence_sha256.json'),artifact_ledger_sha256=sha(R/'artifacts_sha256.json'),closure_sha256=sha(R/'closure.json')),indent=2))
if __name__=='__main__':main()
