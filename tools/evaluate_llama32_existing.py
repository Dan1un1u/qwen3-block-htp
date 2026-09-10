#!/usr/bin/env python3
"""Independent frozen device PPL run; preserves separate generation/replay outcomes."""
import argparse,json,math,shlex,subprocess
from pathlib import Path
from run_llama32_layer import ROOT,adb
from run_llama32_frontend import records
from llama_reference import sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--launch',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
 p=json.loads((a.launch/'protocol.json').read_text());remote=shlex.split(p['command'])[1]
 assert remote.startswith('/data/local/tmp/llama32-htp/'+p['experiment'].lower()+'/') and '..' not in remote
 assert adb('shell','sha256sum '+shlex.quote(remote+'/package/manifest.json')).stdout.split()[0]==p['package_manifest_sha256']
 for name,h in p['builds'].items():assert adb('shell','sha256sum '+shlex.quote(remote+'/'+name)).stdout.split()[0]==h
 for name,h in json.loads((a.reference/'freeze.json').read_text()).items():assert sha256(a.reference/name)==h
 assert adb('shell','sha256sum '+shlex.quote(remote+'/heldout.bin')).stdout.split()[0]==sha256(a.reference/'heldout.bin')
 a.output.mkdir(exist_ok=False)
 command=p['command'].replace(' && ',' && QBH_EVAL_QUIET=1 QBH_EVAL_FILE='+shlex.quote(remote+'/heldout.bin')+' ',1)
 (a.output/'protocol.json').write_text(json.dumps({'experiment':p['experiment'],'launch_protocol_sha256':sha256(a.launch/'protocol.json'),'generation_result_sha256':sha256(a.launch/'result.json'),'generation_status_unchanged':True,'command':command,'dataset_sha256':sha256(a.reference/'dataset.json'),'teacher_sha256':sha256(a.reference/'teacher.json'),'scope':'independent teacher-forced PPL; no reinterpretation of free-generation exact replay gate'},indent=2))
 r=adb('shell',command,check=False);(a.output/'stdout.txt').write_text(r.stdout);(a.output/'stderr.txt').write_text(r.stderr)
 rows=[v for v in records(r.stdout) if isinstance(v,dict) and v.get('record')=='eval_step'];ds=json.loads((a.reference/'dataset.json').read_text());samples=[s for s in ds['samples'] if s.get('split','heldout')=='heldout'];teacher=json.loads((a.reference/'teacher.json').read_text())
 keys={(v['sample_id'],v['step']) for v in rows} if rows and 'step' in rows[0] else None
 complete=r.returncode==0 and len(rows)==len(samples)*16 and all(v['pass'] and v['nll'] is not None and math.isfinite(v['nll']) for v in rows)
 result={'process_exit_code':r.returncode,'evaluation_complete':complete,'rows':len(rows),'ppl':{},'free_generation_replay_promoted':False}
 if complete:
  for lang in ['all','en','zh']:
   chosen={s['id'] for s in samples if lang=='all' or s['language']==lang};vals=[v['nll'] for v in rows if v['sample_id'] in chosen];v={'tokens':len(vals),'device':math.exp(sum(vals)/len(vals))}
   for dtype,label in [('torch.float16','fp16_teacher'),('torch.bfloat16','bf16_teacher')]:
    target=[t for s in teacher['nll'][dtype] if s['id'] in chosen for t in s['nll']];v[label]=math.exp(sum(target)/len(target));v['ratio_to_'+label]=v['device']/v[label]
   v['point_threshold_pass']=v['ratio_to_bf16_teacher'] <= (1.05 if lang=='all' else 1.10);result['ppl'][lang]=v
 result['quality_point_gate']=complete and all(v['point_threshold_pass'] for v in result['ppl'].values())
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not complete:print(r.stdout[-2000:]);raise SystemExit(1)
if __name__=='__main__':main()
