#!/usr/bin/env python3
"""Audit actual Llama A8 replay/generation/PPL evidence; never promote a baseline."""
import json,math,subprocess
from pathlib import Path
import numpy as np
from llama_reference import sha256
from run_llama32_frontend import records
ROOT=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0003')
P=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/frontend-a01')
F=R/'device-frontend-a02'

def read(p):return json.loads(p.read_text())
def main():
 out=R/'validation_summary.json'
 if out.exists():raise FileExistsError(out)
 subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
 replay=[]
 for count,tag in [(1,'device-stack1-a05'),(3,'device-stack3-a01'),(16,'device-stack16-a01')]:
  r=read(R/tag/'result.json');assert r['pass'] and r['process_exit_code']==0
  ps=[p for p in r['records'] if p.get('record')=='replay_profile'];assert len(ps)==2
  for x in ps:
   for key in ['output_mismatches','output_max_lsb','cache_prefix_mismatches','cache_mismatches','cache_structure_mismatches','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','w4u8_qkvo_weight_expand_ticks','w4u8_mlp_weight_expand_ticks','hmx_fp16_tile_pair_count']:assert x[key]==0,(tag,key,x[key])
   assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608
   assert x['vtcm_peak_plan_bytes']==7668960 and x['block_invocation_count']==count
   assert x['w4u8_decode_direct_n_projection_count']==7*count
   assert x['w4u8_decode_direct_n_hmx_command_count']==200*count
   assert x['w4u8_decode_direct_n_weight_ddr_read_bytes']==30408704*count
  replay.append(dict(layers=count,prefill_max_lsb=0,decode_max_lsb=0,kv_mismatches=0,evidence=tag))
 g=read(F/'result.json');assert g['generation_pass'] and g['arithmetic_token_match'] and g['arithmetic_code_match'] and g['evaluation_complete'] and g['eval_process_exit_code']==0 and g['pass'] and not g['quality_gate_applied']
 expected=read(R/'frontend-reference-a01/teacher.json');assert g['generated_ids']==expected['u8_generated_ids']
 gr=records((F/'generation.stdout.txt').read_text());profiles=[p for p in gr if p.get('record')=='generation_profile'];assert len(profiles)==16
 for x in profiles:
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']==7668960
  assert x['block_invocation_count']==16 and x['w4u8_decode_direct_n_projection_count']==113
  assert x['w4u8_decode_direct_n_hmx_command_count']==3701 and x['w4u8_decode_direct_n_weight_ddr_read_bytes']==617873408
  for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','generation_lm_head_expand_ticks','w4u8_qkvo_weight_expand_ticks','w4u8_mlp_weight_expand_ticks','dense_r3_total_calls','hmx_fp16_tile_pair_count','boundary_ddr_write_bytes']:assert x[k]==0,(k,x[k])
 er=records((F/'evaluation.stdout.txt').read_text());steps=[r for r in er if r.get('record')=='eval_step'];ds=read(R/'frontend-reference-a01/dataset.json');samples=ds['samples'];mapping={s['id']:s for s in samples};assert len(samples)==128 and len(steps)==2048
 keys={(r['sample_id'],r['step']) for r in steps};assert keys=={(s['id'],i) for s in samples for i in range(16)}
 for x in steps:
  sample=mapping[x['sample_id']];assert x['teacher_forcing'] and x['target_token']==sample['target_ids'][x['step']]
  assert x['pass'] and x['vocab_count']==128256 and x['nonfinite']==0 and math.isfinite(x['nll'])
  assert x['vtcm_bytes']==8388608 and x['intermediate_read']==x['intermediate_write']==x['spill']==0
  assert x['cache_valid']==64+x['step'] and 0<=x['target_code']<256
 oracle=read(R/'nll-arithmetic-oracle.json');checks=[]
 for o in oracle['rows']:
  actual=next(x for x in steps if (x['sample_id'],x['step'])==(o['sample_id'],o['step']));err=abs(actual['nll']-o['nll']);assert actual['target_code']==o['target_code'] and err<=1e-5
  checks.append(dict(sample_id=o['sample_id'],language=o['language'],nll_error=err,code_match=True))
 quality={}
 for lang in ['all','en','zh']:
  selected=[s for s in samples if lang=='all' or s['language']==lang];chosen={s['id'] for s in selected};vals=[x['nll'] for x in steps if x['sample_id'] in chosen];ppl=math.exp(float(np.mean(vals)));assert abs(ppl/g['ppl'][lang]['device']-1)<1e-10
  teacher={r['id']:r['nll'] for r in expected['nll']['torch.bfloat16']};delta=np.array([np.mean([x['nll'] for x in steps if x['sample_id']==s['id']])-np.mean(teacher[s['id']]) for s in selected]);rng=np.random.default_rng(320003);boot=np.exp(np.mean(delta[rng.integers(0,len(delta),(10000,len(delta)))],axis=1));quality[lang]=dict(g['ppl'][lang],ratio_document_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
 reg=read(R/'regression-w4a16-a01/result.json');assert reg['pass'] and reg['process_exit_code']==0
 old=read(R.parent/'l32-0002/device-stack16-a01/result.json')
 for x,y in zip(reg['steps'],old['steps']):
  for key in ['output_nrmse','output_cosine','output_max_abs']:assert x[key]==y[key],key
 m=read(P/'manifest.json')
 for name,rec in m['files'].items():assert sha256(P/name)==rec['sha256'],name
 result=dict(experiment='L32-0003',recipe='W4A8',rotation='OFF',functional_pass=True,quality_gate_applied=False,usable_text=False,baseline_promoted=False,scope='M64+15 feedback decode; capacity80 row-major U8 KV; no arbitrary-length serving or rotated Llama acceptance',weights='fresh Llama C64 enhanced GPTQ per-output-channel W4 from sealed L32-0002',activations='fresh65536-token Llama static affine minmax',replay=replay,generation=dict(ids=g['generated_ids'],text=g['text'],steps=16,independent_id_and_code_match=True),ppl=quality,nll_arithmetic_audit=checks,physical=dict(vtcm_acquired=8388608,vtcm_peak=7668960,intermediate_ddr=0,spill=0,native_w4_projections_per_token=113,native_w4_commands_per_token=3701,native_w4_weight_bytes_per_token=617873408,w4_to_s8_expand_ticks=0),functional_run_speed=g['functional_run_speed'],formal_profiling=False,w4a16_regression_pass=True,w4a16_quality_gate_pass=False,package_manifest_sha256=sha256(P/'manifest.json'),core_source_head=read(F/'protocol.json')['source_head'],source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),evidence_root=str(R))
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
