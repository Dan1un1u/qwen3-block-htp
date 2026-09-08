#!/usr/bin/env python3
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import device_exp0248 as d
from exp0240_report import FIELDS
R=d.R
def records(tag):
 return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')=='exp0240_profile']
def check(rs,cell,rep):
 assert len(rs)==rep*9
 gold=records('chain_control_02' if cell=='control' else 'chain_refined_02')
 for j,x in enumerate(rs):
  assert x['output_hash']==gold[j%9]['output_hash'],(cell,j,'output_hash')
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608
  assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==0
  assert x['u8_attention_audit_ddr_write_bytes']==0 and x['block_invocation_count']==1
  assert x['w4u8_decode_direct_n_projection_count']==7
  assert sum(x[k] for k in FIELDS)==x['ledger_named_ticks']==x['invocation_ticks']
  assert x['ledger_unattributed_ticks']==0
  assert x['scan_total_kv_length']==64+j%9
  assert x['dsp_status']==3
if __name__=='__main__':
 d.preflight();assert json.loads((R/'numerical_audit.json').read_text())['pass_all']
 provenance=json.loads((R/'device_binary_manifest_attempt3.json').read_text());assert all(d.sha(p)==provenance['binaries'][p.name] for p in d.binaries())
 (R/'formal_provenance.json').write_text(json.dumps(provenance,indent=2))
 for stage,rounds in [('short',5),('formal',10)]:
  for i in range(rounds):
   for rep in [1,10]:
    for cell in (['control','refined'] if i%2==0 else ['refined','control']):
     rs=d.run(cell,rep,f'{stage}_{i:02d}_r{rep}_{cell}',False);check(rs,cell,rep)
   print('PAIR_COMPLETE',stage,i,flush=True)
 (R/'collection_pass.json').write_text(json.dumps(dict(pass_all=True,short_pairs=5,formal_pairs=10,repeat_scopes=[1,10],layer_RPCs=2970,numerical_reference='numerical_audit.json',output_hashes='chain_control_02 / chain_refined_02',physical_gate=True),indent=2))
