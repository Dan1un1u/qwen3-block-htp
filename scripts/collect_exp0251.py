#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
import device_exp0251 as d
from exp0240_report import FIELDS
R=d.R

def records(tag):
 return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')=='exp0240_profile']
def check(rs,cell,rep):
 assert len(rs)==rep*9
 gold=records('audit_'+cell)
 for j,x in enumerate(rs):
  assert x['output_hash']==gold[j%9]['output_hash'],(cell,j,'output_hash')
  assert x['wide_score_mode']==int(cell!='control')
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608
  assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==0
  assert x['u8_attention_audit_ddr_write_bytes']==0 and x['block_invocation_count']==1
  assert x['w4u8_decode_direct_n_projection_count']==7
  assert sum(x[k] for k in FIELDS)==x['ledger_named_ticks']==x['invocation_ticks']
  assert x['ledger_unattributed_ticks']==0 and x['scan_total_kv_length']==64+j%9 and x['dsp_status']==3
if __name__=='__main__':
 d.preflight();gate=json.loads((R/'numerical_audit.json').read_text());assert gate['pass_all']
 cells=[x for x in ['control','wide','r3_wide'] if gate['eligible'][x]]
 provenance=json.loads((R/'device_binary_manifest_attempt1.json').read_text());assert all(d.sha(p)==provenance['binaries'][p.name] for p in d.binaries())
 remote=d.adb('shell',f'cd {d.REMOTE} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so');assert all(h in remote for h in provenance['binaries'].values())
 assert d.adb('shell','cat /proc/sys/kernel/random/boot_id')==provenance['boot']
 with (R/'formal_provenance.json').open('x') as f:json.dump(dict(provenance,collection_source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.S,text=True).strip(),cells=cells,audit_disabled=True),f,indent=2)
 calls=0
 for phase,rounds in [('short',5),('formal',10)]:
  for i in range(rounds):
   order=cells[i%len(cells):]+cells[:i%len(cells)]
   for rep in [1,10]:
    for cell in order:
     rs=d.run(cell,rep,f'{phase}_{i:02d}_r{rep}_{cell}',False);check(rs,cell,rep);calls+=len(rs)
   print('PAIR_COMPLETE',phase,i,flush=True)
 assert d.adb('shell','cat /proc/sys/kernel/random/boot_id')==provenance['boot']
 with (R/'collection_pass.json').open('x') as f:json.dump(dict(pass_all=True,cells=cells,short_pairs=5,formal_pairs=10,repeat_scopes=[1,10],layer_RPCs=calls,physical_gate=True,numerical_reference='numerical_audit.json',unavailable_cells=[x for x in gate['eligible'] if not gate['eligible'][x]]),f,indent=2)
