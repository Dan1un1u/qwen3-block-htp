#!/usr/bin/env python3
from device_exp0257 import *
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized

def main():
 preflight();assert json.loads((R/'seed_slice_gate.json').read_text())['pass_all']
 head=json.loads((R/'seed_head_audit/independent_reference.json').read_text());assert head['summary']['implementation_gate']=='pass'
 a=json.loads((R/'seed_full_nr64/validated.json').read_text());b=json.loads((R/'seed_full_scalar_nr64/validated.json').read_text());assert a['token_ids']==b['token_ids']
 for name in ['seed_full_nr64','seed_full_scalar_nr64','seed_full_sole']:
  records=[json.loads(l) for l in (R/name/'stdout.jsonl').read_text().splitlines() if l.startswith('{')];profiles=[r for r in records if r.get('record')=='generation_profile'];assert len(profiles)==64
  for step,p in enumerate(profiles):
   assert p['prefix_kv_mode']==1 and p['prefix_group_patch_count']==(224 if step==0 else 0)
   assert sum(normalized([p])[k] for _,k in LEDGER)==p['invocation_ticks']
   for i in range(28):
    q=p[f'slice_layer_{i}'];assert q['cache_valid_after']==64+step and q['hidden_ddr_read_bytes']==q['hidden_ddr_write_bytes']==q['layer_unattributed_ticks']==0
 write(R/'full_model_gate.json',dict(pass_all=True,scalar_NR64_all64tokens_exact=True,independent_head=head['summary'],prefix_group_count=224,prefix_metadata_bytes=57344,cache_length_range=[64,127],scope='actual28layers embedding head greedy, all64steps; auxiliary text quality separate',slice_reference=sha(R/'seed_slice_gate.json')))
 print('FULL_MODEL_GATE_PASS',flush=True)
if __name__=='__main__':main()
