import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_exp0275 import records
from summarize_exp0217 import normalized
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0275')
A={'ALL':0,'QKV':1,'FFN':2,'EPILOGUE':12,'LOOKAHEAD':16}
metrics=['hmx_command_count','hmx_u8s8_tile_pair_count','hmx_fp16_tile_pair_count','weight_ddr_read_bytes','w4u8_decode_direct_n_weight_ddr_read_bytes','generation_lm_head_ddr_read_bytes']
count=0;binary_seals=set();pair_count=0
for scope,phase,rounds in [(1,'formal',10),(3,'formal',10),(28,'short',5),(28,'formal',10)]:
 for c in range(rounds):
  ref=None
  for a,mask in A.items():
   d=R/f'l{scope}-{phase}/{c:02d}-{a}';z=json.loads((d/'protocol.json').read_text());seal=z['runtime']['seal'];assert seal['paper_trace'] is False and seal['source_head']=='19c9f728dbf2d06a2f6d0ec6488f4d1547627226'
   arc=Path(z['runtime']['archive'])
   for n,h in seal['files'].items():
    key=(str(arc/Path(n).name),h)
    if key not in binary_seals:assert hashlib.sha256(Path(key[0]).read_bytes()).hexdigest()==h;binary_seals.add(key)
   ps=[v for v in records(d/'stdout.jsonl') if v.get('record')==('generation_profile' if scope==28 else 'exp0240_profile')];assert len(ps)==(160 if scope==28 else 20)
   signature=[]
   for v in ps:
    assert v['paper_format_disable']==0 and v['paper_pipeline_disable']==mask
    assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
    assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
    signature.append([v.get(k,0) for k in metrics]);count+=1
   if ref is None:ref=signature
   else:assert signature==ref,(scope,phase,c,a);pair_count+=len(ps)
trace=[json.loads(p.read_text()) for p in R.glob('trace-*-a0*-events.json')];assert len(trace)==15 and all(v['numerical_pass'] for v in trace)
out=dict(pass_all=True,timed_profiles=count,paired_profile_comparisons=pair_count,arithmetic_and_weight_metrics=metrics,independent_numerical_gates=[json.loads((R/n).read_text())['pass_all'] for n in ['single_gate.json','slice_gate.json','full_gate.json']],production_trace=False,trace_diagnostic_runs=15,trace_complete_runs=sum(v['trace_complete'] for v in trace),trace_status='N/A: incomplete FARF transport; no timeline or utilization claim',production_source='19c9f728dbf2d06a2f6d0ec6488f4d1547627226',quality_claim=False)
assert all(out['independent_numerical_gates'])
with (R/'FINAL_AUDIT.json').open('x') as f:json.dump(out,f,indent=2)
print(out)
