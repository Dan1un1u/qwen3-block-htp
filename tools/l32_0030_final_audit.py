import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from run_llama32_frontend import records as parse_records
def records(p):return parse_records(Path(p).read_text())

R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0030')
A={'ALL':0,'QKV':1,'FFN':2,'EPILOGUE':12,'LOOKAHEAD':16}
metrics=['hmx_command_count','hmx_u8s8_tile_pair_count','hmx_fp16_tile_pair_count','weight_ddr_read_bytes','w4u8_decode_direct_n_weight_ddr_read_bytes','generation_lm_head_ddr_read_bytes']
count=0;binary_seals=set();pair_count=0
for scope,phase,rounds in [(1,'formal',10),(3,'formal',10),(16,'short',5),(16,'formal',10)]:
 for c in range(rounds):
  ref=None
  for a,mask in A.items():
   d=R/f'l{scope}-{phase}-a02/{c:02d}-{a}';z=json.loads((d/'protocol.json').read_text());seal=z['runtime']['seal'];assert seal['paper_trace'] is False and seal['source_head']=='4421949fcdeaff146cf093a4ce529190b13e2ded'
   arc=Path(z['runtime']['archive'])
   for n,h in seal['files'].items():
    key=(str(arc/Path(n).name),h)
    if key not in binary_seals:assert hashlib.sha256(Path(key[0]).read_bytes()).hexdigest()==h;binary_seals.add(key)
   ps=[v for v in records(d/'stdout.txt') if v.get('record')==('generation_profile' if scope==16 else 'replay_profile')];assert len(ps)==(160 if scope==16 else 20)
   signature=[]
   for v in ps:
    assert v['paper_format_disable']==0 and v['paper_pipeline_disable']==mask
    assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
    assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
    assert v['w4u8_qkv_ring_dispatch_count']==0
    signature.append([v.get(k,0) for k in metrics]);count+=1
   if ref is None:ref=signature
   else:assert signature==ref,(scope,phase,c,a);pair_count+=len(ps)
trace=[json.loads(p.read_text()) for p in R.glob('trace-*-a01-events.json')];assert len(trace)==5 and all(v['numerical_pass'] for v in trace)
out=dict(pass_all=True,timed_profiles=count,paired_profile_comparisons=pair_count,arithmetic_and_weight_metrics=metrics,independent_numerical_gates=[json.loads((R/n).read_text())['pass_all'] for n in ['single_gate.json','slice_gate.json','full_gate.json']],production_trace=False,trace_diagnostic_runs=5,trace_complete_runs=sum(v['trace_complete'] for v in trace),trace_status='N/A: incomplete FARF transport; no timeline or utilization claim',production_source='4421949fcdeaff146cf093a4ce529190b13e2ded',quality_claim=False)
assert all(out['independent_numerical_gates'])
with (R/'FINAL_AUDIT.json').open('x') as f:json.dump(out,f,indent=2)
print(out)
