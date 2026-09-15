import sys,json,shlex,statistics
from pathlib import Path
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'scripts'));import f16_baseline_refresh as f
f.preflight();runtime=f.read(f.R/'runtime-l3.json');remote=runtime['remote'];proof=next(z for z in f.read(f.R/'selected-local-provenance.json') if 'stack3' in z['package']);m=f.read(Path(proof['package'])/'manifest.json');rp='/data/local/tmp/llama32-htp/l32-0001/device-stack3-a01/package'
assert f.adb('shell','sha256sum '+rp+'/manifest.json').stdout.split()[0]==proof['manifest_sha256']
names=list(m['files'])
for i in range(0,len(names),32):
 lines=f.adb('shell','sha256sum '+' '.join(shlex.quote(rp+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
 for l in lines:
  h,n=l.split(None,1);assert h==m['files'][n.removeprefix(rp+'/')]['sha256']
for opt in [0,2]:
 tag=f'l3-opt{opt}-a01';d=f.R/tag;d.mkdir(parents=True,exist_ok=False);target=remote+'/'+tag;f.adb('shell','mkdir '+target)
 e=dict(LD_LIBRARY_PATH=remote,DSP_LIBRARY_PATH=remote,ADSP_LIBRARY_PATH=remote,QBH_F16F16_OPT=str(opt),QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_REPLAY_DECODE_STEPS='1',QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',QBH_KV_CACHE_CAPACITY='80',QBH_REPLAY_DUMP_DIR=target)
 if opt:e['QBH_W4F16_DECODE_AUDIT']='1'
 cmd='cd '+remote+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {rp} F16F16 1 '+f.ARGS
 f.write(d/'protocol.json',dict(command=cmd,runtime=runtime,package_proof=proof));z=f.adb('shell',cmd,check=False);(d/'stdout.jsonl').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);f.write(d/'exit.json',dict(returncode=z.returncode));f.adb('pull',target+'/.',f.win(d),check=False)
 assert z.returncode==0,(tag,z.stderr[-2000:],z.stdout[-2200:])
 rs=f.records(d/'stdout.jsonl');ps=[q for q in rs if q.get('record')=='replay_profile'];assert len(ps)==2
 for q in ps:
  assert q['vtcm_requested_bytes']==q['vtcm_acquired_bytes']==8388608 and q['block_invocation_count']==3
  assert all(q[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','w4f16_decode_conversion_audit_mismatches'])
 assert any(q.get('replay_sequence_complete') and q.get('all_steps_pass') for q in rs)
 out=dict(tag=tag,opt=opt,repeat=1,count=3,profiles=2,output_hashes=[q['output_hash'] for q in ps],selected_codes=[],token_sequences=[],layer_output_hashes=[[q[f'slice_layer_{i}']['output_hash'] for i in range(3)] for q in ps],dump_hashes={x.name:f.sha(x) for x in d.glob('*.bin')},physical_pass=True)
 f.write(d/'validated.json',out);print('CHAIN3_PASS',tag,flush=True)
f.compare(['l3-opt0-a01','l3-opt2-a01'])
