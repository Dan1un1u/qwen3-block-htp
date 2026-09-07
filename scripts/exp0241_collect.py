#!/usr/bin/env python3
import json,hashlib,subprocess,sys
from pathlib import Path
import exp0241_device as d

def load(tag):return [json.loads(l) for l in (d.RESULT/tag/'stdout.jsonl').read_text().splitlines() if '"record":"exp0240_profile"' in l]
def main():
 prefix=sys.argv[1] if len(sys.argv)>1 else ""
 d.preflight()
 assert json.loads((d.RESULT/'projection_audit_01_integer_audit.json').read_text())['pass']
 gold={cell:{x['replay_step']:x['output_hash'] for x in load(tag)} for cell,tag in [('control','smoke_control'),('direct','smoke_direct_02'),('lpbq32','smoke_lpbq')]}
 for f in (d.RESULT/'smoke_unit_02').glob('*.bin'):
  assert f.read_bytes()==(d.RESULT/'smoke_control'/f.name).read_bytes(),f.name
 for f in (d.RESULT/'smoke_direct_02').glob('*.bin'):
  assert f.read_bytes()==(d.RESULT/'smoke_lpbq'/f.name).read_bytes(),f.name
 binaries=[d.SOURCE/'android_ReleaseG_aarch64/ship/qwen3_block_cli',d.SOURCE/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',d.SOURCE/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']
 before={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.SOURCE,text=True).strip(),'binaries':{f.name:d.sha(f) for f in binaries},'remote_sha256':d.adb('shell',f'cd {d.REMOTE} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so'),'boot':d.adb('shell','cat /proc/sys/kernel/random/boot_id'),'threshold_percent':10,'execution':'one layer14 M64 prefill then8 real teacher-input M1 steps; runtime computes persistent KV; oneRPC/step','repeat10':'ten complete replays in a loaded prepared runtime, state and cache reinitialized before each prefill; mean per replay, not a frozen snapshot'}
 for h in before['binaries'].values():assert h in before['remote_sha256']
 (d.RESULT/(prefix+'formal_provenance.json')).write_text(json.dumps(before,indent=2)+'\n')
 for phase,rounds in [('short',5),('formal',10)]:
  for round in range(rounds):
   for repeat in [1,10]:
    order=['control','direct','lpbq32']
    shift=(round+(repeat==10))%3;order=order[shift:]+order[:shift]
    if round%2:order=order[::-1]
    for cell in order:
     tag=f'{prefix}{phase}_{round:02d}_r{repeat}_{cell}'
     x=d.run(cell,repeat,tag)
     assert all(v['output_hash']==gold[cell][v['replay_step']] for v in x),tag
     assert all(v['intermediate_ddr_read_bytes']==0 and v['intermediate_ddr_write_bytes']==0 and v['intermediate_spill_fill_count']==0 and v['vtcm_acquired_bytes']==8388608 for v in x),tag
   print(phase,'ROUND_COMPLETE',round,flush=True)
  assert d.adb('shell','cat /proc/sys/kernel/random/boot_id')==before['boot']
 print('COLLECTION COMPLETE; FULL MODEL NOT STARTED',flush=True)
if __name__=='__main__':main()
