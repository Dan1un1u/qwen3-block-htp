from pathlib import Path
import hashlib,json
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0282');runs=[]
for arm in ['A8','SP2']:
 for mode in ['greedy','fixed']:
  name=f'full-{arm}-{mode}-audit';old=R/'A5'/name;new=R/'A5-fair-a02'/name
  a=json.loads((old/'validated.json').read_text());b=json.loads((new/'validated.json').read_text());assert a['selected_codes']==b['selected_codes']
  files=[p.name for p in old.glob('*.bin') if p.name!='eval.bin'];assert files
  for n in files:assert (old/n).read_bytes()==(new/n).read_bytes(),(arm,mode,n)
  runs.append(dict(arm=arm,mode=mode,exact_files=files,source=str(old)))
with (R/'A5-fair-a02/unchanged_full_arithmetic_gate.json').open('x') as f:json.dump(dict(pass_all=True,runs=runs),f,indent=2)
print('FULL_A8_SP2_ARITHMETIC_UNCHANGED',sum(len(x['exact_files']) for x in runs))
