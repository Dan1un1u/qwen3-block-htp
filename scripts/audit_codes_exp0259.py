import json,sys
from pathlib import Path
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0259')
def rec(p):return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
count=0
for arm in [0,1]:
 gold=[(q['selected_token_id'],q['selected_logit_half_bits']) for q in rec(R.parent/f'exp0258/smoke_a{arm}/stdout.jsonl') if 'selected_logit_half_bits' in q];assert len(gold)==16
 for p in sorted(R.glob(f'full_*/*_a{arm}/validated.json')):
  qs=[q for q in rec(p.parent/'stdout.jsonl') if 'selected_logit_half_bits' in q]
  assert all((q['selected_token_id'],q['selected_logit_half_bits'])==gold[i%16] for i,q in enumerate(qs)),str(p)
  count+=len(qs)
print('EXACT_TOKEN_AND_U8_CODE',count)
if '--final' in sys.argv:
 assert count==5280
 with (R/'selected_code_determinism.json').open('x') as f:json.dump(dict(pass_all=True,steps=count,reference='sealed EXP0258 smoke per arm',tokens_and_selected_U8_codes_exact=True,scope='selected maxima only, not full-vocabulary logit equivalence'),f,indent=2)
