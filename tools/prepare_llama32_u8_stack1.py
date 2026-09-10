from pathlib import Path
import json,os
from llama_reference import sha256
base=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/layers-a01');out=base.parent/'stack1-a01';out.mkdir(exist_ok=False)
p=base/'layer0-prefill';d=base/'layer0-decode'
for root in [p,d]:
 for n,v in json.loads((root/'manifest.json').read_text())['files'].items():assert sha256(root/n)==v['sha256']
for n in ['block_input_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin','rope_cos_f16.bin','rope_sin_f16.bin']:os.link(p/n,out/n)
for dst,src in [('replay_decode_input_00_u8.bin','block_input_u8.bin'),('replay_decode_reference_00_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin'),('replay_decode_rope_cos_00_f16.bin','rope_cos_f16.bin'),('replay_decode_rope_sin_00_f16.bin','rope_sin_f16.bin')]:os.link(d/src,out/dst)
l=out/'layer0';l.mkdir()
for f in p.iterdir():
 if f.name=='manifest.json':continue
 src=d/f.name if f.name.startswith('reference_kv_cache') else f
 os.link(src,l/f.name)
m=dict(experiment='L32-0003',recipe='W4A8',layers=1,cache_capacity=80,decode_steps=1,reference='SDK HMX + independent Llama integer math, prefill and one historical-KV decode',files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()});(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
