import sys,json,numpy as np
from pathlib import Path
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
import prototype_llama32_sp2 as p
from llama_u8_reference import unpack_w4_codes
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0040');p.OUT=R/'component';p.OUT.mkdir(exist_ok=False)
z=json.loads((R/'runtime-l1.json').read_text());(p.OUT/'deployment.json').write_text(json.dumps(dict(remote=z['remote'],source_head=z['seal']['source_head'],artifacts=z['seal']['files'])))
v=np.tile(np.arange(-32768,32768,dtype='i4'),2).reshape(16,8192).astype('i2');v[0]=-32768;v[1]=32767
w=unpack_w4_codes(Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016/layer7-a02/layer0'),'down',2048,8192)[:32]
for mode in [1,2]:
 name='int16-complete-grid-mode'+str(mode);p.fixture(name,v,w,mode,dict(contract='uniform INT16 range; frozen actual layer7 weights'));p.run(name,name+'-a01')
# Cover every signed value explicitly; boundary rows above replace some values.
v=np.tile(np.arange(-32768,32768,dtype='i4'),2).reshape(16,8192).astype('i2')
p.fixture('int16-all65536',v,w,1,dict(all_signed16_values=65536));p.run('int16-all65536','int16-all65536-a01')
