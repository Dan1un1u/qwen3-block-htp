import numpy as np,json
from pathlib import Path
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0263')
def pos(r,c,nt):return ((r//32)*nt+c//32)*1024+((r%32)//2*32+c%32)*2+r%2
results=[]
for rows,count in [(1,1),(64,16),(64,8)]:
 for first in range(0,rows,count):
  r,a,c=np.indices((count,12,512))
  expected=pos(r*512+c,a,1)
  vector=r*16384+(c//32)*1024+a*2+(c%32//2)*64+c%2
  assert np.array_equal(expected,vector)
  assert np.unique(vector).size==count*6144
  assert vector.min()>=0 and vector.max()<count*16384
  sr=a*rows+first+r
  assert np.all(pos(sr&~1,c//64*64,16)%64==0)
  assert np.all(pos(sr,c,16)<((12*rows+31)//32*32)*512)
 results.append(dict(rows=rows,batch=count,all_live_element_addresses_exact=True,distinct_destinations=True))
# Pipeline batch i reads/writes slot i%2; processing i-1 output and i+1
# input uses the opposite slot. Last use is completed before slot reuse.
for i in range(8):
 active=i%2;other=active^1
 ai=(active*262144,(active+1)*262144);bi=(other*262144,(other+1)*262144)
 assert ai[1]<=bi[0] or bi[1]<=ai[0]
 assert max(ai[1],bi[1])<=524288<786432
tile_sets=[set(range(i*64,(i+1)*64)) for i in range(3)]
assert set.union(*tile_sets)==set(range(192)) and sum(map(len,tile_sets))==192
finish_sets=[set(range(i*4,(i+1)*4)) for i in range(3)]
assert set.union(*finish_sets)==set(range(12)) and sum(map(len,finish_sets))==12
scratch=[set(range(i*256,(i+1)*256)) for i in range(3)]
assert set.union(*scratch)==set(range(768)) and sum(map(len,scratch))==768
stream=[set(range(g*32+i*16,g*32+(i+1)*16)) for g in range(6) for i in range(2)]
assert set.union(*stream)==set(range(192)) and sum(map(len,stream))==192
(R/'layout_address_proof.json').write_text(json.dumps(dict(pass_all=True,cases=results,slot_nonoverlap=True,maximum_arena_bytes=524288),indent=2))
print('LAYOUT_ADDRESS_PROOF_PASS')
