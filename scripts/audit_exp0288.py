"""Independent full28 hidden, final norm, greedy and native KV checks."""
from common_exp0288 import *
from device_exp0288 import records
from reference_w4u8_hmx import unpack_u8_hmx_activation,load_qparams_bin
from reference_math_exp0288 import norm
import numpy as np

def audit(tag,package="frontend64-a03"):
 preflight();d=R/tag;p=O/package;z=read(d/"validated.json");t=read(R/"frontend64-a03-teacher.json")
 q=load_qparams_bin(p/"generation_qparams_u8.bin");g=np.fromfile(p/"generation_final_norm_weight_f16.bin","<f2")
 steps=len(z["selected_codes"]);assert z["selected_codes"]==[list(v) for v in zip(t["u8_generated_ids"][:steps],t["u8_selected_codes"][:steps])]
 for j in range(steps):
  a=np.fromfile(d/f"generation_hidden_step{j:02d}_f32.bin","<f4")
  b=np.fromfile(p/f"audit_hidden_{j:02d}_f32.bin","<f4").reshape(-1,1024)[-1]
  assert np.array_equal(a,b),(tag,j,"hidden",int(np.count_nonzero(a!=b)))
  a=unpack_u8_hmx_activation(np.fromfile(d/f"generation_norm_step{j:02d}_u8_native.bin","u1"),1024)[:1]
  b=norm(b[None],g,q["generation_final_norm_output"]);assert np.array_equal(a,b),(tag,j,"norm")
 for i in range(28):
  for n in ["k","v"]:
   assert (d/f"generation_prefill_layer{i:02d}_{n}_cache_u8.bin").read_bytes()==(p/f"layer{i}/reference_kv_cache_{n}_hmx_u8_segmented_step00.bin").read_bytes(),(i,n)
 out=dict(pass_all=True,steps=steps,full28_hidden_and_finalnorm_exact=True,greedy_ids_and_codes_exact=True,all_prefill_native_KV_exact=True,physical=z["pass_all"],model_quality_accepted=False)
 write(d/"independent_gate.json",out);print("FULL_INDEPENDENT_PASS",tag,flush=True);return out
if __name__=="__main__":
 import sys
 audit(sys.argv[1])
