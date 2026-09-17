"""Independent actual FP32 residual -> final norm -> FP16 HMX head audit."""
import torch,numpy as np,json
import measure_exp0292 as m
from prepare_exp0288 import tensor
from check_exp0292_components import norm
from common_exp0289 import write
@torch.inference_mode()
def main():
 m.d.preflight();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 weight=tensor('model.embed_tokens.weight').half().cuda()
 gamma=tensor('model.norm.weight').half().cpu().numpy().astype('f4')
 rows=[]
 for step,(token,bits) in enumerate(m.signatures(m.R/'audit/fp32')):
  raw=(m.R/'audit/fp32'/f'generation_hidden_norm_step{step:02d}_f32_f16.bin').read_bytes()
  hidden=np.frombuffer(raw,'<f4',count=1024)
  actual_norm=np.frombuffer(raw,'<f2',offset=4096).copy();assert len(actual_norm)==1024
  expected=norm(hidden[None],gamma)[0]
  delta=abs(actual_norm.astype('f4')-expected.astype('f4'))
  ulps=np.maximum(abs(np.spacing(abs(expected))),np.float16(2**-24)).astype('f4')
  norm_ulp=float((delta/ulps).max())
  logits=torch.nn.functional.linear(torch.from_numpy(actual_norm).cuda().float(),weight.float()).half()
  ref=float(logits[token]);maximum=float(logits.max());actual=float(np.asarray([bits],dtype='<u2').view('<f2')[0])
  ulp=float(abs(np.spacing(np.float16(ref))));maxulp=float(abs(np.spacing(np.float16(maximum))))
  ok=bool(np.isfinite(hidden).all() and np.isfinite(actual_norm).all() and norm_ulp<=1 and abs(ref-actual)<=ulp and maximum-ref<=maxulp)
  rows.append(dict(step=step,norm_max_half_ulp=norm_ulp,selected_token=token,reference_argmax=int(logits.argmax()),logit_error=abs(ref-actual),logit_one_ulp=ulp,selection_gap=maximum-ref,pass_all=ok))
 write(m.R/'head-boundary-checks.json',dict(pass_all=all(z['pass_all'] for z in rows),scope='actual FP32 hidden -> independently ordered FP32 norm -> FP16, actual norm -> independent entire FP16 head; no model quality claim',results=rows))
 print(dict(pass_all=all(z['pass_all'] for z in rows),norm_max_half_ulp=max(z['norm_max_half_ulp'] for z in rows),argmax_mismatches=sum(z['selected_token']!=z['reference_argmax'] for z in rows)),flush=True)
 assert all(z['pass_all'] for z in rows)
if __name__=='__main__':main()
