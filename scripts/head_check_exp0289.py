"""Independent norm/head checks on audited hardware hidden states, not full-model alignment."""
import torch,numpy as np
import measure_exp0289 as m
from prepare_exp0288 import tensor
from reference_w4_exp0289 import RawW4
from common_exp0289 import *
@torch.inference_mode()
def main():
 preflight();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 gamma=tensor('model.norm.weight').half().cuda()
 results={}
 for recipe,(pkg,audit) in m.ARM.items():
  if recipe=='f16f16':weight=tensor('model.embed_tokens.weight').half().cuda()
  else:weight=RawW4(O/pkg,'generation_lm_head',151936,1024).cuda()
  sig=m.signatures(audit);rows=[]
  for step,(token,bits) in enumerate(sig):
   x=torch.from_numpy(np.fromfile(R/audit/f'generation_hidden_norm_step{step:02d}_f16.bin','<f2').copy()).cuda()
   hidden=x[:1024];actual_norm=x[1024:]
   ref_norm=(hidden.float()*torch.rsqrt(hidden.float().square().mean()+1e-6)*gamma.float()).half()
   error=float(torch.linalg.vector_norm(actual_norm.float()-ref_norm.float())/torch.linalg.vector_norm(ref_norm.float()))
   if recipe=='f16f16':logits=torch.nn.functional.linear(actual_norm.float(),weight.float()).half()
   else:logits=(torch.nn.functional.linear(actual_norm.float(),weight.codes.float())*weight.scale.float()).half()
   ref=float(logits[token]);maximum=float(logits.max())
   actual=float(np.asarray([bits],dtype='<u2').view('<f2')[0])
   ulp=float(abs(np.spacing(np.float16(ref))));maxulp=float(abs(np.spacing(np.float16(maximum))))
   logit_error=abs(ref-actual);selection_gap=maximum-ref
   ok=bool(torch.isfinite(x).all() and error<=.003 and logit_error<=ulp and selection_gap<=maxulp)
   rows.append(dict(step=step,norm_nrmse=error,token=token,reference_argmax=int(logits.argmax()),
    actual_selected_logit=actual,independent_selected_logit=ref,reference_max_logit=maximum,
    logit_error=logit_error,one_ulp=ulp,selection_gap=selection_gap,pass_all=ok))
  results[recipe]=rows
 write(R/'head-boundary-checks.json',dict(pass_all=all(z['pass_all'] for rows in results.values() for z in rows),
  scope='actual audited hidden -> independent norm; actual norm -> independent entire head; one FP16 ULP permits matrix reduction rounding, not exact full-model alignment',
  results=results))
 print({k:dict(pass_all=all(z['pass_all'] for z in v),norm_max=max(z['norm_nrmse'] for z in v),head_max_ulp=max(z['logit_error']/z['one_ulp'] for z in v),argmax_mismatches=sum(z['token']!=z['reference_argmax'] for z in v)) for k,v in results.items()},flush=True)
 assert all(z['pass_all'] for rows in results.values() for z in rows)
if __name__=='__main__':main()
