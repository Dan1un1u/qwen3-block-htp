"""Shape-aware independent integer QK/AV with float64 Softmax oracle."""
import numpy as np
def floating_attention(q,k,v,past,cfg):
 # Independent Float64 continuous softmax; raw QK and AV integer conversion
 # stay identical to the retained hardware contract.
 rows=len(q);width=k.shape[1];nh=q.shape[1];dim=q.shape[2];group=nh//len(cfg);score=np.empty((nh,rows,width),'u1');prob=np.zeros_like(score);av=np.empty((rows,nh,dim),'u1')
 for g,c in enumerate(cfg):
  qq=q[:,group*g:group*g+group].transpose(1,0,2).astype('i8');kk=k[g].astype('i8');vv=v[g].astype('i8')
  acc=(qq-int(c[4]))@np.clip(kk-int(c[5]),-128,127).T;div=2**int(c[11]);raw=np.clip((acc+div//2)//div+128,0,255);pp=np.zeros_like(raw)
  for h in range(group):
   for row in range(rows):
    n=past+row+1;xs=raw[h,row,:n].astype('f8');xs=(xs-xs.max())*int(c[12])/2**int(c[2]);ex=np.exp2(xs);pp[h,row,:n]=np.clip(np.floor(255*ex/ex.sum()+.5),0,255)
  vc=vv-int(c[6]);vc=np.clip(np.sign(vc)*((np.abs(vc)*int(c[9])+int(c[10])//2)//int(c[10])),-128,127)
  accum=pp@vc;div=2**int(c[13]);zero=int(c[8]) if int(c[14])==1 else 128;temp=np.clip((accum+div//2)//div+zero,0,255);value=temp if int(c[14])==1 else np.clip((temp-128)*int(c[14])+int(c[8]),0,255)
  av[:,group*g:group*g+group]=value.transpose(1,0,2);score[group*g:group*g+group]=raw;prob[group*g:group*g+group]=pp
 return av,score,prob

