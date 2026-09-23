"""Independent 3B ordinary-A8 oracle: only Down representation differs."""
from llama32_3b_reference import *
from functools import lru_cache
@lru_cache(maxsize=64)
def ordinary_table(path):return np.fromfile(path,"<u2").reshape(256,256)
def layer(x,p,q,c,s,past=None,r4=False):
    cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
    def proj(a,n,iq,oq):
        return project_w4u8(a,p,n,{'q':H,'k':KH*D,'v':KH*D,'gate':F,'up':F}[n],a.shape[1],q[iq],q[oq],cv)
    def raw(a,n,scale,zero=0):
        w=unpack_w4_codes(p,n,H,a.shape[1]);ws=np.fromfile(p/(n+'_weight_w4_scale_f32.bin'),'<f4');acc=oracle(a.astype('i4')-zero,w)
        assert np.max(np.abs(acc))<2**31
        return acc.astype('f4')*(np.float32(scale)*ws)
    a=norm(x,np.fromfile(p/'input_norm_weight_f16.bin','<f2'),q['input_norm'])
    qr=proj(a,'q','input_norm','q_projection');kr=proj(a,'k','input_norm','k_projection');v=proj(a,'v','input_norm','v')
    qr=exact_qk_norm_rope_u8(qr,NH,q['q_projection'],q['q_rope'],None,c,s).reshape(-1,NH,D)
    kr=exact_qk_norm_rope_u8(kr,KH,q['k_projection'],q['k_rope'],None,c,s).reshape(-1,KH,D)
    k=kr.transpose(1,0,2);v=v.reshape(-1,KH,D).transpose(1,0,2);count=0
    if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
    av,score,prob=exact_attention_dynamic(qr,k,v,count,configs(q),cv,divide);av=av.reshape(len(x),H)
    o=raw(av,'o',q['attention_concat']['scale'],q['attention_concat']['zero_point']);res=x+o
    post=norm(res,np.fromfile(p/'post_norm_weight_f16.bin','<f2'),q['post_attention_norm'])
    g=proj(post,'gate','post_attention_norm','gate');u=proj(post,'up','post_attention_norm','up')
    mid=ordinary_table(str(p/'silu_up_lut_u16.bin'))[g,u]
    if r4:
        mid=mid.view('<f2').astype('f4')
        for h in [1<<i for i in range(13)]:
            rv=mid.reshape(len(x),-1,2*h);a=rv[:,:,:h].copy();b=rv[:,:,h:].copy();rv[:,:,:h]=a+b;rv[:,:,h:]=a-b
        mid*=np.float32(1/np.sqrt(8192))
        code=np.float32(mid*np.float32(1/q['middle']['scale']))+np.float32(q['middle']['zero_point'])
        mid=np.clip(np.trunc(np.float32(code+np.float32(.5))),0,255).astype('u1')
    down=raw(mid,'down',q['middle']['scale'],q['middle']['zero_point'])
    return res+down,(k,v),dict(input_norm=a,q=qr,k=kr,attention=av,o=o,residual=res,post=post,gate=g,up=u,middle=mid,down=down,score=score,probability=prob)
