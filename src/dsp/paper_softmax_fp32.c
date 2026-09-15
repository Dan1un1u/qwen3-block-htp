/* A9 FP32 vector reference arm. Vendor QHL vector exp, native score/probability
 * tiles, no intermediate tensor allocation or scalar per-element arithmetic. */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stddef.h>
#include "attention_u8_core.h"
#include "qhmath_hvx_vector.h"
static const int32_t lanes[32] __attribute__((aligned(128)))={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
static inline __attribute__((always_inline)) HVX_Vector splat(float x){union{float f;int32_t i;}u={.f=x};return Q6_V_vsplat_R(u.i);}
static inline __attribute__((always_inline)) HVX_Vector mul(HVX_Vector a,HVX_Vector b){return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));}
static inline __attribute__((always_inline)) HVX_Vector load32(const uint8_t *p){
    /* Aligned128B load stays inside one2048B score tile, including lastrow. */
    HVX_Vector v=*(const HVX_Vector *)((uintptr_t)p&~(uintptr_t)127U);
    v=Q6_V_vror_VR(v,(uintptr_t)p&127U);
    return Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v))));
}
static inline __attribute__((always_inline)) HVX_Vector maximum(HVX_Vector v){
    v=Q6_Vw_vmax_VwVw(v,Q6_V_vror_VR(v,64));v=Q6_Vw_vmax_VwVw(v,Q6_V_vror_VR(v,32));
    v=Q6_Vw_vmax_VwVw(v,Q6_V_vror_VR(v,16));v=Q6_Vw_vmax_VwVw(v,Q6_V_vror_VR(v,8));
    return Q6_Vw_vmax_VwVw(v,Q6_V_vror_VR(v,4));
}
static inline __attribute__((always_inline)) HVX_Vector sum32(HVX_Vector v){
    v=Q6_Vsf_vadd_VsfVsf(v,Q6_V_vror_VR(v,64));v=Q6_Vsf_vadd_VsfVsf(v,Q6_V_vror_VR(v,32));
    v=Q6_Vsf_vadd_VsfVsf(v,Q6_V_vror_VR(v,16));v=Q6_Vsf_vadd_VsfVsf(v,Q6_V_vror_VR(v,8));
    return Q6_Vsf_vadd_VsfVsf(v,Q6_V_vror_VR(v,4));
}
static inline __attribute__((always_inline)) HVX_Vector exptile(HVX_Vector raw,HVX_Vector mx,HVX_Vector scale,uint32_t valid){
    HVX_Vector diff=Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(raw,mx));
    HVX_Vector e=qhmath_hvx_exp_vf(mul(diff,scale));
    HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid),*(const HVX_Vector *)lanes);
    return Q6_V_vmux_QVV(live,e,Q6_V_vzero());
}
static inline __attribute__((always_inline)) void publish(HVX_Vector e,HVX_Vector inv,uint8_t *dst,float *dump){
    HVX_Vector p=mul(e,inv);
    if(dump)*(HVX_Vector *)dump=p;
    /* Nonnegative round-half-up, exactly the retained probability U8 boundary. */
    HVX_Vector w=qhmath_hvx_vw_truncate_vsf(Q6_Vsf_vadd_VsfVsf(mul(p,splat(255.0f)),splat(0.5f)));
    HVX_Vector h=Q6_Vh_vpacke_VwVw(w,w);HVX_Vector b=Q6_Vb_vpacke_VhVh(h,h);
    uint32_t shift=(uintptr_t)dst&127U;
    HVX_VectorPred pred=Q6_Q_and_QQn(Q6_Q_vsetq2_R(shift+32U),Q6_Q_vsetq_R(shift));
    Q6_vmem_QRIV(pred,(HVX_Vector *)dst,Q6_V_vlalign_VVR(b,b,shift));
}
void qbh_attention_fp32_softmax_native(const uint8_t *scores,uint8_t *probability,
    uint32_t heads,uint32_t first_row,uint32_t rows,uint32_t past,
    uint32_t padded,const struct qbh_attention_config *config,float *dump,
    struct qbh_attention_u8_telemetry *telemetry){
    const uint32_t tiles=padded/32U;
    const float scale=(float)config->score_multiplier*(0.69314718055994530942f/(float)(1U<<config->fraction_bits));
    const HVX_Vector sc=splat(scale),zero=Q6_V_vzero();
    for(uint32_t head=0;head<heads;head++)for(uint32_t row=first_row;row<first_row+rows;row++){
        const uint8_t *base=scores+(size_t)head*tiles*2048U+row*32U;
        uint8_t *dst=probability+(size_t)head*tiles*2048U+row*32U;
        uint32_t valid=past+row+1U;
        HVX_Vector a=load32(base),b=tiles>1?load32(base+2048):zero;
        HVX_Vector c=tiles>2?load32(base+4096):zero,d=tiles>3?load32(base+6144):zero;
        HVX_Vector lane=*(const HVX_Vector *)lanes;
        a=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(valid),lane),a,zero);
        b=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-32),lane),b,zero);
        c=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-64),lane),c,zero);
        d=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-96),lane),d,zero);
        HVX_Vector mx=maximum(Q6_Vw_vmax_VwVw(Q6_Vw_vmax_VwVw(a,b),Q6_Vw_vmax_VwVw(c,d)));
        a=exptile(a,mx,sc,valid);
        b=tiles>1?exptile(b,mx,sc,valid>32?valid-32:0):zero;
        c=tiles>2?exptile(c,mx,sc,valid>64?valid-64:0):zero;
        d=tiles>3?exptile(d,mx,sc,valid>96?valid-96:0):zero;
        HVX_Vector sv=sum32(Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(a,b),Q6_Vsf_vadd_VsfVsf(c,d)));
        union{float f;int32_t i;}su={.i=Q6_R_vextract_VR(sv,0)};
        HVX_Vector inv=splat(1.0f/su.f);
        float *fd=dump?dump+((size_t)head*64U+row)*padded:NULL;
        publish(a,inv,dst,fd);
        if(tiles>1)publish(b,inv,dst+2048,fd?fd+32:NULL);
        if(tiles>2)publish(c,inv,dst+4096,fd?fd+64:NULL);
        if(tiles>3)publish(d,inv,dst+6144,fd?fd+96:NULL);
        if(telemetry){uint32_t sum=0;for(uint32_t k=0;k<valid;k++)sum+=dst[(k/32)*2048+k%32];
            if(sum<telemetry->probability_row_sum_min)telemetry->probability_row_sum_min=sum;
            if(sum>telemetry->probability_row_sum_max)telemetry->probability_row_sum_max=sum;}
    }
    asm volatile("barrier":::"memory");
}
