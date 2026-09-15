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
/* Separate phases keep tensor values in explicit caller-owned VTCM scratch.
 * This also bounds register pressure of the inlined vendor exponent polynomial. */
__attribute__((noinline)) static int32_t rowmax(const uint8_t *src,uint32_t tiles,uint32_t valid){
    HVX_Vector mx=Q6_V_vzero();
    for(uint32_t t=0;t<tiles;t++){
        HVX_Vector v=load32(src+t*2048U);
        HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*32),*(const HVX_Vector *)lanes);
        mx=Q6_Vw_vmax_VwVw(mx,Q6_V_vmux_QVV(live,v,Q6_V_vzero()));
    }
    return Q6_R_vextract_VR(maximum(mx),0);
}
__attribute__((noinline)) static void rowexp(const uint8_t *src,float *tmp,uint32_t tiles,uint32_t valid,int32_t mx,float scale){
    for(uint32_t t=0;t<tiles;t++){
        uint32_t n=valid>t*32?valid-t*32:0;
        *(HVX_Vector *)(tmp+t*32)=exptile(load32(src+t*2048),Q6_V_vsplat_R(mx),splat(scale),n);
    }
}
__attribute__((noinline)) static float rowsum(const float *tmp,uint32_t tiles){
    HVX_Vector v=Q6_V_vzero();
    for(uint32_t t=0;t<tiles;t++)v=Q6_Vsf_vadd_VsfVsf(v,*(const HVX_Vector *)(tmp+t*32));
    union{float f;int32_t i;}u={.i=Q6_R_vextract_VR(sum32(v),0)};return u.f;
}
__attribute__((noinline)) static void rowpublish(const float *tmp,uint8_t *dst,float *dump,uint32_t tiles,float inverse){
    for(uint32_t t=0;t<tiles;t++)publish(*(const HVX_Vector *)(tmp+t*32),splat(inverse),dst+t*2048,dump?dump+t*32:NULL);
}
void qbh_attention_fp32_softmax_native(const uint8_t *scores,uint8_t *probability,
    uint32_t heads,uint32_t first_row,uint32_t rows,uint32_t past,
    uint32_t padded,const struct qbh_attention_config *config,float *scratch,float *dump,
    struct qbh_attention_u8_telemetry *telemetry){
    const uint32_t tiles=padded/32U;
    const float scale=(float)config->score_multiplier*(0.69314718055994530942f/(float)(1U<<config->fraction_bits));
    for(uint32_t head=0;head<heads;head++)for(uint32_t row=first_row;row<first_row+rows;row++){
        const uint8_t *base=scores+(size_t)head*tiles*2048U+row*32U;
        uint8_t *dst=probability+(size_t)head*tiles*2048U+row*32U;
        uint32_t valid=past+row+1U;
        int32_t mx=rowmax(base,tiles,valid);
        rowexp(base,scratch,tiles,valid,mx,scale);
        float inverse=1.0f/rowsum(scratch,tiles);
        rowpublish(scratch,dst,dump?dump+((size_t)head*64U+row)*padded:NULL,tiles,inverse);
        if(telemetry){uint32_t sum=0;for(uint32_t k=0;k<valid;k++)sum+=dst[(k/32)*2048+k%32];
            if(sum<telemetry->probability_row_sum_min)telemetry->probability_row_sum_min=sum;
            if(sum>telemetry->probability_row_sum_max)telemetry->probability_row_sum_max=sum;}
    }
    asm volatile("barrier":::"memory");
}
