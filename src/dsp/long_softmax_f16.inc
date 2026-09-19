/* Generic A16 long-context softmax shared by W16A16 and W4A16.
 * Row-major aligned M64 planes; FP32 exp/sum, FP16 stored exponential and
 * probability as in the dynamic floating contract. No tensor-sized DDR scratch. */
static HVX_Vector qbh_long_f16_splat(float x) {
    union {float f;int32_t i;} u={.f=x}; return Q6_V_vsplat_R(u.i);
}
static HVX_Vector qbh_long_f16_mul(HVX_Vector a,HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));
}
void qbh_hvx_long_softmax_f16(const __fp16 *scores,__fp16 *probability,
    uint32_t heads,uint32_t rows,uint32_t padded,uint32_t past,float scale) {
    static const int32_t indices[32] __attribute__((aligned(128)))=
      {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    float tmp[832] __attribute__((aligned(128)));
    for(uint32_t i=0;i<heads*64U*padded/64U;++i)
        ((HVX_Vector *)probability)[i]=Q6_V_vzero();
    for(uint32_t h=0;h<heads;++h)for(uint32_t row=0;row<rows;++row) {
        const __fp16 *src=scores+(h*64U+row)*padded;
        const uint32_t valid=past+row+1U;
        HVX_Vector mx=qbh_long_f16_splat(-INFINITY);
        for(uint32_t t=0;t<padded/64U;++t) {
            HVX_VectorPair x=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(src+t*64U));
            x=Q6_W_vshuff_VVR(Q6_V_hi_W(x),Q6_V_lo_W(x),-4);
            ((HVX_Vector *)tmp)[t*2]=Q6_V_lo_W(x);
            ((HVX_Vector *)tmp)[t*2+1]=Q6_V_hi_W(x);
        }
        for(uint32_t t=0;t<padded/32U;++t) {
            HVX_Vector x=qbh_long_f16_mul(((HVX_Vector *)tmp)[t],qbh_long_f16_splat(scale));
            HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*32),*(const HVX_Vector *)indices);
            x=Q6_V_vmux_QVV(live,x,qbh_long_f16_splat(-INFINITY));
            ((HVX_Vector *)tmp)[t]=x;mx=Q6_Vsf_vmax_VsfVsf(mx,x);
        }
        for(int n=64;n>=4;n>>=1)mx=Q6_Vsf_vmax_VsfVsf(mx,Q6_V_vror_VR(mx,n));
        HVX_Vector sum=Q6_V_vzero();
        for(uint32_t t=0;t<padded/32U;++t) {
            HVX_Vector x=qhmath_hvx_exp_vf(Q6_Vsf_vsub_VsfVsf(((HVX_Vector *)tmp)[t],mx));
            HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*32),*(const HVX_Vector *)indices);
            x=Q6_V_vmux_QVV(live,x,Q6_V_vzero());
            ((HVX_Vector *)tmp)[t]=x;sum=Q6_Vsf_vadd_VsfVsf(sum,x);
        }
        for(int n=64;n>=4;n>>=1)sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sum,n));
        union {int32_t i;float f;} total={.i=Q6_R_vextract_VR(sum,0)};
        HVX_Vector inv=qbh_long_f16_splat(1.0f/total.f);
        for(uint32_t t=0;t<padded/64U;++t) {
            HVX_VectorPair d=Q6_W_vdeal_VVR(((HVX_Vector *)tmp)[t*2+1],((HVX_Vector *)tmp)[t*2],-4);
            HVX_Vector e=Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(d),Q6_V_hi_W(d));
            d=Q6_Wsf_vcvt_Vhf(e);
            e=Q6_Vhf_vcvt_VsfVsf(qbh_long_f16_mul(Q6_V_lo_W(d),inv),qbh_long_f16_mul(Q6_V_hi_W(d),inv));
            *(HVX_Vector *)(probability+(h*64U+row)*padded+t*64U)=e;
        }
    }
    asm volatile("barrier":::"memory");
}

/* EXP0300: disjoint row ownership; caller zeros padding and provides
 * one 3328-byte VTCM scratch per context. Identical per-row math/order. */
void qbh_hvx_long_softmax_f16_partition(const __fp16 *scores,__fp16 *probability,
    uint32_t heads,uint32_t rows,uint32_t padded,uint32_t past,float scale,
    float *tmp,uint32_t first,uint32_t stride) {
    static const int32_t indices[32] __attribute__((aligned(128)))=
      {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    for(uint32_t task=first;task<heads*rows;task+=stride) {
        const uint32_t h=task/rows,row=task%rows;
        const __fp16 *src=scores+(h*64U+row)*padded;
        const uint32_t valid=past+row+1U;
        HVX_Vector mx=qbh_long_f16_splat(-INFINITY);
        const uint32_t active32=(valid+31U)/32U;
        const uint32_t active64=(valid+63U)/64U;
        const HVX_Vector scalev=qbh_long_f16_splat(scale);
        const HVX_Vector neginf=qbh_long_f16_splat(-INFINITY);
        /* Fuse conversion with scale/mask. Preserve low-then-high max order.
         * Entirely masked tiles cannot contribute to max or positive exp sum. */
        for(uint32_t t=0;t<active64;++t) {
            HVX_VectorPair pair=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(src+t*64U));
            pair=Q6_W_vshuff_VVR(Q6_V_hi_W(pair),Q6_V_lo_W(pair),-4);
            HVX_Vector lo=qbh_long_f16_mul(Q6_V_lo_W(pair),scalev);
            HVX_Vector hi=qbh_long_f16_mul(Q6_V_hi_W(pair),scalev);
            HVX_VectorPred live0=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*64),*(const HVX_Vector *)indices);
            HVX_VectorPred live1=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*64-32),*(const HVX_Vector *)indices);
            lo=Q6_V_vmux_QVV(live0,lo,neginf);
            hi=Q6_V_vmux_QVV(live1,hi,neginf);
            ((HVX_Vector *)tmp)[t*2]=lo;
            ((HVX_Vector *)tmp)[t*2+1]=hi;
            mx=Q6_Vsf_vmax_VsfVsf(mx,lo);
            mx=Q6_Vsf_vmax_VsfVsf(mx,hi);
        }
        for(int n=64;n>=4;n>>=1)mx=Q6_Vsf_vmax_VsfVsf(mx,Q6_V_vror_VR(mx,n));
        HVX_Vector sum=Q6_V_vzero();
        for(uint32_t t=0;t<active32;++t) {
            HVX_Vector x=qhmath_hvx_exp_vf(Q6_Vsf_vsub_VsfVsf(((HVX_Vector *)tmp)[t],mx));
            HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*32),*(const HVX_Vector *)indices);
            x=Q6_V_vmux_QVV(live,x,Q6_V_vzero());
            ((HVX_Vector *)tmp)[t]=x;sum=Q6_Vsf_vadd_VsfVsf(sum,x);
        }
        if(active32 & 1U) ((HVX_Vector *)tmp)[active32]=Q6_V_vzero();
        for(int n=64;n>=4;n>>=1)sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sum,n));
        union {int32_t i;float f;} total={.i=Q6_R_vextract_VR(sum,0)};
        HVX_Vector inv=qbh_long_f16_splat(1.0f/total.f);
        for(uint32_t t=0;t<active64;++t) {
            HVX_VectorPair d=Q6_W_vdeal_VVR(((HVX_Vector *)tmp)[t*2+1],((HVX_Vector *)tmp)[t*2],-4);
            HVX_Vector e=Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(d),Q6_V_hi_W(d));
            d=Q6_Wsf_vcvt_Vhf(e);
            e=Q6_Vhf_vcvt_VsfVsf(qbh_long_f16_mul(Q6_V_lo_W(d),inv),qbh_long_f16_mul(Q6_V_hi_W(d),inv));
            *(HVX_Vector *)(probability+(h*64U+row)*padded+t*64U)=e;
        }
    }
    asm volatile("barrier":::"memory");
}
