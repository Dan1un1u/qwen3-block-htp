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

/* C3: native probability producer; scratch is832 FP32 +832 FP16 per context. */
void qbh_hvx_long_softmax_f16_native_partition(const __fp16 *scores,__fp16 *probability,
    uint32_t heads,uint32_t rows,uint32_t padded,uint32_t past,float scale,
    float *tmp,uint32_t first,uint32_t stride) {
    static const int32_t indices[32] __attribute__((aligned(128)))=
      {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const uint32_t pairs=(rows+1U)/2U;
    __fp16 *first_row=(__fp16 *)(tmp+832U);
    for(uint32_t task=first;task<heads*pairs;task+=stride) {
        const uint32_t h=task/pairs,pair_row=(task%pairs)*2U;
        const uint32_t first_active64=(past+pair_row+64U)/64U;
        for(uint32_t rr=0U;rr<2U && pair_row+rr<rows;++rr) {
        const uint32_t row=pair_row+rr;
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
            if(rr==0U && pair_row+1U<rows) {
                ((HVX_Vector *)first_row)[t]=e;
            } else {
                HVX_Vector a=rr==0U ? e :
                    (t<first_active64 ? ((HVX_Vector *)first_row)[t] : Q6_V_vzero());
                HVX_Vector b=rr==0U ? Q6_V_vzero() : e;
                HVX_VectorPair packed=Q6_W_vshuff_VVR(b,a,-2);
                /* Entire row pair belongs to one worker: full-vector stores
                 * have disjoint addresses even when rows is odd. */
                size_t tile=(size_t)h*64U*padded+
                    ((size_t)(pair_row/32U)*(padded/32U)+t*2U)*1024U;
                HVX_Vector *dst=(HVX_Vector *)(probability+tile)+(pair_row%32U)/2U;
                dst[0]=Q6_V_lo_W(packed);dst[16]=Q6_V_hi_W(packed);
            }
        }
        }
    }
    asm volatile("barrier":::"memory");
}

/* EXP0303: generic row-owned padding and tail-only causal masks. */
void qbh_hvx_long_softmax_f16_partition_tail(const __fp16 *scores,__fp16 *probability,
    uint32_t heads,uint32_t rows,uint32_t padded,uint32_t past,float scale,
    float *tmp,uint32_t first,uint32_t stride) {
    static const int32_t indices[32] __attribute__((aligned(128)))=
      {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    for(uint32_t task=first;task<heads*64U;task+=stride) {
        const uint32_t h=task/64U,row=task%64U;
        __fp16 *dst=probability+task*padded;
        if(row>=rows) {
            for(uint32_t t=0;t<padded/64U;++t)
                ((HVX_Vector *)dst)[t]=Q6_V_vzero();
            continue;
        }
        const __fp16 *src=scores+(h*64U+row)*padded;
        const uint32_t valid=past+row+1U;
        HVX_Vector mx=qbh_long_f16_splat(-INFINITY);
        const uint32_t active32=(valid+31U)/32U;
        const uint32_t active64=(valid+63U)/64U;
        for(uint32_t t=active64;t<padded/64U;++t)
            ((HVX_Vector *)dst)[t]=Q6_V_vzero();
        const HVX_Vector scalev=qbh_long_f16_splat(scale);
        const HVX_Vector neginf=qbh_long_f16_splat(-INFINITY);
        /* Fuse conversion with scale/mask. Preserve low-then-high max order.
         * Entirely masked tiles cannot contribute to max or positive exp sum. */
        for(uint32_t t=0;t<active64;++t) {
            HVX_VectorPair pair=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(src+t*64U));
            pair=Q6_W_vshuff_VVR(Q6_V_hi_W(pair),Q6_V_lo_W(pair),-4);
            HVX_Vector lo=qbh_long_f16_mul(Q6_V_lo_W(pair),scalev);
            HVX_Vector hi=qbh_long_f16_mul(Q6_V_hi_W(pair),scalev);
            if(t+1U==active64 && (valid&63U)) {
            HVX_VectorPred live0=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*64),*(const HVX_Vector *)indices);
            HVX_VectorPred live1=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*64-32),*(const HVX_Vector *)indices);
            lo=Q6_V_vmux_QVV(live0,lo,neginf);
            hi=Q6_V_vmux_QVV(live1,hi,neginf);
            }
            ((HVX_Vector *)tmp)[t*2]=lo;
            ((HVX_Vector *)tmp)[t*2+1]=hi;
            mx=Q6_Vsf_vmax_VsfVsf(mx,lo);
            mx=Q6_Vsf_vmax_VsfVsf(mx,hi);
        }
        for(int n=64;n>=4;n>>=1)mx=Q6_Vsf_vmax_VsfVsf(mx,Q6_V_vror_VR(mx,n));
        HVX_Vector sum=Q6_V_vzero();
        for(uint32_t t=0;t<active32;++t) {
            HVX_Vector x=qhmath_hvx_exp_vf(Q6_Vsf_vsub_VsfVsf(((HVX_Vector *)tmp)[t],mx));
            if(t+1U==active32 && (valid&31U)) {
            HVX_VectorPred live=Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R((int32_t)valid-(int32_t)t*32),*(const HVX_Vector *)indices);
            x=Q6_V_vmux_QVV(live,x,Q6_V_vzero());
            }
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
