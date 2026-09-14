/* L32-0016 accuracy prototype. Native packed W4 products, never U8 output.
 * Raw stores are signed24, proven at host preparation for EVERY possible U8
 * input. SP2's reconstructed signed32 bound is proved separately. No DDR spill.
 * Scalar FP32 epilogue deliberately prioritizes an auditable numerical contract;
 * its add time belongs to the O/Down projection ledger (no second residual add).
 */
static inline HVX_Vector qbh_fp32_splat(float x) {
    union {float f;int32_t i;} u={.f=x};return Q6_V_vsplat_R(u.i);
}
static inline HVX_Vector qbh_fp32_mul(HVX_Vector a,HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));
}
static inline HVX_Vector qbh_fp32_dot(HVX_Vector low,HVX_Vector high,
    HVX_Vector sum,uint32_t sp2,int32_t zero) {
    /* Modular operations exactly recover the separately proven S32 result. */
    return sp2 ? Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(low,Q6_Vw_vasl_VwR(high,8)),
            Q6_Vw_vasl_VwR(sum,15))
        : Q6_Vw_vsub_VwVw(low,Q6_Vw_vmpyi_VwRh(sum,zero));
}
static void qbh_llama_fp32_epilogue(const uint8_t *lo,const uint8_t *hi,
    const float *scale,float *dst,uint32_t rows,uint32_t sp2,int32_t zero) {
    HVX_Vector scales=*(const HVX_Vector *)scale,sum=*(const HVX_Vector *)(scale+32U);
    for(uint32_t row=0;row<rows;row+=4U) {
        struct qbh_sp2_words l=qbh_sp2_chunk24(lo,row*32U);
        struct qbh_sp2_words h=sp2?qbh_sp2_chunk24(hi,row*32U):l;
        HVX_Vector lv[4]={l.v0,l.v1,l.v2,l.v3},hv[4]={h.v0,h.v1,h.v2,h.v3};
        for(uint32_t k=0;k<4U && row+k<rows;k++) {
            HVX_Vector acc=qbh_fp32_dot(lv[k],hv[k],sum,sp2,zero);
            HVX_Vector add=qbh_fp32_mul(Q6_Vsf_equals_Vw(acc),scales);
            HVX_Vector *out=(HVX_Vector *)(dst+(row+k)*QBH_BLOCK_HIDDEN);
            *out=Q6_Vsf_vadd_VsfVsf(*out,add);
        }
    }
}
static int qbh_llama_w4_fp32_residual(struct qbh_block_hmx_worker *w) {
#pragma clang fp contract(off)
    uint8_t *lo=w->sp2_scratch,*hi=lo+8192U;
    uint32_t *bias=(uint32_t *)(lo+16384U);
    if(!lo || !w->fp32_residual || w->m_tiles!=1U ||
       (w->fp32_rows!=1U && w->fp32_rows!=64U)) return AEE_EBADPARM;
    for(uint32_t nt=0;nt<w->n_tiles;nt++) {
        const uint8_t *weight=(const uint8_t *)w->weight+(size_t)nt*w->k_tiles*512U;
        const float *scale=(const float *)w->scale_or_bias+nt*64U;
        uint32_t passes=w->fp32_sp2 && w->fp32_rows==64U ? 2U:1U;
        for(uint32_t pass=0;pass<passes;pass++) {
            for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[n+32U]=0U;}
            qbh_hmx_begin_u8s8_output(bias);
            qbh_hmx_accumulate_u8n4_projection(pass?w->sp2_high:w->activation,weight,w->k_tiles);
            read_i32(pass?hi:lo,bias,3U);
        }
        qbh_llama_fp32_epilogue(lo,w->fp32_sp2 ? (w->fp32_rows==64U?hi:lo+128U):NULL,
            scale,w->fp32_residual+w->fp32_first_channel+nt*32U,
            w->fp32_rows,w->fp32_sp2,w->fp32_input_zero);
    }
    asm volatile("barrier":::"memory");
    return AEE_SUCCESS;
}
/* Sequential FP32 reduction across channels, vectorized across 32 rows.
 * Scratch reuses the idle SP2 VTCM raw slots; no tensor on the thread stack. */
static void qbh_llama_fp32_norm(const float *x,const __fp16 *gamma,
    uint8_t *out,const struct qbh_block_qparam *q,uint32_t rows,
    uint32_t width,uint32_t native,uint8_t *scratch) {
#pragma clang fp contract(off)
    HVX_Vector *tile=(HVX_Vector *)scratch;
    float *gf=(float *)(scratch+4096U),*sums=(float *)(scratch+12288U);
    float *values=(float *)(scratch+12416U);
    int32_t *flags=(int32_t *)(scratch+12544U);
    for(uint32_t c=0;c<width;c+=64U) {
        HVX_VectorPair g=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(gamma+c));
        HVX_VectorPair ordered=Q6_W_vshuff_VVR(Q6_V_hi_W(g),Q6_V_lo_W(g),-4);
        *(HVX_Vector *)(gf+c)=Q6_V_lo_W(ordered);
        *(HVX_Vector *)(gf+c+32U)=Q6_V_hi_W(ordered);
    }
    memset(out,q->zero_point,(size_t)QBH_BLOCK_M*width);
    for(uint32_t first=0;first<rows;first+=32U) {
        HVX_Vector sum=Q6_V_vzero();
        if(rows==1U) {
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v=*(const HVX_Vector *)(x+c),sq=qbh_fp32_mul(v,v);
                for(uint32_t k=0;k<32U;k++) {
                    sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vsplat_R(Q6_R_vextract_VR(sq,0)));
                    sq=Q6_V_vror_VR(sq,4);
                }
            }
        } else {
            for(uint32_t c=0;c<width;c+=32U) {
                for(uint32_t r=0;r<32U;r++)tile[r]=*(const HVX_Vector *)(x+(first+r)*width+c);
                /* Interleave rows at 4,8,...64 bytes; columns emerge bit-reversed. */
                for(uint32_t gap=1U;gap<32U;gap*=2U)
                    for(uint32_t r=0;r<32U;r++)if(!(r&gap)) {
                        HVX_VectorPair t=Q6_W_vshuff_VVR(tile[r+gap],tile[r],-(int)(4U*gap));
                        tile[r]=Q6_V_lo_W(t);tile[r+gap]=Q6_V_hi_W(t);
                    }
                for(uint32_t k=0;k<32U;k++) {
                    uint32_t rev=((k&1U)<<4)|((k&2U)<<2)|(k&4U)|((k&8U)>>2)|((k&16U)>>4);
                    HVX_Vector v=tile[rev];
                    sum=Q6_Vsf_vadd_VsfVsf(sum,qbh_fp32_mul(v,v));
                }
            }
        }
        *(HVX_Vector *)sums=sum;
        uint32_t count=rows==1U?1U:32U;
        for(uint32_t r=0;r<count;r++) {
            uint32_t row=first+r;
            float inv=1.0f/sqrtf(sums[r]/(float)width+QBH_MODEL_RMS_EPS);
            HVX_Vector vi=qbh_fp32_splat(inv),vq=qbh_fp32_splat(1.0f/q->scale);
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v=qbh_fp32_mul(*(const HVX_Vector *)(x+row*width+c),vi);
                v=qbh_fp32_mul(v,*(const HVX_Vector *)(gf+c));
                HVX_Vector code=Q6_Vsf_vadd_VsfVsf(qbh_fp32_mul(v,vq),qbh_fp32_splat((float)q->zero_point));
                HVX_Vector z=Q6_Vsf_vadd_VsfVsf(code,qbh_fp32_splat(0.5f));
                HVX_Vector w=Q6_Vw_equals_Vsf(z);
                HVX_Vector frac=Q6_Vsf_vsub_VsfVsf(z,Q6_Vsf_equals_Vw(w));
                HVX_VectorPred edge=Q6_Q_or_QQ(Q6_Q_vcmp_gt_VsfVsf(qbh_fp32_splat(.001f),frac),
                    Q6_Q_vcmp_gt_VsfVsf(frac,qbh_fp32_splat(.999f)));
                edge=Q6_Q_and_QQ(edge,Q6_Q_vcmp_gt_VsfVsf(z,qbh_fp32_splat(0.0f)));
                edge=Q6_Q_and_QQ(edge,Q6_Q_vcmp_gt_VsfVsf(qbh_fp32_splat(256.0f),z));
                HVX_Vector mask=Q6_V_vmux_QVV(edge,Q6_V_vsplat_R(1),Q6_V_vzero());
                HVX_Vector packed=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),
                    Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),w));
                size_t off=native?(size_t)(c/32U)*2048U+row*32U:(size_t)row*width+c;
                memcpy(out+off,&packed,32U);
                HVX_Vector any=mask;
                for(uint32_t sh=64;sh>=4;sh>>=1)any=Q6_V_vor_VV(any,Q6_V_vror_VR(any,sh));
                if(Q6_R_vextract_VR(any,0)) {
                    *(HVX_Vector *)values=v;*(HVX_Vector *)flags=mask;
                    for(uint32_t k=0;k<32U;k++)if(flags[k]) {
                        float exact=roundf(values[k]/q->scale+(float)q->zero_point);
                        out[off+k]=(uint8_t)fminf(255.0f,fmaxf(0.0f,exact));
                    }
                }
            }
        }
    }
    asm volatile("barrier":::"memory");
}
