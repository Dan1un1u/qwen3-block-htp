static const uint32_t qbh_fp32_broadcast0[32] __attribute__((aligned(128)))={0x0,0x4040404,0x8080808,0xc0c0c0c,0x10101010,0x14141414,0x18181818,0x1c1c1c1c,0x20202020,0x24242424,0x28282828,0x2c2c2c2c,0x30303030,0x34343434,0x38383838,0x3c3c3c3c,0x40404040,0x44444444,0x48484848,0x4c4c4c4c,0x50505050,0x54545454,0x58585858,0x5c5c5c5c,0x60606060,0x64646464,0x68686868,0x6c6c6c6c,0x70707070,0x74747474,0x78787878,0x7c7c7c7c};
/* L32-0017 vectorized FP32 boundaries. Native packed W4 products, never U8 output.
 * Raw stores are signed24, proven at host preparation for EVERY possible U8
 * input. SP2's reconstructed signed32 bound is proved separately. No DDR spill.
 * HVX FP32 epilogues retain the independent scalar numerical contract;
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
        : Q6_Vw_vsub_VwVw(low,Q6_Vw_vmpyi_VwRh(sum,(uint32_t)zero*0x00010001U));
}
/* EXP0269 candidate2: explicit register operands avoid indexed vector arrays
 * being materialized as a 1024-byte payload on the DDR thread stack. */
static inline void qbh_fp32_add_row(HVX_Vector l,HVX_Vector h,HVX_Vector sum,
    HVX_Vector scales,float *dst,uint32_t sp2,int32_t zero) {
    HVX_Vector acc=qbh_fp32_dot(l,h,sum,sp2,zero);
    HVX_Vector add=qbh_fp32_mul(Q6_Vsf_equals_Vw(acc),scales);
    HVX_Vector *out=(HVX_Vector *)dst;
    *out=Q6_Vsf_vadd_VsfVsf(*out,add);
}
static void qbh_llama_fp32_epilogue(const uint8_t *lo,const uint8_t *hi,
    const float *scale,float *dst,uint32_t rows,uint32_t sp2,int32_t zero) {
    HVX_Vector scales=*(const HVX_Vector *)scale,sum=*(const HVX_Vector *)(scale+32U);
    for(uint32_t row=0;row<rows;row+=4U) {
        struct qbh_sp2_words l=qbh_sp2_chunk24(lo,row*32U);
        struct qbh_sp2_words h=sp2?qbh_sp2_chunk24(hi,row*32U):l;
        qbh_fp32_add_row(l.v0,h.v0,sum,scales,dst+row*QBH_BLOCK_HIDDEN,sp2,zero);
        if(row+1U<rows)qbh_fp32_add_row(l.v1,h.v1,sum,scales,dst+(row+1U)*QBH_BLOCK_HIDDEN,sp2,zero);
        if(row+2U<rows)qbh_fp32_add_row(l.v2,h.v2,sum,scales,dst+(row+2U)*QBH_BLOCK_HIDDEN,sp2,zero);
        if(row+3U<rows)qbh_fp32_add_row(l.v3,h.v3,sum,scales,dst+(row+3U)*QBH_BLOCK_HIDDEN,sp2,zero);
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
/* EXP0269 candidate3: keep the 16-row transpose in HVX registers. Each
 * vector contains two consecutive channels over 16 rows after transposition.
 * Add its lower half then upper half, retaining ascending-channel FP32 order.
 * Only lanes0..15 are live; no padded transpose plane or payload DDR stack. */
__attribute__((noinline)) static HVX_Vector qbh_fp32_norm_reduce16(const float *x,uint32_t width) {
    HVX_Vector sum=Q6_V_vzero();
    for(uint32_t c=0;c<width;c+=32U) {
        HVX_Vector v0=*(const HVX_Vector *)(x+0U*width+c);
        HVX_Vector v1=*(const HVX_Vector *)(x+1U*width+c);
        HVX_Vector v2=*(const HVX_Vector *)(x+2U*width+c);
        HVX_Vector v3=*(const HVX_Vector *)(x+3U*width+c);
        HVX_Vector v4=*(const HVX_Vector *)(x+4U*width+c);
        HVX_Vector v5=*(const HVX_Vector *)(x+5U*width+c);
        HVX_Vector v6=*(const HVX_Vector *)(x+6U*width+c);
        HVX_Vector v7=*(const HVX_Vector *)(x+7U*width+c);
        HVX_Vector v8=*(const HVX_Vector *)(x+8U*width+c);
        HVX_Vector v9=*(const HVX_Vector *)(x+9U*width+c);
        HVX_Vector v10=*(const HVX_Vector *)(x+10U*width+c);
        HVX_Vector v11=*(const HVX_Vector *)(x+11U*width+c);
        HVX_Vector v12=*(const HVX_Vector *)(x+12U*width+c);
        HVX_Vector v13=*(const HVX_Vector *)(x+13U*width+c);
        HVX_Vector v14=*(const HVX_Vector *)(x+14U*width+c);
        HVX_Vector v15=*(const HVX_Vector *)(x+15U*width+c);
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v1,v0,-4);v0=Q6_V_lo_W(t);v1=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v3,v2,-4);v2=Q6_V_lo_W(t);v3=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v5,v4,-4);v4=Q6_V_lo_W(t);v5=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v7,v6,-4);v6=Q6_V_lo_W(t);v7=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v9,v8,-4);v8=Q6_V_lo_W(t);v9=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v11,v10,-4);v10=Q6_V_lo_W(t);v11=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v13,v12,-4);v12=Q6_V_lo_W(t);v13=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v15,v14,-4);v14=Q6_V_lo_W(t);v15=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v2,v0,-8);v0=Q6_V_lo_W(t);v2=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v3,v1,-8);v1=Q6_V_lo_W(t);v3=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v6,v4,-8);v4=Q6_V_lo_W(t);v6=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v7,v5,-8);v5=Q6_V_lo_W(t);v7=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v10,v8,-8);v8=Q6_V_lo_W(t);v10=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v11,v9,-8);v9=Q6_V_lo_W(t);v11=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v14,v12,-8);v12=Q6_V_lo_W(t);v14=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v15,v13,-8);v13=Q6_V_lo_W(t);v15=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v4,v0,-16);v0=Q6_V_lo_W(t);v4=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v5,v1,-16);v1=Q6_V_lo_W(t);v5=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v6,v2,-16);v2=Q6_V_lo_W(t);v6=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v7,v3,-16);v3=Q6_V_lo_W(t);v7=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v12,v8,-16);v8=Q6_V_lo_W(t);v12=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v13,v9,-16);v9=Q6_V_lo_W(t);v13=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v14,v10,-16);v10=Q6_V_lo_W(t);v14=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v15,v11,-16);v11=Q6_V_lo_W(t);v15=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v8,v0,-32);v0=Q6_V_lo_W(t);v8=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v9,v1,-32);v1=Q6_V_lo_W(t);v9=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v10,v2,-32);v2=Q6_V_lo_W(t);v10=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v11,v3,-32);v3=Q6_V_lo_W(t);v11=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v12,v4,-32);v4=Q6_V_lo_W(t);v12=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v13,v5,-32);v5=Q6_V_lo_W(t);v13=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v14,v6,-32);v6=Q6_V_lo_W(t);v14=Q6_V_hi_W(t);}
        {HVX_VectorPair t=Q6_W_vshuff_VVR(v15,v7,-32);v7=Q6_V_lo_W(t);v15=Q6_V_hi_W(t);}
        {HVX_Vector sq=qbh_fp32_mul(v0,v0);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v8,v8);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v4,v4);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v12,v12);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v2,v2);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v10,v10);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v6,v6);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v14,v14);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v1,v1);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v9,v9);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v5,v5);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v13,v13);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v3,v3);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v11,v11);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v7,v7);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
        {HVX_Vector sq=qbh_fp32_mul(v15,v15);sum=Q6_Vsf_vadd_VsfVsf(sum,sq);sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vror_VR(sq,64));}
    }
    return sum;
}
/* Sequential FP32 reduction across channels, vectorized across 32 rows.
 * Scratch reuses the idle SP2 VTCM raw slots; no tensor on the thread stack. */
static void qbh_llama_fp32_norm(const float *x,const __fp16 *gamma,
    uint8_t *out,const struct qbh_block_qparam *q,uint32_t rows,
    uint32_t width,uint32_t native,uint8_t *scratch) {
#pragma clang fp contract(off)
    HVX_Vector *tile=(HVX_Vector *)scratch;
    float *gf=(float *)(scratch+4096U),*sums=(float *)(scratch+12288U);
    int32_t *flags=(int32_t *)(scratch+12544U);
    for(uint32_t c=0;c<width;c+=64U) {
        HVX_VectorPair g=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(gamma+c));
        HVX_VectorPair ordered=Q6_W_vshuff_VVR(Q6_V_hi_W(g),Q6_V_lo_W(g),-4);
        *(HVX_Vector *)(gf+c)=Q6_V_lo_W(ordered);
        *(HVX_Vector *)(gf+c+32U)=Q6_V_hi_W(ordered);
    }
    if(rows==1U) {
        /* Candidate3 native initialization: identical padding, vector stores
         * instead of a scalar libc traversal of 128KiB VTCM each decode Norm. */
        HVX_Vector fill=Q6_V_vsplat_R((uint32_t)q->zero_point*0x01010101U);
        for(uint32_t off=0;off<QBH_BLOCK_M*width;off+=128U)
            *(HVX_Vector *)(out+off)=fill;
    }
    for(uint32_t first=0;first<rows;first+=32U) {
        HVX_Vector sum=Q6_V_vzero();
        if(rows==1U) {
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v=*(const HVX_Vector *)(x+c),sq=qbh_fp32_mul(v,v);
                for(uint32_t k=0;k<32U;k++) {
                    /* Lane0 visits channels in order; other lanes are not consumed. */
                    sum=Q6_Vsf_vadd_VsfVsf(sum,sq);
                    sq=Q6_V_vror_VR(sq,4);
                }
            }
        } else if(rows==16U) {
            sum=qbh_fp32_norm_reduce16(x,width);
        } else {
            for(uint32_t c=0;c<width;c+=32U) {
                for(uint32_t r=0;r<32U;r++)tile[r]=first+r<rows ? *(const HVX_Vector *)(x+(first+r)*width+c):Q6_V_vzero();
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
        uint32_t count=rows-first<32U?rows-first:32U;
        for(uint32_t r=0;r<count;r++) {
            uint32_t row=first+r;
            float inv=1.0f/sqrtf(sums[r]/(float)width+QBH_MODEL_RMS_EPS);
            HVX_Vector vi=qbh_fp32_splat(inv),vq=qbh_fp32_splat(1.0f/q->scale);
            /* L32-0018: collect rare division-boundary repairs per row.
             * The transpose tile is idle here: reuse its first 2048 bytes for flags.
             * HVX writes native 32-byte segments without a scalar stack roundtrip. */
            HVX_Vector row_mask=Q6_V_vzero();
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v=qbh_fp32_mul(*(const HVX_Vector *)(x+row*width+c),vi);
                v=qbh_fp32_mul(v,*(const HVX_Vector *)(gf+c));
                HVX_Vector code=Q6_Vsf_vadd_VsfVsf(qbh_fp32_mul(v,vq),qbh_fp32_splat((float)q->zero_point));
                HVX_Vector z=Q6_Vsf_vadd_VsfVsf(code,qbh_fp32_splat(0.5f));
                HVX_Vector w=Q6_Vw_equals_Vsf(z);
                HVX_Vector frac=Q6_Vsf_vsub_VsfVsf(z,Q6_Vsf_equals_Vw(w));
                /* Reciprocal/multiply differs from division by <0.0001 in [0,256].
                 * Repair that entire rounding-boundary band using scalar division. */
                HVX_VectorPred edge=Q6_Q_or_QQ(Q6_Q_vcmp_gt_VsfVsf(qbh_fp32_splat(.0001f),frac),
                    Q6_Q_vcmp_gt_VsfVsf(frac,qbh_fp32_splat(.9999f)));
                edge=Q6_Q_and_QQ(edge,Q6_Q_vcmp_gt_VsfVsf(z,qbh_fp32_splat(0.0f)));
                edge=Q6_Q_and_QQ(edge,Q6_Q_vcmp_gt_VsfVsf(qbh_fp32_splat(256.0f),z));
                HVX_Vector mask=Q6_V_vmux_QVV(edge,Q6_V_vsplat_R(1),Q6_V_vzero());
                HVX_Vector packed=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),
                    Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),w));
                size_t off=native?(size_t)(c/32U)*2048U+row*32U:(size_t)row*width+c;
                uint32_t shift=(uintptr_t)(out+off)&127U;
                HVX_VectorPred pred=Q6_Q_and_QQn(Q6_Q_vsetq2_R(shift+32U),Q6_Q_vsetq_R(shift));
                Q6_vmem_QRIV(pred,(HVX_Vector *)(out+off),Q6_V_vlalign_VVR(packed,packed,shift));
                row_mask=Q6_V_vor_VV(row_mask,mask);
                HVX_Vector fm=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),
                    Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),mask));
                uint8_t *fp=scratch+c;
                uint32_t fs=(uintptr_t)fp&127U;
                HVX_VectorPred fq=Q6_Q_and_QQn(Q6_Q_vsetq2_R(fs+32U),Q6_Q_vsetq_R(fs));
                Q6_vmem_QRIV(fq,(HVX_Vector *)fp,Q6_V_vlalign_VVR(fm,fm,fs));
            }
            HVX_Vector any=row_mask;
            for(uint32_t sh=64;sh>=4;sh>>=1)any=Q6_V_vor_VV(any,Q6_V_vror_VR(any,sh));
            if(Q6_R_vextract_VR(any,0)) {
                *(HVX_Vector *)flags=row_mask;
                asm volatile("barrier":::"memory");
                for(uint32_t k=0;k<32U;k++)if(flags[k])
                    for(uint32_t c=k;c<width;c+=32U)if(scratch[c]) {
                        float val=x[row*width+c]*inv;
                        val=val*gf[c];
                        float exact=roundf(val/q->scale+(float)q->zero_point);
                        size_t off=native?(size_t)(c/32U)*2048U+row*32U+c%32U:(size_t)row*width+c;
                        out[off]=(uint8_t)fminf(255.0f,fmaxf(0.0f,exact));
                    }
            }
        }
    }
    asm volatile("barrier":::"memory");
}
static void qbh_llama_fp32_norm_worker(struct qbh_block_w4f16_pool *p,uint32_t worker) {
    uint32_t first=worker*16U;
    qbh_llama_fp32_norm(p->fp32_norm_input+first*QBH_BLOCK_HIDDEN,p->fp32_norm_gamma,
        p->fp32_norm_output+first*32U,p->fp32_norm_qparam,16U,QBH_BLOCK_HIDDEN,1U,
        p->fp32_norm_scratch+worker*16384U);
}
