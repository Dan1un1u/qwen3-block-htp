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
/* Separate noinline consumer ensures a real VTCM handoff, at existing tile
 * granularity, while keeping all arithmetic and stream ownership unchanged. */
static __attribute__((noinline)) void qbh_boundary_residual_add(
    const float *tmp,float *dst,uint32_t rows) {
    for(uint32_t row=0;row<rows;row++) {
        HVX_Vector *out=(HVX_Vector *)(dst+row*QBH_BLOCK_HIDDEN);
        *out=Q6_Vsf_vadd_VsfVsf(*out,*(const HVX_Vector *)(tmp+row*32U));
    }
}
static void qbh_llama_fp32_epilogue(const uint8_t *lo,const uint8_t *hi,
    const float *scale,float *dst,uint32_t rows,uint32_t sp2,int32_t zero,float *tmp) {
    HVX_Vector scales=*(const HVX_Vector *)scale,sum=*(const HVX_Vector *)(scale+32U);
    for(uint32_t row=0;row<rows;row+=4U) {
        struct qbh_sp2_words l=qbh_sp2_chunk24(lo,row*32U);
        struct qbh_sp2_words h=sp2?qbh_sp2_chunk24(hi,row*32U):l;
        HVX_Vector lv[4]={l.v0,l.v1,l.v2,l.v3},hv[4]={h.v0,h.v1,h.v2,h.v3};
        for(uint32_t k=0;k<4U && row+k<rows;k++) {
            HVX_Vector acc=qbh_fp32_dot(lv[k],hv[k],sum,sp2,zero);
            HVX_Vector add=qbh_fp32_mul(Q6_Vsf_equals_Vw(acc),scales);
            HVX_Vector *out=(HVX_Vector *)(dst+(row+k)*QBH_BLOCK_HIDDEN);
            if(tmp) *(HVX_Vector *)(tmp+(row+k)*32U)=add;
            else *out=Q6_Vsf_vadd_VsfVsf(*out,add);
        }
    }
    if(tmp) {
        asm volatile("barrier":::"memory");
        qbh_boundary_residual_add(tmp,dst,rows);
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
        uint32_t passes=w->fp32_sp2 && (w->fp32_rows==64U || w->fp32_split) ? 2U:1U;
        for(uint32_t pass=0;pass<passes;pass++) {
            for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[n+32U]=0U;}
            qbh_hmx_begin_u8s8_output(bias);
            qbh_hmx_accumulate_u8n4_projection(pass?w->sp2_high:w->activation,weight,w->k_tiles);
            read_i32(pass?hi:lo,bias,3U);
        }
        qbh_llama_fp32_epilogue(lo,w->fp32_sp2 ? ((w->fp32_rows==64U || w->fp32_split)?hi:lo+128U):NULL,
            scale,w->fp32_residual+w->fp32_first_channel+nt*32U,
            w->fp32_rows,w->fp32_sp2,w->fp32_input_zero,w->fp32_boundary_tmp);
    }
    asm volatile("barrier":::"memory");
    return AEE_SUCCESS;
}
/* Sequential FP32 reduction across channels, vectorized across 32 rows.
 * Scratch reuses the idle SP2 VTCM raw slots; no tensor on the thread stack. */
static void qbh_llama_fp32_norm(const float *x,const __fp16 *gamma,
    uint8_t *out,const struct qbh_block_qparam *q,uint32_t rows,
    uint32_t width,uint32_t native,uint8_t *scratch,float *normalized) {
#pragma clang fp contract(off)
    HVX_Vector *tile=(HVX_Vector *)scratch;
    float *gf=(float *)(scratch+4096U),*sums=(float *)(scratch+4096U+width*4U);
    int32_t *flags=(int32_t *)(scratch+4352U+width*4U);
    for(uint32_t c=0;c<width;c+=64U) {
        HVX_VectorPair g=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(gamma+c));
        HVX_VectorPair ordered=Q6_W_vshuff_VVR(Q6_V_hi_W(g),Q6_V_lo_W(g),-4);
        *(HVX_Vector *)(gf+c)=Q6_V_lo_W(ordered);
        *(HVX_Vector *)(gf+c+32U)=Q6_V_hi_W(ordered);
    }
    if(rows==1U)memset(out,q->zero_point,(size_t)QBH_BLOCK_M*width);
    for(uint32_t first=0;first<rows;first+=32U) {
        HVX_Vector sum=Q6_V_vzero();
        if(rows==1U) {
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v=*(const HVX_Vector *)(x+c),sq=qbh_fp32_mul(v,v);
                for(uint32_t k=0;k<32U;k++) {
                    sum=Q6_Vsf_vadd_VsfVsf(sum,Q6_V_vdelta_VV(sq,*(const HVX_Vector *)qbh_fp32_broadcast0));
                    sq=Q6_V_vror_VR(sq,4);
                }
            }
        } else {
            for(uint32_t c=0;c<width;c+=32U) {
                if(rows==16U) {
                    tile[0]=*(const HVX_Vector *)(x+0U*width+c);
                    tile[1]=*(const HVX_Vector *)(x+1U*width+c);
                    tile[2]=*(const HVX_Vector *)(x+2U*width+c);
                    tile[3]=*(const HVX_Vector *)(x+3U*width+c);
                    tile[4]=*(const HVX_Vector *)(x+4U*width+c);
                    tile[5]=*(const HVX_Vector *)(x+5U*width+c);
                    tile[6]=*(const HVX_Vector *)(x+6U*width+c);
                    tile[7]=*(const HVX_Vector *)(x+7U*width+c);
                    tile[8]=*(const HVX_Vector *)(x+8U*width+c);
                    tile[9]=*(const HVX_Vector *)(x+9U*width+c);
                    tile[10]=*(const HVX_Vector *)(x+10U*width+c);
                    tile[11]=*(const HVX_Vector *)(x+11U*width+c);
                    tile[12]=*(const HVX_Vector *)(x+12U*width+c);
                    tile[13]=*(const HVX_Vector *)(x+13U*width+c);
                    tile[14]=*(const HVX_Vector *)(x+14U*width+c);
                    tile[15]=*(const HVX_Vector *)(x+15U*width+c);
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[1],tile[0],-4);tile[0]=Q6_V_lo_W(t);tile[1]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[3],tile[2],-4);tile[2]=Q6_V_lo_W(t);tile[3]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[5],tile[4],-4);tile[4]=Q6_V_lo_W(t);tile[5]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[7],tile[6],-4);tile[6]=Q6_V_lo_W(t);tile[7]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[9],tile[8],-4);tile[8]=Q6_V_lo_W(t);tile[9]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[11],tile[10],-4);tile[10]=Q6_V_lo_W(t);tile[11]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[13],tile[12],-4);tile[12]=Q6_V_lo_W(t);tile[13]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[15],tile[14],-4);tile[14]=Q6_V_lo_W(t);tile[15]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[2],tile[0],-8);tile[0]=Q6_V_lo_W(t);tile[2]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[3],tile[1],-8);tile[1]=Q6_V_lo_W(t);tile[3]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[6],tile[4],-8);tile[4]=Q6_V_lo_W(t);tile[6]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[7],tile[5],-8);tile[5]=Q6_V_lo_W(t);tile[7]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[10],tile[8],-8);tile[8]=Q6_V_lo_W(t);tile[10]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[11],tile[9],-8);tile[9]=Q6_V_lo_W(t);tile[11]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[14],tile[12],-8);tile[12]=Q6_V_lo_W(t);tile[14]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[15],tile[13],-8);tile[13]=Q6_V_lo_W(t);tile[15]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[4],tile[0],-16);tile[0]=Q6_V_lo_W(t);tile[4]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[5],tile[1],-16);tile[1]=Q6_V_lo_W(t);tile[5]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[6],tile[2],-16);tile[2]=Q6_V_lo_W(t);tile[6]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[7],tile[3],-16);tile[3]=Q6_V_lo_W(t);tile[7]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[12],tile[8],-16);tile[8]=Q6_V_lo_W(t);tile[12]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[13],tile[9],-16);tile[9]=Q6_V_lo_W(t);tile[13]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[14],tile[10],-16);tile[10]=Q6_V_lo_W(t);tile[14]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[15],tile[11],-16);tile[11]=Q6_V_lo_W(t);tile[15]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[8],tile[0],-32);tile[0]=Q6_V_lo_W(t);tile[8]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[9],tile[1],-32);tile[1]=Q6_V_lo_W(t);tile[9]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[10],tile[2],-32);tile[2]=Q6_V_lo_W(t);tile[10]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[11],tile[3],-32);tile[3]=Q6_V_lo_W(t);tile[11]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[12],tile[4],-32);tile[4]=Q6_V_lo_W(t);tile[12]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[13],tile[5],-32);tile[5]=Q6_V_lo_W(t);tile[13]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[14],tile[6],-32);tile[6]=Q6_V_lo_W(t);tile[14]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(tile[15],tile[7],-32);tile[7]=Q6_V_lo_W(t);tile[15]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[0],-64);tile[0]=Q6_V_lo_W(t);tile[16]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[1],-64);tile[1]=Q6_V_lo_W(t);tile[17]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[2],-64);tile[2]=Q6_V_lo_W(t);tile[18]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[3],-64);tile[3]=Q6_V_lo_W(t);tile[19]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[4],-64);tile[4]=Q6_V_lo_W(t);tile[20]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[5],-64);tile[5]=Q6_V_lo_W(t);tile[21]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[6],-64);tile[6]=Q6_V_lo_W(t);tile[22]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[7],-64);tile[7]=Q6_V_lo_W(t);tile[23]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[8],-64);tile[8]=Q6_V_lo_W(t);tile[24]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[9],-64);tile[9]=Q6_V_lo_W(t);tile[25]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[10],-64);tile[10]=Q6_V_lo_W(t);tile[26]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[11],-64);tile[11]=Q6_V_lo_W(t);tile[27]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[12],-64);tile[12]=Q6_V_lo_W(t);tile[28]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[13],-64);tile[13]=Q6_V_lo_W(t);tile[29]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[14],-64);tile[14]=Q6_V_lo_W(t);tile[30]=Q6_V_hi_W(t);}
                    {HVX_VectorPair t=Q6_W_vshuff_VVR(Q6_V_vzero(),tile[15],-64);tile[15]=Q6_V_lo_W(t);tile[31]=Q6_V_hi_W(t);}
                } else {
                for(uint32_t r=0;r<32U;r++)tile[r]=first+r<rows ? *(const HVX_Vector *)(x+(first+r)*width+c):Q6_V_vzero();
                /* Interleave rows at 4,8,...64 bytes; columns emerge bit-reversed. */
                for(uint32_t gap=1U;gap<32U;gap*=2U)
                    for(uint32_t r=0;r<32U;r++)if(!(r&gap)) {
                        HVX_VectorPair t=Q6_W_vshuff_VVR(tile[r+gap],tile[r],-(int)(4U*gap));
                        tile[r]=Q6_V_lo_W(t);tile[r+gap]=Q6_V_hi_W(t);
                    }

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
            if(normalized) {
                for(uint32_t c=0;c<width;c+=32U) {
                    HVX_Vector v=qbh_fp32_mul(*(const HVX_Vector *)(x+row*width+c),vi);
                    *(HVX_Vector *)(normalized+c)=qbh_fp32_mul(v,*(const HVX_Vector *)(gf+c));
                }
                asm volatile("barrier":::"memory");
            }
            HVX_Vector row_mask=Q6_V_vzero();
            for(uint32_t c=0;c<width;c+=32U) {
                HVX_Vector v;
                if(normalized)v=*(const HVX_Vector *)(normalized+c);
                else {
                    v=qbh_fp32_mul(*(const HVX_Vector *)(x+row*width+c),vi);
                    v=qbh_fp32_mul(v,*(const HVX_Vector *)(gf+c));
                }
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
                        float val;
                        if(normalized)val=normalized[c];
                        else {val=x[row*width+c]*inv;val=val*gf[c];}
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
        p->fp32_norm_output+first*(p->fp32_norm_native?32U:QBH_BLOCK_HIDDEN),p->fp32_norm_qparam,16U,QBH_BLOCK_HIDDEN,p->fp32_norm_native,
        p->fp32_norm_scratch+worker*((4096U+QBH_BLOCK_HIDDEN*4U+512U+4095U)&~4095U),
        p->fp32_norm_tmp?p->fp32_norm_tmp+worker*QBH_BLOCK_HIDDEN:NULL);
}
