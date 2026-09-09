static const int8_t qbh_r4_h12[12][12]={
{1,1,1,1,1,1,1,1,1,1,1,1},
{1,-1,-1,1,-1,-1,-1,1,1,1,-1,1},
{1,1,-1,-1,1,-1,-1,-1,1,1,1,-1},
{1,-1,1,-1,-1,1,-1,-1,-1,1,1,1},
{1,1,-1,1,-1,-1,1,-1,-1,-1,1,1},
{1,1,1,-1,1,-1,-1,1,-1,-1,-1,1},
{1,1,1,1,-1,1,-1,-1,1,-1,-1,-1},
{1,-1,1,1,1,-1,1,-1,-1,1,-1,-1},
{1,-1,-1,1,1,1,-1,1,-1,-1,1,-1},
{1,-1,-1,-1,1,1,1,-1,1,-1,-1,1},
{1,1,-1,-1,-1,1,1,1,-1,1,-1,-1},
{1,-1,1,-1,-1,-1,1,1,1,-1,1,-1}
};

/* EXP0261 full R4 = H12 x H512. Dense matrix factors, never FWHT.
 * Lifetime: gate/up projections fully joined. hmx_activation (768KiB),
 * expanded_weight and expanded_weight_alt (1MiB each) are phase-dead.
 * Stage1 output stays in alt while up to16 tokens of stage2 use the other
 * two arenas; stage2 sign matrix occupies the dead HMX output tile. */
static size_t qbh_r4_pos(uint32_t r,uint32_t c,uint32_t nt) {
    return qbh_hmx_fp16_matrix_tile_offset(r/32U,c/32U,nt)+
           qbh_hmx_fp16_tile_offset(r%32U,c%32U);
}
static void qbh_r4_zero(void *p,uint32_t bytes) {
    for(uint32_t i=0;i<bytes/128U;++i)((HVX_Vector *)p)[i]=Q6_V_vzero();
}
static int qbh_r4_gemm(struct qbh_block_header *h,
    struct qbh_block_hmx_worker *worker,const __fp16 *a,const __fp16 *w,
    void *sc,__fp16 *out,uint32_t mt,uint32_t kt,uint32_t nt) {
    uint64_t t=HAP_perf_get_qtimer_count();
    int rc=qbh_hmx_submit(worker,QBH_BLOCK_HMX_FP16,a,w,sc,out,mt,kt,nt);
    h->dense_r4_matmul_ticks+=HAP_perf_get_qtimer_count()-t;
    ++h->hmx_command_count;++h->dense_r4_hmx_calls;
    h->hmx_fp16_tile_pair_count+=mt*kt*nt;
    return rc;
}
static int qbh_run_dense_r4(struct qbh_block_header *h,uint8_t *shared,
    struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,uint8_t *middle) {
    const uint32_t rows=h->logical_m, p1=(12U*rows+31U)&~31U;
    __fp16 *act=(__fp16 *)b->hmx_activation;
    __fp16 *weight=(__fp16 *)b->expanded_weight;
    __fp16 *stage1=(__fp16 *)b->expanded_weight_alt;
    __fp16 *w12=(__fp16 *)b->hmx_output;
    __fp16 tmp[128] __attribute__((aligned(128)));
    uint8_t codes[128] __attribute__((aligned(128)));
    float scales[32] __attribute__((aligned(128)));
    __fp16 *audit=h->dense_r4_audit_offset?(__fp16 *)(shared+h->dense_r4_audit_offset):NULL;
    if(h->variant!=QBH_BLOCK_W4U8 || h->dense_r4_mode!=1U ||
       (rows!=1U && rows!=64U) || h->w4u8_decode_direct_n_gate_up_swiglu_stream ||
       (audit && !qbh_range_valid(h->dense_r4_audit_offset,3U*786432U,h->shared_bytes)))return -1;
    /* These bounds are the existing owned allocations, no arena growth. */
    if((uintptr_t)b->expanded_weight_alt-(uintptr_t)b->expanded_weight<1048576U ||
       (uintptr_t)b->hmx_output-(uintptr_t)b->expanded_weight_alt<1048576U ||
       (uintptr_t)b->compressed_weight-(uintptr_t)b->hmx_activation<786432U) return -1;
    uint64_t t=HAP_perf_get_qtimer_count();
    qbh_r4_zero(act,p1*512U*2U);
    for(uint32_t tile=0;tile<192U;++tile) {
        uint32_t a=tile/16U,k=tile%16U;
        for(uint32_t r=0;r<rows;r+=4U) {
            qbh_mlp_gate_up_lut_f16_hvx(b->gate+tile*2048U+r*32U,
                b->up+tile*2048U+r*32U,(uint16_t *)tmp,128U,
                (const uint16_t *)b->w4u8_silu_lut,b->w4u8_gather_scratch);
            if(rows==64U) {
                __fp16 *dst=act+qbh_r4_pos(a*rows+r,k*32U,16U);
                ((HVX_Vector *)dst)[0]=Q6_Vh_vshuff_Vh(((HVX_Vector *)tmp)[0]);
                ((HVX_Vector *)dst)[1]=Q6_Vh_vshuff_Vh(((HVX_Vector *)tmp)[1]);
            } else for(uint32_t c=0;c<32U;++c)act[qbh_r4_pos(a,k*32U+c,16U)]=tmp[c];
        }
    }
    /* Materialize H512 from exact dense H128 blocks with parity signs.
     * This generates matrix entries; activations never traverse a butterfly. */
    for(uint32_t n=0;n<16U;++n)for(uint32_t k=0;k<16U;++k) {
        const HVX_Vector *src=(const HVX_Vector *)(qbh_r3_sign_matrix+((n%4U)*4U+k%4U)*1024U);
        HVX_Vector *dst=(HVX_Vector *)(weight+(n*16U+k)*1024U);
        HVX_Vector sign=Q6_Vh_vsplat_R((__builtin_popcount((n/4U)&(k/4U))&1)?0x8000:0);
        for(uint32_t v=0;v<16U;++v)dst[v]=Q6_V_vxor_VV(src[v],sign);
    }
    for(uint32_t c=0;c<32U;++c)scales[c]=(float)(__fp16)0.04419417382415922f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    if(audit)for(uint32_t a=0;a<12U;++a)for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[(a*rows+r)*512U+c]=act[qbh_r4_pos(a*rows+r,c,16U)];
    if(qbh_r4_gemm(h,worker,act,weight,b->channel_scale,stage1,p1/32U,16U,16U))return -1;
    if(audit)for(uint32_t r=0;r<12U*rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[393216U+r*512U+c]=stage1[qbh_r4_pos(r,c,16U)];
    t=HAP_perf_get_qtimer_count();
    qbh_r4_zero(w12,2048U);
    /* Input row is [group], output column is [new group]. */
    for(uint32_t n=0;n<12U;++n)for(uint32_t k=0;k<12U;++k)
        w12[qbh_hmx_fp16_tile_offset(k,n)]=(__fp16)qbh_r4_h12[k][n];
    for(uint32_t c=0;c<32U;++c)scales[c]=(float)(__fp16)0.2886751345948129f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    for(uint32_t first=0;first<rows;first+=16U) {
        uint32_t count=rows-first<16U?rows-first:16U;
        t=HAP_perf_get_qtimer_count();
        qbh_r4_zero(act,count*512U*32U*2U);
        for(uint32_t r=0;r<count;++r)for(uint32_t a=0;a<12U;++a)for(uint32_t c=0;c<512U;++c)
            act[qbh_r4_pos(r*512U+c,a,1U)]=stage1[qbh_r4_pos(a*rows+first+r,c,16U)];
        h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-t;
        if(qbh_r4_gemm(h,worker,act,w12,b->channel_scale,weight,count*16U,1U,1U))return -1;
        t=HAP_perf_get_qtimer_count();
        for(uint32_t tile=0;tile<192U;++tile)for(uint32_t r=0;r<count;r+=4U) {
            uint32_t a=tile/16U,c0=tile%16U*32U,live=count-r<4U?count-r:4U;
            for(uint32_t rr=0;rr<4U;++rr)for(uint32_t c=0;c<32U;++c) {
                __fp16 z=rr<live?weight[qbh_r4_pos((r+rr)*512U+c0+c,a,1U)]:(__fp16)0;
                tmp[rr*32U+c]=z;
                if(audit && rr<live)audit[2U*393216U+(first+r+rr)*6144U+tile*32U+c]=z;
            }
            qbh_hvx_quantize_f16_to_u8(tmp,codes,128U,&h->qparams[QBH_BLOCK_QP_MIDDLE]);
            *(HVX_Vector *)(middle+tile*2048U+(first+r)*32U)=*(HVX_Vector *)codes;
        }
        h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-t;
    }
    ++h->dense_r4_calls;h->dense_r4_rows+=rows;
    if(audit)h->dense_r4_audit_bytes+=3U*rows*6144U*2U;
    return 0;
}
