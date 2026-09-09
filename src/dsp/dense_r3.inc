#include "r3_sign_matrix.inc"
/* EXP0247: explicit dense HMX GEMM. Included after shared block helpers.
 * Gate/up and projection scratch are dead after the QKV completion barrier.
 * Every head uses the same right transform; cached history is never touched. */
static void qbh_r3_pack(const __fp16 *src, __fp16 *dst, uint32_t rows,
                        uint32_t unpack) {
    for (uint32_t r=0;r<rows;r+=2U) for (uint32_t c=0;c<128U;c+=64U) {
        size_t tile=qbh_hmx_fp16_matrix_tile_offset(r/32U,c/32U,4U);
        size_t pair=(r%32U)/2U;
        if (!unpack) {
            HVX_VectorPair v=Q6_W_vshuff_VVR(
                *(const HVX_Vector *)(src+(r+1U)*128U+c),
                *(const HVX_Vector *)(src+r*128U+c),-2);
            ((HVX_Vector *)(dst+tile))[pair]=Q6_V_lo_W(v);
            ((HVX_Vector *)(dst+tile+1024U))[pair]=Q6_V_hi_W(v);
        } else {
            HVX_VectorPair v=Q6_W_vdeal_VVR(
                ((const HVX_Vector *)(src+tile+1024U))[pair],
                ((const HVX_Vector *)(src+tile))[pair],-2);
            *(HVX_Vector *)(dst+r*128U+c)=Q6_V_lo_W(v);
            *(HVX_Vector *)(dst+(r+1U)*128U+c)=Q6_V_hi_W(v);
        }
    }
    asm volatile("barrier" ::: "memory");
}

/* Direct dot-product refinement. Each sign row is explicitly materialized;
 * the reduction sums products within one dot and never performs an FWHT. */
static uint32_t qbh_r3_refine(const __fp16 *raw, __fp16 *result,
    uint32_t rows, uint32_t logical_rows, const struct qbh_block_header *h,
    uint32_t *signs) {
    for (uint32_t n=0;n<128U;++n) for (uint32_t part=0;part<4U;++part)
        for (uint32_t lane=0;lane<32U;++lane) {
            uint32_t k=(part/2U)*64U+2U*lane+part%2U;
            signs[n*128U+part*32U+lane]=(__builtin_popcount(n&k)&1)?0x80000000U:0U;
        }
    uint32_t count=0;
    for (uint32_t r=0;r<rows;++r) {
        const struct qbh_block_qparam *qp=&h->qparams[
            r/logical_rows<16U?QBH_BLOCK_QP_Q_ROPE:QBH_BLOCK_QP_K_ROPE];
        const float inv=1.0f/qp->scale;
        HVX_VectorPair x0=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(raw+r*128U));
        HVX_VectorPair x1=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(raw+r*128U+64U));
        HVX_Vector x[4]={Q6_V_lo_W(x0),Q6_V_hi_W(x0),Q6_V_lo_W(x1),Q6_V_hi_W(x1)};
        for (uint32_t n=0;n<128U;++n) {
            uint16_t bits;memcpy(&bits,result+r*128U+n,2U);
            uint16_t mag=bits&0x7fffU,up=mag+1U;
            __fp16 yy,zz;memcpy(&yy,&mag,2U);memcpy(&zz,&up,2U);
            float err=2.0f*((float)zz-(float)yy)+0.00006103515625f;
            float y=(float)result[r*128U+n];
            int lo=(int)floorf((y-err)*inv+(float)qp->zero_point+0.5f);
            int hi=(int)floorf((y+err)*inv+(float)qp->zero_point+0.5f);
            lo=lo<0?0:lo>255?255:lo;hi=hi<0?0:hi>255?255:hi;
            if (lo==hi) continue;
            const HVX_Vector *sg=(const HVX_Vector *)(signs+n*128U);
            HVX_Vector a=Q6_Vsf_vadd_VsfVsf(Q6_V_vxor_VV(x[0],sg[0]),Q6_V_vxor_VV(x[1],sg[1]));
            HVX_Vector b=Q6_Vsf_vadd_VsfVsf(Q6_V_vxor_VV(x[2],sg[2]),Q6_V_vxor_VV(x[3],sg[3]));
            a=Q6_Vsf_vadd_VsfVsf(a,b);
            for (uint32_t shift=64U;shift>=4U;shift>>=1U)
                a=Q6_Vsf_vadd_VsfVsf(a,Q6_V_vlalign_VVR(a,Q6_V_vzero(),shift));
            uint32_t word=(uint32_t)Q6_R_vextract_VR(a,124);float sum;memcpy(&sum,&word,4U);
            result[r*128U+n]=(__fp16)(sum*0.08837890625f);++count;
        }
    }
    return count;
}

static int qbh_run_dense_r3(struct qbh_block_header *h,uint8_t *shared,
    struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,
    uint32_t logical_rows) {
    const uint32_t rows=24U*logical_rows, padded=(rows+31U)&~31U;
    __fp16 *raw=(__fp16 *)b->gate, *result=(__fp16 *)b->up;
    __fp16 *act=(__fp16 *)b->hmx_activation;
    __fp16 *out=(__fp16 *)(b->hmx_activation+QBH_DENSE_R3_CARRIER_BYTES);
    __fp16 *weight=(__fp16 *)b->expanded_weight;
    const __fp16 normalization=(__fp16)0.08838834764831844f;
    float scales[32] __attribute__((aligned(128)));
    if (h->dense_r3_mode==0U) return 0;
    if (h->dense_r3_optimization>2U ||
        (h->dense_r3_optimization && h->dense_r3_mode!=1U) || h->dense_r3_mode>5U || h->variant!=QBH_BLOCK_W4U8 ||
        (logical_rows!=1U && logical_rows!=64U) ||
        h->u8_norm_reduction_mode < QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE ||
        (h->dense_r3_audit_offset && !qbh_range_valid(h->dense_r3_audit_offset,
            QBH_DENSE_R3_AUDIT_BYTES,h->shared_bytes))) return -1;
    _Static_assert(2U*QBH_DENSE_R3_CARRIER_BYTES<=64U*QBH_BLOCK_MAX_K*2U,
                   "R3 input/output must fit the dead HMX activation arena");
    h->dense_r3_rows=rows; h->dense_r3_hmx_calls=0; h->dense_r3_refined_values=0;
    uint64_t start=HAP_perf_get_qtimer_count();
    if (h->dense_r3_optimization<2U) {
    memset(raw,0,padded*128U*2U);
    for (uint32_t head=0;head<24U;++head) {
        const uint32_t k=head>=16U;
        (h->dense_r3_optimization ? qbh_hvx_r3_prepare_head_vector : qbh_hvx_r3_prepare_head)((k?b->k:b->q)+(head-(k?16U:0U))*8192U,
            logical_rows,&h->qparams[k?QBH_BLOCK_QP_K_PROJECTION:QBH_BLOCK_QP_Q_PROJECTION],
            (const __fp16 *)(k?b->k_norm_weight:b->q_norm_weight),
            b->attention_projection+QBH_QK_ROPE_SF32_CACHE_OFFSET,
            raw+head*logical_rows*128U);
    }
    } /* Streamed prepare already owns raw, joined by QKV ring. */
    qbh_r3_pack(raw,act,padded,0U);
    /* Materialize the full dense matrix, with no fast-transform recurrence. */
    if (h->dense_r3_optimization) {
        for (uint32_t i=0;i<sizeof(qbh_r3_sign_matrix)/128U;++i)
            ((HVX_Vector *)weight)[i]=((const HVX_Vector *)qbh_r3_sign_matrix)[i];
        h->dense_r3_constant_read_bytes+=sizeof(qbh_r3_sign_matrix);
    } else for (uint32_t n=0;n<128U;++n) for (uint32_t k=0;k<128U;++k) {
        size_t pos=((n/32U)*4U+k/32U)*1024U+
            qbh_hmx_fp16_tile_offset(k%32U,n%32U);
        weight[pos]=h->dense_r3_mode==3U ? (__fp16)(n==k) :
            (__fp16)((__builtin_popcount(n&k)&1)?-1.0f:1.0f);
    }
    for (uint32_t i=0;i<32U;++i) scales[i]=h->dense_r3_mode==3U?1.0f:(float)normalization;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r3_prepare_ticks=HAP_perf_get_qtimer_count()-start;
    start=HAP_perf_get_qtimer_count();
    if (h->dense_r3_mode==4U) {
        /* Audit only: reproduce the candidate's HMX state transition before
         * replacing its output with the independent scalar dense result. */
        if (qbh_hmx_submit(worker,QBH_BLOCK_HMX_FP16,act,weight,b->channel_scale,
                           out,padded/32U,4U,4U)!=0) return -1;
        ++h->hmx_command_count; ++h->dense_r3_hmx_calls;
        h->hmx_fp16_tile_pair_count+=(padded/32U)*16U;
    }
    if (h->dense_r3_mode==2U || h->dense_r3_mode==4U) {
        /* Untimed independent scalar dense dot-product oracle, never fallback. */
        for (uint32_t r=0;r<rows;++r) for (uint32_t n=0;n<128U;++n) {
            double sum=0.0;
            for (uint32_t k=0;k<128U;++k)
                sum+=(double)raw[r*128U+k]*((__builtin_popcount(n&k)&1)?-1.0:1.0);
            result[r*128U+n]=(__fp16)(sum*(double)normalization);
        }
    } else {
        if (qbh_hmx_submit(worker,QBH_BLOCK_HMX_FP16,act,weight,b->channel_scale,
                           out,padded/32U,4U,4U)!=0) return -1;
        ++h->hmx_command_count; ++h->dense_r3_hmx_calls;
        h->hmx_fp16_tile_pair_count+=(padded/32U)*16U;
        qbh_r3_pack(out,result,padded,1U);
    }
    if (h->dense_r3_mode==5U) h->dense_r3_refined_values=qbh_r3_refine(raw,result,rows,logical_rows,h,(uint32_t *)b->expanded_weight);
    h->dense_r3_matmul_ticks=HAP_perf_get_qtimer_count()-start;
    start=HAP_perf_get_qtimer_count();
    for (uint32_t head=0;head<24U;++head) {
        uint32_t k=head>=16U;
        if(h->dense_r3_optimization) {
            uint8_t *scratch=b->attention_concat+(k?head-16U:0U)*QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
            qbh_hvx_r3_quantize_head_vector(result+head*logical_rows*128U,
                (k?b->k:b->q)+(head-(k?16U:0U))*8192U,logical_rows,
                &h->qparams[k?QBH_BLOCK_QP_K_ROPE:QBH_BLOCK_QP_Q_ROPE],
                k && logical_rows==64U?&b->attention_configs[head-16U]:NULL,
                (int8_t *)(scratch+QBH_ATTN_U8_K_WEIGHT_OFFSET),
                (uint32_t *)(scratch+QBH_ATTN_U8_QK_BIAS_OFFSET));
        } else qbh_hvx_r3_quantize_head(result+head*logical_rows*128U,
            (k?b->k:b->q)+(head-(k?16U:0U))*8192U,logical_rows,
            &h->qparams[k?QBH_BLOCK_QP_K_ROPE:QBH_BLOCK_QP_Q_ROPE]);
    }
    if (logical_rows==64U && !h->dense_r3_optimization) for (uint32_t k=0;k<8U;++k) {
        uint8_t *scratch=b->attention_concat+k*QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
        qbh_attention_u8_pack_k_native(b->k+k*8192U,&b->attention_configs[k],
            (int8_t *)(scratch+QBH_ATTN_U8_K_WEIGHT_OFFSET),
            (uint32_t *)(scratch+QBH_ATTN_U8_QK_BIAS_OFFSET));
    }
    h->dense_r3_finish_ticks=HAP_perf_get_qtimer_count()-start;
    ++h->dense_r3_total_calls;
    h->dense_r3_total_rows+=rows;
    h->dense_r3_total_hmx_calls+=h->dense_r3_hmx_calls;
    h->dense_r3_total_refined_values+=h->dense_r3_refined_values;
    h->dense_r3_total_prepare_ticks+=h->dense_r3_prepare_ticks;
    h->dense_r3_total_matmul_ticks+=h->dense_r3_matmul_ticks;
    h->dense_r3_total_finish_ticks+=h->dense_r3_finish_ticks;
    h->u8_attention_qk_norm_rope_ticks+=h->dense_r3_prepare_ticks+
        h->dense_r3_matmul_ticks+h->dense_r3_finish_ticks;
    if (h->dense_r3_audit_offset) {
        uint8_t *audit=shared+h->dense_r3_audit_offset;
        memset(audit,0,QBH_DENSE_R3_AUDIT_BYTES);
        memcpy(audit,raw,rows*128U*2U);
        memcpy(audit+QBH_DENSE_R3_CARRIER_BYTES,result,rows*128U*2U);
        memcpy(audit+2U*QBH_DENSE_R3_CARRIER_BYTES,b->q,16U*8192U);
        memcpy(audit+2U*QBH_DENSE_R3_CARRIER_BYTES+16U*8192U,b->k,8U*8192U);
    }
    return 0;
}
