
/* L32-0051: four native query rows per vector. Cross-KV reductions stay
 * vector-valued until the entire causal row is available. Exact-fast division
 * is the same integer probability contract as wide_score=8. */
static inline HVX_VectorPred qbh_long_native_valid(int base) {
    HVX_Vector lane=*(const HVX_Vector *)qbh_attention_u8_lane_index;
    if(base>=32) return Q6_Q_vcmp_eq_VbVb(lane,lane);
    if(base<=-3) return Q6_Q_vcmp_gt_VubVub(lane,lane);
    HVX_VectorPair lh=Q6_Wuh_vunpack_Vub(lane);
    HVX_Vector lo=Q6_V_lo_W(lh),hi=Q6_V_hi_W(lh);
    HVX_Vector clo=Q6_V_vand_VV(lo,Q6_Vh_vsplat_R(31));
    HVX_Vector chi=Q6_V_vand_VV(hi,Q6_Vh_vsplat_R(31));
    lo=Q6_Vh_vadd_VhVh(Q6_Vuh_vlsr_VuhR(lo,5),Q6_Vh_vsplat_R(base));
    hi=Q6_Vh_vadd_VhVh(Q6_Vuh_vlsr_VuhR(hi,5),Q6_Vh_vsplat_R(base));
    lo=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(lo,clo),Q6_Vh_vsplat_R(255),Q6_V_vzero());
    hi=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(hi,chi),Q6_Vh_vsplat_R(255),Q6_V_vzero());
    return Q6_Q_vcmp_gt_VubVub(Q6_Vub_vpack_VhVh_sat(hi,lo),Q6_V_vzero());
}
void qbh_attention_u8_softmax_long_native4(uint8_t *scores,uint8_t *prob,
    uint32_t rows,uint32_t past,uint32_t padded,
    const struct qbh_attention_config *cfg) {
    const unsigned tiles=padded/32;
    const HVX_Vector zero=Q6_V_vzero();
    const HVX_Vector lane=*(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred low64=Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(64),lane);
    const HVX_Vector banks=Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(lane,Q6_Vb_vsplat_R(32)),1);
    memset(prob,0,(size_t)QBH_ATTENTION_Q_HEADS_PER_GROUP*tiles*2048);
    for(unsigned head=0;head<QBH_ATTENTION_Q_HEADS_PER_GROUP;++head)
    for(unsigned row=0;row<rows;row+=4) {
        HVX_Vector maxv=zero;
        HVX_Vector sum0=zero,sum1=zero,sum2=zero,sum3=zero;
        for(unsigned t=0;t<tiles;++t) {
            HVX_Vector raw=*(HVX_Vector *)(scores+((size_t)head*tiles+t)*2048+row*32);
            HVX_Vector v=qbh_attention_u8_requant_centered(raw,cfg->score_multiplier,128);
            maxv=Q6_Vub_vmax_VubVub(maxv,Q6_V_vmux_QVV(qbh_long_native_valid((int)(past+row+1)-(int)t*32),v,zero));
        }
        for(unsigned shift=16;shift;shift>>=1)
            maxv=Q6_Vub_vmax_VubVub(maxv,Q6_V_vlalign_VVR(maxv,zero,shift));
        unsigned m0=Q6_R_vextract_VR(maxv,28)>>24,m1=Q6_R_vextract_VR(maxv,60)>>24;
        unsigned m2=Q6_R_vextract_VR(maxv,92)>>24,m3=Q6_R_vextract_VR(maxv,124)>>24;
        HVX_Vector mx=Q6_V_vmux_QVV(low64,
            Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(32),lane),Q6_Vb_vsplat_R(m0),Q6_Vb_vsplat_R(m1)),
            Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(96),lane),Q6_Vb_vsplat_R(m2),Q6_Vb_vsplat_R(m3)));
        for(unsigned t=0;t<tiles;++t) {
            uint8_t *p=scores+((size_t)head*tiles+t)*2048+row*32;
            HVX_Vector v=qbh_attention_u8_requant_centered(*(HVX_Vector *)p,cfg->score_multiplier,128);
            HVX_VectorPair dh=Q6_Wuh_vunpack_Vub(Q6_Vub_vsub_VubVub_sat(mx,v));
            HVX_Vector round=Q6_Vh_vsplat_R(1U<<(cfg->fraction_bits-1));
            HVX_Vector a=Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(Q6_V_lo_W(dh),round),cfg->fraction_bits),Q6_Vh_vsplat_R(15));
            HVX_Vector b=Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(Q6_V_hi_W(dh),round),cfg->fraction_bits),Q6_Vh_vsplat_R(15));
            HVX_Vector codes=Q6_V_vmux_QVV(qbh_long_native_valid((int)(past+row+1)-(int)t*32),Q6_Vub_vpack_VhVh_sat(b,a),Q6_Vb_vsplat_R(16));
            *(HVX_Vector *)p=codes;
            HVX_VectorPair ch=Q6_Wuh_vunpack_Vub(codes);
            a=Q6_Vh_vasl_VhVh(Q6_Vh_vsplat_R(1),Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(15),Q6_V_lo_W(ch)));
            b=Q6_Vh_vasl_VhVh(Q6_Vh_vsplat_R(1),Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(15),Q6_V_hi_W(ch)));
            HVX_VectorPair aw=Q6_Wuw_vunpack_Vuh(a),bw=Q6_Wuw_vunpack_Vuh(b);
            sum0=Q6_Vw_vadd_VwVw(sum0,Q6_V_lo_W(aw));sum1=Q6_Vw_vadd_VwVw(sum1,Q6_V_hi_W(aw));
            sum2=Q6_Vw_vadd_VwVw(sum2,Q6_V_lo_W(bw));sum3=Q6_Vw_vadd_VwVw(sum3,Q6_V_hi_W(bw));
        }
        uint8_t lut0[128] __attribute__((aligned(128)))={0};
        uint8_t lut1[128] __attribute__((aligned(128)))={0};
        qbh_attention_u8_build_exact_fast_entries(lut0,0,qbh_attention_u8_reduce_sum_words(sum0));
        qbh_attention_u8_build_exact_fast_entries(lut0,16,qbh_attention_u8_reduce_sum_words(sum1));
        qbh_attention_u8_build_exact_fast_entries(lut1,0,qbh_attention_u8_reduce_sum_words(sum2));
        qbh_attention_u8_build_exact_fast_entries(lut1,16,qbh_attention_u8_reduce_sum_words(sum3));
        HVX_VectorPred live=rows-row>=4 ? Q6_Q_vcmp_eq_VbVb(lane,lane) :
            Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R((rows-row)*32),lane);
        for(unsigned t=0;t<tiles;++t) {
            size_t off=((size_t)head*tiles+t)*2048+row*32;
            HVX_Vector codes=*(HVX_Vector *)(scores+off);
            HVX_Vector idx=Q6_Vb_vadd_VbVb(codes,banks);
            HVX_Vector out=Q6_V_vmux_QVV(low64,
                Q6_Vb_vlut32_VbVbR_nomatch(idx,*(HVX_Vector *)lut0,0),
                Q6_Vb_vlut32_VbVbR_nomatch(idx,*(HVX_Vector *)lut1,0));
            HVX_VectorPred valid=Q6_Q_and_QQ(live,Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(16),codes));
            *(HVX_Vector *)(prob+off)=Q6_V_vmux_QVV(valid,out,zero);
        }
    }
    asm volatile("barrier" ::: "memory");
}
