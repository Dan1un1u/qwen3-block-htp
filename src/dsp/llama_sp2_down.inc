/* L32-0009: fused native SP2 producer's exact radix256 Down consumer.
 * All payload scratch is VTCM; existing double-buffered W4 DMA is unchanged.
 * Caller transfers its HVX lock before publishing this command. */
static void read_i32(uint8_t *bytes,uint32_t *bias,uint32_t digits){
 for(uint32_t j=0;j<digits;j++){
  for(uint32_t n=0;n<32;n++){bias[n]=(24U-8U*j)<<10;bias[32+n]=0;}
  Q6_bias_mxmem2_A(bias);
  Q6_mxmem_AR_after_retain_cm_ub(bytes+2048*j,0x700U);
  asm volatile("barrier":::"memory");
 }
}
struct qbh_sp2_words { HVX_Vector v0,v1,v2,v3; };
static inline struct qbh_sp2_words qbh_sp2_chunk24(const uint8_t *raw,uint32_t base) {
    HVX_VectorPair a=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(raw+base));
    HVX_VectorPair b=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(raw+base+2048U));
    HVX_VectorPair c=Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(raw+base+4096U));
    HVX_Vector lo=Q6_V_vor_VV(Q6_V_lo_W(a),Q6_Vh_vasl_VhR(Q6_V_lo_W(b),8));
    HVX_Vector hi=Q6_V_vor_VV(Q6_V_hi_W(a),Q6_Vh_vasl_VhR(Q6_V_hi_W(b),8));
    HVX_VectorPair l=Q6_Wuw_vunpack_Vuh(lo),h=Q6_Wuw_vunpack_Vuh(hi);
    HVX_VectorPair cl=Q6_Ww_vunpack_Vh(Q6_V_lo_W(c)),ch=Q6_Ww_vunpack_Vh(Q6_V_hi_W(c));
    struct qbh_sp2_words out={
      Q6_V_vor_VV(Q6_V_lo_W(l),Q6_Vw_vasl_VwR(Q6_V_lo_W(cl),16)),
      Q6_V_vor_VV(Q6_V_hi_W(l),Q6_Vw_vasl_VwR(Q6_V_hi_W(cl),16)),
      Q6_V_vor_VV(Q6_V_lo_W(h),Q6_Vw_vasl_VwR(Q6_V_lo_W(ch),16)),
      Q6_V_vor_VV(Q6_V_hi_W(h),Q6_Vw_vasl_VwR(Q6_V_hi_W(ch),16))};
    return out;
}
static inline HVX_Vector qbh_sp2_requant(HVX_Vector low,HVX_Vector high,
    HVX_Vector sum,HVX_Vector multiplier,HVX_Vector zp) {
    HVX_Vector acc=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(low,
        Q6_Vw_vasl_VwR(high,8)),Q6_Vw_vasl_VwR(sum,15));
    HVX_Vector value=Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(
        Q6_Vw_vmpye_VwVuh(acc,multiplier),acc,multiplier);
    return Q6_Vw_vmin_VwVw(Q6_V_vsplat_R(255),Q6_Vw_vmax_VwVw(Q6_V_vzero(),Q6_Vw_vadd_VwVw(value,zp)));
}
__attribute__((noinline)) static int qbh_llama_sp2_down(struct qbh_block_hmx_worker *worker) {
    uint8_t *raw0=worker->sp2_scratch, *raw1=raw0+8192U;
    uint32_t *bias=(uint32_t *)(raw0+16384U);
    if (!raw0 || !worker->sp2_high || worker->m_tiles!=1U ||
        (worker->sp2_rows!=4U && worker->sp2_rows!=64U)) return AEE_EBADPARM;
    for(uint32_t nt=0;nt<worker->n_tiles;nt++) {
        const uint8_t *w=(const uint8_t *)worker->weight+(size_t)nt*worker->k_tiles*512U;
        const int32_t *meta=(const int32_t *)worker->scale_or_bias+nt*64U;
        for(uint32_t pass=0;pass<(worker->sp2_rows==64U?2U:1U);pass++) {
            for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[32+n]=0;}
            qbh_hmx_begin_u8s8_output(bias);
            qbh_hmx_accumulate_u8n4_projection(pass?worker->sp2_high:worker->activation,w,worker->k_tiles);
            read_i32(pass?raw1:raw0,bias,3U);
        }
            HVX_Vector sum=*(const HVX_Vector *)(meta+32U),multiplier=*(const HVX_Vector *)meta;
            HVX_Vector zp=Q6_V_vsplat_R(worker->sp2_zero_point);
            for(uint32_t base=0;base<worker->sp2_rows*32U;base+=128U) {
                struct qbh_sp2_words l=qbh_sp2_chunk24(raw0,base);
                struct qbh_sp2_words h=qbh_sp2_chunk24(worker->sp2_rows==64U?raw1:raw0,
                    base+(worker->sp2_rows==64U?0U:128U));
                HVX_Vector y0=qbh_sp2_requant(l.v0,h.v0,sum,multiplier,zp);
                HVX_Vector y1=qbh_sp2_requant(l.v1,h.v1,sum,multiplier,zp);
                HVX_Vector y2=qbh_sp2_requant(l.v2,h.v2,sum,multiplier,zp);
                HVX_Vector y3=qbh_sp2_requant(l.v3,h.v3,sum,multiplier,zp);
                *(HVX_Vector *)((uint8_t *)worker->output+(size_t)nt*2048U+base)=
                    Q6_Vb_vpacke_VhVh(Q6_Vh_vpacke_VwVw(y3,y2),Q6_Vh_vpacke_VwVw(y1,y0));
            }
    }
    asm volatile("barrier":::"memory");
    return AEE_SUCCESS;
}
