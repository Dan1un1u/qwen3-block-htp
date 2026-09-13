/* L32-0009: fused native SP2 producer's exact radix256 Down consumer.
 * All payload scratch is VTCM; existing double-buffered W4 DMA is unchanged.
 * Caller transfers its HVX lock before publishing this command. */
static void read_i32(uint8_t *bytes,uint32_t *bias){
 for(uint32_t j=0;j<4;j++){
  for(uint32_t n=0;n<32;n++){bias[n]=(24U-8U*j)<<10;bias[32+n]=0;}
  Q6_bias_mxmem2_A(bias);
  Q6_mxmem_AR_after_retain_cm_ub(bytes+2048*j,0x700U);
  asm volatile("barrier":::"memory");
 }
}
static void unpack_digits(const uint8_t *bytes,int32_t *wide,uint32_t digits,uint32_t elements){
 for(uint32_t base=0;base<elements;base+=128){
  HVX_Vector value[4]={Q6_V_vzero(),Q6_V_vzero(),Q6_V_vzero(),Q6_V_vzero()};
  for(uint32_t digit=0;digit<digits;digit++){
   HVX_Vector x=*(const HVX_Vector*)(bytes+2048*digit+base);
   HVX_VectorPair h=Q6_Wuh_vunpack_Vub(x);
   HVX_VectorPair wl=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(h)),wh=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(h));
   HVX_Vector w[4]={Q6_V_lo_W(wl),Q6_V_hi_W(wl),Q6_V_lo_W(wh),Q6_V_hi_W(wh)};
   for(uint32_t j=0;j<4;j++)value[j]=Q6_V_vor_VV(value[j],Q6_Vw_vasl_VwR(w[j],8*digit));
  }
  for(uint32_t j=0;j<4;j++)*(HVX_Vector*)(wide+base+32*j)=value[j];
 }
}

static int qbh_llama_sp2_down(struct qbh_block_hmx_worker *worker) {
    uint8_t *raw0=worker->sp2_scratch, *raw1=raw0+8192U;
    int32_t *wide0=(int32_t *)(raw0+16384U), *wide1=wide0+2048U;
    uint32_t *bias=(uint32_t *)(raw0+32768U);
    int32_t *merged=(int32_t *)(raw0+33024U);
    if (!raw0 || !worker->sp2_high || worker->m_tiles!=1U ||
        (worker->sp2_rows!=4U && worker->sp2_rows!=64U)) return AEE_EBADPARM;
    if(qurt_hvx_lock(QURT_HVX_MODE_128B)!=AEE_SUCCESS) return AEE_EFAILED;
    for(uint32_t nt=0;nt<worker->n_tiles;nt++) {
        const uint8_t *w=(const uint8_t *)worker->weight+(size_t)nt*worker->k_tiles*512U;
        const int32_t *meta=(const int32_t *)worker->scale_or_bias+nt*64U;
        for(uint32_t pass=0;pass<(worker->sp2_rows==64U?2U:1U);pass++) {
            for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[32+n]=0;}
            qbh_hmx_begin_u8s8_output(bias);
            qbh_hmx_accumulate_u8n4_projection(pass?worker->sp2_high:worker->activation,w,worker->k_tiles);
            read_i32(pass?raw1:raw0,bias);
        }
        unpack_digits(raw0,wide0,4U,worker->sp2_mode>=2U && worker->sp2_rows==4U ? 256U:2048U);
        if(worker->sp2_rows==64U)unpack_digits(raw1,wide1,4U,2048U);
        if (worker->sp2_mode>=2U) {
            HVX_Vector sum=*(const HVX_Vector *)(meta+32U);
            HVX_Vector multiplier=*(const HVX_Vector *)meta;
            HVX_Vector zero=Q6_V_vzero(),limit=Q6_V_vsplat_R(255);
            HVX_Vector zp=Q6_V_vsplat_R(worker->sp2_zero_point);
            for(uint32_t row=0;row<worker->sp2_rows;row+=4U) {
                HVX_Vector y[4];
                for(uint32_t j=0;j<4;j++) {
                    HVX_Vector low=*(const HVX_Vector *)(wide0+(row+j)*32U);
                    HVX_Vector high=*(const HVX_Vector *)(worker->sp2_rows==64U ?
                        wide1+(row+j)*32U : wide0+(row+j+4U)*32U);
                    HVX_Vector acc=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(low,
                        Q6_Vw_vasl_VwR(high,8)),Q6_Vw_vasl_VwR(sum,15));
                    HVX_Vector value=Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(
                        Q6_Vw_vmpye_VwVuh(acc,multiplier),acc,multiplier);
                    y[j]=Q6_Vw_vmin_VwVw(limit,Q6_Vw_vmax_VwVw(zero,Q6_Vw_vadd_VwVw(value,zp)));
                }
                *(HVX_Vector *)((uint8_t *)worker->output+(size_t)nt*2048U+row*32U)=
                    Q6_Vb_vpacke_VhVh(Q6_Vh_vpacke_VwVw(y[3],y[2]),Q6_Vh_vpacke_VwVw(y[1],y[0]));
            }
        } else
        for(uint32_t row=0;row<worker->sp2_rows;row++) {
            HVX_Vector low=*(const HVX_Vector *)(wide0+row*32U);
            HVX_Vector high=*(const HVX_Vector *)(worker->sp2_rows==64U ?
                wide1+row*32U : wide0+(row+4U)*32U);
            HVX_Vector sum=*(const HVX_Vector *)(meta+32U);
            *(HVX_Vector *)merged=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(low,
                Q6_Vw_vasl_VwR(high,8)),Q6_Vw_vasl_VwR(sum,15));
            /* Initial reference epilogue: exact signed int64 Q31 rounding.
             * Candidate HVX epilogues must match these output codes exactly. */
            uint8_t *out=(uint8_t *)worker->output+(size_t)nt*2048U+row*32U;
            for(uint32_t n=0;n<32;n++) {
                int64_t value=((int64_t)merged[n]*meta[n]+(INT64_C(1)<<30))>>31;
                value+=worker->sp2_zero_point;
                out[n]=(uint8_t)(value<0?0:value>255?255:value);
            }
        }
    }
    asm volatile("barrier":::"memory");
    return qurt_hvx_unlock()==AEE_SUCCESS ? AEE_SUCCESS : AEE_EFAILED;
}
