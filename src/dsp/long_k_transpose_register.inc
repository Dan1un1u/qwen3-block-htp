/* EXP0297: expose the fixed32x32 word transpose to register allocation.
 * Same five shuffle stages, weights, bias and padding as the VTCM control. */
void qbh_attention_u8_pack_k_long_register(const uint8_t *rows,uint32_t valid,uint32_t padded,
 const struct qbh_attention_config *cfg,int8_t *weight,uint32_t *bias) {
    if(QBH_ATTENTION_HEAD_DIM!=128U) {qbh_attention_u8_pack_k_long_transpose(rows,valid,padded,cfg,weight,bias);return;}
    const uint32_t div=1U<<cfg->score_shift,round=cfg->score_shift?div/2U:0U;
    const uint16_t cv=qbh_attention_u8_float_to_half_bits(512.0f/(float)div);
    for(unsigned tile=0;tile<padded/32U;++tile) {
        HVX_Vector v0=tile*32U+0U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+0U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v1=tile*32U+1U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+1U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v2=tile*32U+2U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+2U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v3=tile*32U+3U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+3U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v4=tile*32U+4U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+4U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v5=tile*32U+5U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+5U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v6=tile*32U+6U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+6U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v7=tile*32U+7U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+7U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v8=tile*32U+8U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+8U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v9=tile*32U+9U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+9U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v10=tile*32U+10U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+10U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v11=tile*32U+11U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+11U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v12=tile*32U+12U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+12U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v13=tile*32U+13U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+13U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v14=tile*32U+14U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+14U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v15=tile*32U+15U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+15U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v16=tile*32U+16U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+16U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v17=tile*32U+17U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+17U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v18=tile*32U+18U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+18U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v19=tile*32U+19U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+19U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v20=tile*32U+20U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+20U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v21=tile*32U+21U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+21U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v22=tile*32U+22U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+22U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v23=tile*32U+23U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+23U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v24=tile*32U+24U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+24U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v25=tile*32U+25U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+25U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v26=tile*32U+26U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+26U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v27=tile*32U+27U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+27U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v28=tile*32U+28U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+28U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v29=tile*32U+29U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+29U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v30=tile*32U+30U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+30U)*128U),cfg->k_zero_point):Q6_V_vzero();
        HVX_Vector v31=tile*32U+31U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(tile*32U+31U)*128U),cfg->k_zero_point):Q6_V_vzero();
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v1,v0,-4);v0=Q6_V_lo_W(x);v1=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v3,v2,-4);v2=Q6_V_lo_W(x);v3=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v5,v4,-4);v4=Q6_V_lo_W(x);v5=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v6,-4);v6=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v9,v8,-4);v8=Q6_V_lo_W(x);v9=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v10,-4);v10=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v12,-4);v12=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v14,-4);v14=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v17,v16,-4);v16=Q6_V_lo_W(x);v17=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v19,v18,-4);v18=Q6_V_lo_W(x);v19=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v21,v20,-4);v20=Q6_V_lo_W(x);v21=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v23,v22,-4);v22=Q6_V_lo_W(x);v23=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v25,v24,-4);v24=Q6_V_lo_W(x);v25=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v27,v26,-4);v26=Q6_V_lo_W(x);v27=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v29,v28,-4);v28=Q6_V_lo_W(x);v29=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v31,v30,-4);v30=Q6_V_lo_W(x);v31=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v2,v0,-8);v0=Q6_V_lo_W(x);v2=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v3,v1,-8);v1=Q6_V_lo_W(x);v3=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v6,v4,-8);v4=Q6_V_lo_W(x);v6=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v5,-8);v5=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v10,v8,-8);v8=Q6_V_lo_W(x);v10=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v9,-8);v9=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v12,-8);v12=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v13,-8);v13=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v18,v16,-8);v16=Q6_V_lo_W(x);v18=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v19,v17,-8);v17=Q6_V_lo_W(x);v19=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v22,v20,-8);v20=Q6_V_lo_W(x);v22=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v23,v21,-8);v21=Q6_V_lo_W(x);v23=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v26,v24,-8);v24=Q6_V_lo_W(x);v26=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v27,v25,-8);v25=Q6_V_lo_W(x);v27=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v30,v28,-8);v28=Q6_V_lo_W(x);v30=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v31,v29,-8);v29=Q6_V_lo_W(x);v31=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v4,v0,-16);v0=Q6_V_lo_W(x);v4=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v5,v1,-16);v1=Q6_V_lo_W(x);v5=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v6,v2,-16);v2=Q6_V_lo_W(x);v6=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v3,-16);v3=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v12,v8,-16);v8=Q6_V_lo_W(x);v12=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v9,-16);v9=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v10,-16);v10=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v11,-16);v11=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v20,v16,-16);v16=Q6_V_lo_W(x);v20=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v21,v17,-16);v17=Q6_V_lo_W(x);v21=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v22,v18,-16);v18=Q6_V_lo_W(x);v22=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v23,v19,-16);v19=Q6_V_lo_W(x);v23=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v28,v24,-16);v24=Q6_V_lo_W(x);v28=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v29,v25,-16);v25=Q6_V_lo_W(x);v29=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v30,v26,-16);v26=Q6_V_lo_W(x);v30=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v31,v27,-16);v27=Q6_V_lo_W(x);v31=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v8,v0,-32);v0=Q6_V_lo_W(x);v8=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v9,v1,-32);v1=Q6_V_lo_W(x);v9=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v10,v2,-32);v2=Q6_V_lo_W(x);v10=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v3,-32);v3=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v12,v4,-32);v4=Q6_V_lo_W(x);v12=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v5,-32);v5=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v6,-32);v6=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v7,-32);v7=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v24,v16,-32);v16=Q6_V_lo_W(x);v24=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v25,v17,-32);v17=Q6_V_lo_W(x);v25=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v26,v18,-32);v18=Q6_V_lo_W(x);v26=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v27,v19,-32);v19=Q6_V_lo_W(x);v27=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v28,v20,-32);v20=Q6_V_lo_W(x);v28=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v29,v21,-32);v21=Q6_V_lo_W(x);v29=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v30,v22,-32);v22=Q6_V_lo_W(x);v30=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v31,v23,-32);v23=Q6_V_lo_W(x);v31=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v16,v0,-64);v0=Q6_V_lo_W(x);v16=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v17,v1,-64);v1=Q6_V_lo_W(x);v17=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v18,v2,-64);v2=Q6_V_lo_W(x);v18=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v19,v3,-64);v3=Q6_V_lo_W(x);v19=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v20,v4,-64);v4=Q6_V_lo_W(x);v20=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v21,v5,-64);v5=Q6_V_lo_W(x);v21=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v22,v6,-64);v6=Q6_V_lo_W(x);v22=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v23,v7,-64);v7=Q6_V_lo_W(x);v23=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v24,v8,-64);v8=Q6_V_lo_W(x);v24=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v25,v9,-64);v9=Q6_V_lo_W(x);v25=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v26,v10,-64);v10=Q6_V_lo_W(x);v26=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v27,v11,-64);v11=Q6_V_lo_W(x);v27=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v28,v12,-64);v12=Q6_V_lo_W(x);v28=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v29,v13,-64);v13=Q6_V_lo_W(x);v29=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v30,v14,-64);v14=Q6_V_lo_W(x);v30=Q6_V_hi_W(x);}
        {HVX_VectorPair x=Q6_W_vshuff_VVR(v31,v15,-64);v15=Q6_V_lo_W(x);v31=Q6_V_hi_W(x);}
        HVX_Vector sum=Q6_V_vzero();
        *(HVX_Vector *)(weight+(size_t)tile*4096U+0U)=v0;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v0,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2048U)=v1;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v1,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1024U)=v2;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v2,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3072U)=v3;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v3,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+512U)=v4;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v4,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2560U)=v5;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v5,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1536U)=v6;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v6,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3584U)=v7;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v7,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+256U)=v8;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v8,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2304U)=v9;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v9,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1280U)=v10;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v10,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3328U)=v11;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v11,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+768U)=v12;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v12,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2816U)=v13;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v13,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1792U)=v14;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v14,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3840U)=v15;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v15,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+128U)=v16;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v16,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2176U)=v17;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v17,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1152U)=v18;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v18,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3200U)=v19;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v19,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+640U)=v20;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v20,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2688U)=v21;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v21,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1664U)=v22;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v22,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3712U)=v23;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v23,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+384U)=v24;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v24,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2432U)=v25;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v25,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1408U)=v26;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v26,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3456U)=v27;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v27,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+896U)=v28;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v28,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+2944U)=v29;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v29,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+1920U)=v30;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v30,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(weight+(size_t)tile*4096U+3968U)=v31;
        sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(v31,Q6_Vb_vsplat_R(1)));
        *(HVX_Vector *)(bias+tile*64U)=Q6_V_vsplat_R(cv);
        *(HVX_Vector *)(bias+tile*64U+32U)=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(128*div+round),Q6_Vw_vmpyi_VwRh(sum,cfg->q_zero_point*0x00010001U));
    }
    asm volatile("barrier" ::: "memory");
}
