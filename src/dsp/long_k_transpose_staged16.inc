/* EXP0302: two independent 16-row register transposes followed by the
 * final shuffle. Scratch is the disjoint, not-yet-produced QK output plane.
 * All padded K operands and zero-point biases retain the original bytes. */
void qbh_attention_u8_pack_k_long_staged16(const uint8_t *rows,
 unsigned valid,unsigned padded,const struct qbh_attention_config *cfg,
 int8_t *weight,uint32_t *bias,uint8_t *scratch) {
 if(QBH_ATTENTION_HEAD_DIM!=128U) {
  qbh_attention_u8_pack_k_long_register(rows,valid,padded,cfg,weight,bias);return;
 }
 const unsigned div=1U<<cfg->score_shift,round=cfg->score_shift?div/2U:0U;
 const uint16_t cv=qbh_attention_u8_float_to_half_bits(512.0f/(float)div);
 for(unsigned tile=0;tile<padded/32U;++tile) {
  for(unsigned half=0;half<2U;++half) {
   const unsigned first=tile*32U+half*16U;
   HVX_Vector v0=first+0U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+0U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v1=first+1U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+1U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v2=first+2U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+2U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v3=first+3U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+3U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v4=first+4U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+4U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v5=first+5U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+5U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v6=first+6U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+6U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v7=first+7U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+7U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v8=first+8U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+8U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v9=first+9U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+9U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v10=first+10U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+10U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v11=first+11U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+11U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v12=first+12U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+12U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v13=first+13U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+13U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v14=first+14U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+14U)*128U),cfg->k_zero_point):Q6_V_vzero();
   HVX_Vector v15=first+15U<valid?qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(first+15U)*128U),cfg->k_zero_point):Q6_V_vzero();
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v1,v0,-4);v0=Q6_V_lo_W(x);v1=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v3,v2,-4);v2=Q6_V_lo_W(x);v3=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v5,v4,-4);v4=Q6_V_lo_W(x);v5=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v6,-4);v6=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v9,v8,-4);v8=Q6_V_lo_W(x);v9=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v10,-4);v10=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v12,-4);v12=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v14,-4);v14=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v2,v0,-8);v0=Q6_V_lo_W(x);v2=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v3,v1,-8);v1=Q6_V_lo_W(x);v3=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v6,v4,-8);v4=Q6_V_lo_W(x);v6=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v5,-8);v5=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v10,v8,-8);v8=Q6_V_lo_W(x);v10=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v9,-8);v9=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v12,-8);v12=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v13,-8);v13=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v4,v0,-16);v0=Q6_V_lo_W(x);v4=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v5,v1,-16);v1=Q6_V_lo_W(x);v5=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v6,v2,-16);v2=Q6_V_lo_W(x);v6=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v7,v3,-16);v3=Q6_V_lo_W(x);v7=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v12,v8,-16);v8=Q6_V_lo_W(x);v12=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v9,-16);v9=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v10,-16);v10=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v11,-16);v11=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v8,v0,-32);v0=Q6_V_lo_W(x);v8=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v9,v1,-32);v1=Q6_V_lo_W(x);v9=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v10,v2,-32);v2=Q6_V_lo_W(x);v10=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v11,v3,-32);v3=Q6_V_lo_W(x);v11=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v12,v4,-32);v4=Q6_V_lo_W(x);v12=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v13,v5,-32);v5=Q6_V_lo_W(x);v13=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v14,v6,-32);v6=Q6_V_lo_W(x);v14=Q6_V_hi_W(x);}
   {HVX_VectorPair x=Q6_W_vshuff_VVR(v15,v7,-32);v7=Q6_V_lo_W(x);v15=Q6_V_hi_W(x);}
   ((HVX_Vector *)scratch)[half*16U+0U]=v0;
   ((HVX_Vector *)scratch)[half*16U+1U]=v1;
   ((HVX_Vector *)scratch)[half*16U+2U]=v2;
   ((HVX_Vector *)scratch)[half*16U+3U]=v3;
   ((HVX_Vector *)scratch)[half*16U+4U]=v4;
   ((HVX_Vector *)scratch)[half*16U+5U]=v5;
   ((HVX_Vector *)scratch)[half*16U+6U]=v6;
   ((HVX_Vector *)scratch)[half*16U+7U]=v7;
   ((HVX_Vector *)scratch)[half*16U+8U]=v8;
   ((HVX_Vector *)scratch)[half*16U+9U]=v9;
   ((HVX_Vector *)scratch)[half*16U+10U]=v10;
   ((HVX_Vector *)scratch)[half*16U+11U]=v11;
   ((HVX_Vector *)scratch)[half*16U+12U]=v12;
   ((HVX_Vector *)scratch)[half*16U+13U]=v13;
   ((HVX_Vector *)scratch)[half*16U+14U]=v14;
   ((HVX_Vector *)scratch)[half*16U+15U]=v15;
  }
  HVX_Vector sum=Q6_V_vzero();
  for(unsigned i=0;i<16U;++i) {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[i+16U],((HVX_Vector *)scratch)[i],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   unsigned rev=((i&1U)<<4)|((i&2U)<<2)|(i&4U)|((i&8U)>>2);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[rev]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[rev+1U]=hi;
   sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum=Q6_Vw_vadd_VwVw(sum,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  *(HVX_Vector *)(bias+tile*64U)=Q6_V_vsplat_R(cv);
  *(HVX_Vector *)(bias+tile*64U+32U)=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(128*div+round),Q6_Vw_vmpyi_VwRh(sum,cfg->q_zero_point*0x00010001U));
 }
 asm volatile("barrier" ::: "memory");
}
