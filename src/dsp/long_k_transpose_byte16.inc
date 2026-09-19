/* EXP0302: two independent 16-row register transposes followed by the
 * final shuffle. Scratch is the disjoint, not-yet-produced QK output plane.
 * All padded K operands and zero-point biases retain the original bytes. */
void qbh_attention_u8_pack_k_long_byte16(const uint8_t *rows,
 unsigned valid,unsigned padded,const struct qbh_attention_config *cfg,
 int8_t *weight,uint32_t *bias,uint8_t *scratch) {
 if(QBH_ATTENTION_HEAD_DIM!=128U || cfg->k_zero_point<0 || cfg->k_zero_point>255) {
  qbh_attention_u8_pack_k_long_register(rows,valid,padded,cfg,weight,bias);return;
 }
 const unsigned div=1U<<cfg->score_shift,round=cfg->score_shift?div/2U:0U;
 const uint16_t cv=qbh_attention_u8_float_to_half_bits(512.0f/(float)div);
 const HVX_Vector sign=Q6_Vb_vsplat_R(128),offset=Q6_Vb_vsplat_R(cfg->k_zero_point-128);
 for(unsigned tile=0;tile<padded/32U;++tile) {
  for(unsigned half=0;half<2U;++half) {
   const unsigned first=tile*32U+half*16U;
   HVX_Vector v0;
   HVX_Vector v1;
   HVX_Vector v2;
   HVX_Vector v3;
   HVX_Vector v4;
   HVX_Vector v5;
   HVX_Vector v6;
   HVX_Vector v7;
   HVX_Vector v8;
   HVX_Vector v9;
   HVX_Vector v10;
   HVX_Vector v11;
   HVX_Vector v12;
   HVX_Vector v13;
   HVX_Vector v14;
   HVX_Vector v15;
   if(first+16U<=valid) {
    v0=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+0U)*128U),sign),offset);
    v1=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+1U)*128U),sign),offset);
    v2=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+2U)*128U),sign),offset);
    v3=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+3U)*128U),sign),offset);
    v4=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+4U)*128U),sign),offset);
    v5=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+5U)*128U),sign),offset);
    v6=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+6U)*128U),sign),offset);
    v7=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+7U)*128U),sign),offset);
    v8=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+8U)*128U),sign),offset);
    v9=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+9U)*128U),sign),offset);
    v10=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+10U)*128U),sign),offset);
    v11=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+11U)*128U),sign),offset);
    v12=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+12U)*128U),sign),offset);
    v13=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+13U)*128U),sign),offset);
    v14=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+14U)*128U),sign),offset);
    v15=Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+15U)*128U),sign),offset);
   } else {
    v0=first+0U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+0U)*128U),sign),offset):Q6_V_vzero();
    v1=first+1U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+1U)*128U),sign),offset):Q6_V_vzero();
    v2=first+2U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+2U)*128U),sign),offset):Q6_V_vzero();
    v3=first+3U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+3U)*128U),sign),offset):Q6_V_vzero();
    v4=first+4U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+4U)*128U),sign),offset):Q6_V_vzero();
    v5=first+5U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+5U)*128U),sign),offset):Q6_V_vzero();
    v6=first+6U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+6U)*128U),sign),offset):Q6_V_vzero();
    v7=first+7U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+7U)*128U),sign),offset):Q6_V_vzero();
    v8=first+8U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+8U)*128U),sign),offset):Q6_V_vzero();
    v9=first+9U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+9U)*128U),sign),offset):Q6_V_vzero();
    v10=first+10U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+10U)*128U),sign),offset):Q6_V_vzero();
    v11=first+11U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+11U)*128U),sign),offset):Q6_V_vzero();
    v12=first+12U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+12U)*128U),sign),offset):Q6_V_vzero();
    v13=first+13U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+13U)*128U),sign),offset):Q6_V_vzero();
    v14=first+14U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+14U)*128U),sign),offset):Q6_V_vzero();
    v15=first+15U<valid?Q6_Vb_vsub_VbVb_sat(Q6_V_vxor_VV(*(const HVX_Vector *)(rows+(size_t)(first+15U)*128U),sign),offset):Q6_V_vzero();
   }
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
  HVX_Vector sum0=Q6_V_vzero(),sum1=Q6_V_vzero();
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[16],((HVX_Vector *)scratch)[0],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[0]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[1]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[17],((HVX_Vector *)scratch)[1],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[16]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[17]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[18],((HVX_Vector *)scratch)[2],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[8]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[9]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[19],((HVX_Vector *)scratch)[3],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[24]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[25]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[20],((HVX_Vector *)scratch)[4],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[4]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[5]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[21],((HVX_Vector *)scratch)[5],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[20]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[21]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[22],((HVX_Vector *)scratch)[6],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[12]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[13]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[23],((HVX_Vector *)scratch)[7],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[28]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[29]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[24],((HVX_Vector *)scratch)[8],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[2]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[3]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[25],((HVX_Vector *)scratch)[9],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[18]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[19]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[26],((HVX_Vector *)scratch)[10],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[10]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[11]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[27],((HVX_Vector *)scratch)[11],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[26]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[27]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[28],((HVX_Vector *)scratch)[12],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[6]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[7]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[29],((HVX_Vector *)scratch)[13],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[22]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[23]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[30],((HVX_Vector *)scratch)[14],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[14]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[15]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  {
   HVX_VectorPair x=Q6_W_vshuff_VVR(((HVX_Vector *)scratch)[31],((HVX_Vector *)scratch)[15],-64);
   HVX_Vector lo=Q6_V_lo_W(x),hi=Q6_V_hi_W(x);
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[30]=lo;
   ((HVX_Vector *)(weight+(size_t)tile*4096U))[31]=hi;
   sum0=Q6_Vw_vadd_VwVw(sum0,Q6_Vw_vrmpy_VbVb(lo,Q6_Vb_vsplat_R(1)));
   sum1=Q6_Vw_vadd_VwVw(sum1,Q6_Vw_vrmpy_VbVb(hi,Q6_Vb_vsplat_R(1)));
  }
  HVX_Vector sum=Q6_Vw_vadd_VwVw(sum0,sum1);
  *(HVX_Vector *)(bias+tile*64U)=Q6_V_vsplat_R(cv);
  *(HVX_Vector *)(bias+tile*64U+32U)=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(128*div+round),Q6_Vw_vmpyi_VwRh(sum,cfg->q_zero_point*0x00010001U));
 }
 asm volatile("barrier" ::: "memory");
}
