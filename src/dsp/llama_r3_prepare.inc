/* L32-0019: head64 post-RoPE FP16 producer, streamed after each Q/K head.
 * Llama has no Q/K norm. Dense R3 consumes this before the U8 RoPE boundary. */
static void qbh_llama_r3_prepare(const uint8_t *native,uint32_t rows,
 const struct qbh_block_qparam *qp,const __fp16 *cosine,const __fp16 *sine,__fp16 *out) {
 for(uint32_t row=0;row<rows;row++) {
  uint32_t shift=(row%4U)*32U;
  HVX_Vector a=*(const HVX_Vector *)(native+(row/4U)*128U);
  HVX_Vector b=*(const HVX_Vector *)(native+2048U+(row/4U)*128U);
  HVX_Vector bytes=Q6_V_vmux_QVV(Q6_Q_vsetq_R(32),Q6_V_vror_VR(a,shift),Q6_V_vror_VR(b,(shift+96U)%128U));
  HVX_VectorPair u=Q6_Wuh_vunpack_Vub(bytes);
  HVX_Vector half=Q6_Vhf_vcvt_Vh(Q6_Vh_vsub_VhVh(Q6_V_lo_W(u),Q6_Vh_vsplat_R(qp->zero_point)));
  HVX_VectorPair x=Q6_Wsf_vcvt_Vhf(half),y=Q6_Wsf_vcvt_Vhf(Q6_V_vror_VR(half,64));
  HVX_VectorPair co=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(cosine+row*64U));
  HVX_VectorPair si=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(sine+row*64U));
  HVX_Vector result[2],scale=qbh_fp32_splat(qp->scale);
  for(uint32_t part=0;part<2;part++) {
   HVX_Vector v=qbh_fp32_mul(part?Q6_V_hi_W(x):Q6_V_lo_W(x),scale);
   HVX_Vector w=qbh_fp32_mul(part?Q6_V_hi_W(y):Q6_V_lo_W(y),scale);
   v=qbh_fp32_mul(v,part?Q6_V_hi_W(co):Q6_V_lo_W(co));
   w=qbh_fp32_mul(w,part?Q6_V_hi_W(si):Q6_V_lo_W(si));
   result[part]=Q6_V_vmux_QVV(Q6_Q_vsetq_R(64),Q6_Vsf_vsub_VsfVsf(v,w),Q6_Vsf_vadd_VsfVsf(v,w));
  }
  *(HVX_Vector *)(out+row*64U)=Q6_Vhf_vcvt_VsfVsf(result[0],result[1]);
 }
 asm volatile("barrier":::"memory");
}
