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

/* L32-0071: six-stage normalized H64, exclusively HVX. Preserve the
 * post-RoPE FP16 and post-R3 FP16 boundaries of the existing R3 contract.
 * Scratch is a head-private slice of phase-dead Gate VTCM. */
#include "hvx_fwht8192.h"
static void qbh_llama_r3_quantize(const __fp16 *,uint8_t *,uint32_t,const struct qbh_block_qparam *);
static void qbh_llama_bf3_head(uint8_t *native,uint32_t rows,
 const struct qbh_block_qparam *qi,const struct qbh_block_qparam *qo,
 const __fp16 *co,const __fp16 *si,__fp16 *scratch) {
 qbh_llama_r3_prepare(native,rows,qi,co,si,scratch);
 const HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t r=0;r<rows;r++) {
  HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(*(HVX_Vector *)(scratch+r*64U));
  HVX_VectorPair order=Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);
  HVX_Vector a=Q6_V_lo_W(order),b=Q6_V_hi_W(order);
  for(uint32_t h=1;h<32;h*=2U) {
   HVX_Vector ctl=Q6_V_vsplat_R((h*4U)*0x01010101U);
   HVX_Vector aa=Q6_V_vdelta_VV(a,ctl),bb=Q6_V_vdelta_VV(b,ctl);
   HVX_VectorPred upper=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(h)),Q6_V_vzero());
   a=Q6_V_vmux_QVV(upper,Q6_Vsf_vsub_VsfVsf(aa,a),Q6_Vsf_vadd_VsfVsf(a,aa));
   b=Q6_V_vmux_QVV(upper,Q6_Vsf_vsub_VsfVsf(bb,b),Q6_Vsf_vadd_VsfVsf(b,bb));
  }
  HVX_Vector lo=bf69_mul(Q6_Vsf_vadd_VsfVsf(a,b),bf69_splat(.125f));
  HVX_Vector hi=bf69_mul(Q6_Vsf_vsub_VsfVsf(a,b),bf69_splat(.125f));
  HVX_VectorPair eo=Q6_W_vdeal_VVR(hi,lo,-4);
  *(HVX_Vector *)(scratch+r*64U)=Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(eo),Q6_V_hi_W(eo));
 }
 qbh_llama_r3_quantize(scratch,native,rows,qo);
}
