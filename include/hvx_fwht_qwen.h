#ifndef QBH_HVX_FWHT_QWEN_H
#define QBH_HVX_FWHT_QWEN_H
/* Exact L32-0069 vector kernel, shared with fullmodel L32-0070. */
static const uint32_t bf69_lanes[32] __attribute__((aligned(128)))={
 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
static HVX_Vector bf69_splat(float x){uint32_t u;memcpy(&u,&x,4);return Q6_V_vsplat_R(u);}
static HVX_Vector bf69_mul(HVX_Vector a,HVX_Vector b){return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));}
static __attribute__((noinline)) void bf69_rows(float *x,uint32_t rows) {
 const HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t row=0;row<rows;row++) {
  float *v=x+row*QBH_BLOCK_INTERMEDIATE;
  for(uint32_t j=0;j<QBH_BLOCK_INTERMEDIATE;j+=32U) {
   HVX_Vector a=*(HVX_Vector *)(v+j);
   for(uint32_t h=1;h<32;h*=2U) {
    HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R((h*4U)*0x01010101U));
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(h)),Q6_V_vzero());
    a=Q6_V_vmux_QVV(upper,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));
   }
   *(HVX_Vector *)(v+j)=a;
  }
  for(uint32_t h=32;h<QBH_BLOCK_INTERMEDIATE/12U;h*=2U)
   for(uint32_t base=0;base<QBH_BLOCK_INTERMEDIATE;base+=2U*h)
    for(uint32_t j=0;j<h;j+=32U) {
     HVX_Vector a=*(HVX_Vector *)(v+base+j),b=*(HVX_Vector *)(v+base+h+j);
     *(HVX_Vector *)(v+base+j)=Q6_Vsf_vadd_VsfVsf(a,b);
     *(HVX_Vector *)(v+base+h+j)=Q6_Vsf_vsub_VsfVsf(a,b);
    }
  /* H12 mixes the twelve butterfly blocks; all12 inputs are loaded before stores. */
  for(uint32_t j=0;j<QBH_BLOCK_INTERMEDIATE/12U;j+=32U) {
   HVX_Vector src[12];
   for(uint32_t g=0;g<12U;g++)src[g]=*(HVX_Vector *)(v+g*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   for(uint32_t n=0;n<12U;n++) {
    HVX_Vector z=Q6_V_vzero();
    for(uint32_t g=0;g<12U;g++) z=qbh_r4_h12[n][g]>0?Q6_Vsf_vadd_VsfVsf(z,src[g]):Q6_Vsf_vsub_VsfVsf(z,src[g]);
    *(HVX_Vector *)(v+n*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
  }
 }
}
#endif
