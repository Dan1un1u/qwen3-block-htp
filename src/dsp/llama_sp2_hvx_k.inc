/* Second implementation: SIMD across K, gather-unpack a native W4 tile
 * once, then reuse it for all M rows. Conversion is timed as compute work. */
static void sp59_unpack_k(const uint8_t *w,int8_t *dst,uint32_t k,HVX_Vector *scratch) {
 uint32_t offsets[32] __attribute__((aligned(128)));
 for(uint32_t j=0;j<32U;j++)offsets[j]=(j&15U)*128U;
 HVX_Vector base=*(HVX_Vector *)offsets,mask=Q6_V_vsplat_R(0x0f0f0f0f),eight=Q6_V_vsplat_R(0x08080808);
 for(uint32_t n=0;n<32U;n++) {
  HVX_Vector off=Q6_Vw_vadd_VwVw(base,Q6_V_vsplat_R(n*4U));
  for(uint32_t c=0;c<k;c+=128U) {
   Q6_vgather_ARMVw(scratch,(int32_t)(uintptr_t)(w+(size_t)c*16U),2047U,off);
   HVX_Vector p=*(volatile HVX_Vector *)scratch;
   HVX_Vector lo=Q6_V_vand_VV(p,mask),hi=Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(p,4),mask);
   lo=Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(lo,eight),eight);
   hi=Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(hi,eight),eight);
   HVX_VectorPair pairs=Q6_W_vshuff_VVR(hi,lo,-4);
   *(HVX_Vector *)(dst+(size_t)n*k+c)=Q6_V_lo_W(pairs);
  }
 }
}
static inline HVX_Vector sp59_shift_vector(HVX_Vector w,HVX_Vector meta) {
 HVX_Vector mask=Q6_V_vsplat_R(15),zero=Q6_V_vzero();
 HVX_Vector sign=Q6_Vw_vasr_VwR(meta,31);
 w=Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(w,sign),sign);
 HVX_Vector p=Q6_Vw_vasl_VwVw(w,Q6_V_vand_VV(meta,mask));
 HVX_Vector q=Q6_Vw_vasl_VwVw(w,Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(meta,8),mask));
 p=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(meta,Q6_V_vsplat_R(65536)),zero),zero,p);
 q=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(meta,Q6_V_vsplat_R(131072)),zero),zero,q);
 return Q6_Vw_vadd_VwVw(p,q);
}
static __attribute__((noinline)) void sp59_kernel_k(const int8_t *w,
 const uint32_t *meta,int32_t *out,uint32_t rows,uint32_t k,uint32_t n) {
 for(uint32_t oc=0;oc<32U;oc++)for(uint32_t r=0;r<rows;r++) {
  HVX_Vector a0=Q6_V_vzero();
  #pragma clang loop unroll(disable)
  #pragma clang loop pipeline(disable)
  for(uint32_t c=0;c<k;c+=128U) {
   HVX_VectorPair hh=Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(w+(size_t)oc*k+c));
   HVX_VectorPair l=Q6_Ww_vunpack_Vh(Q6_V_lo_W(hh)),h=Q6_Ww_vunpack_Vh(Q6_V_hi_W(hh));
   const HVX_Vector *e=(const HVX_Vector *)(meta+(size_t)r*k+c);
   a0=Q6_Vw_vadd_VwVw(a0,sp59_shift_vector(Q6_V_lo_W(l),e[0]));
   a0=Q6_Vw_vadd_VwVw(a0,sp59_shift_vector(Q6_V_hi_W(l),e[1]));
   a0=Q6_Vw_vadd_VwVw(a0,sp59_shift_vector(Q6_V_lo_W(h),e[2]));
   a0=Q6_Vw_vadd_VwVw(a0,sp59_shift_vector(Q6_V_hi_W(h),e[3]));
  }
  HVX_Vector sum=a0;
  for(uint32_t shift=64U;shift>=4U;shift>>=1)sum=Q6_Vw_vadd_VwVw(sum,Q6_V_vror_VR(sum,shift));
  out[(size_t)r*n+oc]=Q6_R_vextract_VR(sum,0);
 }
}
