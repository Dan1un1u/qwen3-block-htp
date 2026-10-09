#ifndef QBH_HVX_FWHT_QWEN_H
#define QBH_HVX_FWHT_QWEN_H
/* Exact L32-0069 vector kernel, shared with fullmodel L32-0070. */
static const uint32_t bf69_lanes[32] __attribute__((aligned(128)))={
 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
static HVX_Vector bf69_splat(float x){uint32_t u;memcpy(&u,&x,4);return Q6_V_vsplat_R(u);}
static HVX_Vector bf69_mul(HVX_Vector a,HVX_Vector b){return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));}
static inline __attribute__((always_inline)) HVX_Vector bf329_local(HVX_Vector a,HVX_Vector lanes) {
 {HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R(4U*0x01010101U));
 HVX_VectorPred q=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(1)),Q6_V_vzero());
 a=Q6_V_vmux_QVV(q,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));}
 {HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R(8U*0x01010101U));
 HVX_VectorPred q=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(2)),Q6_V_vzero());
 a=Q6_V_vmux_QVV(q,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));}
 {HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R(16U*0x01010101U));
 HVX_VectorPred q=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(4)),Q6_V_vzero());
 a=Q6_V_vmux_QVV(q,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));}
 {HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R(32U*0x01010101U));
 HVX_VectorPred q=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(8)),Q6_V_vzero());
 a=Q6_V_vmux_QVV(q,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));}
 {HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R(64U*0x01010101U));
 HVX_VectorPred q=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(16)),Q6_V_vzero());
 a=Q6_V_vmux_QVV(q,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));}
 return a;
}
/* Register-resident radix stages preserve every individual add/sub.
 * Four-vector and sixteen-vector schedules amortize permutation controls
 * and remove intermediate VTCM passes without changing the H12 ordering. */
static __attribute__((noinline)) void bf329_blocks(float *v,uint32_t n,uint32_t vectors,HVX_Vector lanes) {
 if(vectors==4U) {
  for(uint32_t base=0;base<n;base+=128U) {
   HVX_Vector a0=bf329_local(*(HVX_Vector *)(v+base+0U),lanes);
   HVX_Vector a1=bf329_local(*(HVX_Vector *)(v+base+32U),lanes);
   HVX_Vector a2=bf329_local(*(HVX_Vector *)(v+base+64U),lanes);
   HVX_Vector a3=bf329_local(*(HVX_Vector *)(v+base+96U),lanes);
   {HVX_Vector t=a0;a0=Q6_Vsf_vadd_VsfVsf(t,a1);a1=Q6_Vsf_vsub_VsfVsf(t,a1);}
   {HVX_Vector t=a2;a2=Q6_Vsf_vadd_VsfVsf(t,a3);a3=Q6_Vsf_vsub_VsfVsf(t,a3);}
   {HVX_Vector t=a0;a0=Q6_Vsf_vadd_VsfVsf(t,a2);a2=Q6_Vsf_vsub_VsfVsf(t,a2);}
   {HVX_Vector t=a1;a1=Q6_Vsf_vadd_VsfVsf(t,a3);a3=Q6_Vsf_vsub_VsfVsf(t,a3);}
   *(HVX_Vector *)(v+base+0U)=a0;
   *(HVX_Vector *)(v+base+32U)=a1;
   *(HVX_Vector *)(v+base+64U)=a2;
   *(HVX_Vector *)(v+base+96U)=a3;
  }
 } else {
  for(uint32_t base=0;base<n;base+=256U) {
   HVX_Vector a0=bf329_local(*(HVX_Vector *)(v+base+0U),lanes);
   HVX_Vector a1=bf329_local(*(HVX_Vector *)(v+base+32U),lanes);
   HVX_Vector a2=bf329_local(*(HVX_Vector *)(v+base+64U),lanes);
   HVX_Vector a3=bf329_local(*(HVX_Vector *)(v+base+96U),lanes);
   HVX_Vector a4=bf329_local(*(HVX_Vector *)(v+base+128U),lanes);
   HVX_Vector a5=bf329_local(*(HVX_Vector *)(v+base+160U),lanes);
   HVX_Vector a6=bf329_local(*(HVX_Vector *)(v+base+192U),lanes);
   HVX_Vector a7=bf329_local(*(HVX_Vector *)(v+base+224U),lanes);
   {HVX_Vector t=a0;a0=Q6_Vsf_vadd_VsfVsf(t,a1);a1=Q6_Vsf_vsub_VsfVsf(t,a1);}
   {HVX_Vector t=a2;a2=Q6_Vsf_vadd_VsfVsf(t,a3);a3=Q6_Vsf_vsub_VsfVsf(t,a3);}
   {HVX_Vector t=a4;a4=Q6_Vsf_vadd_VsfVsf(t,a5);a5=Q6_Vsf_vsub_VsfVsf(t,a5);}
   {HVX_Vector t=a6;a6=Q6_Vsf_vadd_VsfVsf(t,a7);a7=Q6_Vsf_vsub_VsfVsf(t,a7);}
   {HVX_Vector t=a0;a0=Q6_Vsf_vadd_VsfVsf(t,a2);a2=Q6_Vsf_vsub_VsfVsf(t,a2);}
   {HVX_Vector t=a1;a1=Q6_Vsf_vadd_VsfVsf(t,a3);a3=Q6_Vsf_vsub_VsfVsf(t,a3);}
   {HVX_Vector t=a4;a4=Q6_Vsf_vadd_VsfVsf(t,a6);a6=Q6_Vsf_vsub_VsfVsf(t,a6);}
   {HVX_Vector t=a5;a5=Q6_Vsf_vadd_VsfVsf(t,a7);a7=Q6_Vsf_vsub_VsfVsf(t,a7);}
   {HVX_Vector t=a0;a0=Q6_Vsf_vadd_VsfVsf(t,a4);a4=Q6_Vsf_vsub_VsfVsf(t,a4);}
   {HVX_Vector t=a1;a1=Q6_Vsf_vadd_VsfVsf(t,a5);a5=Q6_Vsf_vsub_VsfVsf(t,a5);}
   {HVX_Vector t=a2;a2=Q6_Vsf_vadd_VsfVsf(t,a6);a6=Q6_Vsf_vsub_VsfVsf(t,a6);}
   {HVX_Vector t=a3;a3=Q6_Vsf_vadd_VsfVsf(t,a7);a7=Q6_Vsf_vsub_VsfVsf(t,a7);}
   *(HVX_Vector *)(v+base+0U)=a0;
   *(HVX_Vector *)(v+base+32U)=a1;
   *(HVX_Vector *)(v+base+64U)=a2;
   *(HVX_Vector *)(v+base+96U)=a3;
   *(HVX_Vector *)(v+base+128U)=a4;
   *(HVX_Vector *)(v+base+160U)=a5;
   *(HVX_Vector *)(v+base+192U)=a6;
   *(HVX_Vector *)(v+base+224U)=a7;
  }
 }
}
static __attribute__((noinline)) void bf329_h12(float *v) {
  /* H12 mixes the twelve butterfly blocks; all12 inputs are loaded before stores. */
  for(uint32_t j=0;j<QBH_BLOCK_INTERMEDIATE/12U;j+=32U) {
   HVX_Vector s0=*(HVX_Vector *)(v+0U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s1=*(HVX_Vector *)(v+1U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s2=*(HVX_Vector *)(v+2U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s3=*(HVX_Vector *)(v+3U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s4=*(HVX_Vector *)(v+4U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s5=*(HVX_Vector *)(v+5U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s6=*(HVX_Vector *)(v+6U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s7=*(HVX_Vector *)(v+7U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s8=*(HVX_Vector *)(v+8U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s9=*(HVX_Vector *)(v+9U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s10=*(HVX_Vector *)(v+10U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s11=*(HVX_Vector *)(v+11U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+0U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+1U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+2U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+3U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+4U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+5U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+6U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+7U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+8U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+9U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+10U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+11U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }

  }

}
static __attribute__((noinline)) void bf329_rows(float *x,uint32_t rows,uint32_t opt) {
 HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t r=0;r<rows;++r) {
  float *v=x+r*QBH_BLOCK_INTERMEDIATE;
  bf329_blocks(v,QBH_BLOCK_INTERMEDIATE,opt==7U?4U:8U,lanes);
  for(uint32_t h=opt==7U?128U:256U;h<QBH_BLOCK_INTERMEDIATE/12U;h*=2U)
   for(uint32_t base=0;base<QBH_BLOCK_INTERMEDIATE;base+=2U*h)
    for(uint32_t j=0;j<h;j+=32U) {
     HVX_Vector a=*(HVX_Vector *)(v+base+j),b=*(HVX_Vector *)(v+base+h+j);
     *(HVX_Vector *)(v+base+j)=Q6_Vsf_vadd_VsfVsf(a,b);
     *(HVX_Vector *)(v+base+h+j)=Q6_Vsf_vsub_VsfVsf(a,b);
    }
  bf329_h12(v);
 }
}
static __attribute__((noinline)) void bf69_rows_opt(float *x,uint32_t rows,uint32_t opt) {
 if(opt>=7U){bf329_rows(x,rows,opt);return;}
 const HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t row=0;row<rows;row++) {
  float *v=x+row*QBH_BLOCK_INTERMEDIATE;
  if(opt>=7U) {
   bf329_blocks(v,QBH_BLOCK_INTERMEDIATE,opt==7U?4U:16U,lanes);
   for(uint32_t h=opt==7U?128U:512U;h<QBH_BLOCK_INTERMEDIATE/12U;h*=2U)
    for(uint32_t base=0;base<QBH_BLOCK_INTERMEDIATE;base+=2U*h)
     for(uint32_t j=0;j<h;j+=32U) {
      HVX_Vector a=*(HVX_Vector *)(v+base+j),b=*(HVX_Vector *)(v+base+h+j);
      *(HVX_Vector *)(v+base+j)=Q6_Vsf_vadd_VsfVsf(a,b);
      *(HVX_Vector *)(v+base+h+j)=Q6_Vsf_vsub_VsfVsf(a,b);
     }
  } else {
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
  }
  /* H12 mixes the twelve butterfly blocks; all12 inputs are loaded before stores. */
  for(uint32_t j=0;j<QBH_BLOCK_INTERMEDIATE/12U;j+=32U) {
   HVX_Vector s0=*(HVX_Vector *)(v+0U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s1=*(HVX_Vector *)(v+1U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s2=*(HVX_Vector *)(v+2U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s3=*(HVX_Vector *)(v+3U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s4=*(HVX_Vector *)(v+4U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s5=*(HVX_Vector *)(v+5U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s6=*(HVX_Vector *)(v+6U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s7=*(HVX_Vector *)(v+7U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s8=*(HVX_Vector *)(v+8U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s9=*(HVX_Vector *)(v+9U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s10=*(HVX_Vector *)(v+10U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   HVX_Vector s11=*(HVX_Vector *)(v+11U*(QBH_BLOCK_INTERMEDIATE/12U)+j);
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+0U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+1U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+2U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+3U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+4U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+5U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+6U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+7U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vadd_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vsub_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+8U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vadd_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vsub_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vadd_VsfVsf(z,s11);
    *(HVX_Vector *)(v+9U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vadd_VsfVsf(z,s1);
    z=Q6_Vsf_vsub_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vadd_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vsub_VsfVsf(z,s8);
    z=Q6_Vsf_vadd_VsfVsf(z,s9);
    z=Q6_Vsf_vsub_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+10U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
#ifdef QBH_QWEN_06B
     0.018042195912175808f
#else
     0.01275775907699572f
#endif
    ));
   }
   { HVX_Vector z=Q6_V_vzero();
    z=Q6_Vsf_vadd_VsfVsf(z,s0);
    z=Q6_Vsf_vsub_VsfVsf(z,s1);
    z=Q6_Vsf_vadd_VsfVsf(z,s2);
    z=Q6_Vsf_vsub_VsfVsf(z,s3);
    z=Q6_Vsf_vsub_VsfVsf(z,s4);
    z=Q6_Vsf_vsub_VsfVsf(z,s5);
    z=Q6_Vsf_vadd_VsfVsf(z,s6);
    z=Q6_Vsf_vadd_VsfVsf(z,s7);
    z=Q6_Vsf_vadd_VsfVsf(z,s8);
    z=Q6_Vsf_vsub_VsfVsf(z,s9);
    z=Q6_Vsf_vadd_VsfVsf(z,s10);
    z=Q6_Vsf_vsub_VsfVsf(z,s11);
    *(HVX_Vector *)(v+11U*(QBH_BLOCK_INTERMEDIATE/12U)+j)=bf69_mul(z,bf69_splat(
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
