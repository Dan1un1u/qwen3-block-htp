/* EXP0331: same FP32 full H12 x H512 arithmetic, no approximation.
 * OPT11 fuses physical FP16 preparation with the first eight radix stages.
 * OPT12 moves output quantization into H12, retaining four-row native stores.
 * OPT13 folds the final H512 add/sub into H12 input loads and directly writes
 * native INT16 byte planes. Input workspace remains immutable during H12.
 * Each worker owns 16 whole token rows; no shared scratch or new buffers.
 * Audit pointers capture the actual optimized raw input and normalized result.
 */
static inline __attribute__((always_inline)) HVX_VectorPair bf331_load64(
 const __fp16 *p,uint32_t odd) {
 HVX_VectorPair d=Q6_W_vdeal_VVR(*(const HVX_Vector *)(p+1024U),*(const HVX_Vector *)p,-2);
 HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(odd?Q6_V_hi_W(d):Q6_V_lo_W(d));
 return Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);
}
static __attribute__((noinline)) void bf331_prepare_blocks(struct bf70_task *t,uint32_t first,uint32_t nr) {
 const __fp16 *act=(const __fp16 *)t->b->hmx_activation;
 float *x=(float *)t->b->expanded_weight;
 HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t r=first;r<first+nr;++r)for(uint32_t a=0;a<12U;++a)for(uint32_t c=0;c<512U;c+=256U) {
  uint32_t sr=a*64U+r;
  HVX_VectorPair p0=bf331_load64(act+qbh_r4_pos(sr&~1U,c+0U,16U),sr&1U);
  HVX_Vector a0=Q6_V_lo_W(p0),a1=Q6_V_hi_W(p0);
  HVX_VectorPair p1=bf331_load64(act+qbh_r4_pos(sr&~1U,c+64U,16U),sr&1U);
  HVX_Vector a2=Q6_V_lo_W(p1),a3=Q6_V_hi_W(p1);
  HVX_VectorPair p2=bf331_load64(act+qbh_r4_pos(sr&~1U,c+128U,16U),sr&1U);
  HVX_Vector a4=Q6_V_lo_W(p2),a5=Q6_V_hi_W(p2);
  HVX_VectorPair p3=bf331_load64(act+qbh_r4_pos(sr&~1U,c+192U,16U),sr&1U);
  HVX_Vector a6=Q6_V_lo_W(p3),a7=Q6_V_hi_W(p3);
  if(t->audit_input) {
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+0U)=a0;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+32U)=a1;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+64U)=a2;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+96U)=a3;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+128U)=a4;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+160U)=a5;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+192U)=a6;
   *(HVX_Vector *)(t->audit_input+r*6144U+a*512U+c+224U)=a7;
  }
  a0=bf329_local(a0,lanes);
  a1=bf329_local(a1,lanes);
  a2=bf329_local(a2,lanes);
  a3=bf329_local(a3,lanes);
  a4=bf329_local(a4,lanes);
  a5=bf329_local(a5,lanes);
  a6=bf329_local(a6,lanes);
  a7=bf329_local(a7,lanes);
  {HVX_Vector z=a0;a0=Q6_Vsf_vadd_VsfVsf(z,a1);a1=Q6_Vsf_vsub_VsfVsf(z,a1);}
  {HVX_Vector z=a2;a2=Q6_Vsf_vadd_VsfVsf(z,a3);a3=Q6_Vsf_vsub_VsfVsf(z,a3);}
  {HVX_Vector z=a4;a4=Q6_Vsf_vadd_VsfVsf(z,a5);a5=Q6_Vsf_vsub_VsfVsf(z,a5);}
  {HVX_Vector z=a6;a6=Q6_Vsf_vadd_VsfVsf(z,a7);a7=Q6_Vsf_vsub_VsfVsf(z,a7);}
  {HVX_Vector z=a0;a0=Q6_Vsf_vadd_VsfVsf(z,a2);a2=Q6_Vsf_vsub_VsfVsf(z,a2);}
  {HVX_Vector z=a1;a1=Q6_Vsf_vadd_VsfVsf(z,a3);a3=Q6_Vsf_vsub_VsfVsf(z,a3);}
  {HVX_Vector z=a4;a4=Q6_Vsf_vadd_VsfVsf(z,a6);a6=Q6_Vsf_vsub_VsfVsf(z,a6);}
  {HVX_Vector z=a5;a5=Q6_Vsf_vadd_VsfVsf(z,a7);a7=Q6_Vsf_vsub_VsfVsf(z,a7);}
  {HVX_Vector z=a0;a0=Q6_Vsf_vadd_VsfVsf(z,a4);a4=Q6_Vsf_vsub_VsfVsf(z,a4);}
  {HVX_Vector z=a1;a1=Q6_Vsf_vadd_VsfVsf(z,a5);a5=Q6_Vsf_vsub_VsfVsf(z,a5);}
  {HVX_Vector z=a2;a2=Q6_Vsf_vadd_VsfVsf(z,a6);a6=Q6_Vsf_vsub_VsfVsf(z,a6);}
  {HVX_Vector z=a3;a3=Q6_Vsf_vadd_VsfVsf(z,a7);a7=Q6_Vsf_vsub_VsfVsf(z,a7);}
  *(HVX_Vector *)(x+r*6144U+a*512U+c+0U)=a0;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+32U)=a1;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+64U)=a2;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+96U)=a3;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+128U)=a4;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+160U)=a5;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+192U)=a6;
  *(HVX_Vector *)(x+r*6144U+a*512U+c+224U)=a7;
 }
}
static __attribute__((noinline)) void bf331_h12_words(struct bf70_task *t,uint32_t r) {
 float *v=(float *)t->b->expanded_weight+r*6144U;
 HVX_Vector norm=bf69_splat(0.01275775907699572f),inv=bf69_splat(t->inv);
 for(uint32_t j=0;j<512U;j+=32U) {
  uint32_t c=j;
  HVX_Vector s0=*(HVX_Vector *)(v+0U+j);
  HVX_Vector s1=*(HVX_Vector *)(v+512U+j);
  HVX_Vector s2=*(HVX_Vector *)(v+1024U+j);
  HVX_Vector s3=*(HVX_Vector *)(v+1536U+j);
  HVX_Vector s4=*(HVX_Vector *)(v+2048U+j);
  HVX_Vector s5=*(HVX_Vector *)(v+2560U+j);
  HVX_Vector s6=*(HVX_Vector *)(v+3072U+j);
  HVX_Vector s7=*(HVX_Vector *)(v+3584U+j);
  HVX_Vector s8=*(HVX_Vector *)(v+4096U+j);
  HVX_Vector s9=*(HVX_Vector *)(v+4608U+j);
  HVX_Vector s10=*(HVX_Vector *)(v+5120U+j);
  HVX_Vector s11=*(HVX_Vector *)(v+5632U+j);
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+0U+c)=z;
   *(HVX_Vector *)(v+0U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+512U+c)=z;
   *(HVX_Vector *)(v+512U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+1024U+c)=z;
   *(HVX_Vector *)(v+1024U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+1536U+c)=z;
   *(HVX_Vector *)(v+1536U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+2048U+c)=z;
   *(HVX_Vector *)(v+2048U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+2560U+c)=z;
   *(HVX_Vector *)(v+2560U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+3072U+c)=z;
   *(HVX_Vector *)(v+3072U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+3584U+c)=z;
   *(HVX_Vector *)(v+3584U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+4096U+c)=z;
   *(HVX_Vector *)(v+4096U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+4608U+c)=z;
   *(HVX_Vector *)(v+4608U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+5120U+c)=z;
   *(HVX_Vector *)(v+5120U+c)=qbh_r4_i16_words(z,inv);
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+5632U+c)=z;
   *(HVX_Vector *)(v+5632U+c)=qbh_r4_i16_words(z,inv);
  }
 }
}
static __attribute__((noinline)) void bf331_h12_direct(struct bf70_task *t,uint32_t r) {
 float *v=(float *)t->b->expanded_weight+r*6144U;
 HVX_Vector norm=bf69_splat(0.01275775907699572f),inv=bf69_splat(t->inv);
 for(uint32_t half=0;half<2U;++half)for(uint32_t j=0;j<256U;j+=32U) {
  uint32_t c=j+half*256U;
  HVX_Vector l0=*(HVX_Vector *)(v+0U+j),u0=*(HVX_Vector *)(v+256U+j);
  HVX_Vector s0=half?Q6_Vsf_vsub_VsfVsf(l0,u0):Q6_Vsf_vadd_VsfVsf(l0,u0);
  HVX_Vector l1=*(HVX_Vector *)(v+512U+j),u1=*(HVX_Vector *)(v+768U+j);
  HVX_Vector s1=half?Q6_Vsf_vsub_VsfVsf(l1,u1):Q6_Vsf_vadd_VsfVsf(l1,u1);
  HVX_Vector l2=*(HVX_Vector *)(v+1024U+j),u2=*(HVX_Vector *)(v+1280U+j);
  HVX_Vector s2=half?Q6_Vsf_vsub_VsfVsf(l2,u2):Q6_Vsf_vadd_VsfVsf(l2,u2);
  HVX_Vector l3=*(HVX_Vector *)(v+1536U+j),u3=*(HVX_Vector *)(v+1792U+j);
  HVX_Vector s3=half?Q6_Vsf_vsub_VsfVsf(l3,u3):Q6_Vsf_vadd_VsfVsf(l3,u3);
  HVX_Vector l4=*(HVX_Vector *)(v+2048U+j),u4=*(HVX_Vector *)(v+2304U+j);
  HVX_Vector s4=half?Q6_Vsf_vsub_VsfVsf(l4,u4):Q6_Vsf_vadd_VsfVsf(l4,u4);
  HVX_Vector l5=*(HVX_Vector *)(v+2560U+j),u5=*(HVX_Vector *)(v+2816U+j);
  HVX_Vector s5=half?Q6_Vsf_vsub_VsfVsf(l5,u5):Q6_Vsf_vadd_VsfVsf(l5,u5);
  HVX_Vector l6=*(HVX_Vector *)(v+3072U+j),u6=*(HVX_Vector *)(v+3328U+j);
  HVX_Vector s6=half?Q6_Vsf_vsub_VsfVsf(l6,u6):Q6_Vsf_vadd_VsfVsf(l6,u6);
  HVX_Vector l7=*(HVX_Vector *)(v+3584U+j),u7=*(HVX_Vector *)(v+3840U+j);
  HVX_Vector s7=half?Q6_Vsf_vsub_VsfVsf(l7,u7):Q6_Vsf_vadd_VsfVsf(l7,u7);
  HVX_Vector l8=*(HVX_Vector *)(v+4096U+j),u8=*(HVX_Vector *)(v+4352U+j);
  HVX_Vector s8=half?Q6_Vsf_vsub_VsfVsf(l8,u8):Q6_Vsf_vadd_VsfVsf(l8,u8);
  HVX_Vector l9=*(HVX_Vector *)(v+4608U+j),u9=*(HVX_Vector *)(v+4864U+j);
  HVX_Vector s9=half?Q6_Vsf_vsub_VsfVsf(l9,u9):Q6_Vsf_vadd_VsfVsf(l9,u9);
  HVX_Vector l10=*(HVX_Vector *)(v+5120U+j),u10=*(HVX_Vector *)(v+5376U+j);
  HVX_Vector s10=half?Q6_Vsf_vsub_VsfVsf(l10,u10):Q6_Vsf_vadd_VsfVsf(l10,u10);
  HVX_Vector l11=*(HVX_Vector *)(v+5632U+j),u11=*(HVX_Vector *)(v+5888U+j);
  HVX_Vector s11=half?Q6_Vsf_vsub_VsfVsf(l11,u11):Q6_Vsf_vadd_VsfVsf(l11,u11);
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+0U+c)=z;
   size_t off=(size_t)((0U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+512U+c)=z;
   size_t off=(size_t)((512U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+1024U+c)=z;
   size_t off=(size_t)((1024U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+1536U+c)=z;
   size_t off=(size_t)((1536U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+2048U+c)=z;
   size_t off=(size_t)((2048U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+2560U+c)=z;
   size_t off=(size_t)((2560U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+3072U+c)=z;
   size_t off=(size_t)((3072U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+3584U+c)=z;
   size_t off=(size_t)((3584U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+4096U+c)=z;
   size_t off=(size_t)((4096U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+4608U+c)=z;
   size_t off=(size_t)((4608U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+5120U+c)=z;
   size_t off=(size_t)((5120U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
  {HVX_Vector z=Q6_V_vzero();
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
   z=bf69_mul(z,norm);
   if(t->audit_rotated)*(HVX_Vector *)(t->audit_rotated+r*6144U+5632U+c)=z;
   size_t off=(size_t)((5632U+c)/32U)*2048U+r*32U;
   qbh_r4_i16_store32(t->middle,t->high,off,qbh_r4_i16_words(z,inv));
   /* Keep independent output sums from exceeding the register file. */
   asm volatile("" : "+v"(s0),"+v"(s1),"+v"(s2),"+v"(s3),"+v"(s4),"+v"(s5),"+v"(s6),"+v"(s7),"+v"(s8),"+v"(s9),"+v"(s10),"+v"(s11) : : "memory");
  }
 }
}
static __attribute__((noinline)) void bf331_tail(struct bf70_task *t,uint32_t first,uint32_t nr) {
 float *x=(float *)t->b->expanded_weight;
 for(uint32_t r=first;r<first+nr;++r) {
  float *v=x+r*6144U;
  if(t->h->dense_r4_optimization!=13U) {
   for(uint32_t base=0;base<6144U;base+=512U)for(uint32_t j=0;j<256U;j+=32U) {
    HVX_Vector a=*(HVX_Vector *)(v+base+j),b=*(HVX_Vector *)(v+base+j+256U);
    *(HVX_Vector *)(v+base+j)=Q6_Vsf_vadd_VsfVsf(a,b);
    *(HVX_Vector *)(v+base+j+256U)=Q6_Vsf_vsub_VsfVsf(a,b);
   }
  }
  if(t->h->dense_r4_optimization==13U)bf331_h12_direct(t,r);
  else if(t->h->dense_r4_optimization==12U)bf331_h12_words(t,r);
  else bf329_h12(v);
 }
}
