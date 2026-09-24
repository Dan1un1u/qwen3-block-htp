/* L32-0063: fused vector DQ / floating SwiGLU / uniform INT16 Q.
 * Deliberately explicit IEEE-rounded operations define an independent CPU
 * reference. No scalar element loop, model LUT, or full floating tensor. */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "qhmath_hvx_vector.h"
#include "fp_island_swiglu.h"
static inline HVX_Vector fs(float x){union{float f;int i;}v={.f=x};return Q6_V_vsplat_R(v.i);}
static inline HVX_Vector fm(HVX_Vector a,HVX_Vector b){return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));}
static inline HVX_Vector load32(const uint8_t *p){
 HVX_Vector a=*(const HVX_Vector *)((uintptr_t)p&~(uintptr_t)127U);
 a=Q6_V_vror_VR(a,(uintptr_t)p&127U);
 return Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(a))));
}
static inline void store32(uint8_t *p,HVX_Vector w){
 HVX_Vector h=Q6_Vh_vpacke_VwVw(w,w),b=Q6_Vb_vpacke_VhVh(h,h);
 uint32_t s=(uintptr_t)p&127U;
 HVX_VectorPred q=Q6_Q_and_QQn(Q6_Q_vsetq2_R(s+32U),Q6_Q_vsetq_R(s));
 Q6_vmem_QRIV(q,(HVX_Vector *)p,Q6_V_vlalign_VVR(b,b,s));
}
static HVX_Vector exp_negative(HVX_Vector x){
 /* x in [-80,0]. Degree7 Taylor on [0,ln2), exact power-of-two scaling. */
 HVX_Vector nf=qhmath_hvx_vsf_floor_vsf(fm(x,fs(1.4426950408889634f)));
 HVX_Vector n=qhmath_hvx_vw_truncate_vsf(nf);
 HVX_Vector r=Q6_Vsf_vsub_VsfVsf(x,fm(nf,fs(0.6931471805599453f)));
 HVX_Vector p=fs(1.0f/5040.0f);
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f/720.0f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f/120.0f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f/24.0f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f/6.0f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(0.5f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f));
 p=Q6_Vsf_vadd_VsfVsf(fm(p,r),fs(1.0f));
 return fm(p,Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n,Q6_V_vsplat_R(127)),23));
}
void qbh_fp_island_swiglu(const uint8_t *g,const uint8_t *u,uint8_t *lo,
 uint8_t *hi,size_t elements,const float params[5]){
 const HVX_Vector zero=Q6_V_vzero(),one=fs(1.0f),half=fs(0.5f);
 for(size_t off=0;off<elements;off+=32U){
  HVX_Vector gf=fm(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(load32(g+off),Q6_V_vsplat_R((int)params[1]))),fs(params[0]));
  HVX_Vector uf=fm(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(load32(u+off),Q6_V_vsplat_R((int)params[3]))),fs(params[2]));
  HVX_Vector ag=Q6_V_vand_VV(gf,Q6_V_vsplat_R(0x7fffffff));
  ag=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(ag,fs(80.0f)),fs(80.0f),ag);
  HVX_Vector e=exp_negative(Q6_Vsf_vsub_VsfVsf(zero,ag));
  HVX_Vector den=Q6_Vsf_vadd_VsfVsf(one,e),inv=fs(0.75f);
  for(uint32_t j=0;j<5U;j++)inv=fm(inv,Q6_Vsf_vsub_VsfVsf(fs(2.0f),fm(den,inv)));
  HVX_Vector sig=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(zero,gf),fm(e,inv),inv);
  HVX_Vector v=fm(fm(fm(gf,sig),uf),fs(1.0f/params[4]));
  HVX_Vector av=Q6_V_vand_VV(v,Q6_V_vsplat_R(0x7fffffff));
  av=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(av,fs(32767.0f)),fs(32767.0f),av);
  HVX_Vector i=qhmath_hvx_vw_truncate_vsf(av);
  HVX_Vector frac=Q6_Vsf_vsub_VsfVsf(av,Q6_Vsf_equals_Vw(i));
  HVX_VectorPred gt=Q6_Q_vcmp_gt_VsfVsf(frac,half);
  HVX_VectorPred eq=Q6_Q_vcmp_eq_VwVw(frac,half);
  HVX_VectorPred odd=Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(i,Q6_V_vsplat_R(1)),Q6_V_vsplat_R(1));
  i=Q6_Vw_vadd_VwVw(i,Q6_V_vmux_QVV(Q6_Q_or_QQ(gt,Q6_Q_and_QQ(eq,odd)),Q6_V_vsplat_R(1),zero));
  i=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(zero,v),Q6_Vw_vsub_VwVw(zero,i),i);
  i=Q6_Vw_vadd_VwVw(i,Q6_V_vsplat_R(32768));
  store32(lo+off,i);store32(hi+off,Q6_Vuw_vlsr_VuwR(i,8));
 }
 asm volatile("barrier" ::: "memory");
}


/* EXP0312. Noinline passes force materialization of caller-owned VTCM values.
 * Exactly the same rounded HVX operations and quantization as the fused arm. */
__attribute__((noinline))
void qbh_fp_island_swiglu_phase(uint32_t phase,const uint8_t *g,const uint8_t *u,
 uint8_t *lo,uint8_t *hi,size_t elements,const float params[5],float *scratch){
 const HVX_Vector zero=Q6_V_vzero(),one=fs(1.0f),half=fs(0.5f);
 float *fg=scratch,*fu=scratch+elements;
 if(phase==0U){
  for(size_t off=0;off<elements;off+=32U){
   *(HVX_Vector *)(fg+off)=fm(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(load32(g+off),Q6_V_vsplat_R((int)params[1]))),fs(params[0]));
   *(HVX_Vector *)(fu+off)=fm(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(load32(u+off),Q6_V_vsplat_R((int)params[3]))),fs(params[2]));
  }
 }else if(phase==1U){
  for(size_t off=0;off<elements;off+=32U){
   HVX_Vector gf=*(const HVX_Vector *)(fg+off),uf=*(const HVX_Vector *)(fu+off);
  HVX_Vector ag=Q6_V_vand_VV(gf,Q6_V_vsplat_R(0x7fffffff));
  ag=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(ag,fs(80.0f)),fs(80.0f),ag);
  HVX_Vector e=exp_negative(Q6_Vsf_vsub_VsfVsf(zero,ag));
  HVX_Vector den=Q6_Vsf_vadd_VsfVsf(one,e),inv=fs(0.75f);
  for(uint32_t j=0;j<5U;j++)inv=fm(inv,Q6_Vsf_vsub_VsfVsf(fs(2.0f),fm(den,inv)));
  HVX_Vector sig=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(zero,gf),fm(e,inv),inv);

   *(HVX_Vector *)(fg+off)=fm(fm(gf,sig),uf);
  }
 }else{
  for(size_t off=0;off<elements;off+=32U){
   HVX_Vector v=fm(*(const HVX_Vector *)(fg+off),fs(1.0f/params[4]));
  HVX_Vector av=Q6_V_vand_VV(v,Q6_V_vsplat_R(0x7fffffff));
  av=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(av,fs(32767.0f)),fs(32767.0f),av);
  HVX_Vector i=qhmath_hvx_vw_truncate_vsf(av);
  HVX_Vector frac=Q6_Vsf_vsub_VsfVsf(av,Q6_Vsf_equals_Vw(i));
  HVX_VectorPred gt=Q6_Q_vcmp_gt_VsfVsf(frac,half);
  HVX_VectorPred eq=Q6_Q_vcmp_eq_VwVw(frac,half);
  HVX_VectorPred odd=Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(i,Q6_V_vsplat_R(1)),Q6_V_vsplat_R(1));
  i=Q6_Vw_vadd_VwVw(i,Q6_V_vmux_QVV(Q6_Q_or_QQ(gt,Q6_Q_and_QQ(eq,odd)),Q6_V_vsplat_R(1),zero));
  i=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(zero,v),Q6_Vw_vsub_VwVw(zero,i),i);
  i=Q6_Vw_vadd_VwVw(i,Q6_V_vsplat_R(32768));
  store32(lo+off,i);store32(hi+off,Q6_Vuw_vlsr_VuwR(i,8));
  }
 }
 asm volatile("barrier" ::: "memory");
}
