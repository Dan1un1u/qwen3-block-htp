/* L32-0060: all tensor storage below is a dead-stage VTCM overlay. */
#include "llama_sp2_hvx_k.inc"
static HVX_Vector sp60_meta(HVX_Vector v) {
 HVX_Vector z=Q6_V_vzero(),s=Q6_Vw_vasr_VwR(v,31),one=Q6_V_vsplat_R(1);
 HVX_Vector a=Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v,s),s);
 HVX_Vector e=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(31),Q6_Vuw_vcl0_Vuw(a));
 HVX_Vector rem=Q6_Vw_vsub_VwVw(a,Q6_Vw_vasl_VwVw(one,e));
 HVX_Vector f=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(31),Q6_Vuw_vcl0_Vuw(rem));
 HVX_Vector flags=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(a,z),Q6_V_vsplat_R(65536),z);
 flags=Q6_V_vor_VV(flags,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(rem,z),Q6_V_vsplat_R(131072),z));
 return Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(e,Q6_V_vsplat_R(15)),
 Q6_Vw_vasl_VwR(Q6_V_vand_VV(f,Q6_V_vsplat_R(15)),8)),
 Q6_V_vor_VV(flags,Q6_V_vand_VV(s,Q6_V_vsplat_R(0x80000000U))));
}
static HVX_Vector sp60_bytes32(const uint8_t *p) {
 HVX_VectorPair h=Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)p);
 return Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(h)));
}
static void sp60_worker(struct qbh_block_w4f16_pool *p,uint32_t id) {
 const uint32_t k=QBH_BLOCK_INTERMEDIATE, rows=p->sp60_rows;
 const uint8_t *w=p->sp60_weight+(size_t)id*k*16U;
 int8_t *expanded=p->sp60_expanded[id];
 int32_t *acc=p->sp60_output+id*64U*32U;
 sp59_unpack_k(w,expanded,k,p->sp60_gather+id);
 sp59_kernel_k(expanded,p->sp60_meta,acc,rows,k,32U);
 HVX_Vector scale=*(const HVX_Vector *)(p->sp60_scale+id*64U);
 float *dst=(float *)p->attention_buffers->residual+p->sp60_first+id*32U;
 for(uint32_t r=0;r<rows;r++) {
  HVX_Vector a=Q6_Vsf_equals_Vw(*(HVX_Vector *)(acc+r*32U));
  HVX_Vector *d=(HVX_Vector *)(dst+r*QBH_BLOCK_HIDDEN);
  *d=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a,scale),*d));
 }
}
