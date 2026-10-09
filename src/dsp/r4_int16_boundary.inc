/* EXP0329: uniform signed INT16, nearest with ties away from zero.
 * Scale is frozen per layer; byte planes feed native packed W4 unchanged.
 * Stores own32 channels of one token, without an intermediate conversion. */
static HVX_Vector qbh_r4_i16_splat(float f) {uint32_t u;memcpy(&u,&f,4);return Q6_V_vsplat_R(u);}
static inline HVX_Vector qbh_r4_i16_words(HVX_Vector v,HVX_Vector inv) {
    v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v,inv));
    HVX_Vector sign=Q6_V_vand_VV(v,Q6_V_vsplat_R((int32_t)0x80000000U));
    HVX_Vector half=Q6_V_vor_VV(sign,qbh_r4_i16_splat(.5f));
    HVX_Vector w=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(v,half));
    return Q6_Vw_vmax_VwVw(Q6_V_vsplat_R(-32768),Q6_Vw_vmin_VwVw(w,Q6_V_vsplat_R(32767)));
}
static inline void qbh_r4_i16_store32(uint8_t *lo,uint8_t *hi,size_t off,HVX_Vector w) {
    HVX_Vector l=Q6_V_vand_VV(w,Q6_V_vsplat_R(255));
    HVX_Vector u=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(w,8),Q6_V_vsplat_R(128));
    l=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),l));
    u=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),u));
    uint32_t shift=off&127U;
    HVX_VectorPred pred=Q6_Q_and_QQn(Q6_Q_vsetq2_R(shift+32U),Q6_Q_vsetq_R(shift));
    Q6_vmem_QRIV(pred,(HVX_Vector *)(lo+off),Q6_V_vlalign_VVR(l,l,shift));
    Q6_vmem_QRIV(pred,(HVX_Vector *)(hi+off),Q6_V_vlalign_VVR(u,u,shift));
}
static inline void qbh_r4_i16_half64(uint8_t *lo,uint8_t *hi,size_t off,HVX_Vector hf,HVX_Vector inv) {
    HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(hf);
    HVX_VectorPair ordered=Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);
    qbh_r4_i16_store32(lo,hi,off,qbh_r4_i16_words(Q6_V_lo_W(ordered),inv));
    qbh_r4_i16_store32(lo,hi,off+2048U,qbh_r4_i16_words(Q6_V_hi_W(ordered),inv));
}

static inline HVX_Vector qbh_r4_i16_halfwords(HVX_Vector hf,HVX_Vector inv) {
 HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(hf);
 HVX_VectorPair o=Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);
 return Q6_Vh_vpack_VwVw_sat(qbh_r4_i16_words(Q6_V_hi_W(o),inv),qbh_r4_i16_words(Q6_V_lo_W(o),inv));
}
static inline HVX_Vector qbh_r4_i16_low_pair(HVX_Vector b,HVX_Vector a) {
 HVX_Vector mask=Q6_V_vsplat_R(0x00ff00ff);
 return Q6_Vub_vpack_VhVh_sat(Q6_V_vand_VV(b,mask),Q6_V_vand_VV(a,mask));
}
static inline HVX_Vector qbh_r4_i16_high_pair(HVX_Vector b,HVX_Vector a) {
 HVX_Vector bias=Q6_Vh_vsplat_R(128);
 return Q6_Vub_vpack_VhVh_sat(Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VhR(b,8),bias),Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VhR(a,8),bias));
}
