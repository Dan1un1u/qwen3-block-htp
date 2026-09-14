/* L32-0019: explicit dense H64, inherited Qwen producer/HMX/consumer order.
 * Raw Gate and result Up are phase-dead; packed output uses the joined QKV
 * alternate weight arena so the existing FP32 HMX activation need not grow. */
static void qbh_llama_r3_pack(const __fp16 *src,__fp16 *dst,uint32_t rows,uint32_t unpack) {
 for(uint32_t r=0;r<rows;r+=2U) {
  size_t tile=qbh_hmx_fp16_matrix_tile_offset(r/32U,0U,2U)+(r%32U)/2U*64U;
  if(!unpack) {
   HVX_VectorPair v=Q6_W_vshuff_VVR(*(const HVX_Vector *)(src+(r+1U)*64U),*(const HVX_Vector *)(src+r*64U),-2);
   *(HVX_Vector *)(dst+tile)=Q6_V_lo_W(v);*(HVX_Vector *)(dst+tile+1024U)=Q6_V_hi_W(v);
  } else {
   HVX_VectorPair v=Q6_W_vdeal_VVR(*(const HVX_Vector *)(src+tile+1024U),*(const HVX_Vector *)(src+tile),-2);
   *(HVX_Vector *)(dst+r*64U)=Q6_V_lo_W(v);*(HVX_Vector *)(dst+(r+1U)*64U)=Q6_V_hi_W(v);
  }
 }
 asm volatile("barrier":::"memory");
}
static inline HVX_Vector qbh_llama_r3_quant_half(HVX_Vector x,HVX_Vector inv,HVX_Vector offset) {
 HVX_VectorPair v=Q6_Wsf_vcvt_Vhf(x);
 HVX_Vector a=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(qbh_fp32_mul(Q6_V_lo_W(v),inv),offset));
 HVX_Vector b=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(qbh_fp32_mul(Q6_V_hi_W(v),inv),offset));
 HVX_VectorPair order=Q6_W_vshuff_VVR(b,a,-4);
 return Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(order),Q6_V_lo_W(order));
}
static void qbh_llama_r3_quantize(const __fp16 *x,uint8_t *out,uint32_t rows,const struct qbh_block_qparam *q) {
 HVX_Vector inv=qbh_fp32_splat(1.0f/q->scale),offset=qbh_fp32_splat((float)q->zero_point+.5f);
 if(rows==64U)for(uint32_t r=0;r<64U;r+=4U) {
  HVX_Vector h0=qbh_llama_r3_quant_half(*(const HVX_Vector *)(x+r*64U),inv,offset);
  HVX_Vector h1=qbh_llama_r3_quant_half(*(const HVX_Vector *)(x+(r+1U)*64U),inv,offset);
  HVX_Vector h2=qbh_llama_r3_quant_half(*(const HVX_Vector *)(x+(r+2U)*64U),inv,offset);
  HVX_Vector h3=qbh_llama_r3_quant_half(*(const HVX_Vector *)(x+(r+3U)*64U),inv,offset);
  HVX_VectorPair native=Q6_W_vdeal_VVR(Q6_Vub_vpack_VhVh_sat(h3,h2),Q6_Vub_vpack_VhVh_sat(h1,h0),-32);
  *(HVX_Vector *)(out+r*32U)=Q6_V_lo_W(native);
  *(HVX_Vector *)(out+2048U+r*32U)=Q6_V_hi_W(native);
 } else {
  HVX_Vector packed=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),qbh_llama_r3_quant_half(*(const HVX_Vector *)x,inv,offset));
  for(uint32_t c=0;c<2;c++)Q6_vmem_QRIV(Q6_Q_vsetq_R(32U),(HVX_Vector *)(out+c*2048U),Q6_V_vror_VR(packed,c*32U));
 }
 asm volatile("barrier":::"memory");
}
static int qbh_run_dense_r3(struct qbh_block_header *h,uint8_t *shared,
 struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,uint32_t logical_rows) {
 const uint32_t heads=QBH_BLOCK_HEADS+QBH_BLOCK_KV_HEADS,rows=heads*logical_rows,padded=(rows+31U)&~31U;
 __fp16 *raw=(__fp16 *)b->gate,*result=(__fp16 *)b->up,*act=(__fp16 *)b->hmx_activation;
 __fp16 *weight=(__fp16 *)b->expanded_weight,*out=(__fp16 *)b->expanded_weight_alt;
 if(h->dense_r3_mode!=1U || h->dense_r3_optimization!=2U || (logical_rows!=1U && logical_rows!=64U))return -1;
 uint64_t start=HAP_perf_get_qtimer_count();
 qbh_llama_r3_pack(raw,act,padded,0U);
 /* H64 is the top-left dense submatrix of the immutable H128 table.
  * Copy four native tiles; never regenerate4096 scalar matrix entries. */
 for(uint32_t n=0;n<2U;n++)for(uint32_t k=0;k<2U;k++) {
  const HVX_Vector *src=(const HVX_Vector *)(qbh_r3_sign_matrix+(n*4U+k)*1024U);
  HVX_Vector *dst=(HVX_Vector *)(weight+(n*2U+k)*1024U);
  for(uint32_t v=0;v<16U;v++)dst[v]=src[v];
 }
 float scales[32] __attribute__((aligned(128)));for(uint32_t i=0;i<32;i++)scales[i]=.125f;
 qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
 h->dense_r3_prepare_ticks=HAP_perf_get_qtimer_count()-start;start=HAP_perf_get_qtimer_count();
 if(qbh_hmx_submit(worker,QBH_BLOCK_HMX_FP16,act,weight,b->channel_scale,out,padded/32U,2U,2U))return -1;
 h->hmx_command_count++;h->hmx_fp16_tile_pair_count+=(padded/32U)*4U;
 qbh_llama_r3_pack(out,result,padded,1U);
 h->dense_r3_matmul_ticks=HAP_perf_get_qtimer_count()-start;start=HAP_perf_get_qtimer_count();
 for(uint32_t head=0;head<heads;head++) {
  uint32_t k=head>=QBH_BLOCK_HEADS,idx=head-(k?QBH_BLOCK_HEADS:0U);
  qbh_llama_r3_quantize(result+head*logical_rows*64U,(k?b->k:b->q)+idx*4096U,logical_rows,&h->qparams[k?QBH_BLOCK_QP_K_ROPE:QBH_BLOCK_QP_Q_ROPE]);
  if(k && logical_rows==64U) {
   uint8_t *scratch=b->attention_concat+idx*QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
   qbh_attention_u8_pack_k_native(b->k+idx*4096U,&b->attention_configs[idx],(int8_t *)(scratch+QBH_ATTN_U8_K_WEIGHT_OFFSET),(uint32_t *)(scratch+QBH_ATTN_U8_QK_BIAS_OFFSET));
  }
 }
 h->dense_r3_finish_ticks=HAP_perf_get_qtimer_count()-start;
 h->dense_r3_rows=rows;h->dense_r3_hmx_calls=1U;h->dense_r3_refined_values=0;
 h->dense_r3_total_calls++;h->dense_r3_total_rows+=rows;h->dense_r3_total_hmx_calls++;
 h->dense_r3_total_prepare_ticks+=h->dense_r3_prepare_ticks;h->dense_r3_total_matmul_ticks+=h->dense_r3_matmul_ticks;h->dense_r3_total_finish_ticks+=h->dense_r3_finish_ticks;
 h->u8_attention_qk_norm_rope_ticks+=h->dense_r3_prepare_ticks+h->dense_r3_matmul_ticks+h->dense_r3_finish_ticks;
 if(h->dense_r3_audit_offset) {
  uint8_t *audit=shared+h->dense_r3_audit_offset;
  memcpy(audit,raw,rows*64U*2U);memcpy(audit+QBH_DENSE_R3_CARRIER_BYTES,result,rows*64U*2U);
  memcpy(audit+2U*QBH_DENSE_R3_CARRIER_BYTES,b->q,QBH_BLOCK_M*QBH_BLOCK_HIDDEN);
  memcpy(audit+2U*QBH_DENSE_R3_CARRIER_BYTES+QBH_BLOCK_M*QBH_BLOCK_HIDDEN,b->k,QBH_BLOCK_M*QBH_BLOCK_KV_HIDDEN);
 }
 return 0;
}
