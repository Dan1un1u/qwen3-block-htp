/* L32-0070: full8192 FWHT, ordinary U8 Down. No HMX/INT16/SP2 rotation.
 * Gate/Up stream already wrote prequantization FP16 SwiGLU in act.
 * Two adjacent phase-dead expanded-weight slots hold FP32 [64,8192].
 * Three persistent pool workers plus caller own disjoint 16-row ranges.
 */
#include "hvx_fwht8192.h"
struct bf70_task {struct qbh_block_header *h;struct qbh_block_buffers *b;uint8_t *middle;uint32_t phase;float inv;float zp;};
static void qbh_bf70_worker(void *context,uint32_t id){
 struct bf70_task *t=context;uint32_t rows=t->h->logical_m,nr=rows==64U?16U:1U,first=rows==64U?id*16U:0U;
 float *x=(float *)t->b->expanded_weight;
 if(t->phase==0U){
  const __fp16 *act=(const __fp16 *)t->b->hmx_activation;
  for(uint32_t r=first;r<first+nr;r++)for(uint32_t a=0;a<16U;a++)for(uint32_t c=0;c<512U;c+=64U){
   uint32_t sr=a*rows+r;const __fp16 *s=act+qbh_r4_pos(sr&~1U,c,16U);
   HVX_VectorPair v=Q6_W_vdeal_VVR(*(const HVX_Vector *)(s+1024U),*(const HVX_Vector *)s,-2);
   HVX_VectorPair f=Q6_Wsf_vcvt_Vhf((sr&1U)?Q6_V_hi_W(v):Q6_V_lo_W(v));
   HVX_VectorPair o=Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);
   *(HVX_Vector *)(x+r*8192U+a*512U+c)=Q6_V_lo_W(o);*(HVX_Vector *)(x+r*8192U+a*512U+c+32U)=Q6_V_hi_W(o);
  }
 }else if(t->phase==1U){bf69_rows(x+first*8192U,nr);}
 else{
  for(uint32_t r=first;r<first+nr;r++)for(uint32_t c=0;c<8192U;c+=32U){
   HVX_Vector a=Q6_Vsf_vadd_VsfVsf(bf69_mul(*(HVX_Vector *)(x+r*8192U+c),bf69_splat(t->inv)),bf69_splat(t->zp));
   HVX_Vector w=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(a,bf69_splat(.5f)));
   HVX_Vector packed=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),w));
   size_t off=(size_t)(c/32U)*2048U+r*32U;uint32_t shift=off&127U;
   HVX_VectorPred pred=Q6_Q_and_QQn(Q6_Q_vsetq2_R(shift+32U),Q6_Q_vsetq_R(shift));
   Q6_vmem_QRIV(pred,(HVX_Vector *)(t->middle+off),Q6_V_vlalign_VVR(packed,packed,shift));
  }
 }
}
static int bf70_dispatch(struct qbh_block_w4f16_pool *pool,struct bf70_task *t){
 if(t->h->logical_m==1U){qbh_bf70_worker(t,0U);return 0;}
 if(!pool||pool->worker_count<3U||pool->active_worker_count)return -1;
 for(uint32_t i=0;i<3U;i++)if(pool->jobs[i].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
 pool->active_worker_count=3U;pool->extra_expand_worker_index=UINT32_MAX;
 for(uint32_t i=0;i<3U;i++){pool->jobs[i].r4_convert_context=t;pool->jobs[i].command_kind=QBH_BLOCK_HVX_POOL_R4_BUTTERFLY;}
 asm volatile("barrier":::"memory");for(uint32_t i=0;i<3U;i++)qurt_sem_up(&pool->command_ready[i]);
 qbh_bf70_worker(t,3U);qbh_w4f16_pool_wait(pool);pool->active_worker_count=0U;return 0;
}
static int qbh_run_butterfly_r4(struct qbh_block_header *h,uint8_t *shared,struct qbh_block_buffers *b,struct qbh_block_w4f16_pool *pool,uint8_t *middle){
 const uint32_t rows=h->logical_m;
 if(QBH_BLOCK_INTERMEDIATE!=8192U||QBH_LLAMA_SP2(h)||!QBH_FP32_RESIDUAL(h)||(rows!=1U&&rows!=64U)||
    (uintptr_t)b->expanded_weight_alt-(uintptr_t)b->expanded_weight<1048576U||
    (uintptr_t)b->hmx_output-(uintptr_t)b->expanded_weight<2097152U)return -1;
 struct bf70_task task={.h=h,.b=b,.middle=middle,.inv=1.0f/h->qparams[QBH_BLOCK_QP_MIDDLE].scale,.zp=(float)h->qparams[QBH_BLOCK_QP_MIDDLE].zero_point};
 if(rows==64U&&!b->r4_prefill_prepared)return -1;
 if(rows==1U&&!h->w4u8_decode_direct_n_gate_up_swiglu_stream)return -1;
 uint64_t start=HAP_perf_get_qtimer_count();
 /* Decode padding is required by the physical M64 carrier, only four rows are consumed. */
 if(rows==1U)for(uint32_t c=0;c<256U;c++)*(HVX_Vector *)(middle+c*2048U)=Q6_Vb_vsplat_R((int)task.zp);
 if(bf70_dispatch(pool,&task))return -1;h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-start;
 if(h->dense_r4_audit_offset)memcpy(shared+h->dense_r4_audit_offset,b->expanded_weight,rows*8192U*4U);
 task.phase=1U;start=HAP_perf_get_qtimer_count();if(bf70_dispatch(pool,&task))return -1;
 h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-start; /* exact core wall, HMX ticks remain zero */
 if(h->dense_r4_audit_offset)memcpy(shared+h->dense_r4_audit_offset+2097152U,b->expanded_weight,rows*8192U*4U);
 task.phase=2U;start=HAP_perf_get_qtimer_count();if(bf70_dispatch(pool,&task))return -1;
 h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-start;
 h->dense_r4_calls++;h->dense_r4_rows+=rows;return 0;
}
