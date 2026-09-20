/* No new arena allocation: Q..Up are dead after SwiGLU joins.
 * Packed DMA uses lower halves; S8 worker tiles use upper halves of both
 * existing weight slots. Next DMA cannot touch live expanded tiles. */
static int sp60_down(struct qbh_block_header *h,const uint8_t *shared,
 struct qbh_block_buffers *b,struct qbh_block_w4f16_pool *p,const uint8_t *lo) {
 const uint32_t k=QBH_BLOCK_INTERMEDIATE, packed=128U*k/2U;
 const struct qbh_block_projection_desc *d=&h->projections[QBH_BLOCK_PROJ_DOWN];
 uint8_t *slots[2]={b->expanded_weight,b->expanded_weight_alt};
 uint8_t *bias[2]={b->sp2_scratch+1024U,b->sp2_scratch+2048U};
 struct qbh_dma_aligned_desc_1d dma[3] __attribute__((aligned(64)));
 uint32_t count=0U;
 if(!p || p->worker_count<3U || QBH_LLAMA_SP2(h)!=8U || !QBH_FP32_RESIDUAL(h) ||
 h->dense_r3_mode || h->dense_r4_mode || h->paper_format_disable ||
 (size_t)(b->middle-b->q)<64U*k*4U ||
 QBH_BLOCK_HIDDEN*512U < packed+64U*k) return -1;
 for(uint32_t i=0;i<p->worker_count;i++)if(p->jobs[i].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
 p->attention_buffers=b;p->sp60_rows=h->logical_m==1U?1U:64U;
 p->sp60_meta=(uint32_t *)b->q;p->sp60_output=(int32_t *)b->normalized;
 p->sp60_gather=(HVX_Vector *)b->sp2_scratch;
 for(uint32_t i=0;i<4U;i++)p->sp60_expanded[i]=(int8_t *)(slots[i/2U]+packed+(i%2U)*32U*k);
 const uint8_t *hi=h->logical_m==1U?lo+128U:b->sp2_high;
 for(uint32_t r=0;r<p->sp60_rows;r++)for(uint32_t c=0;c<k;c+=32U) {
  size_t off=(size_t)(c/32U)*2048U+r*32U;
  HVX_Vector v=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(sp60_bytes32(lo+off),
   Q6_Vw_vasl_VwR(sp60_bytes32(hi+off),8)),Q6_V_vsplat_R(32768));
  *(HVX_Vector *)(p->sp60_meta+(size_t)r*k+c)=sp60_meta(v);
 }
#ifdef QBH_LLAMA_3B
 b->down_prefetched=0U; /* Existing producer prefetch is counted, not hidden. */
#endif
 uint64_t t=HAP_perf_get_qtimer_count();
 int rc=qbh_dma_start_w4u8_direct_n_prefetch(dma,slots[0],shared+d->direct_n_weight_offset,
 packed,bias[0],shared+d->bias_offset,1024U,1U,&count);
 if(!rc)rc=qbh_dma_wait_w4u8_direct_n_prefetch(dma,count);
 h->weight_dma_ticks+=HAP_perf_get_qtimer_count()-t;
 h->weight_ddr_read_bytes+=packed+1024U;h->weight_dma_descriptor_count+=count;
 if(rc)return -1;
 for(uint32_t first=0,slot=0;first<QBH_BLOCK_HIDDEN;first+=128U,slot^=1U) {
  uint32_t next=first+128U;
  p->sp60_weight=slots[slot];p->sp60_scale=(const float *)bias[slot];p->sp60_first=first;
  p->active_worker_count=3U;
  for(uint32_t i=0;i<3U;i++)p->jobs[i].command_kind=QBH_BLOCK_HVX_POOL_SP2_SHIFTADD;
  asm volatile("barrier":::"memory");
  for(uint32_t i=0;i<3U;i++)qurt_sem_up(&p->command_ready[i]);
  if(next<QBH_BLOCK_HIDDEN) {
   t=HAP_perf_get_qtimer_count();
   rc=qbh_dma_start_w4u8_direct_n_prefetch(dma,slots[slot^1U],
    shared+d->direct_n_weight_offset+(size_t)next*k/2U,packed,bias[slot^1U],
    shared+d->bias_offset+(size_t)(next/32U)*256U,1024U,1U,&count);
   h->weight_dma_ticks+=HAP_perf_get_qtimer_count()-t;
   h->weight_ddr_read_bytes+=packed+1024U;h->weight_dma_descriptor_count+=count;
  }
  sp60_worker(p,3U);
  qbh_w4f16_pool_wait(p);p->active_worker_count=0U;
  if(rc)return -1;
  if(next<QBH_BLOCK_HIDDEN) {
   t=HAP_perf_get_qtimer_count();rc=qbh_dma_wait_w4u8_direct_n_prefetch(dma,count);
   h->weight_dma_ticks+=HAP_perf_get_qtimer_count()-t;
   if(rc)return -1;
  }
 }
 asm volatile("barrier":::"memory");
 return 0;
}
